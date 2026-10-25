from typing import Collection, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms as T
from torchvision import tv_tensors

from omr.end2end.codec.dataset import SmartTokenizer
from omr.steps.end2end.sample_builder import Sample

WIDTH_STRIDE = 32


def preprocess_pil_image(img: Image.Image, target_height: int) -> torch.Tensor:
    """Scales to target height keeping aspect ratio and stretches the width up to the next
    multiple of 32 (Swin requirement), matching swin_batch_predict.preprocess_image."""
    img = img.convert('RGB')
    w, h = img.size
    new_w = max(int(w * (target_height / h)), WIDTH_STRIDE)
    remainder = new_w % WIDTH_STRIDE
    if remainder != 0:
        new_w += WIDTH_STRIDE - remainder

    transform = T.Compose([
        T.Resize((target_height, new_w), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return transform(img)


def collate_end2end(batch):
    """Pads images to the batch-max width (zeros), target sequences with the <pad> token (0)
    and localisation targets with NaN. Returns (images, widths, targets, locs)."""
    images, targets, locs = zip(*batch)

    max_w = max(img.shape[2] for img in images)
    target_w = ((max_w + WIDTH_STRIDE - 1) // WIDTH_STRIDE) * WIDTH_STRIDE

    images_tensor = torch.stack([F.pad(img, (0, target_w - img.shape[2], 0, 0), value=0) for img in images])
    widths = torch.tensor([img.shape[2] for img in images], dtype=torch.long)
    targets_tensor = pad_sequence(targets, batch_first=True, padding_value=0)
    locs_tensor = pad_sequence(locs, batch_first=True, padding_value=float('nan'))
    return images_tensor, widths, targets_tensor, locs_tensor


class End2EndTorchDataset(TorchDataset):
    """Items are (image (3, H, W), token ids (L,), locs (L, 2)).

    `locs[i]` is the position of the symbol that token i encodes, in units of the network
    input height (x / H, y / H); NaN for non-symbol tokens and for symbols that left the image."""

    def __init__(self, samples: List[Sample], tokenizer: SmartTokenizer, height: int,
                 symbol_token_ids: Collection[int], transform: Optional[object] = None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.height = height
        self.symbol_token_ids = set(symbol_token_ids)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample.image()
        w, h = image.size
        points = torch.tensor([[fx * w, fy * h] for fx, fy in sample.symbol_locs], dtype=torch.float32).reshape(-1, 2)
        if self.transform:
            image, points = self.transform(image, tv_tensors.KeyPoints(points, canvas_size=(h, w)))
            points = points.as_subclass(torch.Tensor)

        image_tensor = preprocess_pil_image(image, self.height)
        target_ids = torch.tensor(self.tokenizer.encode(sample.gt), dtype=torch.long)

        locs = torch.full((len(target_ids), 2), float('nan'))
        symbol_positions = [i for i, t in enumerate(target_ids.tolist()) if t in self.symbol_token_ids]
        if len(symbol_positions) == len(points):
            inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
            # fractions of the image -> units of the (stretched) network input height
            scaled = torch.stack([points[:, 0] / w * image_tensor.shape[2] / self.height, points[:, 1] / h], dim=-1)
            scaled[~inside] = float('nan')
            locs[symbol_positions] = scaled
        return image_tensor, target_ids, locs
