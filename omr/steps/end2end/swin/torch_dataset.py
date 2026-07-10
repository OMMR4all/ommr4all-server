from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms as T

from omr.end2end.codec.dataset import SmartTokenizer
from omr.steps.end2end.sample_builder import Sample

DEFAULT_HEIGHT = 224


def preprocess_pil_image(img: Image.Image, target_height: int = DEFAULT_HEIGHT) -> torch.Tensor:
    """Scales to target height keeping aspect ratio and pads the width up to the next
    multiple of 32 (Swin requirement), matching swin_batch_predict.preprocess_image."""
    img = img.convert('RGB')
    w, h = img.size
    new_w = max(int(w * (target_height / h)), 32)
    remainder = new_w % 32
    if remainder != 0:
        new_w += 32 - remainder

    transform = T.Compose([
        T.Resize((target_height, new_w), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return transform(img)


def collate_end2end(batch):
    """Pads images to the batch-max width rounded up to a multiple of 32 and pads the
    target sequences with the <pad> token (0)."""
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    images, targets = zip(*batch)

    stride = 32
    max_w = max(img.shape[2] for img in images)
    target_w = ((max_w + stride - 1) // stride) * stride

    images_tensor = torch.stack([F.pad(img, (0, target_w - img.shape[2], 0, 0), value=0) for img in images])
    targets_tensor = pad_sequence(targets, batch_first=True, padding_value=0)
    return images_tensor, targets_tensor


class End2EndTorchDataset(TorchDataset):
    def __init__(self, samples: List[Sample], tokenizer: SmartTokenizer,
                 height: int = DEFAULT_HEIGHT, transform: Optional[object] = None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.height = height
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample.image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_tensor = preprocess_pil_image(image, self.height)
        target_ids = torch.tensor(self.tokenizer.encode(sample.gt), dtype=torch.long)
        return image_tensor, target_ids
