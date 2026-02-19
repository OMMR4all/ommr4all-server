import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import math

from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.network import SwinTransformerOMR2d, SwinTransformerOMR


def get_args():
    parser = argparse.ArgumentParser(description="OMR Batch Prediction Script (Swin)")
    parser.add_argument("--input_dir", type=str, default="/home/alexanderh/unsloth2/images/",
                        help="Path to folder with input images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to save text files (Default: same as input)")
    parser.add_argument("--suffix", type=str, default="_pred_swin_1d_small",
                        help="Suffix for output files")
    parser.add_argument("--model_path", type=str,
                        default="/home/alexanderh/unsloth/swin/smallswin1ddata/swinarch1d_pis.pth",
                        help="Path to the trained .pth model")

    parser.add_argument("--codec_path", type=str, default="/home/alexanderh/unsloth/swin/smallswin1ddata/codec.txt",
                        help="Path to codec/vocabulary file")

    parser.add_argument("--image_height", type=int, default=224,
                        help="Height to scale the image to")
    parser.add_argument("--max_len", type=int, default=1500,
                        help="Max sequence length")
    return parser.parse_args()


def preprocess_image(image_path, target_height=224):
    """
    Loads image, scales to target_height, ensures width is divisible by 32,
    and normalizes for Swin Transformer.
    """
    try:
        img = Image.open(image_path).convert('RGB')

        w, h = img.size

        new_w = int(w * (target_height / h))

        remainder = new_w % 32
        if remainder != 0:
            new_w = new_w + (32 - remainder)

        transform = T.Compose([
            T.Resize((target_height, new_w), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),

        ])

        img_tensor = transform(img)
        return img_tensor

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def predict(model, image_tensor, tokenizer, device, max_len=1500):
    """
    Inference for a single image (Greedy Decoding).
    """
    model.eval()

    img = image_tensor.unsqueeze(0).to(device)

    sos_id = tokenizer.token2idx['<sos>']
    eos_id = tokenizer.token2idx['<eos>']

    curr_seq = torch.tensor([[sos_id]], device=device)

    predicted_ids = []

    with torch.no_grad():
        for _ in range(max_len):

            logits = model(img, curr_seq)

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            if next_token_id == eos_id:
                break

            predicted_ids.append(next_token_id)

            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            curr_seq = torch.cat([curr_seq, next_token_tensor], dim=1)

    decoded_string = tokenizer.decode(predicted_ids)
    return decoded_string


def main():
    args = get_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"ERROR: Input folder '{input_path}' does not exist.")
        return

    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(ext)))
    image_files.sort()

    if not image_files:
        print(f"No images found in '{input_path}'.")
        return

    print(f"Found: {len(image_files)} images. Starting processing...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    print(f"Loading Tokenizer from: {args.codec_path}")
    tokenizer = SmartTokenizer(args.codec_path)

    print(f"Loading Model from: {args.model_path}")
    try:

        model = SwinTransformerOMR(vocab_size=len(tokenizer))

        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")

        if "size mismatch" in str(e):
            print("Tip: Check if your 'codec.txt' matches the one used for training.")
        return

    try:
        iterator = tqdm(image_files, desc="Processing")
    except ImportError:
        iterator = image_files
        print("Tip: Install 'tqdm' for a progress bar.")

    success_count = 0

    for img_file in iterator:
        out_filename = f"{img_file.stem}{args.suffix}.txt"
        out_file_path = output_path / out_filename

        img_tensor = preprocess_image(img_file, target_height=args.image_height)

        if img_tensor is None:
            continue

        try:
            result = predict(model, img_tensor, tokenizer, device, max_len=args.max_len)

            with open(out_file_path, "w", encoding="utf-8") as f:
                f.write(result)

            success_count += 1

            if not isinstance(iterator, tqdm):
                print(f"Saved: {out_filename}")

        except Exception as e:
            print(f"\nError processing file {img_file.name}: {e}")

    print("\n" + "=" * 40)
    print(f"DONE. {success_count}/{len(image_files)} images processed.")
    print(f"Results saved to: {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
