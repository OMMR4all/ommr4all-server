import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.network import ResNetTransformer, ResNetTransformerOld


def get_args():
    parser = argparse.ArgumentParser(description="OMR Batch Prediction Script")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Pfad zum Ordner mit den Eingabebildern")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Pfad zum Speicherort der Textdateien (Standard: gleicher Ordner wie Input)")
    parser.add_argument("--suffix", type=str, default="_pred_lm_sw",
                        help="Suffix für die Ausgabedatei (z.B. bild01.jpg -> bild01_pred_lm.txt)")

    parser.add_argument("--model_path", type=str, default="/home/alexanderh/unsloth/best_model3.pth",
                        help="Pfad zum trainierten Model (.pth)")
    parser.add_argument("--codec_path", type=str, default="/home/alexanderh/unsloth/codec.txt",
                        help="Pfad zur Codec/Vokabular Datei")
    parser.add_argument("--image_height", type=int, default=128,
                        help="Höhe, auf die das Bild skaliert wird")
    parser.add_argument("--max_len", type=int, default=1500,
                        help="Maximale Länge der vorhergesagten Sequenz")
    return parser.parse_args()


def preprocess_image(image_path, target_height=128):
    try:
        img = Image.open(image_path).convert('RGB')

        w, h = img.size

        new_w = int(w * (target_height / h))

        transform = T.Compose([
            T.Resize((target_height, new_w)),
            T.ToTensor(),

        ])

        img_tensor = transform(img)
        return img_tensor

    except Exception as e:
        print(f"Fehler beim Laden des Bildes {image_path}: {e}")
        return None


def predict(model, image_tensor, tokenizer, device, max_len=1500):
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
        print(f"ordner '{input_path}' existiert nicht.")
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
        print(f"Keine Bilder im Ordner '{input_path}' gefunden.")
        return

    print(f"{len(image_files)} Bilder")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = SmartTokenizer(args.codec_path)

    try:
        model = ResNetTransformer(len(tokenizer))
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"error beim Laden des Modells: {e}")
        return

    try:
        iterator = tqdm(image_files, desc="Verarbeite Bilder")
    except ImportError:
        iterator = image_files

    success_count = 0

    for img_file in iterator:

        print(f"{img_file}")

        out_filename = f"{img_file.stem}{args.suffix}.txt"
        out_file_path = output_path / out_filename

        img_tensor = preprocess_image(img_file, target_height=args.image_height)

        if img_tensor is None:
            continue

        try:

            result = predict(model, img_tensor, tokenizer, device, max_len=args.max_len)
            print(f"Result: {result}")

            with open(out_file_path, "w", encoding="utf-8") as f:
                f.write(result)

            success_count += 1

            if not isinstance(iterator, tqdm):
                print(f"Gespeichert: {out_filename}")

        except Exception as e:
            print(f"\nFehler bei Datei {img_file.name}: {e}")

    print("\n" + "=" * 40)
    print(f"{success_count}/{len(image_files)} Bilder erfolgreich verarbeitet.")
    print(f"gespeichert in: {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
