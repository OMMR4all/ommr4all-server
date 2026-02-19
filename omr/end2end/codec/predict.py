import argparse
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.network import ResNetTransformer, ResNetTransformerOld


def get_args():
    parser = argparse.ArgumentParser(description="OMR Prediction Script")
    parser.add_argument("--image_path", type=str, required=True, help="Pfad zum Eingabebild")
    parser.add_argument("--model_path", type=str, default="/tmp/best_model3.pth",
                        help="Pfad zum trainierten Model (.pth)")
    parser.add_argument("--codec_path", type=str, default="/tmp/unsloth/codec.txt",
                        help="Pfad zur Codec/Vokabular Datei")
    parser.add_argument("--image_height", type=int, default=128,
                        help="Höhe, auf die das Bild skaliert wird (muss mit Training übereinstimmen)")
    parser.add_argument("--max_len", type=int, default=10000, help="Maximale Länge der vorhergesagten Sequenz")
    return parser.parse_args()


def preprocess_image(image_path, target_height=128):
    """
    Lädt ein Bild, skaliert es auf target_height (unter Beibehaltung der Ratio),
    konvertiert es zu Graustufen (oder RGB je nach Modell) und normalisiert es.
    """
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
        print(f"Fehler beim Laden des Bildes: {e}")
        return None


def predict(model, image_tensor, tokenizer, device, max_len=1500):
    """
    Führt die Vorhersage für ein einzelnes Bild durch (Greedy Decoding).
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benutze Device: {device}")

    print(f"Lade Tokenizer von: {args.codec_path}")
    tokenizer = SmartTokenizer(args.codec_path)
    print(f"Vokabulargröße: {len(tokenizer)}")

    print(f"Lade Modell von: {args.model_path}")
    try:
        model = ResNetTransformer(len(tokenizer))
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("Modell erfolgreich geladen.")
    except FileNotFoundError:
        print(f"FEHLER: Modelldatei unter '{args.model_path}' nicht gefunden.")
        return
    except Exception as e:
        print(f"FEHLER beim Laden des Modells: {e}")
        return

    print(f"Verarbeite Bild: {args.image_path}")
    img_tensor = preprocess_image(args.image_path, target_height=args.image_height)

    if img_tensor is None:
        return

    print("Starte Inferenz...")
    try:
        result = predict(model, img_tensor, tokenizer, device, max_len=args.max_len)

        print("\n" + "=" * 40)
        print("VORHERSAGE ERGEBNIS:")
        print("=" * 40)
        print(result)
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"Ein Fehler ist während der Vorhersage aufgetreten: {e}")


if __name__ == "__main__":
    main()
