from datetime import datetime
import time

import torch
from nltk import RegexpTokenizer

from omr.end2end.codec.augmentation import get_train_transforms
from omr.end2end.codec.dataset import OMRDataset, CharTokenizer, collate_fn, SmartTokenizer, OMRJsonlDataset
from omr.end2end.codec.network import ResNetTransformer
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

JSON_DATASET_PATH = "/home/alexanderh/unsloth/exp5_blocks/train_data.jsonl"
IMG_ROOT_DIR = "/home/alexanderh/unsloth/exp5_blocks/"
CODECS_PATH = "/home/alexanderh/unsloth/codec.txt"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
VAL_SPLIT = 0.1
IMAGE_HEIGHT = 128
LOG_FILE = "/tmp/training_log_n_a.txt"
SAVE_PATH = "/tmp/best_model_n_a.pth"


def log_print(msg, file_path):
    print(msg)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def predict_single2(model, image_tensor, tokenizer, device, max_len=150):
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)
    curr_seq = torch.tensor([[tokenizer.token2idx['<sos>']]], device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(img, curr_seq)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            if next_token.item() == tokenizer.token2idx['<eos>']:
                break
            curr_seq = torch.cat([curr_seq, next_token], dim=1)

    return tokenizer.decode(curr_seq[0])


def predict_single(model, image_tensor, tokenizer, device, max_len=1500):
    model.eval()
    img = image_tensor.unsqueeze(0).to(device)

    sos_id = tokenizer.token2idx['<sos>']
    eos_id = tokenizer.token2idx['<eos>']

    curr_seq = torch.tensor([[sos_id]], device=device)

    raw_ids = [sos_id]

    with torch.no_grad():
        for i in range(max_len):
            logits = model(img, curr_seq)

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            raw_ids.append(next_token_id)

            if next_token_id == eos_id:
                break

            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            curr_seq = torch.cat([curr_seq, next_token_tensor], dim=1)

    print(f"\n[DEBUG] Raw Predicted IDs: {raw_ids}")
    decoded = tokenizer.decode(raw_ids)
    print(f"[DEBUG] Decoded String: '{decoded}'")

    return decoded


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    with open(LOG_FILE, "w") as f:
        f.write(f"Training Started: {datetime.now()}\n")
        f.write(f"Device: {device}\n\n")

    tokenizer = SmartTokenizer(CODECS_PATH)
    print(f"Vocab Size: {len(tokenizer)}")

    full_dataset = OMRJsonlDataset(JSON_DATASET_PATH, IMG_ROOT_DIR, tokenizer, height=IMAGE_HEIGHT)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    log_print(f"Dataset Split -> Train: {train_size} | Val: {val_size}", LOG_FILE)
    train_aug = get_train_transforms()
    train_set.dataset.transform = train_aug

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = ResNetTransformer(len(tokenizer)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx['<pad>'])

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_train_loss = 0

        for i, (imgs, tgts) in enumerate(train_loader):
            imgs, tgts = imgs.to(device), tgts.to(device)

            optimizer.zero_grad()
            output = model(imgs, tgts[:, :-1])

            loss = criterion(output.reshape(-1, len(tokenizer)), tgts[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if i % 10 == 0:
                print(f"\rEpoch {epoch + 1} [Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f}", end="")

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                output = model(imgs, tgts[:, :-1])
                loss = criterion(output.reshape(-1, len(tokenizer)), tgts[:, 1:].reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        val_iter = iter(val_loader)
        sample_img, sample_tgt = next(val_iter)
        print(f"\n[DEBUG] Raw GT Tensor: {sample_tgt[0].tolist()}")
        pred_str = predict_single(model, sample_img[0], tokenizer, device)
        gt_str = tokenizer.decode(sample_tgt[0])

        epoch_time = time.time() - start_time
        log_msg = (f"\n\n=== End of Epoch {epoch + 1}/{EPOCHS} ({epoch_time:.1f}s) ===\n"
                   f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
                   f"--------------------------------------------------\n"
                   f"GT:   {gt_str[:300]}...\n"
                   f"PRED: {pred_str[:300]}...\n"
                   f"--------------------------------------------------")

        log_print(log_msg, LOG_FILE)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            log_print(f"--> New Best Model saved (Loss: {best_val_loss:.4f})", LOG_FILE)


if __name__ == "__main__":
    train()
