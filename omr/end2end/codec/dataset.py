import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import math
import random
import os


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3

        self.char2idx['<pad>'] = self.PAD
        self.char2idx['<sos>'] = self.SOS
        self.char2idx['<eos>'] = self.EOS
        self.char2idx['<unk>'] = self.UNK

        self.idx2char[self.PAD] = '<pad>'
        self.idx2char[self.SOS] = '<sos>'
        self.idx2char[self.EOS] = '<eos>'
        self.idx2char[self.UNK] = '<unk>'

    def build_vocab(self, sentences):
        """Scans all GT strings to build vocabulary"""
        unique_chars = set("".join(sentences))
        for i, char in enumerate(sorted(list(unique_chars))):
            idx = i + 4
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, text):
        return [self.SOS] + [self.char2idx.get(c, self.UNK) for c in text] + [self.EOS]

    def decode(self, tokens):
        res = []
        for t in tokens:
            if t == self.EOS: break
            if t == self.SOS or t == self.PAD: continue
            res.append(self.idx2char.get(t, '?'))
        return "".join(res)

    def __len__(self):
        return len(self.char2idx)


import re


class SmartTokenizer:
    def __init__(self, codecs_path):
        self.token2idx = {}
        self.idx2token = {}
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2
        self.UNK = 3

        self.specials = ['<pad>', '<sos>', '<eos>', '<unk>']
        for i, t in enumerate(self.specials):
            self.token2idx[t] = i
            self.idx2token[i] = t
        if ' ' not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[' '] = idx
            self.idx2token[idx] = ' '

        self.load_codecs(codecs_path)

        tokens_sorted = sorted(list(self.token2idx.keys()), key=len, reverse=True)
        tokens_escaped = [re.escape(t) for t in tokens_sorted if t not in self.specials]

        self.pattern = re.compile("|".join(tokens_escaped))

    def __len__(self):
        return len(self.token2idx)

    def load_codecs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            t = line.strip()
            if t and t not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[t] = idx
                self.idx2token[idx] = t

    def encode(self, text):

        matches = [m.group() for m in self.pattern.finditer(text)]

        return [self.token2idx['<sos>']] + \
            [self.token2idx.get(t, self.token2idx['<unk>']) for t in matches] + \
            [self.token2idx['<eos>']]

    def decode(self, ids):
        res = []
        for i in ids:
            if isinstance(i, torch.Tensor): i = i.item()
            if i == self.token2idx['<eos>']: break
            if i in [self.token2idx['<sos>'], self.token2idx['<pad>']]: continue

            if i in self.idx2token:
                res.append(self.idx2token[i])
            else:

                print(f"[Warnung] ID {i} hat kein Text-Pendant!")
                res.append(f"<{i}?>")

        return "".join(res)


class OMRJsonlDataset(Dataset):
    def __init__(self, jsonl_path, img_root_dir, tokenizer, height=128, transform=None):
        """
        Args:
            jsonl_path (str): Pfad zur .jsonl Datei (z.B. "data/train.jsonl")
            img_root_dir (str): Ordner, in dem der "images"-Unterordner liegt.
            tokenizer (SmartTokenizer): Dein initialisierter Tokenizer.
            height (int): Zielhöhe der Bilder (Standard 128px).
        """
        self.data = []
        self.img_root_dir = img_root_dir
        self.tokenizer = tokenizer
        self.height = height

        print(f"Lade Daten aus {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:

                if line.strip():
                    self.data.append(json.loads(line))
        print(f"-> {len(self.data)} Zeilen geladen.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        try:

            rel_path = entry['messages'][0]['content'][0]['image']

            gt_text = entry['messages'][1]['content'][0]['text']

        except (IndexError, KeyError) as e:
            print(f"Fehler bei Index {idx}: Struktur passt nicht zum Schema.")
            raise e

        full_img_path = os.path.join(self.img_root_dir, rel_path)

        try:
            image = Image.open(full_img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WARNUNG: Bild nicht gefunden: {full_img_path}")

            image = Image.new('RGB', (100, self.height), color=(0, 0, 0))
        if self.transform:
            image = self.transform(image)

        w, h = image.size
        new_w = int(w * (self.height / h))

        max_w = 2000
        if new_w > max_w:
            new_w = max_w

        image = image.resize((new_w, self.height), Image.BILINEAR)
        image_tensor = transforms.ToTensor()(image)

        target_ids = torch.tensor(self.tokenizer.encode(gt_text), dtype=torch.long)

        return image_tensor, target_ids


class OMRJsonlPageDataset(Dataset):
    def __init__(self, jsonl_path, img_root_dir, tokenizer, height=800, max_width=1280, transform=None):
        """
        Args:
            jsonl_path: Pfad zur .jsonl Datei.
            img_root_dir: Basisverzeichnis für Bilder.
            tokenizer: Tokenizer mit encode-Methode.
            height: Feste Zielhöhe für Swin (z.B. 800).
            max_width: Die maximale Breite (aus dem Skript ermittelt), auf die gepaddet wird.
            transform: Optionale Albumentations oder torchvision Transforms.
        """
        self.data = []
        self.img_root_dir = img_root_dir
        self.tokenizer = tokenizer
        self.height = height
        self.max_width = max_width
        self.transform = transform

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        rel_path = entry['messages'][0]['content'][0]['image']
        gt_text = entry['messages'][1]['content'][0]['text']
        full_img_path = os.path.join(self.img_root_dir, rel_path)

        try:
            image = Image.open(full_img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (self.max_width, self.height), color=(0, 0, 0))

        w, h = image.size
        scale = self.height / h
        new_w = min(int(w * scale), self.max_width)
        image = image.resize((new_w, self.height), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        image = transforms.ToTensor()(image)

        c, curr_h, curr_w = image.shape
        if curr_w < self.max_width:
            pad_w = self.max_width - curr_w

            image = torch.nn.functional.pad(image, (0, pad_w, 0, 0), value=0)

        target_ids = torch.tensor(self.tokenizer.encode(gt_text), dtype=torch.long)

        return image, target_ids


class OMRDataset(Dataset):
    def __init__(self, data_list, tokenizer, height=128, transform=None):
        """
        data_list: List of tuples (image_path, gt_text)
        """
        self.data = data_list
        self.tokenizer = tokenizer
        self.height = height
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text = self.data[idx]

        image = self._generate_dummy_image(text)

        w, h = image.size
        new_w = int(w * (self.height / h))
        image = image.resize((new_w, self.height), Image.BILINEAR)

        image_tensor = transforms.ToTensor()(image)

        target_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        return image_tensor, target_ids

    def _generate_dummy_image(self, text):
        width = 400 + random.randint(0, 200)
        img = Image.new('RGB', (width, 200), color=(random.randint(200, 255), 200, 200))
        return img


def collate_fn(batch):
    """
    Pad images to max width in batch
    Pad targets to max length in batch
    """
    images, targets = zip(*batch)

    max_w = max([img.shape[2] for img in images])
    padded_imgs = []
    for img in images:
        diff = max_w - img.shape[2]
        padded_img = F.pad(img, (0, 0, diff, 0), fill=0)
        padded_imgs.append(padded_img)
    images_tensor = torch.stack(padded_imgs)

    from torch.nn.utils.rnn import pad_sequence
    targets_tensor = pad_sequence(targets, batch_first=True, padding_value=0)

    return images_tensor, targets_tensor


def collate_fn_page(batch):
    """
    Pads images to the maximum width in the batch (must be multiple of 32).
    Pads targets to the maximum length in the batch.
    """
    images, targets = zip(*batch)

    current_max_w = max([img.shape[2] for img in images])

    stride = 32
    target_w = ((current_max_w + stride - 1) // stride) * stride

    padded_imgs = []
    for img in images:
        diff_w = target_w - img.shape[2]

        padded_img = F.pad(img, (0, diff_w, 0, 0), value=0)
        padded_imgs.append(padded_img)

    images_tensor = torch.stack(padded_imgs)

    from torch.nn.utils.rnn import pad_sequence

    targets_tensor = pad_sequence(targets, batch_first=True, padding_value=0)

    return images_tensor, targets_tensor


if "__main__" == __name__:
    tokenizer = SmartTokenizer("/tmp/unsloth/codec.txt")

    test_str = "*[(clef|f|7)] po[(note|0|8|2)]"
    ids = tokenizer.encode(test_str)
    dex = tokenizer.decode(ids)
    print(f"IDs: {ids}")
    print(f"DEX: {dex}")
