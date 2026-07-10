import logging
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from database import DatabaseBook
from omr.dataset import DatasetParams
from omr.dataset.datafiles import EmptyDataSetException
from omr.end2end.codec.augmentation import get_train_transforms
from omr.end2end.codec.codec import gen_codec
from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.network import SwinTransformerOMR
from omr.steps.algorithm import AlgorithmTrainerParams, AlgorithmTrainerSettings, AlgorithmMeta, TrainerCallback

from ..sample_builder import collect_unique_chars
from ..trainer import End2EndTrainerBase
from .torch_dataset import End2EndTorchDataset, collate_end2end, DEFAULT_HEIGHT

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


class SwinTrainer(End2EndTrainerBase):
    @staticmethod
    def meta() -> Type['AlgorithmMeta']:
        from .meta import Meta
        return Meta

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_epoch=100,
            n_iter=100,  # = epochs; drives the client progress bar
            l_rate=1e-4,
            display=10,
            early_stopping_max_keep=5,
            processes=1,
        )

    @staticmethod
    def force_dataset_params(params: DatasetParams):
        params.gt_required = True
        params.height = DEFAULT_HEIGHT

    def _train(self, target_book: Optional[DatabaseBook] = None, callback: Optional[TrainerCallback] = None):
        if callback:
            callback.resolving_files()

        train_samples = self.train_dataset.to_end2end_samples(callback)
        val_samples = self.validation_dataset.to_end2end_samples(callback)
        if len(train_samples) == 0:
            raise EmptyDataSetException()
        if len(val_samples) == 0:
            val_samples = train_samples
        logger.info(f"End2End training on {len(train_samples)} train / {len(val_samples)} val samples")

        # The codec travels with the checkpoint so vocabulary and weights always match.
        codec_path = self.settings.model.local_file('codec.txt')
        unique_chars = collect_unique_chars(self.train_dataset.files + self.validation_dataset.files)
        gen_codec(codec_path, unique_chars, melody=False)
        tokenizer = SmartTokenizer(codec_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SwinTransformerOMR(vocab_size=len(tokenizer)).to(device)

        if self.params.model_to_load():
            load_path = self.params.model_to_load().local_file('model_best.pth')
            try:
                model.load_state_dict(torch.load(load_path, map_location=device))
                logger.info(f"Loaded pretrained weights from {load_path}")
            except (FileNotFoundError, RuntimeError) as e:
                # RuntimeError covers vocab-size mismatches with a differing codec
                logger.warning(f"Could not load pretrained model {load_path}: {e}. Training from scratch.")

        train_loader = DataLoader(
            End2EndTorchDataset(train_samples, tokenizer, height=self.settings.dataset_params.height,
                                transform=get_train_transforms()),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_end2end)
        val_loader = DataLoader(
            End2EndTorchDataset(val_samples, tokenizer, height=self.settings.dataset_params.height),
            batch_size=1, shuffle=False, collate_fn=collate_end2end)

        optimizer = optim.Adam(model.parameters(), lr=self.params.l_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD)

        n_epochs = self.params.n_epoch
        best_val_loss = float('inf')
        epochs_since_best = 0

        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0
            for imgs, tgts in train_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                optimizer.zero_grad()
                output = model(imgs, tgts[:, :-1])
                loss = criterion(output.reshape(-1, len(tokenizer)), tgts[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / max(len(train_loader), 1)

            model.eval()
            total_val_loss = 0
            correct_tokens = 0
            total_tokens = 0
            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    output = model(imgs, tgts[:, :-1])
                    loss = criterion(output.reshape(-1, len(tokenizer)), tgts[:, 1:].reshape(-1))
                    total_val_loss += loss.item()

                    pred = output.argmax(dim=-1)
                    mask = tgts[:, 1:] != tokenizer.PAD
                    correct_tokens += ((pred == tgts[:, 1:]) & mask).sum().item()
                    total_tokens += mask.sum().item()
            avg_val_loss = total_val_loss / max(len(val_loader), 1)
            val_acc = correct_tokens / max(total_tokens, 1)

            logger.info(f"Epoch {epoch + 1}/{n_epochs}: train loss {avg_train_loss:.4f}, "
                        f"val loss {avg_val_loss:.4f}, val token acc {val_acc:.4f}")
            if callback:
                callback.next_iteration(epoch + 1, avg_train_loss, val_acc)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_since_best = 0
                torch.save(model.state_dict(), self.settings.model.local_file('model_best.pth'))
                if callback:
                    callback.next_best_model(epoch + 1, val_acc, epochs_since_best)
            else:
                epochs_since_best += 1
                if 0 < self.params.early_stopping_max_keep <= epochs_since_best:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    if callback:
                        callback.early_stopping()
                    break


if __name__ == '__main__':
    import os
    import random
    import numpy as np

    random.seed(1)
    np.random.seed(1)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ommr4all.settings')
    import django

    django.setup()

    from database.file_formats.performance.pageprogress import Locks
    from omr.dataset.datafiles import dataset_by_locked_pages, LockState

    book = DatabaseBook('Graduel_Part_1_gt')
    train_pcgts, val_pcgts = dataset_by_locked_pages(
        0.8, [LockState(Locks.SYMBOLS, True), LockState(Locks.TEXT, True)], True, [book])

    settings = AlgorithmTrainerSettings(
        DatasetParams(gt_required=True),
        train_pcgts,
        val_pcgts,
    )
    trainer = SwinTrainer(settings)
    trainer.train(book)
