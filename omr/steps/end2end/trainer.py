import logging
import math
import os
import shutil
from abc import ABC
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from database import DatabaseBook
from database.file_formats.performance.pageprogress import Locks
from omr.dataset import DatasetParams
from omr.dataset.datafiles import EmptyDataSetException, LockState
from omr.end2end.codec.augmentation import get_train_transforms
from omr.end2end.codec.codec import codec_lines, merge_codec_lines, read_codec, write_codec
from omr.end2end.codec.dataset import SmartTokenizer
from omr.steps.algorithm import AlgorithmTrainer, AlgorithmTrainerParams, TrainerCallback

from .dataset import registered_class_specs
from .model_io import has_loc_feedback, load_pretrained, symbol_token_ids
from .networks import previous_symbol_locations
from .sample_builder import collect_unique_chars
from .torch_dataset import End2EndTorchDataset, collate_end2end

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
# weight of the localisation loss relative to the token cross entropy; 0.2 instead of 0.5 raised hSAR
# by ~3.5 points on the screening (docs/end2end-experiments.md, finding 13)
LOC_LOSS_WEIGHT = 0.2
GRAD_CLIP_NORM = 1.0
# learning rate of everything but the pretrained backbone relative to `l_rate`
HEAD_LR_FACTOR = 3.0
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
LABEL_SMOOTHING = 0.1
# the learning rate is multiplied by LR_DECAY after every LR_PATIENCE epochs without a new best model
LR_PATIENCE = 2
LR_DECAY = 0.3
# no LR decay and no early stopping during the first FULL_LR_EPOCHS epochs: the validation loss
# plateaus for several epochs before the localisation "clicks" (x error 20 % -> 5 % of the crop
# height); runs whose LR decayed inside that plateau ended up to 11 points hSAR lower (docs/end2end-experiments.md)
FULL_LR_EPOCHS = 25
# model selection / early stopping / LR decay on the validation token loss only (False) or on
# token + weighted localisation loss (True)
SELECT_WITH_LOC_LOSS = False
# feed the location of the previous symbol into the decoder (`networks.LocationFeedback`);
# on the 150-page screening it raised hSAR from 72.8 to 85.4 % (docs/end2end-experiments.md)
LOC_FEEDBACK = True
# std of the jitter on the fed-back GT locations during training, in units of the input height
LOC_FEEDBACK_NOISE = 0.02
# DataLoader workers pass every batch through /dev/shm (Docker default: 64 MB)
MIN_SHM_BYTES = 1 << 30


def loader_workers(requested: int) -> int:
    """`requested` data loader workers, or 0 if /dev/shm is too small to hold their batches;
    running out of shm crashes training with "unable to allocate shared memory"."""
    if requested <= 0:
        return 0
    try:
        free = shutil.disk_usage('/dev/shm').free
    except OSError:
        return requested
    if free < MIN_SHM_BYTES:
        logger.warning(f"Only {free >> 20} MB free in /dev/shm, loading data in the main process. "
                       f"Give the container more (docker: shm_size / --shm-size) to use {requested} workers.")
        return 0
    return requested


class End2EndTrainerBase(AlgorithmTrainer, ABC):
    """Trains any `End2EndNetwork`; the architecture comes from `meta().create_network`.

    A pretrained model (``params.load``) may have been trained with a different codec (other
    lyric characters, fewer symbol classes) and without the localisation head: the codec is
    extended rather than replaced and the weights are carried over token by token."""

    @staticmethod
    def required_locks() -> List[LockState]:
        # the GT sequence interleaves symbols and syllables; crops come from the layout (music block +
        # lyric line) and the symbol targets sit on the staff lines, so all of them must be final
        return [LockState(Locks.STAFF_LINES, True), LockState(Locks.LAYOUT, True),
                LockState(Locks.SYMBOLS, True), LockState(Locks.TEXT, True)]

    @staticmethod
    def default_params() -> AlgorithmTrainerParams:
        return AlgorithmTrainerParams(
            n_epoch=100,
            n_iter=100,  # = epochs; drives the client progress bar
            l_rate=1e-4,
            display=10,
            early_stopping_max_keep=5,
            processes=4,  # data loader workers
        )

    @classmethod
    def force_dataset_params(cls, params: DatasetParams):
        params.gt_required = True
        params.height = cls.meta().default_height()

    def _build_codec(self, codec_path: str, base_codec: Optional[List[str]]) -> SmartTokenizer:
        specs = registered_class_specs(self.settings.dataset_params.symbol_label_sets)
        unique_chars = collect_unique_chars(self.train_dataset.files + self.validation_dataset.files)
        lines = codec_lines(unique_chars, melody=False, class_specs=specs)
        if base_codec is not None:
            lines = merge_codec_lines(base_codec, lines)
        # the codec travels with the checkpoint so vocabulary and weights always match
        write_codec(codec_path, lines)
        return SmartTokenizer(codec_path)

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        base_model = self.params.model_to_load()
        base_state, base_codec_path = None, None
        if base_model:
            base_codec_path = base_model.local_file('codec.txt')
            try:
                base_state = torch.load(base_model.local_file('model_best.pth'), map_location='cpu')
            except FileNotFoundError as e:
                logger.warning(f"Could not load pretrained model {base_model.local_file('')}: {e}. "
                               f"Training from scratch.")
                base_state = None

        codec_path = self.settings.model.local_file('codec.txt')
        tokenizer = self._build_codec(codec_path, read_codec(base_codec_path) if base_state is not None else None)
        # a pretrained model with feedback keeps it
        loc_feedback = LOC_FEEDBACK or (base_state is not None and has_loc_feedback(base_state))
        model = self.meta().create_network(len(tokenizer), localization=True, pretrained_backbone=base_state is None,
                                           loc_feedback=loc_feedback)
        if base_state is not None:
            specs = {s.class_id: s for s in registered_class_specs(self.settings.dataset_params.symbol_label_sets)}
            try:
                fresh = load_pretrained(model, base_state, SmartTokenizer(base_codec_path), tokenizer, specs)
                logger.info(f"Fine-tuning {base_model.local_file('')}; newly initialised: "
                            f"{sorted({k.split('.')[0] for k in fresh}) or 'nothing'}")
            except (RuntimeError, KeyError) as e:
                # e.g. a checkpoint of another architecture
                logger.warning(f"Could not take over the pretrained model: {e}. Training from scratch.")
        model.to(device)

        symbol_ids = symbol_token_ids(tokenizer)
        symbol_id_tensor = torch.tensor(sorted(symbol_ids), device=device)
        height = self.settings.dataset_params.height
        workers = loader_workers(self.params.processes)
        # Explicitly forked: the task worker itself runs in a *spawned* process (restapi/operationworker);
        # spawned loader workers would each boot a fresh interpreter and re-import torch and the omr
        # stack. Forking is safe here, the loader workers only decode JPEGs and never touch CUDA.
        loader_args = dict(collate_fn=collate_end2end, batch_size=BATCH_SIZE, num_workers=workers,
                           persistent_workers=workers > 0, multiprocessing_context='fork' if workers > 0 else None)
        train_loader = DataLoader(
            End2EndTorchDataset(train_samples, tokenizer, height, symbol_ids, transform=get_train_transforms()),
            shuffle=True, **loader_args)
        val_loader = DataLoader(End2EndTorchDataset(val_samples, tokenizer, height, symbol_ids),
                                shuffle=False, **loader_args)

        # the pretrained backbone keeps the base rate, the freshly initialised rest learns faster
        backbone = [p for n, p in model.named_parameters() if n.startswith('backbone')]
        rest = [p for n, p in model.named_parameters() if not n.startswith('backbone')]
        optimizer = torch.optim.AdamW([dict(params=backbone, lr=self.params.l_rate),
                                       dict(params=rest, lr=self.params.l_rate * HEAD_LR_FACTOR)],
                                      weight_decay=WEIGHT_DECAY)
        base_lrs = [g['lr'] for g in optimizer.param_groups]
        warmup_steps = max(1, min(WARMUP_STEPS, len(train_loader)))
        lr_factor = 1.0   # lowered whenever the validation loss stops improving
        step = 0
        train_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD, label_smoothing=LABEL_SMOOTHING)
        val_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD)
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
            if device.type == 'cuda' else None
        scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)

        def run_batch(imgs, widths, tgts, locs, criterion, feedback_noise=0.0):
            imgs, widths, tgts, locs = imgs.to(device), widths.to(device), tgts.to(device), locs.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                enc = model.encode(imgs, widths)
                previous = None
                if model.has_loc_feedback:
                    # teacher forcing with the GT locations, jittered during training so that the
                    # decoder copes with the imperfect predicted ones at inference
                    previous = previous_symbol_locations(locs)[:, :-1]
                    previous = previous + feedback_noise * torch.randn_like(previous)
                hidden = model.decode(tgts[:, :-1], enc, previous)
                logits = model.logits(hidden)
                token_loss = criterion(logits.reshape(-1, logits.shape[-1]).float(), tgts[:, 1:].reshape(-1))

                # the state whose input is symbol token t localises symbol t
                loc_targets = locs[:, :-1]
                selected = ~torch.isnan(loc_targets[..., 0])
                loc_loss = torch.zeros((), device=device)
                loc_error, n_loc = 0.0, int(selected.sum())
                if model.has_localization and n_loc > 0:
                    loc_logits = model.localization_logits(hidden, enc)[selected]
                    valid = enc.loc_valid[:, None, :].expand(-1, hidden.shape[1], -1)[selected]
                    centers = model.cell_centers(enc)
                    head = model.loc_head
                    loc_loss = head.loss(loc_logits, valid, loc_targets[selected].float(), centers,
                                         head.spacing(enc.image_height))
                    with torch.no_grad():
                        w = enc.loc_keys.shape[-1]
                        predicted = head.locate(loc_logits.float(), centers, w)
                        loc_error = (predicted - loc_targets[selected]).abs()[:, 0].sum().item()
            return logits, token_loss, loc_loss, loc_error, n_loc

        n_epochs = self.params.n_epoch
        best_val_loss = float('inf')
        epochs_since_best = 0
        best_path = self.settings.model.local_file('model_best.pth')

        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0.0
            for imgs, widths, tgts, locs in train_loader:
                step += 1
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = base_lr * lr_factor * min(1.0, step / warmup_steps)
                optimizer.zero_grad(set_to_none=True)
                _, token_loss, loc_loss, _, _ = run_batch(imgs, widths, tgts, locs, train_criterion,
                                                          LOC_FEEDBACK_NOISE)
                loss = token_loss + LOC_LOSS_WEIGHT * loc_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / max(len(train_loader), 1)

            model.eval()
            total_token_loss, total_loc_loss = 0.0, 0.0
            correct_tokens, total_tokens = 0, 0
            correct_symbols, total_symbols = 0, 0
            loc_error_sum, n_loc_sum = 0.0, 0
            with torch.no_grad():
                for imgs, widths, tgts, locs in val_loader:
                    logits, token_loss, loc_loss, loc_error, n_loc = run_batch(imgs, widths, tgts, locs,
                                                                               val_criterion)
                    total_token_loss += token_loss.item()
                    total_loc_loss += loc_loss.item()
                    loc_error_sum += loc_error
                    n_loc_sum += n_loc
                    target = tgts[:, 1:].to(device)
                    mask = target != tokenizer.PAD
                    correct = logits.argmax(dim=-1) == target
                    correct_tokens += (correct & mask).sum().item()
                    total_tokens += mask.sum().item()
                    symbols = torch.isin(target, symbol_id_tensor)
                    correct_symbols += (correct & symbols).sum().item()
                    total_symbols += symbols.sum().item()
            n_val = max(len(val_loader), 1)
            # the localisation loss swings strongly between epochs; selecting on it would stop the
            # training (and decay the learning rate) while the transcription still improves
            val_loss = (total_token_loss + (LOC_LOSS_WEIGHT * total_loc_loss if SELECT_WITH_LOC_LOSS else 0)) / n_val
            val_acc = correct_tokens / max(total_tokens, 1)
            symbol_acc = correct_symbols / max(total_symbols, 1)
            # mean |dx| of the symbols in percent of the crop height
            x_error = 100 * loc_error_sum / n_loc_sum if n_loc_sum else math.nan

            logger.info(f"Epoch {epoch + 1}/{n_epochs}: train loss {avg_train_loss:.4f}, "
                        f"val token loss {total_token_loss / n_val:.4f}, val loc loss {total_loc_loss / n_val:.4f}, "
                        f"val token acc {val_acc:.4f} (symbols {symbol_acc:.4f}), "
                        f"val symbol x error {x_error:.2f}% of crop height, "
                        f"lr factor {lr_factor:g}")
            if callback:
                callback.next_iteration(epoch + 1, avg_train_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_best = 0
                torch.save(model.state_dict(), best_path)
                if callback:
                    callback.next_best_model(epoch + 1, val_acc, epochs_since_best)
            elif epoch + 1 > FULL_LR_EPOCHS:
                epochs_since_best += 1
                if 0 < self.params.early_stopping_max_keep <= epochs_since_best:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    if callback:
                        callback.early_stopping()
                    break
                if epochs_since_best % LR_PATIENCE == 0:
                    lr_factor *= LR_DECAY

        if not os.path.exists(best_path):
            torch.save(model.state_dict(), best_path)
