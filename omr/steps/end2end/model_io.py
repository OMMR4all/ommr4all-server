"""Loading end-to-end checkpoints and carrying weights over to a changed codec.

A model directory holds ``codec.txt`` (the vocabulary, one token per line), ``model_best.pth``
(the state dict) and ``dataset_params.json``. Models trained before the localisation head and
the registered symbol classes existed have the same files, so they load unchanged.
"""
import logging
from typing import Dict, List, Optional, Set

import torch

from omr.end2end.codec.codec import is_symbol_token, symbol_token
from omr.end2end.codec.dataset import SmartTokenizer
from omr.end2end.codec.parse import symbol_parse

from .networks import End2EndNetwork, LOC_FEEDBACK_PREFIX, LOC_HEAD_PREFIX

logger = logging.getLogger(__name__)


def symbol_token_ids(tokenizer: SmartTokenizer) -> Set[int]:
    return {i for t, i in tokenizer.token2idx.items() if is_symbol_token(t)}


def has_localization(state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(LOC_HEAD_PREFIX) for k in state)


def has_loc_feedback(state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(LOC_FEEDBACK_PREFIX) for k in state)


def base_token_of(token: str, class_specs: Dict) -> Optional[str]:
    """For a registered-class token the built-in token of its base class (same position and
    connection), which is the natural initialisation of the new embedding row."""
    if not is_symbol_token(token) or '|@' not in token:
        return None
    s = symbol_parse(token[1:-1], skip_gapped=True, class_specs=class_specs)
    return symbol_token(s, ())   # without class ids: the built-in token, None for 'other'


def load_pretrained(model: End2EndNetwork, state: Dict[str, torch.Tensor], old_tokenizer: SmartTokenizer,
                    new_tokenizer: SmartTokenizer, class_specs: Dict = None) -> List[str]:
    """Loads `state` (trained with `old_tokenizer`) into `model` (built for `new_tokenizer`).

    Vocabulary-sized parameters are remapped row by row via the token strings, so tokens keep
    their weights whatever their index. A token new to the codec starts from its base class's
    row if it is a registered-class token and keeps its random initialisation otherwise.
    Returns the parameter names left at their initialisation (e.g. the localisation head
    when the pretrained model has none)."""
    state = dict(state)
    own = model.state_dict()
    for name in model.VOCAB_PARAMS:
        old = state.pop(name)
        new = own[name].clone()
        n_copied = 0
        for token, new_i in new_tokenizer.token2idx.items():
            old_i = old_tokenizer.token2idx.get(token)
            if old_i is None:
                base = base_token_of(token, class_specs or {})
                old_i = old_tokenizer.token2idx.get(base) if base else None
            if old_i is not None and old_i < old.shape[0]:
                new[new_i] = old[old_i]
                n_copied += 1
        state[name] = new
        logger.info(f"{name}: {n_copied}/{new.shape[0]} rows taken over from the pretrained model")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Pretrained model has unknown parameters: {unexpected[:5]}...")
    # the localisation parts may be new: they are initialised freshly
    not_loc = [k for k in missing if not k.startswith((LOC_HEAD_PREFIX, LOC_FEEDBACK_PREFIX))]
    if not_loc:
        raise RuntimeError(f"Pretrained model lacks parameters: {not_loc[:5]}...")
    return missing
