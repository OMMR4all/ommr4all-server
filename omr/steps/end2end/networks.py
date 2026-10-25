"""Networks of the end-to-end (image -> music + lyrics sequence) algorithms.

Every network is an image encoder plus an autoregressive transformer decoder over the codec
tokens (``omr.end2end.codec``). Optionally it carries a `PointerLocalizationHead` that
localises every emitted symbol token: the decoder state of the position whose *input* is the
symbol token queries a stride-8 feature map, and the soft-argmax of that attention is the
symbol position. The head adds no tokens, so the sequence format stays identical with and
without it, and checkpoints without the head (all models trained before it existed) keep
loading.

With `LocationFeedback` the decoder additionally sees, at every position, where the latest
symbol before it was localised, so it keeps track of its place on the staff.
"""
import math
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from omr.end2end.codec.network import PositionalEncoding, PositionEmbedding2D

LOC_HEAD_PREFIX = 'loc_head.'
LOC_FEEDBACK_PREFIX = 'loc_feedback.'


def previous_symbol_locations(locs: torch.Tensor) -> torch.Tensor:
    """`locs` (B, T, 2): the location of token t if it is a localised symbol, NaN otherwise.
    Returns (B, T, 2): for every t the location of the latest symbol strictly before t (NaN if
    there is none), i.e. what the decoder knows at position t during inference."""
    b, t, _ = locs.shape
    positions = torch.arange(t, device=locs.device).expand(b, t)
    latest = torch.where(~torch.isnan(locs[..., 0]), positions, -1).cummax(dim=1).values
    filled = torch.gather(locs.nan_to_num(), 1, latest.clamp_min(0)[..., None].expand(b, t, 2))
    filled = filled.masked_fill((latest < 0)[..., None], float('nan'))
    previous = torch.full_like(locs, float('nan'))
    previous[:, 1:] = filled[:, :-1]
    return previous


class LocationFeedback(nn.Module):
    """Embeds a location (x, y in units of the input height) into the decoder input; a learned
    vector stands in while no symbol has been localised yet (NaN)."""

    def __init__(self, d_model: int, n_frequencies: int = 16):
        super().__init__()
        # wavelengths from 1/50 to 30 image heights: fine enough for neighbouring noteheads,
        # unique over the longest crops
        wavelengths = torch.logspace(math.log10(0.02), math.log10(30), n_frequencies)
        self.register_buffer('frequencies', 2 * math.pi / wavelengths, persistent=False)
        self.proj = nn.Linear(4 * n_frequencies, d_model)
        self.unknown = nn.Parameter(torch.zeros(d_model))

    def forward(self, locs: torch.Tensor) -> torch.Tensor:
        """(..., 2) -> (..., d_model)"""
        known = ~torch.isnan(locs[..., 0])
        angles = locs.nan_to_num().float()[..., None] * self.frequencies        # (..., 2, F)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)   # (..., 4F)
        embedded = self.proj(features.to(self.proj.weight.dtype))
        return torch.where(known[..., None], embedded, self.unknown.to(embedded.dtype))


class EncodedImage(NamedTuple):
    memory: torch.Tensor                        # (B, L, d_model)
    memory_padding: Optional[torch.Tensor]      # (B, L), True = padded image area
    loc_keys: Optional[torch.Tensor]            # (B, d_model, h, w) stride-8 keys of the localisation head
    loc_valid: Optional[torch.Tensor]           # (B, h * w), False = padded image area
    image_height: int


def causal_mask(size: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def _project(mha: nn.MultiheadAttention, x: torch.Tensor, which: int) -> torch.Tensor:
    """The query (0), key (1) or value (2) input projection of `mha` applied to `x` (B, T, d)."""
    d = mha.embed_dim
    return F.linear(x, mha.in_proj_weight[which * d:(which + 1) * d], mha.in_proj_bias[which * d:(which + 1) * d])


def _heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, t, d = x.shape
    return x.view(b, t, num_heads, d // num_heads).transpose(1, 2)


def _attend(mha: nn.MultiheadAttention, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Attention of projected heads (B, h, T, d_h) followed by the output projection of `mha`
    (inference: no dropout). `mask` is True where attending is allowed."""
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    b, h, t, dh = out.shape
    return mha.out_proj(out.transpose(1, 2).reshape(b, t, h * dh))


def valid_columns(widths: Optional[torch.Tensor], n_cols: int, stride: int, batch: int, device) -> torch.Tensor:
    """(B, n_cols) True for feature columns that cover image content (not the batch padding)."""
    if widths is None:
        return torch.ones(batch, n_cols, dtype=torch.bool, device=device)
    cols = torch.arange(n_cols, device=device)[None, :]
    return cols * stride < widths.to(device)[:, None]


class PointerLocalizationHead(nn.Module):
    """Localises a decoder state on a feature map by attention (pointer) + soft-argmax.

    Positions are expressed in units of the input image height, i.e. (x / H, y / H).
    """

    def __init__(self, feature_dim: int, d_model: int, stride: int):
        super().__init__()
        self.stride = stride
        self.key_proj = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.key_pos = PositionEmbedding2D(d_model)
        self.query_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.scale = 1 / math.sqrt(d_model)

    def keys(self, features: torch.Tensor) -> torch.Tensor:
        return self.key_pos(self.key_proj(features))

    def forward(self, hidden: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Attention logits (B, T, h * w) of each decoder state over the feature cells, in float32
        even under autocast: the soft-argmax weights follow the logit differences closely."""
        with torch.autocast(hidden.device.type, enabled=False):
            q = self.query_proj(hidden.float())
            logits = torch.einsum('btd,bdn->btn', q, keys.float().flatten(2)) * self.scale
        return logits.masked_fill(~valid[:, None, :], torch.finfo(logits.dtype).min)

    def cell_centers(self, h: int, w: int, image_height: int, device) -> torch.Tensor:
        """(h * w, 2) centres (x, y) of the feature cells in units of the image height."""
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * self.stride / image_height
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * self.stride / image_height
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    def spacing(self, image_height: int) -> float:
        """Distance of neighbouring cell centres in units of the image height."""
        return self.stride / image_height

    def loss(self, logits: torch.Tensor, valid: torch.Tensor, targets: torch.Tensor, centers: torch.Tensor,
             spacing: float) -> torch.Tensor:
        """`logits` and `valid` (N, h * w) of N symbols with `targets` (N, 2) in image height units.

        Cross entropy against a Gaussian heat map (sigma = one cell) sharpens the attention onto
        the symbol; the L1 of the soft-argmax (in cells) refines it below the cell size."""
        log_p = F.log_softmax(logits.float(), dim=-1)
        d2 = ((centers[None, :, :] - targets[:, None, :]) ** 2).sum(-1)       # (N, h * w)
        heat = torch.exp(-d2 / (2 * spacing ** 2)) * valid
        heat = heat / heat.sum(-1, keepdim=True).clamp_min(1e-12)
        ce = -(heat * log_p.masked_fill(~valid, 0)).sum(-1)
        expected = expected_position(log_p.exp(), centers)
        l1 = (expected - targets).abs().sum(-1) / spacing
        return (ce + l1).mean()

    @staticmethod
    def locate(logits: torch.Tensor, centers: torch.Tensor, w: int, radius: int = 1) -> torch.Tensor:
        """Positions (..., 2) of attention `logits` (..., h * w): soft-argmax restricted to the
        (2 * radius + 1)^2 cells around the maximum, so a second, weaker mode cannot pull the
        estimate between two symbols."""
        best = logits.argmax(-1, keepdim=True)
        cells = torch.arange(logits.shape[-1], device=logits.device)
        near = ((cells // w - best // w).abs() <= radius) & ((cells % w - best % w).abs() <= radius)
        weights = F.softmax(logits.float().masked_fill(~near, float('-inf')), dim=-1)
        return expected_position(weights, centers)


def expected_position(weights: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """`weights` (..., h * w) @ `centers` (h * w, 2) in float32 even under autocast: in bfloat16 an x
    of several image heights would be rounded to 1/32 of the height or more, i.e. several pixels."""
    with torch.autocast(weights.device.type, enabled=False):
        return weights.float() @ centers.float()


class End2EndNetwork(nn.Module):
    """Common interface. Subclasses implement `encode`; the decoder, the token head and the
    optional localisation head are shared."""

    # parameters with one row per codec token, remapped when the codec changes
    VOCAB_PARAMS = ('embedding.weight', 'head.weight', 'head.bias')

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int,
                 loc_feature_dim: Optional[int], dropout: float = 0.1, loc_feedback: bool = False):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.loc_head = PointerLocalizationHead(loc_feature_dim, d_model, stride=8) \
            if loc_feature_dim is not None else None
        # feeds the location of the previous symbol back; needs the head that produces it
        self.loc_feedback = LocationFeedback(d_model) if loc_feedback and self.loc_head is not None else None

    @property
    def has_localization(self) -> bool:
        return self.loc_head is not None

    @property
    def has_loc_feedback(self) -> bool:
        return self.loc_feedback is not None

    def encode(self, img: torch.Tensor, widths: Optional[torch.Tensor] = None) -> EncodedImage:
        raise NotImplementedError()

    def decode(self, tgt: torch.Tensor, enc: EncodedImage, previous_locs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decoder states (B, T, d_model); position t has seen the tokens tgt[:, :t + 1] and, with
        location feedback, `previous_locs[:, t]` (see `previous_symbol_locations`)."""
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        if self.loc_feedback is not None:
            tgt_emb = tgt_emb + self.loc_feedback(previous_locs)
        return self.decoder(tgt_emb, enc.memory, tgt_mask=causal_mask(tgt.size(1), tgt.device),
                            tgt_key_padding_mask=(tgt == 0), memory_key_padding_mask=enc.memory_padding,
                            tgt_is_causal=True)

    def start_decoding(self, enc: EncodedImage) -> List[dict]:
        """Cache for `decode_next`: per decoder layer the projected keys / values of the image
        memory (computed once) and of the self attention (grown by every step)."""
        cache = []
        for layer in self.decoder.layers:
            mha = layer.multihead_attn
            cache.append(dict(mem_k=_heads(_project(mha, enc.memory, 1), mha.num_heads),
                              mem_v=_heads(_project(mha, enc.memory, 2), mha.num_heads),
                              self_k=None, self_v=None, position=0))
        return cache

    def decode_next(self, tokens: torch.Tensor, cache: List[dict], enc: EncodedImage,
                    previous_loc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference only: the decoder state (B, 1, d_model) of the next position, whose input
        is `tokens` (B,) (and, with location feedback, `previous_loc` (B, 2), the location of the
        latest symbol so far), equal to the last state of `decode` over all tokens so far.

        Nothing of the earlier positions is recomputed (see `start_decoding`), so a step costs
        time linear in the sequence length instead of re-running the decoder over the prefix."""
        position = cache[0]['position']
        x = self.embedding(tokens)[:, None, :] * math.sqrt(self.d_model) + \
            self.pos_encoder.pe[:, position:position + 1]
        if self.loc_feedback is not None:
            x = x + self.loc_feedback(previous_loc[:, None, :])
        memory_mask = None if enc.memory_padding is None else ~enc.memory_padding[:, None, None, :]
        for layer, c in zip(self.decoder.layers, cache):
            sa, ca = layer.self_attn, layer.multihead_attn
            k, v = _heads(_project(sa, x, 1), sa.num_heads), _heads(_project(sa, x, 2), sa.num_heads)
            c['self_k'] = k if c['self_k'] is None else torch.cat([c['self_k'], k], dim=2)
            c['self_v'] = v if c['self_v'] is None else torch.cat([c['self_v'], v], dim=2)
            c['position'] = position + 1
            x = layer.norm1(x + _attend(sa, _heads(_project(sa, x, 0), sa.num_heads), c['self_k'], c['self_v']))
            x = layer.norm2(x + _attend(ca, _heads(_project(ca, x, 0), ca.num_heads), c['mem_k'], c['mem_v'],
                                        memory_mask))
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return x

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def localization_logits(self, hidden: torch.Tensor, enc: EncodedImage) -> torch.Tensor:
        return self.loc_head(hidden, enc.loc_keys, enc.loc_valid)

    def cell_centers(self, enc: EncodedImage) -> torch.Tensor:
        _, _, h, w = enc.loc_keys.shape
        return self.loc_head.cell_centers(h, w, enc.image_height, enc.loc_keys.device)

    def forward(self, img: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.logits(self.decode(tgt, self.encode(img)))

    def _loc_keys(self, features: torch.Tensor, widths: Optional[torch.Tensor]):
        if self.loc_head is None:
            return None, None
        keys = self.loc_head.keys(features)
        b, _, h, w = keys.shape
        valid = valid_columns(widths, w, self.loc_head.stride, b, keys.device)
        return keys, valid[:, None, :].expand(b, h, w).reshape(b, h * w)


class SwinTransformerOMR(End2EndNetwork):
    """Swin-Tiny encoder (stride 32, 1D positional encoding over the flattened feature map).

    The attribute names match the checkpoints trained before the localisation head existed,
    which therefore load unchanged (`localization=False`) or as fine-tuning base."""

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, localization: bool = False,
                 loc_feedback: bool = False):
        super().__init__(vocab_size, d_model, nhead, num_layers, loc_feature_dim=192 if localization else None,
                         loc_feedback=loc_feedback)
        from transformers import SwinModel
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.feature_dim = 768
        self.adapter = nn.Linear(self.feature_dim, d_model)

    def encode(self, img: torch.Tensor, widths: Optional[torch.Tensor] = None) -> EncodedImage:
        outputs = self.backbone(img, output_hidden_states=self.has_localization)
        memory = self.pos_encoder(self.adapter(outputs.last_hidden_state))

        b, _, h_img, w_img = img.shape
        h32, w32 = math.ceil(h_img / 32), math.ceil(w_img / 32)
        padding = None
        if widths is not None and h32 * w32 == memory.shape[1]:
            valid = valid_columns(widths, w32, 32, b, img.device)
            padding = ~valid[:, None, :].expand(b, h32, w32).reshape(b, h32 * w32)

        loc_keys, loc_valid = self._loc_keys(outputs.reshaped_hidden_states[1], widths) \
            if self.has_localization else (None, None)
        return EncodedImage(memory, padding, loc_keys, loc_valid, h_img)


class ConvNeXtTransformerOMR(End2EndNetwork):
    """ConvNeXt-Tiny encoder truncated at stride 16 with a 2D sinusoidal positional encoding.

    Compared to `SwinTransformerOMR` the memory has twice the resolution in both directions
    (a 256 px crop yields 16 rows instead of 7) and keeps the 2D layout explicit, which
    matters for telling staff positions apart and for placing lyrics below their neumes.
    The stride-8 stage feeds the localisation head."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, localization: bool = True,
                 pretrained_backbone: bool = True, loc_feedback: bool = False):
        super().__init__(vocab_size, d_model, nhead, num_layers, loc_feature_dim=192 if localization else None,
                         loc_feedback=loc_feedback)
        import torchvision
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
        features = torchvision.models.convnext_tiny(weights=weights).features
        self.backbone_s8 = features[:4]     # stem + stage 1 + downsample + stage 2: stride 8, 192 ch
        self.backbone_s16 = features[4:6]   # downsample + stage 3: stride 16, 384 ch
        self.adapter = nn.Sequential(nn.Linear(384, d_model), nn.LayerNorm(d_model))
        self.memory_pos = PositionEmbedding2D(d_model)
        self.register_buffer('mean', torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)

    def encode(self, img: torch.Tensor, widths: Optional[torch.Tensor] = None) -> EncodedImage:
        f8 = self.backbone_s8((img - self.mean) / self.std)
        f16 = self.backbone_s16(f8)
        b, _, h16, w16 = f16.shape
        memory = self.adapter(f16.permute(0, 2, 3, 1))                       # (B, h, w, d)
        memory = self.memory_pos(memory.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        padding = None
        if widths is not None:
            valid = valid_columns(widths, w16, 16, b, img.device)
            padding = ~valid[:, None, :].expand(b, h16, w16).reshape(b, h16 * w16)
        loc_keys, loc_valid = self._loc_keys(f8, widths)
        return EncodedImage(memory, padding, loc_keys, loc_valid, img.shape[2])
