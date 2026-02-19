import torch
import torch.nn as nn
import torchvision.models as models
import math

import torch
import torch.nn as nn
import torchvision.models as models
import math
from transformers import SwinModel
from transformers import SwinModel, SwinConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionEmbedding2D(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=500):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model muss durch 2 teilbar sein für 2D Encoding")

        self.d_model = d_model
        dim_t = d_model // 2

        div_term = torch.exp(torch.arange(0, dim_t, 2).float() * (-math.log(10000.0) / dim_t))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """
        x: Input Tensor der Form (Batch, C, H, W)
        """
        b, c, h, w = x.shape

        pe_y = torch.zeros(h, w, self.d_model // 2, device=x.device)

        pos_y = torch.arange(0, h, dtype=torch.float, device=x.device).unsqueeze(1)

        freq_y = pos_y * self.div_term

        freq_y = freq_y.unsqueeze(1)

        pe_y[:, :, 0::2] = torch.sin(freq_y)
        pe_y[:, :, 1::2] = torch.cos(freq_y)

        pe_x = torch.zeros(h, w, self.d_model // 2, device=x.device)

        pos_x = torch.arange(0, w, dtype=torch.float, device=x.device).unsqueeze(1)

        freq_x = pos_x * self.div_term

        freq_x = freq_x.unsqueeze(0)

        pe_x[:, :, 0::2] = torch.sin(freq_x)
        pe_x[:, :, 1::2] = torch.cos(freq_x)

        pe_2d = torch.cat([pe_y, pe_x], dim=2).permute(2, 0, 1).unsqueeze(0)

        return x + pe_2d


class SwinTransformerOMR2d(nn.Module):
    def __init__(self, vocab_size, d_model=256):
        super().__init__()

        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        self.adapter = nn.Linear(768, d_model)

        self.pos_encoder_2d = PositionEmbedding2D(d_model)

        self.pos_encoder_text = PositionalEncoding(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def forward(self, img, tgt):

        outputs = self.backbone(img)
        last_hidden_state = outputs.last_hidden_state

        memory = self.adapter(last_hidden_state)

        B, L, C = memory.shape
        H_feat = img.shape[2] // 32
        W_feat = img.shape[3] // 32

        if H_feat * W_feat == L:

            memory_2d = memory.permute(0, 2, 1).view(B, C, H_feat, W_feat)

            memory_2d = self.pos_encoder_2d(memory_2d)

            memory_seq = memory_2d.flatten(2).permute(0, 2, 1)
        else:

            memory_seq = memory

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder_text(tgt_emb)

        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)) * float('-inf'), diagonal=1).to(tgt.device)
        pad_mask = (tgt == 0)

        out = self.decoder(tgt_emb, memory_seq, tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)

        return self.head(out)


class SwinTransformerOMR(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()

        print("Lade Swin Transformer Backbone...")
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        self.feature_dim = 768

        self.adapter = nn.Linear(self.feature_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def forward(self, img, tgt):
        outputs = self.backbone(img)

        features = outputs.last_hidden_state

        memory = self.adapter(features)

        memory = self.pos_encoder(memory)

        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)) * float('-inf'), diagonal=1).to(tgt.device)
        pad_mask = (tgt == 0)

        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)
        return self.head(out)


class ResNetTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dropout=0.2):
        super(ResNetTransformer, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        resnet.maxpool = nn.Identity()

        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)

        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.cnn_feature_height = 32
        self.cnn_feature_channels = 512

        self.flattened_dim = self.cnn_feature_height * self.cnn_feature_channels

        self.feature_projection = nn.Linear(self.flattened_dim, d_model)
        self.feature_dropout = nn.Dropout(p=dropout)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src_img, tgt_seq):
        """
        src_img: (Batch, 3, 128, Width) - Höhe MUSS 128 sein, Breite variabel
        tgt_seq: (Batch, Seq_Len)       - Target Tokens (z.B. für Teacher Forcing)
        """

        features = self.backbone(src_img)

        b, c, h, w = features.shape

        assert h == self.cnn_feature_height, f"Bildhöhe falsch! Erwartet Feature-Höhe {self.cnn_feature_height}, bekam {h}. Input-Bild muss 128px hoch sein."

        features = features.permute(0, 3, 1, 2).contiguous()

        memory = features.view(b, w, c * h)

        memory = self.feature_projection(memory)
        memory = self.feature_dropout(memory)
        memory = self.pos_encoder(memory)

        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        seq_len = tgt_seq.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        tgt_mask = tgt_mask.to(tgt_seq.device)

        tgt_pad_mask = (tgt_seq == 0).to(tgt_seq.device)

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask

        )

        logits = self.output_head(output)

        return logits


class ResNetTransformer1(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dropout=0.2):
        super(ResNetTransformer1, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        resnet.maxpool = nn.Identity()

        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)

        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        for param in list(resnet.parameters())[:len(list(resnet.parameters())) // 2]:
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.conv1x1 = nn.Conv2d(512, d_model, kernel_size=1)

        self.feature_dropout = nn.Dropout(p=dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=10000)
        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   batch_first=True, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_dropout = nn.Dropout(p=dropout)
        self.output_head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def forward(self, src_img, tgt_seq):
        features = self.backbone(src_img)
        features = self.conv1x1(features)

        features = self.avgpool(features)

        b, c, h, w = features.shape
        memory = features.view(b, c, h * w).permute(0, 2, 1)

        memory = self.pos_encoder(memory)

        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)

        tgt_emb = self.pos_encoder(tgt_emb)

        seq_len = tgt_seq.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(tgt_seq.device)
        tgt_pad_mask = (tgt_seq == 0)

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        output = self.output_dropout(output)
        logits = self.output_head(output)

        return logits


class ResNetTransformerOld(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dropout=0.2):
        super(ResNetTransformerOld, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        resnet.maxpool = nn.Identity()

        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)

        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        for param in list(resnet.parameters())[:len(list(resnet.parameters())) // 2]:
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.conv1x1 = nn.Conv2d(512, d_model, kernel_size=1)

        self.feature_dropout = nn.Dropout(p=dropout)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=10000)
        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   batch_first=True, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_dropout = nn.Dropout(p=dropout)
        self.output_head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def forward(self, src_img, tgt_seq):
        features = self.backbone(src_img)
        features = self.conv1x1(features)

        b, c, h, w = features.shape
        memory = features.view(b, c, h * w).permute(0, 2, 1)

        memory = self.pos_encoder(memory)

        tgt_emb = self.embedding(tgt_seq) * math.sqrt(self.d_model)

        tgt_emb = self.pos_encoder(tgt_emb)

        seq_len = tgt_seq.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(tgt_seq.device)
        tgt_pad_mask = (tgt_seq == 0)

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        output = self.output_dropout(output)
        logits = self.output_head(output)

        return logits


class PositionalEncoding1(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class PositionEmbedding2DPage(nn.Module):
    """
    Normalisiertes 2D Sinusoidal Encoding für variable Bildgrößen.
    """

    def __init__(self, d_model):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model muss durch 4 teilbar sein.")
        self.d_model = d_model

    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device

        y_embed = torch.arange(1, h + 1, device=device).view(h, 1).expand(h, w)
        x_embed = torch.arange(1, w + 1, device=device).view(1, w).expand(h, w)

        y_embed = y_embed / (h + 1e-6) * 2 * math.pi
        x_embed = x_embed / (w + 1e-6) * 2 * math.pi

        dim_t = torch.arange(self.d_model // 2, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (self.d_model // 2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
        return x + pos


class PositionalEncoding1DPage(nn.Module):
    """Klassisches 1D Encoding für den Text-Decoder."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SwinTransformerOMRPage(nn.Module):
    def __init__(self, vocab_size, d_model=256, target_resolution=(800, 1280), max_seq_len=2048):
        super().__init__()

        config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window12-384")
        config.image_size = target_resolution

        config.output_attentions = True

        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window12-384",
            config=config,
            add_pooling_layer=False
        )

        self.adapter = nn.Linear(1024, d_model)
        self.pos_encoder_2d = PositionEmbedding2DPage(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_text = PositionalEncoding1DPage(d_model, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True)
            for _ in range(6)
        ])

        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, img, tgt):
        outputs = self.backbone(img)

        memory = outputs.last_hidden_state
        backbone_attns = outputs.attentions

        memory = self.adapter(memory)

        b, l, c = memory.shape
        h_feat, w_feat = img.shape[2] // 32, img.shape[3] // 32
        memory = memory.permute(0, 2, 1).view(b, c, h_feat, w_feat)
        memory = self.pos_encoder_2d(memory)
        memory = memory.flatten(2).permute(0, 2, 1)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder_text(tgt_emb)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)

        cross_attentions = []
        x = tgt_emb

        for layer in self.decoder_layers:
            x = layer.norm1(x + layer.self_attn(x, x, x, attn_mask=tgt_mask)[0])

            attn_out, attn_weights = layer.multihead_attn(x, memory, memory)
            x = layer.norm2(x + attn_out)

            x = layer.norm3(x + layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))

            cross_attentions.append(attn_weights)

        logits = self.head(x)

        return {
            "logits": logits,
            "backbone_attentions": backbone_attns,
            "cross_attentions": cross_attentions
        }

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
