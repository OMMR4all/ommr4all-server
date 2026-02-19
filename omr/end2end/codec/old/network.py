import torch
import torch.nn as nn
import torchvision.models as models
import math


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
        return self.dropout(x)  # <--- WICHTIG: Dropout anwenden
