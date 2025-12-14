import torch
import torch.nn as nn

# =============================================================
# Fusion Transformer PRO v2 (Inference-Ready)
# Updated to match: train_fusion_classifier.py
# =============================================================

D_PROJ_DEFAULT = 256

# -----------------------------
# Projection MLP (shared for 3 embeddings)
# -----------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Fusion Transformer Module
# -----------------------------
class FusionTransformer(nn.Module):
    def __init__(self, d_model=D_PROJ_DEFAULT, heads=4, layers=1, dropout=0.1):
        super().__init__()

        # Trainable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer Encoder
        # Config khớp với file train: dim_feedforward=d_model*4, activation="gelu"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.norm = nn.LayerNorm(d_model)

        # NOTE: Đã xóa post_ffn để khớp với file train_fusion_classifier.py

    def forward(self, x):
        # x shape: (B, 3, D)
        B = x.size(0)

        cls = self.cls_token.expand(B, 1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1)          # (B,4,D)

        x = self.encoder(x)

        # Lấy CLS token ra và chuẩn hóa
        # Logic khớp file train: return self.norm(x[:, 0])
        cls_out = self.norm(x[:, 0])         

        return cls_out

# -----------------------------
# Full Fusion Classifier
# -----------------------------
class FusionClassifier(nn.Module):
    def __init__(self, ds, df, dc, num_classes, d_proj=D_PROJ_DEFAULT):
        super().__init__()

        # Projection for each embedding type
        self.p_sc = ProjectionMLP(ds, d_proj)
        self.p_fs = ProjectionMLP(df, d_proj)
        self.p_cf = ProjectionMLP(dc, d_proj)

        # Transformer Fusion
        self.fusion = FusionTransformer(d_model=d_proj, layers=1)

        # Classification Head
        # Khớp file train: Linear -> ReLU -> Dropout(0.25) -> Linear
        self.clf = nn.Sequential(
            nn.Linear(d_proj, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, sc, fs, cf):
        # sc, fs, cf are shape (B, D)

        z_sc = self.p_sc(sc)
        z_fs = self.p_fs(fs)
        z_cf = self.p_cf(cf)

        # Build 3-token sequence (B, 3, D_proj)
        # QUAN TRỌNG: Thứ tự stack phải khớp file train: [z_fs, z_sc, z_cf]
        tokens = torch.stack([z_fs, z_sc, z_cf], dim=1)

        fused = self.fusion(tokens)

        return self.clf(fused)

# =============================================================
# END OF FILE
# =============================================================