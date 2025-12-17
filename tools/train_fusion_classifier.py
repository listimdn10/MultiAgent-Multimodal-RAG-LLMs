# ================================================================
# Fusion Transformer for Smart Contract Vulnerability Detection
# Label Merging: External Call Vulnerabilities
# ================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ================================================================
# 1. Load data
# ================================================================

df = pd.read_excel("solidity_code_with_full_embeddings_final.xlsx")

# ================================================================
# 2. Parse embedding string -> numpy
# ================================================================

def parse_embedding(x):
    if isinstance(x, str):
        x = x.strip("[]")
        return np.array([float(i) for i in x.split(",")], dtype=np.float32)
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    else:
        raise ValueError("Invalid embedding format")

df["graphcodebert_embeddings"] = df["graphcodebert_embeddings"].apply(parse_embedding)
df["all_mpnet_base_v2_embeddings"] = df["all_mpnet_base_v2_embeddings"].apply(parse_embedding)
df["gatv2_embedding"] = df["gatv2_embedding"].apply(parse_embedding)

# ================================================================
# 3. Merge labels (KEY STEP)
# ================================================================

def merge_label(label):
    if label in [
        "reentrancy",
        "unchecked_low_level_calls",
        "dangerous delegatecall (DE)"
    ]:
        return "external_call_vulnerability"
    return label

df["label"] = df["label"].apply(merge_label)

print("\nLabel distribution after merge:")
print(df["label"].value_counts())

# ================================================================
# 4. Encode labels
# ================================================================

le = LabelEncoder()
y = le.fit_transform(df["label"])

print("\nLabel mapping:")
for i, c in enumerate(le.classes_):
    print(f"{i} -> {c}")

# ================================================================
# 5. Build feature matrices
# ================================================================

X_scode = np.vstack(df["graphcodebert_embeddings"].values)
X_fsem  = np.vstack(df["all_mpnet_base_v2_embeddings"].values)
X_cfg   = np.vstack(df["gatv2_embedding"].values)

d_scode = X_scode.shape[1]
d_fsem  = X_fsem.shape[1]
d_cfg   = X_cfg.shape[1]

print("\nEmbedding dimensions:")
print("SCode:", d_scode, "FSem:", d_fsem, "CFG:", d_cfg)

# ================================================================
# 6. Train / Test split
# ================================================================

(
    X_scode_tr, X_scode_te,
    X_fsem_tr,  X_fsem_te,
    X_cfg_tr,   X_cfg_te,
    y_tr,       y_te
) = train_test_split(
    X_scode, X_fsem, X_cfg, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================================================================
# 7. Torch Dataset
# ================================================================

class FusionDataset(Dataset):
    def __init__(self, sc, fs, cfg, y):
        self.sc = torch.tensor(sc, dtype=torch.float32)
        self.fs = torch.tensor(fs, dtype=torch.float32)
        self.cfg = torch.tensor(cfg, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.sc[idx], self.fs[idx], self.cfg[idx], self.y[idx]

train_loader = DataLoader(FusionDataset(X_scode_tr, X_fsem_tr, X_cfg_tr, y_tr),
                          batch_size=32, shuffle=True)
test_loader  = DataLoader(FusionDataset(X_scode_te, X_fsem_te, X_cfg_te, y_te),
                          batch_size=32)

# ================================================================
# 8. Model
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_PROJ = 256

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.net(x)

class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, heads=4):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])

class FusionClassifier(nn.Module):
    def __init__(self, ds, df, dc, num_classes):
        super().__init__()
        self.p_sc = ProjectionMLP(ds, D_PROJ)
        self.p_fs = ProjectionMLP(df, D_PROJ)
        self.p_cf = ProjectionMLP(dc, D_PROJ)

        self.fusion = FusionTransformer(D_PROJ)

        self.classifier = nn.Sequential(
            nn.Linear(D_PROJ, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, sc, fs, cfg):
        z_sc = self.p_sc(sc)
        z_fs = self.p_fs(fs)
        z_cf = self.p_cf(cfg)

        tokens = torch.stack([z_fs, z_sc, z_cf], dim=1)
        fused = self.fusion(tokens)

        return self.classifier(fused)

model = FusionClassifier(d_scode, d_fsem, d_cfg, len(le.classes_)).to(DEVICE)

# ================================================================
# 9. Training setup
# ================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

EPOCHS = 50
best_acc = 0
patience = 8
wait = 0

# ================================================================
# 10. Training loop
# ================================================================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for sc, fs, cfg, yb in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}"):
        sc, fs, cfg, yb = sc.to(DEVICE), fs.to(DEVICE), cfg.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(sc, fs, cfg)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * yb.size(0)

    scheduler.step()

    # Evaluate
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for sc, fs, cfg, yb in test_loader:
            sc, fs, cfg = sc.to(DEVICE), fs.to(DEVICE), cfg.to(DEVICE)
            logits = model(sc, fs, cfg)
            preds.append(torch.argmax(logits, dim=1).cpu())
            labels.append(yb)

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    acc = accuracy_score(labels.numpy(), preds.numpy())
    print(f"Epoch {epoch+1:02d} | Loss={total_loss/len(train_loader.dataset):.4f} | Acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        wait = 0
        torch.save(model.state_dict(), "fusion_external_call_best.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

# ================================================================
# 11. Final report
# ================================================================

print("\n=== FINAL REPORT ===")
print(classification_report(labels.numpy(), preds.numpy(), target_names=le.classes_))
print("🔥 Best Accuracy:", best_acc)




import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
