import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import os
# Cấu hình thiết bị & Random Seed để đảm bảo công bằng
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Running on: {DEVICE}")

# ================================================================
# 1. LOAD & PREPROCESS DATA
# ================================================================

def parse_embedding(x):
    """Chuyển chuỗi string embedding thành numpy array"""
    if isinstance(x, str):
        x = x.strip("[]")
        # Xử lý trường hợp chuỗi rỗng hoặc lỗi format
        if not x: return np.zeros(1)
        return np.array([float(i) for i in x.split(",")], dtype=np.float32)
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=np.float32)
    else:
        return np.zeros(1)

print("--- [1/5] Loading Data ---")
try:
    df = pd.read_excel("tools/solidity_code_with_full_embeddings_final.xlsx")
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file excel. Hãy đảm bảo file nằm cùng thư mục với script này.")
    exit()

# Parse embeddings
print("Parsing embeddings...")
df["graphcodebert_embeddings"] = df["graphcodebert_embeddings"].apply(parse_embedding)
df["all_mpnet_base_v2_embeddings"] = df["all_mpnet_base_v2_embeddings"].apply(parse_embedding)
df["gatv2_embedding"] = df["gatv2_embedding"].apply(parse_embedding)

# Merge labels (Logic của bạn)
def merge_label(label):
    if label in ["reentrancy", "unchecked_low_level_calls", "dangerous delegatecall (DE)"]:
        return "external_call_vulnerability"
    return label

df["label"] = df["label"].apply(merge_label)
print("Label distribution:", df["label"].value_counts().to_dict())

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])
NUM_CLASSES = len(le.classes_)

# Build Feature Matrices
X_scode = np.vstack(df["graphcodebert_embeddings"].values)
X_fsem  = np.vstack(df["all_mpnet_base_v2_embeddings"].values)
X_cfg   = np.vstack(df["gatv2_embedding"].values)

d_scode = X_scode.shape[1]
d_fsem  = X_fsem.shape[1]
d_cfg   = X_cfg.shape[1]

print(f"Dimensions: Code={d_scode}, Text={d_fsem}, Graph={d_cfg}")

# Train/Test Split
(X_sc_tr, X_sc_te, X_fs_tr, X_fs_te, X_cf_tr, X_cf_te, y_tr, y_te) = train_test_split(
    X_scode, X_fsem, X_cfg, y, test_size=0.2, stratify=y, random_state=SEED
)

# Dataset Class
class FusionDataset(Dataset):
    def __init__(self, sc, fs, cfg, y):
        self.sc = torch.tensor(sc, dtype=torch.float32)
        self.fs = torch.tensor(fs, dtype=torch.float32)
        self.cfg = torch.tensor(cfg, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.sc[idx], self.fs[idx], self.cfg[idx], self.y[idx]

train_loader = DataLoader(FusionDataset(X_sc_tr, X_fs_tr, X_cf_tr, y_tr), batch_size=32, shuffle=True)
test_loader  = DataLoader(FusionDataset(X_sc_te, X_fs_te, X_cf_te, y_te), batch_size=32)

# ================================================================
# 2. DEFINE MODELS (THE CONTENDERS)
# ================================================================

# --- Model A: Early Fusion (Concatenation) ---
class EarlyFusionModel(nn.Module):
    def __init__(self, d_sc, d_fs, d_cfg, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_sc + d_fs + d_cfg, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, sc, fs, cfg):
        cat = torch.cat([sc, fs, cfg], dim=1)
        return self.fc(cat)

# --- Model B: Late Fusion (Voting/Averaging) ---
class LateFusionModel(nn.Module):
    def __init__(self, d_sc, d_fs, d_cfg, n_classes):
        super().__init__()
        self.net_sc = nn.Sequential(nn.Linear(d_sc, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.net_fs = nn.Sequential(nn.Linear(d_fs, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.net_cf = nn.Sequential(nn.Linear(d_cfg, 128), nn.ReLU(), nn.Linear(128, n_classes))
    def forward(self, sc, fs, cfg):
        o1 = self.net_sc(sc)
        o2 = self.net_fs(fs)
        o3 = self.net_cf(cfg)
        return (o1 + o2 + o3) / 3

# --- Model C: Fusion Transformer (YOUR MODEL) ---
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(0.15)
        )
    def forward(self, x): return self.net(x)

class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, heads=4):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model*4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B = x.size(0)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])

class TransformerFusionModel(nn.Module):
    def __init__(self, ds, df, dc, num_classes):
        super().__init__()
        D_PROJ = 256
        self.p_sc = ProjectionMLP(ds, D_PROJ)
        self.p_fs = ProjectionMLP(df, D_PROJ)
        self.p_cf = ProjectionMLP(dc, D_PROJ)
        self.fusion = FusionTransformer(D_PROJ)
        self.classifier = nn.Sequential(
            nn.Linear(D_PROJ, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, num_classes)
        )
    def forward(self, sc, fs, cfg):
        z_sc, z_fs, z_cf = self.p_sc(sc), self.p_fs(fs), self.p_cf(cfg)
        tokens = torch.stack([z_fs, z_sc, z_cf], dim=1)
        fused = self.fusion(tokens)
        return self.classifier(fused)

# ================================================================
# 3. TRAINING & EVALUATION LOOP
# ================================================================

def run_experiment(model, name, train_loader, test_loader, epochs=20):
    print(f"\n--- Training {name} ---")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0

    # Train Loop
    for epoch in range(epochs):
        model.train()
        for sc, fs, cfg, yb in train_loader:
            sc, fs, cfg, yb = sc.to(DEVICE), fs.to(DEVICE), cfg.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(sc, fs, cfg)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Final Evaluation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for sc, fs, cfg, yb in test_loader:
            sc, fs, cfg = sc.to(DEVICE), fs.to(DEVICE), cfg.to(DEVICE)
            logits = model(sc, fs, cfg)
            preds.append(torch.argmax(logits, dim=1).cpu())
            targets.append(yb)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')

    print(f"✅ {name} Completed: Acc={acc:.4f}, F1={f1:.4f}")
    return {"Model": name, "Accuracy": acc, "F1-Score": f1}

# ================================================================
# 4. MAIN EXECUTION
# ================================================================

print("\n--- [2/5] Initializing Models ---")
m1 = EarlyFusionModel(d_scode, d_fsem, d_cfg, NUM_CLASSES)
m2 = LateFusionModel(d_scode, d_fsem, d_cfg, NUM_CLASSES)
m3 = TransformerFusionModel(d_scode, d_fsem, d_cfg, NUM_CLASSES)

print("\n--- [3/5] Starting Benchmark ---")
results = []
# Số epoch có thể giảm xuống 15-20 để test nhanh, hoặc tăng lên 50 để chính xác
EPOCHS_TO_RUN = 30

results.append(run_experiment(m1, "Early Fusion", train_loader, test_loader, EPOCHS_TO_RUN))
results.append(run_experiment(m2, "Late Fusion", train_loader, test_loader, EPOCHS_TO_RUN))
results.append(run_experiment(m3, "Transformer Fusion (Ours)", train_loader, test_loader, EPOCHS_TO_RUN))
# ================================================================
# 5. SAVE BEST MODEL (EARLY FUSION) TO TOOLS/
# ================================================================

print("\n--- [4/5] Saving Best Model (Early Fusion) ---")

# 1. Tạo thư mục 'tools' nếu chưa có
if not os.path.exists("tools"):
    os.makedirs("tools")
    print("📁 Created directory: tools/")

# 2. Lưu Model State Dict
save_path_model = "tools/early_fusion.pth"
torch.save(m1.state_dict(), save_path_model)
print(f"💾 Saved model to: {save_path_model}")

# 3. Lưu Label Encoder (để khi predict decode được label)
save_path_le = "tools/early_fusion_label_encoder.pkl"
with open(save_path_le, "wb") as f:
    pickle.dump(le, f)
print(f"💾 Saved label encoder to: {save_path_le}")

# ================================================================
# 6. VISUALIZATION
# ================================================================

print("\n--- [5/5] Generating Plot ---")
df_res = pd.DataFrame(results)
print(df_res)

plt.figure(figsize=(8, 6))
df_melt = df_res.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Performance Comparison", fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)
plt.show()

print("\n🎉 DONE! All files saved in 'tools/' folder.")