import torch
import torch.nn as nn

# ================================================================
# Early Fusion Model (Inference-Ready)
# This model concatenates embeddings and processes them with an MLP.
# It achieved the highest accuracy in previous benchmarks.
# ================================================================

class EarlyFusionModel(nn.Module):
    def __init__(self, d_sc, d_fs, d_cfg, n_classes):
        super().__init__()
        # Total features after concatenation
        total_features = d_sc + d_fs + d_cfg

        self.fc = nn.Sequential(
            nn.Linear(total_features, 512), # First layer takes concatenated features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes) # Output layer for classification
        )

    def forward(self, sc, fs, cfg):
        # Concatenate the three input embeddings along the feature dimension
        cat = torch.cat([sc, fs, cfg], dim=1)
        # Pass the concatenated vector through the fully connected layers
        return self.fc(cat)

# Example usage (assuming d_scode, d_fsem, d_cfg, NUM_CLASSES are defined)
# model = EarlyFusionModel(d_scode, d_fsem, d_cfg, NUM_CLASSES)
# print(model)
