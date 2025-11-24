import torch
import torch.nn as nn
from src.cnn_model import SimpleCNNLA
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data_utils import load_protocol, add_audio_path
from data.build_dataset import ASVspoofLADataset
from data.feature_matrix import extract_mfcc_features


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleCNNLA(n_mels=64).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.float().to(device)  # BCEWithLogits expects float targets

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # validation step: compute AUC, EER, etc.
