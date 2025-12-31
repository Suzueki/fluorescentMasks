import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TRAINING_PATH = r"/content/drive/MyDrive/FluoroMask/DepthData"
SAVE_MODEL_PATH = r"/content/drive/MyDrive/FluoroMask/realistic_depth_model.pth"
# May run data generation on CPU locally and mount drive, and perform training on Colab GPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 1e-4

class DepthDataset(Dataset):
    def __init__(self, sample_paths, target_size=(240, 240)):
        self.sample_paths = sample_paths
        self.target_size = target_size

    def __len__(self): return len(self.sample_paths)

    def __getitem__(self, idx):
        path = self.sample_paths[idx]
        f_2d = np.load(os.path.join(path, "fluo_2d.npy"))
        m_2d = np.load(os.path.join(path, "mri_ref.npy"))
        d_gt = np.load(os.path.join(path, "depth_gt.npy"))
        
        fluo_t = torch.from_numpy(f_2d).unsqueeze(0).float()
        mri_t = torch.from_numpy(m_2d).unsqueeze(0).float()
        depth_t = torch.from_numpy(d_gt).unsqueeze(0).float()
        
        # Consistent Resizing (Handles variable BraTS dimensions)
        fluo_t = F.interpolate(fluo_t.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        mri_t = F.interpolate(mri_t.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        depth_t = F.interpolate(depth_t.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)
        
        return fluo_t, mri_t, depth_t

class SurgicalDepthNet(nn.Module):
    def __init__(self):
        super(SurgicalDepthNet, self).__init__()
        self.enc_fluo = self._conv_block(1, 32)
        self.enc_mri = self._conv_block(1, 32)
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1), nn.ReLU()
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )

    def forward(self, fluo, mri):
        combined = torch.cat([self.enc_fluo(fluo), self.enc_mri(mri)], dim=1)
        return self.regressor(self.fusion(combined))

def surgical_safety_loss(pred, target):
    diff = pred - target
    loss_depth = torch.where(diff > 0, (diff**2) * 15.0, diff**2)
    no_tumor_mask = (target == 0).float()
    false_positive_penalty = torch.mean(F.relu(pred) * no_tumor_mask) * 50.0
    return loss_depth.mean() + false_positive_penalty

def main():
    # 1. Setup Samples (Patient-wise split to avoid leakage)
    all_samples = [os.path.join(TRAINING_PATH, d) for d in os.listdir(TRAINING_PATH) if os.path.isdir(os.path.join(TRAINING_PATH, d))]
    patient_ids = sorted(list(set([os.path.basename(s).split('_sample_')[0] for s in all_samples])))
    split_idx = int(0.85 * len(patient_ids))
    train_pts, val_pts = patient_ids[:split_idx], patient_ids[split_idx:]
    
    train_samples = [s for s in all_samples if any(pt in s for pt in train_pts)]
    val_samples = [s for s in all_samples if any(pt in s for pt in val_pts)]

    train_loader = DataLoader(DepthDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DepthDataset(val_samples), batch_size=BATCH_SIZE)

    # 2. Initialize / Resume Model
    model = SurgicalDepthNet().to(DEVICE)
    if os.path.exists(SAVE_MODEL_PATH):
        print(f"Resuming training from {SAVE_MODEL_PATH}...")
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
    
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    
    # --- SCHEDULER ---
    # Reduces LR by factor of 0.5 if Val Loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"Active Training on {len(train_samples)} augmented samples...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for f, m, d in train_loader:
            f, m, d = f.to(DEVICE), m.to(DEVICE), d.to(DEVICE)
            optimizer.zero_grad()
            loss = surgical_safety_loss(model(f, m), d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevents loss explosions
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for f, m, d in val_loader:
                f, m, d = f.to(DEVICE), m.to(DEVICE), d.to(DEVICE)
                val_loss += surgical_safety_loss(model(f, m), d).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Training phase complete.")

if __name__ == "__main__":
    main()