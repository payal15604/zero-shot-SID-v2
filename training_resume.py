import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from dataset import HazeDataset
from pytorch_msssim import ssim
from Statistical_Transmission.bounding_fun import bounding_function
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN
from utils import DarkChannel, AtmLight
from INet.models.dehazeformer import DehazeFormer

# Compute transmission
def compute_transmission(hazy_img, device):
    batch_size = hazy_img.shape[0]
    transmission_list = []

    for i in range(batch_size):
        img = hazy_img[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        _, transmission, _ = bounding_function(img, zeta=1)
        transmission_list.append(transmission)

    transmission_tensor = torch.tensor(np.stack(transmission_list), dtype=torch.float32, device=device)
    return transmission_tensor.unsqueeze(1)

# Estimate atmospheric light
def estimate_atmospheric_light(hazy_img):
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    batch_A = []
    for img in hazy_np:
        dark = DarkChannel(img)
        A = AtmLight(img, dark)
        batch_A.append(A)

    A_tensor = torch.tensor(batch_A, dtype=torch.float32, device=hazy_img.device).unsqueeze(2).unsqueeze(3)
    return A_tensor

# Resume training config
resume_training = True
resume_model_path = "dehazeformer_trained.pth"
resume_epoch = 100              # Epochs already trained
additional_epochs = 50         # New epochs to train
learning_rate = 0.05           # New LR for resumed training

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU:', device)

# Data setup
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load BetaCNN
haze_net = BetaCNN().to(device)
haze_net.load_state_dict(torch.load("../Gamma_Estimation/beta_cnn.pth"))
haze_net.eval()

# Initialize DehazeFormer
i_net = DehazeFormer().to(device)
start_epoch = 0
total_epochs = additional_epochs

if resume_training and os.path.exists(resume_model_path):
    print(f"Resuming from {resume_model_path}")
    i_net.load_state_dict(torch.load(resume_model_path))
    start_epoch = resume_epoch
    total_epochs = resume_epoch + additional_epochs
else:
    # Load base pretrained weights if not resuming from your own model
    checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/INet/models/dehazeformer-t.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    i_net.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    print("Starting from base DehazeFormer pretrained weights.")

i_net.train()
optimizer = optim.Adam(i_net.parameters(), lr=learning_rate)

# Training loop
for epoch in range(start_epoch, total_epochs):
    epoch_loss = 0.0
    print(f"\nEpoch [{epoch + 1}/{total_epochs}]")

    for hazy_img in dataloader:
        hazy_img = hazy_img.to(device)

        with torch.no_grad():
            gamma = haze_net(hazy_img)

        transmission = compute_transmission(hazy_img, device)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        A = estimate_atmospheric_light(hazy_img)
        A = A.squeeze().view(-1, 3, 1, 1)

        J_haze_free = i_net(hazy_img)
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free

        loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Loss: {epoch_loss / len(dataloader):.4f}")

    # Optional: Save model after every epoch
    torch.save(i_net.state_dict(), resume_model_path)

print(f"\nTraining complete. Model saved to {resume_model_path}")
