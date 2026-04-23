import torch

# Load test data and the trained Psl-CNN model
test_dict = torch.load("spectrum_data/test_data.pt")
test_data = test_dict["data"]
test_labels = test_dict["labels"]   # 0 = noise, 1 = anomaly

model_psi = AE_Classifier1d(n_channels=2, n_classes=0, nf=16, k=3, use_dropout=True)
model_psi.load_state_dict(torch.load("spectrum_data/psl_cnn_ae.pth"))
model_psi.eval()

with torch.no_grad():
    recon = model_psi.AE(test_data)
    mse = torch.mean((test_data - recon)**2, dim=[1, 2])

print("🔍 Mean Reconstruction Error (Psl-CNN):")
print(f"  Pure noise samples   : {mse[test_labels == 0].mean():.5f}")
print(f"  Anomaly samples      : {mse[test_labels == 1].mean():.5f}")
print(f"  Difference (noise - anomaly): {mse[test_labels == 0].mean() - mse[test_labels == 1].mean():.5f}")