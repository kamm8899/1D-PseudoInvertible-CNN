import torch

from spectrum_paths import get_psinn_test_data_path, assert_psinn_full_channel_metadata
from psinn_layer_1d import AE_Classifier1d

# Load test data and the trained Psl-CNN model
_psinn_test_path = get_psinn_test_data_path()
test_dict = torch.load(_psinn_test_path)
assert_psinn_full_channel_metadata(test_dict)
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