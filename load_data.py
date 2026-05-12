import torch
from torch.utils.data import DataLoader, TensorDataset

from spectrum_paths import get_psinn_test_data_path, assert_psinn_full_channel_metadata

train_noise = torch.load("spectrum_data/train_noise.pt")
_psinn_test_path = get_psinn_test_data_path()
test_dict = torch.load(_psinn_test_path)
assert_psinn_full_channel_metadata(test_dict)
test_data = test_dict["data"]
test_labels = test_dict["labels"]   # 0 = noise, 1 = anomaly
test_snrs = test_dict["snrs"]

print(f"Train shape: {train_noise.shape} (pure noise)")
print(f"Test shape : {test_data.shape} ({test_labels.sum().item()} anomalies)")

train_loader = DataLoader(TensorDataset(train_noise), batch_size=128, shuffle=True)
test_loader  = DataLoader(TensorDataset(test_data), batch_size=128, shuffle=False)