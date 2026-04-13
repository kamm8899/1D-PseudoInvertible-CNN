import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_dict = torch.load("spectrum_data/test_data.pt")
test_data = test_dict["data"]
test_labels = test_dict["labels"].numpy()

# Load models
model_psi = AE_Classifier1d(n_channels=2, n_classes=0, nf=16, k=3, use_dropout=True).to(device)
model_base = AE_Baseline_Classifier1d(n_channels=2, n_classes=0, nf=16, k=3, use_dropout=True).to(device)

model_psi.load_state_dict(torch.load("spectrum_data/psl_cnn_ae.pth"))
model_base.load_state_dict(torch.load("spectrum_data/baseline_ae.pth"))
model_psi.eval()
model_base.eval()

def get_anomaly_scores(model, data):
    scores = []
    with torch.no_grad():
        for i in range(0, len(data), 128):
            batch = data[i:i+128].to(device)
            recon = model.AE(batch)
            mse = torch.mean((batch - recon)**2, dim=[1,2]).cpu()
            scores.append(mse)
    return torch.cat(scores).numpy()

print("Computing anomaly scores...")
scores_psi = get_anomaly_scores(model_psi, test_data)
scores_base = get_anomaly_scores(model_base, test_data)

fpr_psi, tpr_psi, _ = roc_curve(test_labels, scores_psi)
fpr_base, tpr_base, _ = roc_curve(test_labels, scores_base)
auc_psi = auc(fpr_psi, tpr_psi)
auc_base = auc(fpr_base, tpr_base)

param_psi = sum(p.numel() for p in model_psi.parameters())
param_base = sum(p.numel() for p in model_base.parameters())

print(f"\n=== FINAL RESULTS ===")
print(f"Psl-CNN AUC: {auc_psi:.4f}")
print(f"Baseline AUC: {auc_base:.4f}")
print(f"Psl-CNN parameters: {param_psi:,}")
print(f"Baseline parameters: {param_base:,}")

# Save evaluation summary
Path("spectrum_data").mkdir(exist_ok=True)
with open("spectrum_data/evaluation_results.txt", "w") as f:
    f.write("=== EVALUATION RESULTS ===\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Psl-CNN AUC: {auc_psi:.4f}\n")
    f.write(f"Baseline AUC: {auc_base:.4f}\n")
    f.write(f"Psl-CNN parameters: {param_psi:,}\n")
    f.write(f"Baseline parameters: {param_base:,}\n")

# Save ROC plot
plt.figure(figsize=(8,6))
plt.plot(fpr_psi, tpr_psi, label=f'Psl-CNN (AUC = {auc_psi:.3f})')
plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {auc_base:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Modulation-Agnostic Anomaly Detection')
plt.legend()
plt.grid()
plt.savefig("spectrum_data/roc_comparison.png")
plt.close()

print("✅ Results saved in spectrum_data/ (CSV, TXT, and PNG)")