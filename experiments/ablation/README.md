# Ablation experiments

This folder contains the architecture, decoder, and input-channel ablation
code and all saved ablation outputs.

## Current paper ablation

The paper uses Model C, a conventional autoencoder with the same
`1 -> 16 -> 64 -> 128` encoder dimensions as Psi-NN and an independently
trained symmetric decoder.

Run its corrected evaluation from the project root with:

```bash
.venv/bin/python experiments/ablation/evaluate_model_c.py
```

The checkpoint and resulting scores are in `output/`:

- `compare_C_lowcap-symdec.pth`
- `model_c_corrected_results.txt`
- `scores_calib_model_c.npy`
- `scores_test_model_c.npy`
- `pd_vs_snr_model_c.npy`
- `snr_points_model_c.npy`

## Other ablation code

- `compare_lowcap_convae.py` — historical Models A--E architecture chain.
- `cae_decoder_ablation.py` — CAE decoder variants.
- `run_cae_ablation_report.py` — channel-condition CAE report.
- `evaluate_channel_ablation.py` — I-only versus I/Q comparison.

The current manuscript cites only the corrected Model C comparison. Other
outputs are retained for provenance and should not replace the paper result
without reevaluation under the current calibration and test conditions.
