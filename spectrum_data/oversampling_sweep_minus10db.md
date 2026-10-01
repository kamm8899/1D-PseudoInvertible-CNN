# Oversampling sweep at -10 dB

Detection probability at target $P_{fa}=0.01$. Each value uses 4,000 signal blocks (1,000 per modulation).

| Samples/symbol | CAV/MME L | Psi-NN | CAE | CAV | MME | ED, known σ² | ED, ±2 dB noise uncertainty |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.3015 | 0.0862 | 0.4248 | 0.4338 | 0.6092 | 0.0000 |
| 4 | 4 | 0.7980 | 0.3962 | 0.7685 | 0.7208 | 0.5917 | 0.0000 |
| 8 | 8 | 0.9365 | 0.8175 | 0.9135 | 0.9012 | 0.5853 | 0.0000 |

The 2-sps result confirms that the known-noise energy detector is strongest when the sampled signal is nearly white. At 8 sps, Psi-NN is strongest and closely matches the advisor sanity value (approximately 0.93).
