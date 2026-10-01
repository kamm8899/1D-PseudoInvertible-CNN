# Complexity verification

Counts use one multiply--accumulate (MAC) for each convolution or lag-product term.
Batch normalization, activation, normalization, interpolation, absolute values, and divisions are excluded.

## Psi-NN convolutional reconstruction

| Stage | Input length | Output length | Channels | MACs |
|---|---:|---:|---:|---:|
| Encoder 1 | 1024 | 511 | 1 to 16 | 40,880 |
| Encoder 2 | 511 | 255 | 16 to 64 | 1,305,600 |
| Encoder 3 | 255 | 127 | 64 to 128 | 5,201,920 |
| Decoder 1 | 127 | 256 | 128 to 64 | 5,201,920 |
| Decoder 2 | 256 | 514 | 64 to 16 | 1,310,720 |
| Decoder 3 | 514 | 1030 | 16 to 1 | 41,120 |

Encoder total: **6,548,400 MACs**.

Decoder total: **6,553,760 MACs**.

Reconstruction total: **13,102,160 MACs per block** (approximately **13.1 million**).

The current PyTorch code forms three pseudoinverse matrices during a forward call. Their two matrix-multiplication stages add **11,141,920 MACs**, in addition to inversions of matrices sized 5x5, 64x64, and 128x128. The 13.1-million figure therefore assumes that the pseudoinverses are cached after training, as they can be for fixed inference weights. Without caching, a batch-one call requires at least 24,244,080 MACs plus the three matrix inversions.

## CAV lag-covariance statistic

The implemented CAV uses the Toeplitz lag-autocorrelation representation rather than forming every covariance entry independently.

| Lag dimension | Lag-product MACs per block |
|---:|---:|
| 4 | 4,090 |
| 8 | 8,164 |

A few additional scalar operations form the CAV numerator and denominator. Thus, the detector core costs approximately 4.1 thousand MACs for the main L=4 result and 8.2 thousand MACs for L=8. The advisor's rough 10,000-operation statement is a reasonable order-of-magnitude description for L=8, but it is not the exact main-result count.
