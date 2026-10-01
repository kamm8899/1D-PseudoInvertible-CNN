"""Reproduce the paper-facing Psi-NN and CAV operation counts."""

from pathlib import Path


N = 1024
K = 5


def conv1d_length(length, kernel=5, stride=2, padding=1):
    return (length + 2 * padding - (kernel - 1) - 1) // stride + 1


def transpose_length(length, kernel=5, stride=2, padding=1, output_padding=1):
    return (length - 1) * stride - 2 * padding + kernel + output_padding


# Psi-NN encoder: 1 -> 16 -> 64 -> 128.
channels = [(1, 16), (16, 64), (64, 128)]
length = N
encoder = []
for cin, cout in channels:
    out_length = conv1d_length(length)
    macs = out_length * cout * cin * K
    encoder.append((length, out_length, cin, cout, macs))
    length = out_length

# The custom decoder applies the same matrices in reverse. Its intermediate
# lengths follow PsiNNConv1d.back(), before the final interpolation to N=1024.
decoder = []
for cin, cout in reversed(channels):
    out_length = transpose_length(length)
    macs = length * cout * cin * K
    decoder.append((length, out_length, cout, cin, macs))
    length = out_length

encoder_macs = sum(row[-1] for row in encoder)
decoder_macs = sum(row[-1] for row in decoder)
psinn_macs = encoder_macs + decoder_macs

# rightInverse(A) evaluates A A^T and A^T(AA^T)^-1. For an m-by-n
# wide matrix, those two matrix products cost 2*m*m*n MACs. The implemented
# layers form right inverses of matrices sized 5x16, 64x80, and 128x320.
pseudoinverse_shapes = [(5, 16), (64, 80), (128, 320)]
pseudoinverse_matmul_macs = sum(2 * m * m * n for m, n in pseudoinverse_shapes)


def cav_lag_product_macs(n, lag_dimension):
    """Products accumulated by lag_autocorr() for lags 0,...,L-1."""
    return sum(n - lag for lag in range(lag_dimension))


lines = [
    "# Complexity verification",
    "",
    "Counts use one multiply--accumulate (MAC) for each convolution or lag-product term.",
    "Batch normalization, activation, normalization, interpolation, absolute values, and divisions are excluded.",
    "",
    "## Psi-NN convolutional reconstruction",
    "",
    "| Stage | Input length | Output length | Channels | MACs |",
    "|---|---:|---:|---:|---:|",
]
for i, (lin, lout, cin, cout, macs) in enumerate(encoder, 1):
    lines.append(f"| Encoder {i} | {lin} | {lout} | {cin} to {cout} | {macs:,} |")
for i, (lin, lout, cin, cout, macs) in enumerate(decoder, 1):
    lines.append(f"| Decoder {i} | {lin} | {lout} | {cin} to {cout} | {macs:,} |")
lines += [
    "",
    f"Encoder total: **{encoder_macs:,} MACs**.",
    "",
    f"Decoder total: **{decoder_macs:,} MACs**.",
    "",
    f"Reconstruction total: **{psinn_macs:,} MACs per block** (approximately **{psinn_macs / 1e6:.1f} million**).",
    "",
    "The current PyTorch code forms three pseudoinverse matrices during a forward call. "
    f"Their two matrix-multiplication stages add **{pseudoinverse_matmul_macs:,} MACs**, "
    "in addition to inversions of matrices sized 5x5, 64x64, and 128x128. "
    "The 13.1-million figure therefore assumes that the pseudoinverses are cached after training, "
    "as they can be for fixed inference weights. Without caching, a batch-one call requires at least "
    f"{psinn_macs + pseudoinverse_matmul_macs:,} MACs plus the three matrix inversions.",
    "",
    "## CAV lag-covariance statistic",
    "",
    "The implemented CAV uses the Toeplitz lag-autocorrelation representation rather than forming every covariance entry independently.",
    "",
    "| Lag dimension | Lag-product MACs per block |",
    "|---:|---:|",
]
for lag_dimension in (4, 8):
    lines.append(f"| {lag_dimension} | {cav_lag_product_macs(N, lag_dimension):,} |")
lines += [
    "",
    "A few additional scalar operations form the CAV numerator and denominator. "
    "Thus, the detector core costs approximately 4.1 thousand MACs for the main L=4 result "
    "and 8.2 thousand MACs for L=8. The advisor's rough 10,000-operation statement is a reasonable order-of-magnitude description for L=8, but it is not the exact main-result count.",
]

report = "\n".join(lines) + "\n"
Path("spectrum_data/complexity_verification.md").write_text(report)
print(report)
