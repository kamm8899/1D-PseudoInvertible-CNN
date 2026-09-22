import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import unittest

import numpy as np
import torch

from generate_spectrum_dataset import generate_iq_dataset
from rf_nonlinearity import rapp, receiver, transmitter
from evaluate_nonlinearity import paired_interval, wilson, scores


class NonlinearityTests(unittest.TestCase):
    def test_complex_saturation_and_phase(self):
        x = torch.tensor([[0., 3., -3000.], [0., 4., 4000.]])
        y = rapp(x, 2.)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((torch.linalg.vector_norm(y, dim=0) <= 2.00001).all())
        torch.testing.assert_close(x[0] * y[1], x[1] * y[0])
        torch.testing.assert_close(y[:, 0], x[:, 0])
        torch.testing.assert_close(rapp(x, 1e9), x)

    def test_backoff_and_fixed_receiver_reference(self):
        x = torch.tensor([[3., 6.], [4., 8.]])
        power = x.square().sum(dim=0).mean().item()
        torch.testing.assert_close(transmitter(x, 3.), rapp(x, np.sqrt(power * 10**.3)))
        torch.testing.assert_close(receiver(x, 3.), rapp(x, np.sqrt(2 * 10**.3)))
        self.assertIs(transmitter(x, None), x)
        self.assertIs(receiver(x, None), x)
        self.assertFalse(torch.allclose(receiver(2*x, 3.), 2*receiver(x, 3.)))

    def test_paired_dataset_and_receiver_order(self):
        options = dict(num_train=2, length=64, samples_per_mod_per_snr=3,
                       noise_per_snr=4, snr_points=[0], seed=123, snr_uncertainty_db=2.)
        linear = generate_iq_dataset(**options)
        tx = generate_iq_dataset(**options, tx_ibo_db=0.)
        rx = generate_iq_dataset(**options, rx_backoff_db=3.)
        both = generate_iq_dataset(**options, tx_ibo_db=0., rx_backoff_db=3.)
        h0 = linear[4] == 0
        torch.testing.assert_close(linear[3][h0], tx[3][h0], rtol=0, atol=0)
        torch.testing.assert_close(rx[3], receiver(linear[3], 3.), rtol=0, atol=0)
        torch.testing.assert_close(both[3], receiver(tx[3], 3.), rtol=0, atol=0)
        self.assertFalse(torch.equal(tx[3][~h0], linear[3][~h0]))
        # Existing clean-noise training set is retained even for receiver test shift.
        torch.testing.assert_close(linear[0], both[0], rtol=0, atol=0)
        torch.testing.assert_close(rx[2].mean(dim=(1, 2)), torch.zeros(len(rx[2])), atol=1e-6, rtol=0)
        torch.testing.assert_close(rx[2].std(dim=(1, 2)), torch.ones(len(rx[2])))
        self.assertEqual(both[7]["nonlinearity"]["tx_ibo_db"], 0.)

    def test_invalid_parameters(self):
        x = torch.ones(2, 4)
        for saturation in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                rapp(x, saturation)
        with self.assertRaises(ValueError):
            rapp(x, 1., 0.)
        with self.assertRaises(ValueError):
            rapp(torch.ones(3, 4), 1.)

    def test_statistics(self):
        p, lo, hi = wilson(np.zeros(100, dtype=bool))
        self.assertEqual(p, 0.)
        self.assertLess(lo, 1e-12)
        self.assertGreater(hi, 0.)
        a = np.array([True, False, True, False])
        self.assertEqual(paired_interval(a, a, np.random.default_rng(1), 100), (0., 0., 0.))
        self.assertEqual(paired_interval(np.ones(20, bool), np.zeros(20, bool),
                                        np.random.default_rng(1), 100), (1., 1., 1.))

    def test_legacy_reconstruction_crop(self):
        class LongerDecoder(torch.nn.Module):
            def AE(self, x):
                return torch.nn.functional.pad(x, (0, 14))
        beta = scores(LongerDecoder(), torch.randn(3, 2, 64), 2, "cpu", 2)
        np.testing.assert_array_equal(beta, np.ones(3))


if __name__ == "__main__":
    unittest.main()
