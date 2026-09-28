import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_reaction_statistics import (
    DAMKOHLERS, METRICS, draw_metric, make_plots, snapshot_statistics,
)


class ReactionStatisticsTests(unittest.TestCase):
    def fields(self, n=8):
        y = (np.arange(n)+.5)/n
        u = np.broadcast_to((2*y-1)[:, None], (n, n)).copy()
        v = np.zeros((n+1, n))
        c1 = np.broadcast_to(y[:, None], (n, n)).copy()
        return u, v, c1, 1-c1

    def test_laminar_shear_not_counted_as_reynolds_stress(self):
        u, v, c1, c2 = self.fields()
        stats, profile = snapshot_statistics(u, v, c1, c2, 1/8)
        for key in ('R11', 'R22', 'R12', 'covariance_streamwise'):
            self.assertEqual(stats[key], 0.)
        self.assertAlmostEqual(stats['maximum_mean_shear'], 2.)
        self.assertAlmostEqual(stats['velocity_jump'], 1.75)
        self.assertAlmostEqual(stats['vorticity_thickness'], .875)
        self.assertAlmostEqual(stats['covariance_global'], -np.var(c1))
        np.testing.assert_array_equal(profile['y'], (np.arange(8)+.5)/8)

    def test_mac_collocation_and_signed_moments(self):
        u, v, c1, c2 = self.fields(4)
        face_mode = np.array([1., 0., -1., 0.])
        centered_mode = np.array([.5, -.5, -.5, .5])
        u += face_mode
        v[1:-1] = -2*centered_mode
        c1 += .1*centered_mode
        c2 -= .2*centered_mode
        originals = [a.copy() for a in (u, v, c1, c2)]
        stats, profiles = snapshot_statistics(u, v, c1, c2, .25)
        np.testing.assert_allclose(profiles['R11'], .25)
        np.testing.assert_allclose(profiles['R22'], [.25, 1., 1., .25])
        np.testing.assert_allclose(profiles['R12'], [-.25, -.5, -.5, -.25])
        self.assertAlmostEqual(stats['R11'], .25)
        self.assertAlmostEqual(stats['R22'], .625)
        self.assertAlmostEqual(stats['R12'], -.375)
        self.assertAlmostEqual(stats['covariance_streamwise'], -.005)
        self.assertAlmostEqual(stats['covariance_global'],
                               stats['covariance_streamwise']+stats['covariance_between_y'])
        for actual, original in zip((u, v, c1, c2), originals):
            np.testing.assert_array_equal(actual, original)

    def test_means_removed_and_scalar_negatives_not_clipped(self):
        u, v, c1, c2 = self.fields()
        c1[:, ::2] -= .5
        c2[:, ::2] += .25
        first, _ = snapshot_statistics(u, v, c1, c2, .125)
        shifted, _ = snapshot_statistics(u+7, v-3, c1+2, c2-5, .125)
        for key in METRICS:
            self.assertAlmostEqual(first[key], shifted[key], places=13)
        self.assertLess(c1.min(), 0.)
        clipped, _ = snapshot_statistics(u, v, np.maximum(c1, 0), c2, .125)
        self.assertNotAlmostEqual(first['covariance_global'], clipped['covariance_global'])

    def test_tanh_thickness_converges_to_twice_shear_parameter(self):
        thickness = .05
        errors = []
        for n in (32, 64, 128):
            u, v, c1, c2 = self.fields(n)
            y = (np.arange(n)+.5)/n
            u[:] = np.tanh((y[:, None]-.5)/thickness)
            stats, _ = snapshot_statistics(u, v, c1, c2, 1/n)
            errors.append(abs(stats['vorticity_thickness']-2*thickness))
        self.assertGreater(errors[0]/errors[1], 3.9)
        self.assertGreater(errors[1]/errors[2], 3.9)

    def test_bad_shapes_nonfinite_and_undefined_thickness_rejected(self):
        u, v, c1, c2 = self.fields()
        with self.assertRaisesRegex(ValueError, 'channel MAC'):
            snapshot_statistics(u, v[:-1], c1, c2, .125)
        with self.assertRaisesRegex(ValueError, 'finite'):
            snapshot_statistics(u*np.nan, v, c1, c2, .125)
        with self.assertRaisesRegex(ValueError, 'positive'):
            snapshot_statistics(u, v, c1, c2, 0.)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            snapshot_statistics(np.ones_like(u), v, c1, c2, .125)

    def test_six_lines_correct_styles_all_panels_and_files(self):
        rows = []
        for da in DAMKOHLERS:
            for method in ('DNS', 'MF'):
                for k in range(8):
                    rows.append(dict(Da=da, method=method, time=k*.1,
                                     **{key: (k+1)/da for key in METRICS}))
        fig, ax = plt.subplots()
        draw_metric(ax, rows, 'R11')
        self.assertEqual(len(ax.lines), 6)
        for line in ax.lines:
            self.assertEqual(line.get_linestyle(), '-' if line.get_label().endswith('DNS') else '--')
            self.assertEqual(len(line.get_xdata()), 8)
        plt.close(fig)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            make_plots(rows, destination)
            for name in ('reynolds_stresses', 'vorticity_thickness', 'unmixedness'):
                for extension in ('png', 'pdf'):
                    self.assertGreater((destination/f'{name}.{extension}').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
