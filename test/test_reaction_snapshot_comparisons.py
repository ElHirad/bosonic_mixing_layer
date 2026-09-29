import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mixing_layer_mean_field import MeanFieldConfig
from plot_mean_field_results import field_coordinates
from plot_reaction_snapshot_comparisons import (
    color_limits, field_figure, profile_figure, save_figure,
)


class SnapshotComparisonTests(unittest.TestCase):
    def cases(self):
        config = MeanFieldConfig(n=4, scalar_species=3, damkohler=1,
                                 reynolds=100, peclet=100)
        cases = {}
        for da in (1, 10, 100):
            c1 = np.broadcast_to(np.arange(4)[None, :, None]/3, (2, 4, 4)).copy()
            c1[0, 0, 0] = -.02
            omega = np.zeros((2, 5, 4))
            omega[0, 1:4] = -40
            omega[1, 1:4] = -4
            data = {}
            for method in ('DNS', 'MF'):
                delta = 0 if method == 'DNS' else .01
                data[method] = dict(times=np.array([0., .1]), c1=c1+delta,
                                    c2=1-c1-delta, vorticity=omega+delta,
                                    concentration=c1+delta, c3=np.zeros_like(c1),
                                    u=np.ones_like(c1), v=np.zeros_like(omega))
            cases[da] = (replace(config, damkohler=da), data)
        return cases

    def test_profile_all_eight_panels_have_six_correctly_styled_y_profiles(self):
        times = np.arange(8)/10
        rows = [dict(Da=da, method=method, time=time, y=(j+.5)/4,
                     shear_stress=j/10+time/100, unmixedness=-j/10-time/100)
                for time in times for da in (1, 10, 100) for method in ('DNS', 'MF') for j in range(4)]
        for quantity in ('shear_stress', 'unmixedness'):
            fig = profile_figure(rows, quantity, times, (-1, 1))
            self.assertEqual(len(fig.axes), 8)
            for ax, time in zip(fig.axes, times):
                self.assertEqual(ax.get_xlabel(), 'y')
                self.assertEqual(len(ax.lines), 6)
                self.assertEqual(ax.get_ylim(), (-1, 1))
                self.assertIn(f'{time:.6f}', ax.get_title())
                for line in ax.lines:
                    np.testing.assert_array_equal(line.get_xdata(), (np.arange(4)+.5)/4)
                    np.testing.assert_allclose(line.get_ydata(), (np.arange(4)/10+time/100)*(1 if quantity=='shear_stress' else -1))
                    self.assertEqual(line.get_linestyle(), '-' if line.get_label().endswith('DNS') else '--')
            plt.close(fig)
        with self.assertRaisesRegex(ValueError, 'only'):
            profile_figure(rows, 'R11', times, (-1, 1))
        with self.assertRaisesRegex(ValueError, 'all six'):
            profile_figure(rows[:-4], 'shear_stress', times, (-1, 1))

    def test_field_arrays_difference_sign_and_color_scales(self):
        cases = self.cases()
        for field in ('c1', 'c2', 'vorticity'):
            fig, metrics = field_figure(cases, field)
            self.assertEqual(len(metrics), 6)
            self.assertEqual(len(fig.axes), 21 if field == 'vorticity' else 20)
            for case_index, (da, (config, data)) in enumerate(cases.items()):
                for k in range(2):
                    bounds, error = color_limits(cases, field, k)
                    expected_arrays = (data['DNS'][field][k], data['MF'][field][k],
                                       data['MF'][field][k]-data['DNS'][field][k])
                    for i, values in enumerate(expected_arrays):
                        mesh = fig.axes[(3*case_index+i)*2+k].collections[0]
                        expected = field_coordinates(values, config, field)[2]
                        np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(expected.shape), expected)
                        self.assertEqual(mesh.get_clim(), bounds if i < 2 else (-error, error))
                    metric = next(row for row in metrics if row['Da']==da and row['snapshot']==k)
                    self.assertAlmostEqual(metric['relative_l2'], np.linalg.norm(expected_arrays[2])/np.linalg.norm(expected_arrays[0]))
            plt.close(fig)

    def test_scale_rules_preserve_scalar_extrema_and_initial_vorticity(self):
        cases = self.cases()
        c0, e0 = color_limits(cases, 'c1', 0)
        c1, e1 = color_limits(cases, 'c1', 1)
        self.assertEqual(c0, c1)
        self.assertEqual(e0, e1)
        self.assertAlmostEqual(c0[0], -.02)
        self.assertAlmostEqual(c0[1], 1.01)
        v0, ve0 = color_limits(cases, 'vorticity', 0)
        v1, ve1 = color_limits(cases, 'vorticity', 1)
        self.assertEqual(v0, (-40, 40))
        self.assertEqual(v1, (-4, 4))
        self.assertEqual(ve0, ve1)

    def test_single_combined_png_output_without_mutation(self):
        cases = self.cases()
        config, data = cases[1]
        before = {method: {key: array.copy() for key, array in values.items()}
                  for method, values in data.items()}
        fig, _ = field_figure(cases, 'c1')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'c1'
            save_figure(fig, base)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['c1.png'])
            self.assertGreater(base.with_suffix('.png').stat().st_size, 1000)
        for method, values in before.items():
            for key, array in values.items():
                np.testing.assert_array_equal(data[method][key], array)

    def test_mismatched_snapshot_times_rejected(self):
        cases = self.cases()
        cases[10][1]['MF']['times'] = np.array([0., .2])
        with self.assertRaisesRegex(ValueError, 'snapshot times'):
            field_figure(cases, 'c1')


if __name__ == '__main__':
    unittest.main()
