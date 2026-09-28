import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mixing_layer_mean_field import MeanFieldConfig
from plot_mean_field_results import field_coordinates
from plot_reaction_snapshot_comparisons import (
    color_limits, field_figure, profile_figure, save_figure, snapshot_name,
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
                                    c2=1-c1-delta, vorticity=omega+delta)
            cases[da] = (replace(config, damkohler=da), data)
        return cases

    def test_profile_six_lines_are_y_not_time_and_have_requested_styles(self):
        rows = [dict(Da=da, method=method, time=.1, y=(j+.5)/4,
                     shear_stress=j/10, unmixedness=-j/10)
                for da in (1, 10, 100) for method in ('DNS', 'MF') for j in range(4)]
        for quantity in ('shear_stress', 'unmixedness'):
            fig = profile_figure(rows, quantity, .1, (-1, 1))
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlabel(), 'y')
            self.assertEqual(len(ax.lines), 6)
            self.assertEqual(ax.get_ylim(), (-1, 1))
            for line in ax.lines:
                np.testing.assert_array_equal(line.get_xdata(), (np.arange(4)+.5)/4)
                np.testing.assert_allclose(line.get_ydata(), np.arange(4)/10*(1 if quantity=='shear_stress' else -1))
                self.assertEqual(line.get_linestyle(), '-' if line.get_label().endswith('DNS') else '--')
            plt.close(fig)
        with self.assertRaisesRegex(ValueError, 'only'):
            profile_figure(rows, 'R11', .1, (-1, 1))
        with self.assertRaisesRegex(ValueError, 'all six'):
            profile_figure(rows[:-4], 'shear_stress', .1, (-1, 1))

    def test_field_arrays_difference_sign_and_color_scales(self):
        cases = self.cases()
        config, data = cases[1]
        for field in ('c1', 'c2', 'vorticity'):
            bounds, error = color_limits(cases, field, 1)
            fig, relative = field_figure(data, config, field, 1, bounds, error)
            expected_arrays = (data['DNS'][field][1], data['MF'][field][1],
                               data['MF'][field][1]-data['DNS'][field][1])
            for i, values in enumerate(expected_arrays):
                mesh = fig.axes[i].collections[0]
                expected = field_coordinates(values, config, field)[2]
                np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(expected.shape), expected)
                self.assertEqual(mesh.get_clim(), bounds if i < 2 else (-error, error))
            self.assertAlmostEqual(relative, np.linalg.norm(expected_arrays[2])/np.linalg.norm(expected_arrays[0]))
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

    def test_unique_snapshot_filenames_and_png_pdf_output_without_mutation(self):
        names = [snapshot_name(k, step) for k, step in enumerate((0,149,297,446,594,743,891,1040))]
        self.assertEqual(len(set(names)), 8)
        self.assertEqual(names[-1], 'snapshot_07_step_1040')
        cases = self.cases()
        config, data = cases[1]
        before = {method: {key: array.copy() for key, array in values.items()}
                  for method, values in data.items()}
        bounds, error = color_limits(cases, 'c1', 1)
        fig, _ = field_figure(data, config, 'c1', 1, bounds, error)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'fields'/'c1'/names[1]
            save_figure(fig, base)
            for extension in ('png', 'pdf'):
                self.assertGreater(base.with_suffix('.'+extension).stat().st_size, 1000)
        for method, values in before.items():
            for key, array in values.items():
                np.testing.assert_array_equal(data[method][key], array)


if __name__ == '__main__':
    unittest.main()
