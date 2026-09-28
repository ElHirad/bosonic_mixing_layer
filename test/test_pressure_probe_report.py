"""Pressure reporting distinguishes convergence from full step acceptance."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from mixing_layer_mean_field import MeanFieldConfig
from report_pressure_probes import make_report


class PressureReportTests(unittest.TestCase):
    def fixtures(self, root):
        for da in (1, 10, 100):
            cfg = asdict(MeanFieldConfig(n=128, reynolds=200, peclet=200, dt=.0003125,
                transition=.01875, scalar_species=3, damkohler=da, time_integrator="forward-euler",
                pressure_max_steps=192000))
            converged = da != 100
            accepted = da == 1
            probe = {"config": cfg, "fingerprint": "test-fixture", "slurm_job_id": "test",
                "state": "one_step_passed" if accepted else "rejected_candidate",
                "accepted_steps": int(accepted), "pressure_converged": converged,
                "pressure_history": [
                    {"pressure_iterations": 0, "pressure_residual": 1., "pressure_pseudo_time": 0., "elapsed_seconds": 0.},
                    {"pressure_iterations": 20, "pressure_residual": 5e-9 if converged else 1e-7,
                     "pressure_pseudo_time": 20*.125/128**2, "elapsed_seconds": 1.}],
                "error": None if accepted else "test-only rejection"}
            if converged:
                probe["candidate_diagnostics"] = {"relative_divergence": 1e-10,
                    "scalar_mass_error": 1e-12 if accepted else 1e-6, "coherent_eigenstate_defect": 1e-7}
            if da == 10:
                probe["failed_gates"] = {"scalar_mass_error": {"value": 1e-6, "limit": 1e-7}}
            reference = {"config": cfg, "mean_field_source_fingerprint": "test-fixture", "completed_steps": 2080,
                "species_ranges_all_steps": {name: {"minimum": 0., "maximum": 1.} for name in ("c1", "c2", "c3")},
                "worst_invariant_error": 1e-15}
            (root/f"mf_euler_probe_da{da}_substeps8.json").write_text(json.dumps(probe))
            (root/f"dns_preflight_da{da}.json").write_text(json.dumps(reference))

    def test_report_and_plot_do_not_confuse_pressure_success_with_all_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixtures(root)
            results = make_report(root)
            self.assertEqual([r["pressure_converged"] for r in results], [True, True, False])
            self.assertEqual([r["accepted_steps"] for r in results], [1, 0, 0])
            report = (root/"PRESSURE_CONVERGENCE.md").read_text()
            self.assertIn("one-step startup diagnostics only", report)
            self.assertIn("scalar_mass_error", report)
            self.assertIn("not completed trajectories", report)
            self.assertGreater((root/"pressure_convergence.png").stat().st_size, 1000)
            self.assertEqual(len((root/"pressure_convergence.csv").read_text().splitlines()), 7)

    def test_report_refuses_unfinished_or_mismatched_cases(self):
        for change in ("running", "mismatch"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixtures(root)
                path = root/"mf_euler_probe_da1_substeps8.json"
                probe = json.loads(path.read_text())
                if change == "running":
                    probe["state"] = "running"
                else:
                    probe["fingerprint"] = "different-source"
                path.write_text(json.dumps(probe))
                with self.assertRaises(ValueError):
                    make_report(root)


if __name__ == "__main__":
    unittest.main()
