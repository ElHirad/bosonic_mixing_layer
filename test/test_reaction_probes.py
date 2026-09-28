"""Configurable screening/probe sizes and diagnostic output, without Slurm."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from reaction_dns_preflight import run_preflight
from reaction_euler_mf_probe import probe


class ReactionProbeTests(unittest.TestCase):
    def test_dns_size_and_pressure_metadata_are_not_hardcoded(self):
        result = run_preflight(10, n=4, reynolds=200, peclet=200, dt=.0003125, pressure_max_steps=192000)
        cfg = result["config"]
        self.assertEqual((cfg["n"], cfg["reynolds"], cfg["peclet"]), (4, 200, 200))
        self.assertEqual(result["completed_steps"], 2080)
        self.assertEqual(cfg["pressure_max_steps"], 192000)
        self.assertEqual(result["dns_time_integrator"], "forward-euler")
        self.assertTrue(result["conservation_and_divergence_checks_passed"])

    def test_mf_probe_preserves_size_and_writes_actual_pressure_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"progress.json"
            result = probe(1, n=4, reynolds=200, peclet=200, dt=.0001,
                           pressure_max_steps=1000, progress_path=path)
            self.assertEqual(result["config"]["n"], 4)
            self.assertEqual(result["config"]["reynolds"], 200)
            self.assertEqual(json.loads(path.read_text()), result)
            history = result["pressure_history"]
            self.assertGreater(len(history), 1)
            self.assertEqual(history[0]["pressure_iterations"], 0)
            self.assertEqual(result["pressure_converged"], history[-1]["pressure_residual"] <= 1e-8)
            if result["state"] == "one_step_passed":
                self.assertTrue(result["pressure_converged"])
                self.assertEqual(result["accepted_steps"], 1)
            else:
                self.assertEqual(result["state"], "rejected_candidate")
                self.assertEqual(result["accepted_steps"], 0)

    def test_128_wrapper_preserves_requested_rates_and_separate_output_path(self):
        script = Path(__file__).resolve().parents[1]/"hpc"/"reaction128_euler_probe.sbatch"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"hpc").mkdir()
            capture = 'import json,os,sys; print(json.dumps(dict(args=sys.argv[1:],output=os.environ["PROBE_OUTPUT"])))'
            (root/"hpc"/"reaction_euler_probe.sbatch").write_text(
                f"#!/bin/bash\nexec python3 -c '{capture}' \"$@\"\n")
            environment = os.environ.copy()
            environment.pop("PROBE_OUTPUT", None)
            environment["PROJECT_ROOT"] = str(root)
            result = subprocess.run(["bash", str(script), "--damkohler", "100"], env=environment,
                                    capture_output=True, text=True, check=True)
            captured = json.loads(result.stdout)
            args = dict(zip(captured["args"][::2], captured["args"][1::2]))
            self.assertEqual(args, {"--n": "128", "--re": "200", "--pe": "200", "--dt": ".0003125",
                                    "--pressure-max-steps": "192000", "--damkohler": "100"})
            self.assertIn("reaction_128x128_re200_pe200_euler_series", captured["output"])


if __name__ == "__main__":
    unittest.main()
