"""Exercise the 64-cell batch wrapper without submitting jobs or evolving fields."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from mixing_layer_mean_field import MeanFieldConfig


SCRIPT = Path(__file__).resolve().parents[1]/"hpc"/"mean_field64_cpu.sbatch"


class BatchPresetTests(unittest.TestCase):
    def wrapper(self, overrides=None, arguments=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"hpc").mkdir()
            # Capture the environment and arguments passed to the existing
            # workflow. This fixture never invokes Slurm or touches scratch.
            capture = (
                "import json,os,sys; print(json.dumps(dict(arguments=sys.argv[1:], "
                "run_dir=os.environ.get(\"MF_RUN_DIR\"), "
                "output_dir=os.environ.get(\"MF_OUTPUT_DIR\"))))"
            )
            (root/"hpc"/"mean_field32_cpu.sbatch").write_text(
                f"#!/bin/bash\nexec python3 -c '{capture}' \"$@\"\n"
            )
            environment = os.environ.copy()
            for key in ("MF_REYNOLDS", "MF_PECLET", "MF_RUN_DIR", "MF_OUTPUT_DIR"):
                environment.pop(key, None)
            environment.update(PROJECT_ROOT=str(root))
            environment.update(overrides or {})
            result = subprocess.run(["bash", str(SCRIPT), *arguments],
                                    env=environment, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

    def test_default_grid_thickness_timestep_and_unchanged_reynolds(self):
        captured = self.wrapper()
        args = dict(zip(captured["arguments"][::2], captured["arguments"][1::2]))
        self.assertEqual(args, {"--n": "64", "--re": "50", "--pe": "50",
            "--transition": "0.01875", "--dt": "0.000625", "--final-time": "0.65",
            "--pressure-max-steps": "48000", "--checkpoint-interval": "5"})
        cfg = MeanFieldConfig(n=int(args["--n"]), transition=float(args["--transition"]),
                              dt=float(args["--dt"]), pressure_max_steps=int(args["--pressure-max-steps"]))
        self.assertEqual(cfg.steps, 1040)
        self.assertEqual(cfg.pressure_tolerance, 1e-8)
        self.assertEqual(cfg.pressure_cfl, .125)
        self.assertIn("sites64-re50-pe50", captured["run_dir"])
        self.assertIn("64x64_re50_pe50", captured["output_dir"])

    def test_optional_reynolds_changes_use_separate_directories(self):
        captured = self.wrapper({"MF_REYNOLDS": "200"})
        args = dict(zip(captured["arguments"][::2], captured["arguments"][1::2]))
        self.assertEqual((args["--re"], args["--pe"]), ("200", "200"))
        self.assertIn("sites64-re200-pe200", captured["run_dir"])
        self.assertIn("64x64_re200_pe200", captured["output_dir"])

    def test_explicit_paths_and_preflight_overrides_are_preserved(self):
        captured = self.wrapper({"MF_RUN_DIR": "/tmp/mf64-run-fixture",
                                 "MF_OUTPUT_DIR": "/tmp/mf64-output-fixture", "MF_PECLET": "100"},
                                ("--max-steps", "3", "--checkpoint-interval", "1"))
        args = dict(zip(captured["arguments"][::2], captured["arguments"][1::2]))
        self.assertEqual(captured["run_dir"], "/tmp/mf64-run-fixture")
        self.assertEqual(captured["output_dir"], "/tmp/mf64-output-fixture")
        self.assertEqual(args["--pe"], "100")
        self.assertEqual(args["--max-steps"], "3")
        self.assertEqual(args["--checkpoint-interval"], "1")


if __name__ == "__main__":
    unittest.main()
