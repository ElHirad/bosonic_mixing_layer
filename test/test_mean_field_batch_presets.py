"""Exercise batch presets without submitting jobs or evolving fields."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

from mixing_layer_mean_field import MeanFieldConfig


SCRIPT = Path(__file__).resolve().parents[1]/"hpc"/"mean_field64_cpu.sbatch"


class BatchPresetTests(unittest.TestCase):
    def wrapper(self, overrides=None, arguments=(), script=SCRIPT):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"hpc").mkdir()
            # Capture the environment and arguments passed to the existing
            # workflow. This fixture never invokes Slurm or touches scratch.
            capture = (
                "import json,os,sys; print(json.dumps(dict(arguments=sys.argv[1:], "
                "run_dir=os.environ.get(\"MF_RUN_DIR\"), "
                "output_dir=os.environ.get(\"MF_OUTPUT_DIR\"), "
                "reynolds=os.environ.get(\"MF_REYNOLDS\"), peclet=os.environ.get(\"MF_PECLET\"), "
                "cutoff=os.environ.get(\"MF_CUTOFF\"), pressure_cfl=os.environ.get(\"MF_PRESSURE_CFL\"), "
                "substeps=os.environ.get(\"MF_SUBSTEPS\"))))"
            )
            (root/"hpc"/"mean_field32_cpu.sbatch").write_text(
                f"#!/bin/bash\nexec python3 -c '{capture}' \"$@\"\n"
            )
            (root/"hpc"/"mean_field64_cpu.sbatch").write_text(
                f"#!/bin/bash\nexec python3 -c '{capture}' \"$@\"\n"
            )
            # Exercise the real 128 wrapper too when testing its tolerance-only
            # extension; the 32 wrapper above still intercepts actual execution.
            shutil.copy2(SCRIPT.with_name("mean_field128_reaction_cpu.sbatch"),
                         root/"hpc"/"mean_field128_reaction_cpu.sbatch")
            environment = os.environ.copy()
            for key in ("MF_REYNOLDS", "MF_PECLET", "MF_RUN_DIR", "MF_OUTPUT_DIR", "MF_DAMKOHLER"):
                environment.pop(key, None)
            environment.update(PROJECT_ROOT=str(root))
            environment.update(overrides or {})
            result = subprocess.run(["bash", str(script), *arguments],
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

    def test_three_reactive_cases_use_separate_paths_and_same_re_pe(self):
        paths = []
        for da in ("1", "10", "100"):
            captured = self.wrapper({"MF_DAMKOHLER": da}, script=SCRIPT.with_name("mean_field64_reaction_cpu.sbatch"))
            args = dict(zip(captured["arguments"][::2], captured["arguments"][1::2]))
            self.assertEqual(args, {"--scalar-species": "3", "--damkohler": da,
                                    "--time-integrator": "forward-euler"})
            self.assertEqual((captured["reynolds"], captured["peclet"]), ("100", "100"))
            self.assertIn(f"-da{da}-three-species", captured["run_dir"])
            self.assertIn(f"_da{da}_reaction", captured["output_dir"])
            self.assertIn("euler", captured["run_dir"])
            self.assertIn("euler", captured["output_dir"])
            paths.append(captured["run_dir"])
        self.assertEqual(len(set(paths)), 3)

    def test_reaction_preset_requires_an_explicit_valid_damkohler(self):
        for environment in ({}, {"MF_DAMKOHLER": "2"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.wrapper(environment, script=SCRIPT.with_name("mean_field64_reaction_cpu.sbatch"))

    def test_128_reaction_preset_matches_successful_probe_configuration(self):
        paths = []
        for da in ("1", "10", "100"):
            captured = self.wrapper({"MF_DAMKOHLER": da, "MF_CUTOFF": "4",
                                     "MF_PRESSURE_CFL": "0.2", "MF_SUBSTEPS": "2"},
                                    script=SCRIPT.with_name("mean_field128_reaction_cpu.sbatch"))
            args = dict(zip(captured["arguments"][::2], captured["arguments"][1::2]))
            self.assertEqual(args, {"--n": "128", "--re": "200", "--pe": "200",
                "--transition": "0.01875", "--dt": "0.0003125", "--final-time": "0.65",
                "--scalar-species": "3", "--damkohler": da, "--time-integrator": "forward-euler",
                "--pressure-tolerance": "1e-8", "--pressure-max-steps": "192000",
                "--checkpoint-interval": "1"})
            self.assertEqual((captured["reynolds"], captured["peclet"]), ("200", "200"))
            self.assertEqual((captured["cutoff"], captured["pressure_cfl"], captured["substeps"]),
                             ("12", "0.125", "8"))
            cfg = MeanFieldConfig(n=int(args["--n"]), reynolds=float(args["--re"]),
                peclet=float(args["--pe"]), transition=float(args["--transition"]),
                dt=float(args["--dt"]), final_time=float(args["--final-time"]),
                scalar_species=int(args["--scalar-species"]), damkohler=float(da),
                time_integrator=args["--time-integrator"],
                pressure_tolerance=float(args["--pressure-tolerance"]),
                pressure_max_steps=int(args["--pressure-max-steps"]),
                boson_cutoff=int(captured["cutoff"]), pressure_cfl=float(captured["pressure_cfl"]),
                predictor_substeps=int(captured["substeps"]), correction_substeps=int(captured["substeps"]))
            self.assertEqual(cfg.steps, 2080)
            probe = SCRIPT.parents[1]/"outputs"/"reaction_128x128_re200_pe200_euler_series"/f"mf_euler_probe_da{da}_substeps8.json"
            self.assertEqual(asdict(cfg), json.loads(probe.read_text())["config"])
            self.assertIn(f"sites128-re200-pe200-da{da}-three-species-euler", captured["run_dir"])
            self.assertIn(f"128x128_re200_pe200_da{da}_reaction_euler", captured["output_dir"])
            paths.append(captured["run_dir"])
        self.assertEqual(len(set(paths)), 3)

    def test_128_preset_requires_valid_rate_and_preserves_restart_overrides(self):
        script = SCRIPT.with_name("mean_field128_reaction_cpu.sbatch")
        for environment in ({}, {"MF_DAMKOHLER": "2"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.wrapper(environment, script=script)
        captured = self.wrapper({"MF_DAMKOHLER": "10", "MF_RUN_DIR": "/tmp/mf128-run-fixture",
                                 "MF_OUTPUT_DIR": "/tmp/mf128-output-fixture"},
                                ("--max-steps", "3"), script=script)
        self.assertEqual(captured["run_dir"], "/tmp/mf128-run-fixture")
        self.assertEqual(captured["output_dir"], "/tmp/mf128-output-fixture")
        self.assertEqual(captured["arguments"][-2:], ["--max-steps", "3"])
        directives = script.read_text()
        self.assertIn("#SBATCH --qos=htc-htc-ll", directives)
        self.assertIn("#SBATCH --time=21-00:00:00", directives)
        self.assertIn("#SBATCH --signal=B:USR1@3600", directives)

    def test_relaxed_preset_changes_only_pressure_tolerance_and_paths(self):
        paths = []
        for da in ("1", "10", "100"):
            original = self.wrapper({"MF_DAMKOHLER": da},
                                    script=SCRIPT.with_name("mean_field128_reaction_cpu.sbatch"))
            relaxed = self.wrapper({"MF_DAMKOHLER": da},
                                   script=SCRIPT.with_name("mean_field128_reaction_ptol1e7_cpu.sbatch"))
            original_args = dict(zip(original["arguments"][::2], original["arguments"][1::2]))
            relaxed_args = dict(zip(relaxed["arguments"][::2], relaxed["arguments"][1::2]))
            self.assertEqual(relaxed_args, {**original_args, "--pressure-tolerance": "1e-7"})
            for key in ("reynolds", "peclet", "cutoff", "pressure_cfl", "substeps"):
                self.assertEqual(relaxed[key], original[key])
            self.assertEqual(relaxed["run_dir"], original["run_dir"]+"-ptol1e7")
            self.assertEqual(Path(relaxed["output_dir"]).name,
                             Path(original["output_dir"]).name+"_ptol1e7")
            paths.append(relaxed["run_dir"])
        self.assertEqual(len(set(paths)), 3)

    def test_relaxed_preset_requires_valid_rate_and_retains_test_step_limit(self):
        script = SCRIPT.with_name("mean_field128_reaction_ptol1e7_cpu.sbatch")
        for environment in ({}, {"MF_DAMKOHLER": "2"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.wrapper(environment, script=script)
        captured = self.wrapper({"MF_DAMKOHLER": "1", "MF_RUN_DIR": "/tmp/mf128-relaxed-fixture"},
                                ("--max-steps", "5"), script=script)
        self.assertEqual(captured["run_dir"], "/tmp/mf128-relaxed-fixture")
        self.assertEqual(captured["arguments"][-2:], ["--max-steps", "5"])

    def test_64_rk4_reaction_preset_selects_requested_series_and_fresh_paths(self):
        script = SCRIPT.with_name("mean_field64_reaction_rk4_cpu.sbatch")
        paths = []
        for da in ("1", "10", "100"):
            captured = self.wrapper({"MF_DAMKOHLER": da}, script=script)
            args = dict(zip(captured["arguments"][::2], captured["arguments"][1::2]))
            self.assertEqual(args, {"--scalar-species": "3", "--damkohler": da,
                "--time-integrator": "rk4", "--pressure-tolerance": "1e-7", "--checkpoint-interval": "1"})
            self.assertEqual((captured["reynolds"], captured["peclet"]), ("100", "100"))
            self.assertEqual((captured["cutoff"], captured["pressure_cfl"], captured["substeps"]),
                             ("12", "0.125", "8"))
            self.assertIn(f"sites64-re100-pe100-da{da}-three-species-rk4", captured["run_dir"])
            self.assertIn(f"64x64_re100_pe100_da{da}_reaction_rk4_ptol1e7", captured["output_dir"])
            paths.append(captured["run_dir"])
        self.assertEqual(len(set(paths)), 3)
        self.assertIn("#SBATCH --time=3-00:00:00", script.read_text())
        for environment in ({}, {"MF_DAMKOHLER": "2"}):
            with self.assertRaises(subprocess.CalledProcessError):
                self.wrapper(environment, script=script)


if __name__ == "__main__":
    unittest.main()
