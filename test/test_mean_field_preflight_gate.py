"""Ensure partial or failed tests cannot release long production jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1]/"hpc"/"mean_field128_reaction_preflight.sbatch"


class PreflightGateTests(unittest.TestCase):
    script = SCRIPT
    pipeline = "mean_field128_reaction_ptol1e7_cpu.sbatch"
    required_steps, total_steps, n, re_pe, dt, method = 5, 2080, 128, 200., .0003125, "forward-euler"

    def fixture(self, *, completed=None, checkpoint_step=None, state="partial", exit_code=0,
                pressure_tolerance=1e-7, history_steps=None, arguments=()):
        completed = self.required_steps if completed is None else completed
        checkpoint_step = self.required_steps if checkpoint_step is None else checkpoint_step
        history_steps = range(1, self.required_steps+1) if history_steps is None else history_steps
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"hpc").mkdir()
            # Fake solver: never submits work or evolves fields. The persisted
            # fixtures below model its terminal checkpoint/status only.
            (root/"hpc"/self.pipeline).write_text(
                f'#!/bin/bash\n[[ "$*" == "--max-steps {self.required_steps}" ]] || exit 64\n'
                f'exit {exit_code}\n')
            (root/"reaction_dns_preflight.py").write_text("# Test fixture: DNS stage succeeds without evolution.\n")
            run = root/"run"
            run.mkdir()
            (run/"run_status.json").write_text(json.dumps(dict(
                state=state, completed_step=completed, requested_steps=self.total_steps)))
            config = dict(n=self.n, reynolds=self.re_pe, peclet=self.re_pe, scalar_species=3,
                          damkohler=10., dt=self.dt, final_time=.65,
                          time_integrator=self.method, pressure_tolerance=pressure_tolerance)
            np.savez(run/"checkpoint.npz", completed_step=checkpoint_step,
                     requested_steps=self.total_steps, config_json=json.dumps(config),
                     history_json=json.dumps([dict(step=step) for step in history_steps]))
            environment = {**os.environ, "PROJECT_ROOT": str(root), "MF_RUN_DIR": str(run),
                           "MF_DAMKOHLER": "10", "MF_PYTHON": sys.executable,
                           "DNS_PREFLIGHT_OUTPUT": str(root/"dns")}
            return subprocess.run(["bash", str(self.script), *arguments], env=environment,
                                  capture_output=True, text=True)

    def test_only_required_accepted_steps_release_production(self):
        result = self.fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PREFLIGHT_PASSED", result.stdout)

    def test_solver_failure_is_not_overridden_by_a_stale_success_status(self):
        result = self.fixture(exit_code=7)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("PREFLIGHT_PASSED", result.stdout)

    def test_graceful_early_stop_or_failed_status_does_not_pass(self):
        for changes in (dict(completed=2), dict(state="failed")):
            result = self.fixture(**changes)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("PREFLIGHT_PASSED", result.stdout)

    def test_checkpoint_history_or_configuration_mismatch_does_not_pass(self):
        for changes in (dict(checkpoint_step=4), dict(history_steps=(1, 2, 4, 5)),
                        dict(pressure_tolerance=1e-8)):
            result = self.fixture(**changes)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("PREFLIGHT_PASSED", result.stdout)

    def test_cli_cannot_skip_steps_or_override_tolerance(self):
        result = self.fixture(arguments=("--max-steps", "1"))
        self.assertEqual(result.returncode, 64)


class RK4PreflightGateTests(PreflightGateTests):
    script = SCRIPT.with_name("mean_field64_reaction_rk4_preflight.sbatch")
    pipeline = "mean_field64_reaction_rk4_cpu.sbatch"
    required_steps, total_steps, n, re_pe, dt, method = 10, 1040, 64, 100., .000625, "rk4"


if __name__ == "__main__":
    unittest.main()
