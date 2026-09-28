#!/usr/bin/env python3
"""Fast independent DNS safety check for the requested reaction series.

Initial physical fields are measured from the same initial local kets as MF.
This is not a mean-field trajectory or a replacement for the post-run comparison.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_mean_field_dns import diagnostics, dns_config, matched_dns_integrator
from mean_field_bosons import LocalBosons
from mean_field_operators import Layout
import mixing_layer_dns as dns
from mixing_layer_mean_field import MeanFieldConfig, atomic_json, initial_amplitudes, source_fingerprint
import reacting_dns


def run_preflight(damkohler, *, n=64, reynolds=100, peclet=100,
                  time_integrator="forward-euler", dt=.000625, pressure_max_steps=48000,
                  pressure_tolerance=1e-8):
    cfg = MeanFieldConfig(n=n, reynolds=float(reynolds), peclet=float(peclet), transition=.01875,
                          dt=dt, pressure_max_steps=pressure_max_steps, scalar_species=3,
                          damkohler=float(damkohler), time_integrator=time_integrator,
                          pressure_tolerance=pressure_tolerance)
    dns_method = matched_dns_integrator(cfg)
    # Initialization needs only the layout and exact initial Fock kets, not
    # construction of large mean-field generators that this DNS never uses.
    bosons, layout = LocalBosons(cfg.boson_cutoff), Layout(cfg.n, cfg.boundary_y, cfg.scalar_species)
    states = bosons.coherent_states(layout.pack(*initial_amplitudes(cfg)))
    measured = layout.unpack(bosons.amplitudes(states))
    fields = {name: values*({"u": cfg.velocity_scale, "v": cfg.velocity_scale,
                            "phi": cfg.pressure_scale}.get(name, cfg.scalar_scale))
              for name, values in measured.items()}
    c = np.stack([fields[f"c{i}"] for i in (1, 2, 3)])
    u, v = fields["u"], fields["v"]
    reference = np.array([(c[0]+c[2]).mean(), (c[1]+c[2]).mean()])
    physical = dns_config(cfg)
    rows = [dict(step=0, time=0., **diagnostics(u, v, c, np.zeros_like(u), physical, reference, damkohler))]
    for step in range(cfg.steps):
        if cfg.dt > dns.stable_timestep(u, v, physical)*(1+1e-12):
            raise FloatingPointError("reactive DNS preflight exceeds transport safety bound")
        u, v, p, c = reacting_dns.advance_one_step(u, v, c, cfg.dt, physical, damkohler, method=dns_method)
        rows.append(dict(step=step+1, time=(step+1)*cfg.dt,
                         **diagnostics(u, v, c, p, physical, reference, damkohler)))
    source_dir = Path(__file__).resolve().parent
    return {"kind": "independent DNS preflight only, not a mean-field result", "completed_steps": cfg.steps,
            "config": asdict(cfg), "mean_field_source_fingerprint": source_fingerprint(cfg),
            "dns_time_integrator": dns_method,
            "source_sha256": {name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest()
                              for name in ("reaction_dns_preflight.py", "reacting_dns.py", "mixing_layer_dns.py",
                                           "compare_mean_field_dns.py", "mixing_layer_mean_field.py",
                                           "mean_field_operators.py", "mean_field_bosons.py")},
            "conservation_and_divergence_checks_passed": True,
            "positivity_preserved": all(row["scalar_minimum"] >= -1e-12 for row in rows),
            "species_ranges_all_steps": {name: {"minimum": min(row[name+"_minimum"] for row in rows),
                                               "maximum": max(row[name+"_maximum"] for row in rows)}
                                          for name in ("c1", "c2", "c3")},
            "minimum_reaction_rate": min(row["reaction_rate_minimum"] for row in rows),
            "maximum_negative_rate_fraction": max(row["reaction_negative_rate_fraction"] for row in rows),
            "first_negative_step": next((i for i, row in enumerate(rows) if row["scalar_minimum"] < -1e-12), None),
            "worst_invariant_error": max(row["scalar_mass_error"] for row in rows),
            "worst_relative_divergence": max(row["relative_divergence"] for row in rows),
            "sign_statistics": {name: {
                "maximum_negative_fraction": max(row[name+"_negative_fraction"] for row in rows),
                "maximum_material_negative_fraction": max(row[name+"_material_negative_fraction"] for row in rows),
                "maximum_positive_integral": max(row[name+"_positive_integral"] for row in rows),
                "minimum_negative_integral": min(row[name+"_negative_integral"] for row in rows)}
                for name in ("c1", "c2", "c3")},
            "history": rows,
            "final": rows[-1], "scalar_clipping": False, "mass_repair": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--pe", type=float, default=100)
    parser.add_argument("--time-integrator", choices=("forward-euler", "rk4"), default="forward-euler")
    parser.add_argument("--dt", type=float, default=.000625)
    parser.add_argument("--pressure-max-steps", type=int, default=48000,
                        help="MF configuration metadata only; DNS uses its independent projection")
    parser.add_argument("--pressure-tolerance", type=float, default=1e-8,
                        help="MF configuration metadata only")
    parser.add_argument("--damkohler", type=int, choices=(1, 10, 100), nargs="+", default=[1, 10, 100])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for da in args.damkohler:
        destination = args.output_dir/f"dns_preflight_da{da}.json"
        csv_path = args.output_dir/f"dns_sign_history_da{da}.csv"
        if destination.exists() or csv_path.exists():
            raise FileExistsError("DNS diagnostics exist; choose a fresh output directory")
        report = run_preflight(da, n=args.n, reynolds=args.re, peclet=args.pe,
                               time_integrator=args.time_integrator, dt=args.dt,
                               pressure_max_steps=args.pressure_max_steps,
                               pressure_tolerance=args.pressure_tolerance)
        rows = report.pop("history")
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report["sign_history_file"] = csv_path.name
        atomic_json(destination, report)
        print(json.dumps({"Da": da, "steps": report["completed_steps"],
            "ranges": report["species_ranges_all_steps"], "invariant_error": report["worst_invariant_error"],
            "minimum_rate": report["minimum_reaction_rate"]}), flush=True)


if __name__ == "__main__":
    main()
