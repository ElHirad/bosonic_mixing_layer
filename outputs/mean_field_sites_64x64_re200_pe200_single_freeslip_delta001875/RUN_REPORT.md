# 64×64 explicit single-site mean-field mixing layer

Completed and validated 1040 steps to t=0.65. Production Slurm job: 11244737. This job resumed from step 3; its measured solver time is 1475.17 minutes, excluding earlier checkpointed work.

Re=200, Pe=200; free-slip y boundaries, periodic x. One shear at y=0.5, tanh thickness 0.01875, KH width 0.12. Mode 2 amplitude 2.5; mode 1 amplitude 0.5. Physical dt=0.000625; boson cutoff=12.

Predictor, pressure relaxation, and correction all evolve explicit single-site bosonic operators. DNS below is a separate projected-midpoint benchmark, initialized from the exact measured mean-field initial fields; it is not a stage of the mean-field solver. No scalar-mass repair or clipping is used.

## Plots

- [Vorticity and velocity](./mean_field_vorticity.png)
- [Concentration](./mean_field_concentration.png)
- [Pairing and numerical diagnostics](./mean_field_diagnostics.png)
- [MF vs DNS vorticity](./mean_field_vs_dns_vorticity.png)
- [MF vs DNS concentration](./mean_field_vs_dns_concentration.png)
- [MF vs DNS errors, pairing, and conservation](./mean_field_vs_dns_diagnostics.png)

## Validation and limitations

Maximum raw relative scalar-mass drift: 1.066e-14. Maximum relative divergence: 2.530e-11. All-step concentration range: [-0.059699, 1.0625]. Centered transport is not bound-preserving; any overshoots are retained.

Final MF/DNS relative L2 differences: velocity 4.582%, vorticity 11.71%, concentration 2.63%. The two methods use different temporal splitting.

Final mode-1/mode-2 ratio: 1.5856. Inspect the spatial evolution to assess roll-up and pairing: subharmonic dominance alone can also result from faster decay of the primary mode. These large initial perturbations do not test linear KH instability. This run by itself does not establish grid or timestep convergence.

[Full numerical summary](./summary.json) · [Mean-field validation](./validation.json) · [DNS validation](./dns_matched_snapshots.validation.json)

## Post-run spatial inspection

The mean-field concentration and vorticity snapshots show clear spiral roll-up
and stronger mutual vortex rotation than the Re=Pe=50 case. Two distinct cores
remain at t=0.65, so a completed merger is not demonstrated. The larger MF/DNS
discrepancy warrants a timestep-convergence study before claiming quantitative
accuracy; its cause has not been isolated by this run. Passing the conservation,
pressure, and local-state gates does not substitute for that study.

This assessment was added after reviewing the generated plots. The production
Slurm allocation lasted 24 hours 36 minutes 12 seconds, including validation,
DNS comparison, and plotting; earlier preflight work is additional.
