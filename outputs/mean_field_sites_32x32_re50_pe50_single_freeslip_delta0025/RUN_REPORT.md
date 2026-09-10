# 32×32 explicit single-site mean-field mixing layer

Completed and validated 520 steps to t=0.65. Production Slurm job: 11216428. This job resumed from step 28; its measured solver time is 57.57 minutes, excluding earlier checkpointed work.

Re=50, Pe=50; free-slip y boundaries, periodic x. One shear at y=0.5, tanh thickness 0.025, KH width 0.12. Mode 2 amplitude 2.5; mode 1 amplitude 0.5. Physical dt=0.00125; boson cutoff=12.

Predictor, pressure relaxation, and correction all evolve explicit single-site bosonic operators. DNS below is a separate projected-midpoint benchmark, initialized from the exact measured mean-field initial fields; it is not a stage of the mean-field solver. No scalar-mass repair or clipping is used.

## Plots

- [Vorticity and velocity](./mean_field_vorticity.png)
- [Concentration](./mean_field_concentration.png)
- [Pairing and numerical diagnostics](./mean_field_diagnostics.png)
- [MF vs DNS vorticity](./mean_field_vs_dns_vorticity.png)
- [MF vs DNS concentration](./mean_field_vs_dns_concentration.png)
- [MF vs DNS errors, pairing, and conservation](./mean_field_vs_dns_diagnostics.png)

## Validation and limitations

Maximum raw relative scalar-mass drift: 3.797e-14. Maximum relative divergence: 8.323e-11. All-step concentration range: [-0.0200416, 1.02454]. Centered transport is not bound-preserving; any overshoots are retained.

Final MF/DNS relative L2 differences: velocity 0.5508%, vorticity 1.261%, concentration 0.3339%. The two methods use different temporal splitting.

Final mode-1/mode-2 ratio: 4.8698. Inspect the spatial evolution to assess roll-up and pairing: subharmonic dominance alone can also result from faster decay of the primary mode. These large initial perturbations do not test linear KH instability. This run by itself does not establish grid or timestep convergence.

## Post-run spatial inspection

Inspection of the saved vorticity and concentration snapshots shows early
roll-up followed by broadening and weakening of the vortices. A strong, clean
pairing/merger event is not established by these fields. The increasing
subharmonic-to-primary ratio must not be presented as proof of such a merger.
This assessment was added after reviewing the automatically generated plots.

[Full numerical summary](./summary.json) · [Mean-field validation](./validation.json) · [DNS validation](./dns_matched_snapshots.validation.json)
