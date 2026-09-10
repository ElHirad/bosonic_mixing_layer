# Mixing layer with explicit single-site bosonic mean field

This is the implemented, operator-based alternative to TDVP. The production
state is a product of **local Fock vectors**, with explicit annihilation,
creation, and identity matrices at every site. It is not an amplitude-only
solver and does not call the DNS, an FFT/DCT pressure solver, a sparse linear
solver, or TDVP. Predictor, pressure relaxation, and correction all use the
same single-site operator evolution.

The original [coherent-state derivation](./COHERENT_MEAN_FIELD_MIXING_LAYER.md)
explains why this is possible. That note also discusses direct-amplitude and
direct-Poisson alternatives; **those alternatives are not the implemented
production algorithm**. Numerical integration of local quantum states is
still a classical computation, not a claim of quantum-hardware execution.

## 1. State and explicit matrices

For each MAC variable/site $j$, store

$$
|\psi_j\rangle=\sum_{m=0}^{N_b}\psi_{j,m}|m\rangle,\qquad
|\Psi\rangle=\bigotimes_j|\psi_j\rangle.
$$

The local matrices are

$$
a=\sum_{m=1}^{N_b}\sqrt m\,|m-1\rangle\langle m|,
\quad a^\dagger=a^{\mathsf T},\quad I=I_{N_b+1}.
$$

Set $\alpha_j=\langle\psi_j|a|\psi_j\rangle/
\langle\psi_j|\psi_j\rangle$. Physical observables are
$u=8\alpha_u$, $v=8\alpha_v$, $\phi=4\alpha_\phi$,
and $c=4\alpha_c$ in the reference case. Here $\phi=\Delta t\,p$.
The amplitudes are **measured to form coupling coefficients**, not integrated
as independent state variables.

Only initialization constructs coherent kets, using the recurrence
$\psi_{j,m}=\psi_{j,m-1}\alpha_j/\sqrt m$ and normalization.
During evolution, kets are never reset or projected onto coherent states.

## 2. Fully decouple every operator product

The non-Hermitian bosonic generator has the normally ordered form

$$
\mathcal G=\sum_r a_r^\dagger F_r(a),\qquad
F_r(a)=\sum_s L_{rs}a_s+\sum_{s,t}Q_{rst}a_sa_t.
$$

Expand every operator around its mean and drop products of fluctuations:

$$
a_r^\dagger a_s\simeq
\alpha_s a_r^\dagger+\alpha_r^*a_s-\alpha_r^*\alpha_s I,
$$

$$
a_r^\dagger a_sa_t\simeq
\alpha_s\alpha_t a_r^\dagger
+\alpha_r^*\alpha_t a_s+\alpha_r^*\alpha_s a_t
-2\alpha_r^*\alpha_s\alpha_t I.
$$

This linearizes repeated-site terms as well; when $s=t$, both annihilation
contributions must be counted. Summing all monomials gives

$$
\mathcal G_{\rm MF}=\sum_j K_j,\qquad
K_j=f_j a_j^\dagger+b_j a_j+c_jI,
$$

$$
f_j=F_j(\alpha),\qquad
b_j=\sum_r\alpha_r^*\frac{\partial F_r(\alpha)}{\partial\alpha_j},
\qquad c_j=-\alpha_jb_j.
$$

The distribution of scalar terms among sites is a normalization/phase
convention; their total equals that of the full factorization. In particular,
the implementation keeps the $b_j a_j$ terms: it does not use the
creation-only coherent-tangent shortcut.

For normalized local kets, evolve

$$
\frac{d|\psi_j\rangle}{dt}
=\left[K_j-\operatorname{Re}\langle K_j\rangle I\right]|\psi_j\rangle.
$$

Indeed, differentiating $\langle\psi_j|\psi_j\rangle$ gives
$2\operatorname{Re}\langle K_j\rangle-2\operatorname{Re}\langle K_j\rangle=0$.
This subtraction and endpoint normalization fix the norm of a quantum ket;
neither alters a scalar field to enforce its mass. RK4 advances the **Fock
vectors**, rebuilding $\alpha$, $f$, $b$, and $c$ at all four stages.

## 3. All three dynamical generators

The conservative MAC coefficients are explicitly assembled into bosonic
monomials in [mean_field_operators.py](./mean_field_operators.py). Sparse
matrices store coefficients of $a_r^\dagger a_s$; they are not inverse
pressure matrices or field propagators. Quadratic terms store
$a_r^\dagger a_sa_t$ and are expanded from the centered face averages.
The full spatial derivation is in Sections 5–7 of the coherent-state note.

| Stage | Creation coefficient $f$ before local-state evolution | Evolution interval |
|---|---|---|
| Predictor | $f_u=-s_uN_u(\alpha_u,\alpha_v)+\nu L_N\alpha_u$; $f_v=-s_uN_v+\nu L_D\alpha_v$; $f_c=-s_uN_c+\kappa L_N\alpha_c$; $f_\phi=0$ | Physical $\Delta t$ |
| Pressure | $f_\phi=L_N\alpha_\phi-(s_u/s_\phi)D\boldsymbol\alpha_u$; other creation coefficients zero | Artificial time, until residual tolerance |
| Correction | $f_u=-(s_\phi/s_u)G_x\alpha_\phi$; $f_v=-(s_\phi/s_u)G_y\alpha_\phi$; $f_\phi=f_c=0$ | Auxiliary interval $[0,1]$ |

For **each** row, derive $b$ and $c$ by the same rule in Section 2 and apply
the explicit matrices to **all** local kets. There is no direct update
$\mathbf u\leftarrow\mathbf u^\star-G\phi$ in the production code.
The pressure residual is measured from $f_\phi$, but pressure is reduced
only by repeated single-site evolution, never by solving a linear system.

The pressure generator contains annihilation terms on nominally frozen
velocity sites; the correction generator contains annihilation terms on
pressure sites. On ideal coherent kets those operators act as scalars.
Finite cutoff and integration error can cause leakage. The code does not
delete these terms or freeze their measured fields by hand: it evolves them
and measures pressure-stage velocity/scalar leakage and correction-stage
pressure/scalar leakage. Excessive leakage fails the run.

Pressure relaxation warm-starts from the existing **pressure kets**, not a
re-encoded pressure array. Its RK4 step is
$\Delta\tau=\texttt{pressure\_cfl}/n^2$. The production value is
$0.125/16^2$, with a residual target of $10^{-8}$. This is stricter than the
old MPS target $10^{-3}$ and does not change Re, Pe, or the physical timestep.
The pressure mean is monitored, not subtracted. No scalar-mean repair or
uniform displacement repair is applied.

The physical step remains the predictor–relaxation–correction Chorin split.
Using RK4 within each stage does **not** make the complete split fourth order.

## 4. Grid and last physical parameter series

The unit square has 16×16 cells. The channel stores 256 $u$, 272 $v$,
256 pressure, and 256 concentration sites: 1,040 local kets. Both wall-$v$
rows are explicit vacuum states. Their operators are eliminated from the
generators, so their velocity stays zero without post-step clamping.

| Parameter | Value |
|---|---|
| Reynolds / Péclet | Re = Pe = 50; $\nu=\kappa=0.02$ |
| Boundaries | Periodic $x$; $\partial_yu=0$, $v=0$, $\partial_y\phi=\partial_yc=0$ at both $y$ walls |
| Shear | One tanh layer at $y=0.5$, thickness 0.04 |
| KH seed | Width 0.12; mode 2 amplitude 2.5; mode 1 amplitude 0.5, phase 0 |
| Time | $\Delta t=0.0025$, $T=0.65$: 260 physical steps |
| Output steps | 0, 37, 74, 111, 149, 186, 223, 260 |
| Production local basis | $N_b=12$: occupations 0,…,12 (dimension 13) |
| Local RK4 subdivisions | Eight predictor and eight correction substeps |

The initialization uses a discrete streamfunction curl, so no initial pressure
projection is necessary. The concentration is the same normalized single-tanh
0-to-1 profile as the previous run. Its initial mean is 1/2.

The old occupation cutoff 4 is not carried over silently: its initial
maximum coherent-eigenstate defect is about $7.9\times10^{-4}$ for this seed.
At cutoff 12 it is about $2.7\times10^{-11}$, and at cutoff 16 about
$1.6\times10^{-15}$. Production is checked against cutoff 16 and a separate
run with half-sized local integration steps. Cutoff convergence does not
establish spatial or physical-timestep convergence.

## 5. Run, restart, and validation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s test -p 'test_*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 mixing_layer_mean_field.py --output-dir /path/to/new/run
python3 mixing_layer_mean_field.py --validate-results /path/to/new/run/mean_field_snapshots.npz
sbatch --clusters=htc hpc/mean_field_cpu.sbatch
```

Use `--resume` with the same configuration and output directory. Checkpoints
store the actual local kets, their complete snapshot history, per-step
diagnostics, and a source/configuration/library fingerprint. Restarts refuse
incompatible fingerprints or concurrent writers. Slurm freezes the three
production source files into the run directory. A stop signal requests an
atomic checkpoint at the next physical-step boundary. Failed candidates do
not replace the last accepted state.

The result file contains local-state snapshots as well as measured velocity,
pressure impulse/pressure, concentration, and vorticity. The independent
validator recomputes fields from the saved Fock vectors and checks initial
encoding, terminal state, output times, all per-step gates, and conservation.
Tests also compare operator coefficients to independently implemented MAC
stencils. Such reference computations are tests, not stages of the solver.

The main quality limits are raw relative scalar-mass drift $10^{-7}$,
relative divergence $10^{-7}$, pressure residual $10^{-8}$, stage leakage
$10^{-7}$, ceiling probability $10^{-8}$, and coherent-eigenstate defect
$10^{-5}$. Actual measured errors, not just pass/fail, are retained.

## 6. Interpretation and plotting

Use [plot_mean_field_results.py](./plot_mean_field_results.py) on a completed
result. It plots measured bosonic observables and can compare with existing
saved MPS/DNS data; it does not run a DNS simulation. The Fourier transform
in the pairing diagnostic is post-processing, not a pressure solver.

The separate [compare_mean_field_dns.py](./compare_mean_field_dns.py) program
generates a matched reference from the **exact measured initial** $u,v,c$
fields of a validated mean-field result. It runs the independent
projected-midpoint DNS with the same grid, physical parameters, timestep,
and output schedule. Neither that program nor the DNS module is imported by
the mean-field solver. See the [comparison plots and reproduction commands](./outputs/mean_field_sites_16x16_re50_pe50_single_freeslip/RUN_REPORT.md).

At 16×16 and nominal Re=50, these deliberately large disturbances test
finite-amplitude roller interaction, not a grid-converged linear KH
instability. Assess roll-up and pairing using spatial vorticity and scalar
snapshots alongside the mode-1/mode-2 diagnostic. A growing ratio alone can
also result from faster viscous decay of mode 2 and is not proof of merger.

Conservation is not the same as positivity. The inherited centered scalar
flux is not bound-preserving: the completed production trajectory briefly
reaches approximately $-0.073\le c\le1.082$. These values are retained and
shown in the all-step diagnostics, not clipped or repaired. A requirement
that concentration remain in $[0,1]$ would need a separately designed
bound-preserving transport formulation, not just a larger boson basis.
