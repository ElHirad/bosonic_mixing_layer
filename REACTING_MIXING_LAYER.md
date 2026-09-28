# Three-species reacting mixing layer

This extends the explicit single-site bosonic calculation to the irreversible
mass-action reaction

\[
c_1+c_2\longrightarrow c_3,\qquad R=\mathrm{Da}\,c_1c_2.
\]

The current requested cases use Da = 1, 10, and 100 with **RK4**
and **Re=Pe=100 on 64×64**. Tanh thickness remains 0.01875,
dt=0.000625, and final time is 0.65 (1,040 steps).
The KH mode-2/mode-1 seeds retain amplitudes 2.5/0.5,
width 0.12, and phase zero. All three species have Pe=100. The initial
profiles are the previous normalized tanh scalar c1, c2=1-c1, and c3=0.
These are initial conditions only: c2 is independently evolved, not inferred
as 1-c1 after reaction starts.

The current workflow performs full-time DNS screening and ten accepted MF
steps per case before conditionally launching full MF/DNS comparisons.
Positive and negative concentrations are recorded at every physical step,
without clipping. Pressure tolerance remains the approved 1e-7; all other
gates are unchanged. See [64×64 RK4 status](./outputs/reaction_64x64_re100_pe100_rk4_series/STATUS.md).
Earlier 64×64/128×128 Euler results and failures are preserved as history.
The historical midpoint benchmark is not substituted for reacting RK4 DNS.

The scalar concentrations are passive with respect to momentum: no heat
release, density change, buoyancy, or chemical feedback on viscosity is added.
The x direction is periodic; y retains free-slip velocity and zero normal
species flux. These assumptions define the requested reaction extension;
they are not a compressible reacting-flow or chemical master-equation model.

## Equations and nondimensionalization

Let dimensional concentrations be C_i=C_ref c_i, length x=L x', velocity
u=U u', and time t=(L/U)t'. If the dimensional bimolecular rate constant is
k, then Da=k C_ref L/U. With the existing unit reference velocity and length,
the nondimensional coefficient in the code is exactly Da, without a further
factor of grid spacing, initial shear thickness, or boson encoding scale.
Dropping primes,

\[
\partial_t\mathbf u+\nabla\cdot(\mathbf u\mathbf u)
=-\nabla p+\mathrm{Re}^{-1}\nabla^2\mathbf u,
\qquad \nabla\cdot\mathbf u=0,
\]

\[
\partial_t c_i=-\nabla\cdot(\mathbf u c_i)
+\mathrm{Pe}^{-1}\nabla^2c_i+\sigma_i\mathrm{Da}\,c_1c_2,
\qquad (\sigma_1,\sigma_2,\sigma_3)=(-1,-1,+1).
\]

The same conservative centered MAC fluxes and Neumann scalar Laplacian are
used for each species. The reaction is local to a cell. Adding equations
shows that c1+c3 and c2+c3 obey reaction-free advection-diffusion equations.
Periodic/no-flux boundaries therefore conserve

\[
M_{13}=\int(c_1+c_3)\,dA,\qquad M_{23}=\int(c_2+c_3)\,dA.
\]

Their initial values are both 0.5 on the unit square. Consequently the
integral of c1+c2+2c3 is conserved. The individual reactant amounts decrease
and the product increases for nonnegative concentrations. The unweighted
sum c1+c2+c3 is **not** conserved. Pointwise uniformity of these combinations
is not imposed during the temporarily divergent Chorin predictor.

## Reaction as explicit bosonic mean-field operators

Each cell has separate local Fock vectors for c1, c2, and c3, with physical
concentration c_i=s_c alpha_i, alpha_i=<a_i>, and s_c=4. The existing u, v,
and pressure-impulse states are retained. There are 98,432 local vectors at
128×128 (24,640 at 64×64), each with dimension 13 (cutoff 12). The reaction amplitude equations
are represented by the normally ordered generator

\[
G_R=q(-a_1^\dagger-a_2^\dagger+a_3^\dagger)a_1a_2,
\qquad q=\mathrm{Da}\,s_c.
\]

The factor follows from dot(alpha_i)=dot(c_i)/s_c:
sigma_i Da (s_c alpha_1)(s_c alpha_2)/s_c.
This is a deterministic coherent-state embedding, not a stochastic chemical
master-equation propensity generator.

For the full normally ordered generator G=sum_r a_r^dagger F_r(a), the
implemented first-order coherent mean-field decoupling gives

\[
K_j=f_j a_j^\dagger+b_j a_j+d_j I,
\qquad f_j=F_j(\alpha),\quad
b_j=\sum_r\alpha_r^*\frac{\partial F_r}{\partial\alpha_j},\quad
d_j=-\alpha_j b_j.
\]

Writing B=-alpha_1^*-alpha_2^*+alpha_3^*, the reaction contributions are

\[
(f_1,f_2,f_3)=q\alpha_1\alpha_2(-1,-1,+1),
\quad b_1=qB\alpha_2,\quad b_2=qB\alpha_1,\quad b_3=0,
\quad d_j=-\alpha_j b_j.
\]

All three local terms are kept, including annihilation and identity terms.
The transport and reaction coefficients are combined in the predictor;
they are recomputed from the current local states at every RK4 stage.
The actual dynamical variable remains the ket, evolved by

\[
\frac{d|\psi_j\rangle}{dt}
=\bigl(K_j-\operatorname{Re}\langle K_j\rangle I\bigr)|\psi_j\rangle.
\]

For the current RK4 path, write the coupled ket derivative above as D(psi).
Each substep h computes

\[
k_1=D(\psi),\quad k_2=D(\psi+h k_1/2),\quad
k_3=D(\psi+h k_2/2),\quad k_4=D(\psi+h k_3),
\]
\[
\psi^{\rm new}=\operatorname{normalize}\left[
\psi+\frac{h}{6}(k_1+2k_2+2k_3+k_4)\right].
\]

All explicit local operator coefficients/actions are reevaluated at each
stage, not applied to an amplitude-only surrogate. Predictor and correction
each use eight subdivisions. Pressure relaxation also uses RK4, with
pseudo-step 0.125/n². Fourth-order substeps do not establish fourth-order
accuracy of the complete Chorin-split MF trajectory.

For reference, the archived forward-Euler path applies each substep literally
to the Fock vector:

\[
|\widetilde\psi_j\rangle=|\psi_j\rangle+h\dot{|\psi_j\rangle},
\qquad |\psi_j^{\rm new}\rangle=
|\widetilde\psi_j\rangle/\|\widetilde\psi_j\|.
\]

Only the ket norm is normalized; its shape is not replaced by a coherent
state. Each substep evaluates the operator derivative once. There are no RK
or exponential-integrator substeps in the forward-Euler path. The previous
eight predictor/correction subdivisions are retained initially; pressure
relaxation in those archived runs also uses forward Euler with pseudo-step
0.125/n². Its diffusion stability ceiling is pressure_cfl<=0.25 versus the
RK4 bound 0.30.
First-order ket integration need not preserve coherence or field invariants
to the old tolerances; the existing gates are retained and failures reported.

There is no separate classical reaction update, amplitude-only evolution,
exact chemical ODE replacement, clipping, scalar-mass repair, or coherent
state reset. Pressure relaxation and velocity correction continue to evolve
the explicit local states under their own mean-field operators, without DNS
or a direct pressure inversion. Reaction acts only over physical predictor
time, never during pressure pseudo-time or the correction parameter interval.
Finite-cutoff coherent-state defects and inactive-field leakage remain
subject to the previous gates; exact coherent invariance is not assumed at
finite truncation.

## Separate DNS and comparison

The current benchmark independently evaluates the coupled velocity/scalar
equations with classical four-stage RK4. Each momentum RHS is projected by
the linear discrete Helmholtz projector P: F(Y)=(P momentum_rhs, species_rhs),
Y=(u,v,c1,c2,c3). Intermediate velocities are divergence-free; species use
the same RK stage fields. Stored pressure is the RK-stage-weighted step
average, not a fourth-order endpoint-pressure claim. Its initial u, v, c1, c2,
and c3 are copied from the saved measured
mean-field initial fields, not reconstructed from nominal profiles. Da,
Re, Pe, grid, boundary conditions, timestep, and eight snapshot times must
match. The integrator is also checked; a midpoint reference cannot be labelled
as current reacting RK4 DNS. A source fingerprint additionally
prevents mixing up the Da cases.
The DNS pressure projection is confined to the benchmark and tests.

For each species, the output includes separate MF and DNS snapshots with
matched colour limits, a side-by-side MF/DNS/signed-difference figure, and
relative L2 and maximum-absolute differences at every snapshot. c3 is included
as a reaction diagnostic in addition to the requested c1 and c2. The CSV,
JSON summary, and report preserve all three species. The historical NPZ
`concentration` key aliases c1; c2/c3 are separately stored and validated.

At t=0 and every accepted physical step, both solvers record species minima,
maxima, positive/negative/zero cell counts and fractions, and signed amounts
M+=mean(max(c,0)), M-=mean(min(c,0)). Unit domain area makes these means
integrals, and M+ + M- equals the species amount. M- is nonpositive, not its
absolute magnitude. Separate c<-1e-12 counts distinguish material undershoots
from raw c<0 roundoff. These are observations, not alterations of the fields.
CSV and plots compare the signed histories; internal RK-stage extrema are
not included in the physical-step statistics.

## Validation and numerical limitations

Tests cover initial complementarity, reaction coefficients including all
local operator terms, transport/reaction stencils, reaction invariants,
zero-Da recovery, passive momentum, exact restart, stored-state validation,
matched DNS species/Da, and end-to-end plot generation. For a homogeneous
mixture c1(0)=c2(0)=a and c3(0)=0, the analytic solution is

\[
c_1(t)=c_2(t)=\frac{a}{1+\mathrm{Da}\,a t},\qquad
c_3(t)=a-c_1(t).
\]

This checks the rate normalization for Da=1,10,100. New tests verify the
literal normalized Euler ket update, a single derivative per substep,
forward Euler in all three mean-field stages, and first-order convergence
of both a fixed-operator ket problem and homogeneous DNS chemistry. Legacy
RK4/midpoint tests are retained as regression checks. New tests verify
fourth-order DNS chemistry and discrete diffusion in both boundary geometries,
four coupled RHS evaluations, unclipped signs, and complete per-step CSV.
None of these establishes
convergence of the full mixing-layer calculation.

Every accepted MF step checks the two integral invariants (maximum relative
drift 1e-7), divergence, pressure residual (currently 1e-7), walls, and the unchanged
local-state gates. In the three-species data `scalar_mass_error` means the
maximum of the M13/M23 relative errors, while `scalar_mass` is simply the
mean across all three species, **not** a conserved mass. Explicit
`c1_mass`, `c2_mass`, `c3_mass`, and invariant diagnostics remove that ambiguity.

The predictor substep additionally requires
dt/substeps * Da * max(|c1|+|c2|) <= 0.25; DNS uses the same safety check
with its full dt. Advective/diffusive checks still apply. Only pressure was
relaxed from 1e-8 to 1e-7 at the user's request; coherence remains limited to 1e-5.

The inherited centered transport is **not positivity-preserving**. Negative
species concentrations can produce negative Da c1 c2 and negative product,
which are numerical artifacts, not physically valid reverse reaction. The
implementation reports species extrema, minimum/mean rate, and the fraction
of cells with negative rate, without clipping or silently changing the
transport discretization. Numerical gate success does not certify physical
positivity. A positivity-preserving spatial discretization would be a
separate change to both solvers. The initial shear remains only a few cells
wide; no grid-convergence or completed-vortex-merger claim is made.

## Cluster execution

The current 64×64 workflow uses `hpc/mean_field64_reaction_rk4_preflight.sbatch`
for ten MF steps plus full DNS screening per Da. Full
`hpc/mean_field64_reaction_rk4_cpu.sbatch` jobs wait for all three preflights
to pass, resume checkpoints, and publish comparisons. See
[cluster instructions](./hpc/README.md) for dependencies and frozen paths.

The preceding 128×128 request used the DNS-first screening and diagnostic-only
wrapper below. Its pressure cap is 192,000, four times the 64×64 cap because
the pseudo-step is four times smaller. The maximum pseudo-time and pressure
tolerance are unchanged.

```bash
OPENBLAS_NUM_THREADS=1 python3 reaction_dns_preflight.py outputs/reaction_128x128_re200_pe200_euler_series --n 128 --re 200 --pe 200 --dt .0003125 --pressure-max-steps 192000
sbatch --clusters=htc hpc/reaction128_euler_probe.sbatch --damkohler 1
sbatch --clusters=htc hpc/reaction128_euler_probe.sbatch --damkohler 10
sbatch --clusters=htc hpc/reaction128_euler_probe.sbatch --damkohler 100
```

Read the DNS extrema before submitting those pressure probes. Residual
histories are measured without changing the dynamics, and pressure convergence
is distinguished from acceptance of all final candidate-step gates. Diagnostics
refuse to overwrite existing reports; use a fresh `PROBE_OUTPUT` for repeats.

### Earlier 64×64 production preset

Each rate has its own source-frozen run directory, checkpoint, and publication
directory; previous nonreacting data are not overwritten. Submit a short
three-step preflight, then resume the same rate after success:

```bash
sbatch --clusters=htc --export=ALL,MF_DAMKOHLER=1 --time=01:00:00 --signal=B:USR1@600 hpc/mean_field64_reaction_cpu.sbatch --max-steps 3 --checkpoint-interval 1
sbatch --clusters=htc --export=ALL,MF_DAMKOHLER=1 hpc/mean_field64_reaction_cpu.sbatch
```

Repeat separately with `MF_DAMKOHLER=10` and `100`. Wait for each preflight to
finish before submitting its continuation, or use `--dependency=afterok:JOBID`.
The wrapper requires an explicit choice from these three values. It requests
one Turin CPU, 4 GiB, and 48 hours, with checkpoint/restart and a stop signal
15 minutes before the wall limit. A stopped partial trajectory must be resumed;
it is never published as a final result.

Completed, validated trajectories automatically generate the independent DNS,
plots, `comparison_DNS.csv`, and `RUN_REPORT.md` in
`outputs/mean_field_sites_64x64_re100_pe100_da{1,10,100}_reaction_euler`.
Only `postprocess_status.json` state `complete` establishes that the whole
publication pipeline finished. Startup checks in
`outputs/reaction_64x64_re100_pe100_euler_series` are not completed MF results.

The diagnostic-only batch `hpc/reaction_euler_probe.sbatch` applies one actual
64×64 mean-field step for each Da, recording all stage diagnostics and any
failed gates. It does not bypass failures to publish an accepted trajectory.
The full-time DNS screening command is:

```bash
OPENBLAS_NUM_THREADS=1 python3 reaction_dns_preflight.py outputs/reaction_64x64_re100_pe100_euler_series --re 100 --pe 100 --time-integrator forward-euler
```

Adding reaction changes source fingerprints. Old result files remain readable;
old nonreacting checkpoints must use their original frozen sources rather
than being resumed under the new implementation.
