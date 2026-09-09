# Mixing-layer simulation by coherent amplitudes instead of TDVP

This note derives a direct coherent-amplitude formulation for the velocity,
pressure impulse, and conserved concentration in this repository. It describes
the proposed replacement for the bosonic MPS time evolution; it does not claim
that a new solver has already been implemented or run.

The central result is

$$
\frac{d}{dt}|z\rangle=G|z\rangle,\qquad
G=\sum_r a_r^\dagger F_r(a)
\quad\Longrightarrow\quad
\boxed{\dot\alpha_r=F_r(\alpha).}
$$

For an initial coherent product state and canonical, untruncated bosons, this
is an exact reduction on the coherent-state manifold. The usual mean-field
factorization is exact there. We can therefore advance a few arrays of
amplitudes, without constructing MPOs or running TDVP. For this encoding the
resulting equations are the original classical, spatially discretized fluid
equations.

The operator and stencil conventions below follow
[mixing_layer_mps_mac.jl](./mixing_layer_mps_mac.jl), especially
`build_channel_predictor_mpo`, `build_channel_pressure_relaxation_mpo`,
`build_channel_pressure_correction_mpo`, and `chorin_step`.
The existing array implementation in
[mixing_layer_dns.py](./mixing_layer_dns.py) supplies a reference.

## 1. Which evolution equation are we reducing?

The repository calls its bosonic operators Hamiltonians, but the solver
advances a **non-Hermitian generator** with positive real evolution time:

$$
\partial_t|z\rangle=G|z\rangle.
$$

It does not apply the unitary Schrödinger propagator
$\exp(-itH)$ for a Hermitian $H$. To write the same equation as
$i\partial_t|z\rangle=H|z\rangle$, one would set $H=iG$, generally
non-Hermitian. This distinction fixes the sign and the absence of $-i$ in
the amplitude equations below.

The generator has the normally ordered form

$$
G=\sum_r a_r^\dagger
\left[b_r+\sum_s A_{rs}a_s+\sum_{s,t}B_{rst}a_sa_t\right].
$$

There is exactly one creation operator in each monomial. Linear terms
$a_r^\dagger a_s$ encode diffusion, pressure relaxation, and pressure
correction; cubic terms $a_r^\dagger a_sa_t$ encode quadratic momentum
and scalar-advection fluxes. Keeping these nonlinear terms is essential.
Replacing the whole generator by its scalar expectation value would remove
the field evolution.

This type of embedding relates nonlinear classical equations to linear
evolution in a bosonic Hilbert space. For background, see Kowalski,
[*Nonlinear dynamical systems and classical orthogonal polynomials*,
J. Math. Phys. 38, 2483–2505 (1997)](https://arxiv.org/abs/solv-int/9801018).
The derivation here is specialized directly to our generator and MAC grid.

## 2. Coherent states and the exact amplitude equation

### 2.1 Use unnormalized states to expose the algebra

Let $[a_r,a_s^\dagger]=\delta_{rs}$ and define

$$
|\alpha))=\exp\left(\sum_r\alpha_ra_r^\dagger\right)|0\rangle.
$$

Then

$$
a_r|\alpha))=\alpha_r|\alpha)),\qquad
\frac{\partial|\alpha))}{\partial\alpha_r}
=a_r^\dagger|\alpha)).
$$

For any polynomial $F_r$ in annihilation operators,

$$
G|\alpha))
=\sum_r F_r(\alpha)a_r^\dagger|\alpha)).
$$

On the other hand,

$$
\frac{d}{dt}|\alpha(t)))
=\sum_r\dot\alpha_r a_r^\dagger|\alpha(t))).
$$

Thus the coherent trajectory solves the generator equation when

$$
\boxed{
\dot\alpha_r=b_r+\sum_s A_{rs}\alpha_s+
\sum_{s,t}B_{rst}\alpha_s\alpha_t.
}
$$

This result is valid while the classical solution exists and the operator
action is defined. It applies to the polynomial stencils used here. No
correlation hierarchy or TDVP optimization is needed on this manifold.

### 2.2 Normalization does not alter the amplitudes

The normalized state is

$$
|\alpha\rangle=e^{-\|\alpha\|^2/2}|\alpha)).
$$

For $\dot\alpha=F(\alpha)$, its derivative is

$$
\frac{d}{dt}|\alpha\rangle
=\left[
G-\operatorname{Re}\sum_r\alpha_r^*F_r(\alpha)
\right]|\alpha\rangle.
$$

Since $\langle G\rangle=\sum_r\alpha_r^*F_r(\alpha)$, the scalar
subtraction is precisely the normalization term for non-Hermitian evolution.
In particular, $\langle a_r\rangle=\alpha_r$ still obeys $\dot\alpha=F$.

One can verify this using the general normalized-expectation identity

$$
\frac{d}{dt}\langle O\rangle
=\langle G^\dagger O+OG\rangle
-\langle G^\dagger+G\rangle\langle O\rangle.
$$

For $O=a_r$, normal ordering gives
$\langle a_rG\rangle=\alpha_r\langle G\rangle+F_r(\alpha)$ and
$\langle G^\dagger a_r\rangle=\alpha_r\langle G^\dagger\rangle$.
The normalization contributions cancel, leaving $F_r(\alpha)$.
Using only a Hermitian-Hamiltonian commutator formula would miss this
normalization issue.

## 3. Mean-field treatment of the operator terms

### 3.1 Bilinear terms

Write $a_r=\alpha_r+\delta a_r$. Expanding gives

$$
a_r^\dagger a_s
=\alpha_s a_r^\dagger+\alpha_r^*a_s-\alpha_r^*\alpha_s
+\delta a_r^\dagger\delta a_s.
$$

The conventional mean-field decoupling drops the last term:

$$
a_r^\dagger a_s
\ \approx\
\alpha_s a_r^\dagger+\alpha_r^*a_s-\alpha_r^*\alpha_s.
$$

On the current coherent state,
$\delta a_s|\alpha\rangle=0$, so the discarded term annihilates that
state. Moreover, the last two retained terms cancel in their action:

$$
a_r^\dagger a_s|\alpha\rangle
=\alpha_s a_r^\dagger|\alpha\rangle.
$$

This statement also holds for $r=s$, including the number operator
$a_r^\dagger a_r$. It is an equality of action on a coherent state,
not an operator identity on arbitrary states.

Therefore a term $C_{rs}a_r^\dagger a_s$ contributes simply

$$
\dot\alpha_r\mathrel{+}=C_{rs}\alpha_s.
$$

For the full constant linear generator,
$G_A=\sum_{r,s}A_{rs}a_r^\dagger a_s$, the amplitudes evolve exactly as

$$
\alpha(t+\Delta t)=e^{\Delta t A}\alpha(t).
$$

This exponential acts on the vector of field amplitudes, not a Fock-space
state. Sparse stencils or spectral transforms usually avoid even assembling
a dense matrix $A$.

### 3.2 Nonlinear advection terms

The corresponding action for a cubic monomial is

$$
a_r^\dagger a_sa_t|\alpha\rangle
=\alpha_s\alpha_t a_r^\dagger|\alpha\rangle,
$$

so $B_{rst}a_r^\dagger a_sa_t$ contributes
$B_{rst}\alpha_s\alpha_t$ to $\dot\alpha_r$. This remains true when
indices coincide, provided the operator is normally ordered first.

For example, a conservative flux term in our scalar generator is

$$
-\frac{s_u}{2h}\,
a_{c,j,i}^\dagger a_{u,j,i+1}
(a_{c,j,i}+a_{c,j,i+1}).
$$

It gives

$$
\dot\alpha_{c,j,i}\mathrel{+}=
-\frac{s_u}{2h}\alpha_{u,j,i+1}
(\alpha_{c,j,i}+\alpha_{c,j,i+1}).
$$

In physical variables this is exactly
$-u_{j,i+1}(c_{j,i}+c_{j,i+1})/(2h)$.
All amplitudes in these products must be evaluated at the current time
integration stage. Freezing them for a whole step introduces a time
discretization approximation even though the coherent reduction is exact.

### 3.3 Optional local-state interpretation

Once the amplitudes are known, each normalized local state can be
reconstructed independently. For a finite increment,

$$
D_r(\delta\alpha_r)
=\exp(\delta\alpha_r a_r^\dagger-\delta\alpha_r^*a_r),\qquad
D_r(\delta\alpha_r)|\alpha_r\rangle
=e^{i\theta_r}|\alpha_r+\delta\alpha_r\rangle.
$$

An equivalent state-dependent Hermitian displacement Hamiltonian is

$$
H_{\mathrm{disp}}(t)
=i\sum_r\left[
F_r(\alpha(t))a_r^\dagger-F_r(\alpha(t))^*a_r
\right].
$$

Its Schrödinger evolution follows the same normalized coherent trajectory
up to a global phase. It is determined by the evolving classical solution;
it is not a claim that the original $G$ is Hermitian. For field simulation,
storing $\alpha$ directly is sufficient.

## 4. Fluid equations and field encoding

On the unit square, the dimensionless equations are

$$
\begin{aligned}
\partial_t u&=-\partial_x(u^2)-\partial_y(uv)
-\partial_xp+\nu\nabla^2u,\\
\partial_t v&=-\partial_x(uv)-\partial_y(v^2)
-\partial_yp+\nu\nabla^2v,\\
\partial_t c&=-\partial_x(uc)-\partial_y(vc)+\kappa\nabla^2c,\\
0&=\partial_xu+\partial_yv,
\end{aligned}
\qquad
\nu=\frac1{\mathrm{Re}},\quad \kappa=\frac1{\mathrm{Pe}}.
$$

The concentration is passive: it is transported by the velocity and does
not feed back into momentum. Pressure enforces incompressibility.

Each MAC cell has amplitude variables for $u,v,\phi,c$, where the
pressure impulse is $\phi=\Delta t\,p$:

$$
u=s_u\alpha_u,\quad v=s_u\alpha_v,\quad
\phi=s_\phi\alpha_\phi,\quad c=s_c\alpha_c.
$$

The current channel case uses $s_u=8$ and $s_\phi=s_c=4$.
Real physical initial conditions and real stencil coefficients keep the
ideal amplitudes real. These scales are encoding choices; viscosity,
Reynolds number, and physical velocities do not change with them.

More generally, if $q_r=s_r\alpha_r$ and $\dot q_r=R_r(q)$, then

$$
F_r(\alpha)=\frac{1}{s_r}R_r(S\alpha),\qquad S=\operatorname{diag}(s_r).
$$

A physical linear coefficient $L_{rs}$ becomes
$L_{rs}s_s/s_r$, and a quadratic coefficient $Q_{rst}$ becomes
$Q_{rst}s_ss_t/s_r$. This explains why velocity-scale factors appear
in the nonlinear MPO coefficients and cancel appropriately in physical
scalar advection.

## 5. MAC grid, boundaries, and discrete projection

Use one-based array indices $i$ in $x$ and $j$ in $y$, with $h=1/n$:

| Field | Position | Channel storage |
|---|---|---|
| $c_{j,i},\phi_{j,i}$ | $((i-\tfrac12)h,(j-\tfrac12)h)$ | $n\times n$ |
| $u_{j,i}$ | $((i-1)h,(j-\tfrac12)h)$ | $n\times n$ |
| $v_{j,i}$ | $((i-\tfrac12)h,(j-1)h)$ | $n\times n$ in MPS; $(n+1)\times n$ in DNS |

Indices wrap in $x$. The channel conditions at $y=0,1$ are

$$
\partial_yu=0,\qquad v=0,\qquad
\partial_y\phi=0,\qquad \partial_yc=0.
$$

Thus the current implementation uses impermeable free-slip walls. It does
not impose a zero normal derivative on every velocity component.
The MPS layout includes the bottom $v$ face and treats the top
$v_{n+1,i}=0$ as implicit. A direct array implementation may store both
walls explicitly.

Define the divergence and pressure gradient by

$$
(D\mathbf u)_{j,i}
=\frac{u_{j,i+1}-u_{j,i}+v_{j+1,i}-v_{j,i}}h,
$$

$$
(G_x\phi)_{j,i}=\frac{\phi_{j,i}-\phi_{j,i-1}}h,\qquad
(G_y\phi)_{j,i}=\frac{\phi_{j,i}-\phi_{j-1,i}}h
\quad(2\le j\le n).
$$

Set $G_y\phi=0$ on the two walls. Then $L_N=DG$ is the cell-centered
Neumann Laplacian. In an interior row,

$$
(L_Nq)_{j,i}
=\frac{q_{j,i+1}+q_{j,i-1}+q_{j+1,i}+q_{j-1,i}-4q_{j,i}}{h^2}.
$$

At the bottom wall,

$$
(L_Nq)_{1,i}
=\frac{q_{1,i+1}+q_{1,i-1}+q_{2,i}-3q_{1,i}}{h^2},
$$

with the analogous top-row formula. This operator applies to $u,c,\phi$.
Interior $v$ diffusion instead uses the five-point stencil with the
prescribed zero wall faces (denote it $L_D$); wall $v$ values are never
evolved.

The identities $D=-G^\mathsf T$ on the allowed face space and $DG=L_N$
are the basis of the discrete incompressibility projection. Here $G=(G_x,G_y)$
denotes the discrete gradient; its use in the fluid equations is distinct
from the bosonic generator $G$ in Sections 1–3.
For the legacy periodic double layer, wrap $j$ as well and use periodic
Laplacians and pressure transforms.

## 6. Conservative momentum and concentration fluxes

Define arithmetic face averages for each target $u$ face:

$$
\begin{aligned}
U_E&=(u_{j,i}+u_{j,i+1})/2,&
U_W&=(u_{j,i-1}+u_{j,i})/2,\\
U_N&=(u_{j,i}+u_{j+1,i})/2,&
U_S&=(u_{j-1,i}+u_{j,i})/2,\\
V_N&=(v_{j+1,i-1}+v_{j+1,i})/2,&
V_S&=(v_{j,i-1}+v_{j,i})/2.
\end{aligned}
$$

The conservative $u$ advection is

$$
N_u(\mathbf u)_{j,i}
=\frac{U_E^2-U_W^2+U_NV_N-U_SV_S}{h}.
$$

The north/south $uv$ wall flux is zero; do not access an out-of-domain
average there.

For an interior $v$ face, define

$$
\begin{aligned}
V_E&=(v_{j,i}+v_{j,i+1})/2,&
V_W&=(v_{j,i-1}+v_{j,i})/2,\\
V_N'&=(v_{j,i}+v_{j+1,i})/2,&
V_S'&=(v_{j-1,i}+v_{j,i})/2,\\
U_E'&=(u_{j-1,i+1}+u_{j,i+1})/2,&
U_W'&=(u_{j-1,i}+u_{j,i})/2.
\end{aligned}
$$

Then

$$
N_v(\mathbf u)_{j,i}
=\frac{U_E'V_E-U_W'V_W+(V_N')^2-(V_S')^2}{h}.
$$

Near the top/bottom boundaries these averages use the zero-valued wall
$v$ faces. In particular, the top-adjacent $(v+0)^2/4$ flux is retained.

For a concentration cell, define

$$
\begin{aligned}
J_E&=u_{j,i+1}(c_{j,i}+c_{j,i+1})/2,&
J_W&=u_{j,i}(c_{j,i-1}+c_{j,i})/2,\\
J_N&=v_{j+1,i}(c_{j,i}+c_{j+1,i})/2,&
J_S&=v_{j,i}(c_{j-1,i}+c_{j,i})/2.
\end{aligned}
$$

Set $J_N=0$ at the top and $J_S=0$ at the bottom. The scalar advection is

$$
N_c(\mathbf u,c)_{j,i}=(J_E-J_W+J_N-J_S)/h.
$$

The pressure-free physical residuals are therefore

$$
R_u=-N_u+\nu L_Nu,\qquad
R_v=-N_v+\nu L_Dv,\qquad
R_c=-N_c+\kappa L_Nc.
$$

For example, the diffusion operator
$\nu h^{-2}a_{u,j,i}^\dagger a_{u,j+1,i}$ contributes
$\nu h^{-2}\alpha_{u,j+1,i}$ to the target amplitude derivative.
Adding its other neighbors and the diagonal number term recovers
$\nu L_N\alpha_u$.

## 7. All three bosonic substeps as amplitude equations

### 7.1 Momentum and scalar predictor: physical time

Momentum advection is homogeneous quadratic in velocity, while scalar
advection is bilinear in velocity and concentration. Therefore

$$
\begin{aligned}
\dot\alpha_u
&=-s_uN_u(\alpha_u,\alpha_v)+\nu L_N\alpha_u,\\
\dot\alpha_v
&=-s_uN_v(\alpha_u,\alpha_v)+\nu L_D\alpha_v,\\
\dot\alpha_c
&=-s_uN_c((\alpha_u,\alpha_v),\alpha_c)+\kappa L_N\alpha_c,\\
\dot\alpha_\phi&=0.
\end{aligned}
$$

Evaluate the local fluxes as array operations. This replaces the entire
predictor MPO sum, including its quadratic advection.

### 7.2 Pressure relaxation: artificial time

Hold the predictor velocity $\mathbf u^\star$ and scalar fixed and evolve

$$
\partial_\tau\alpha_\phi
=L_N\alpha_\phi-\frac{s_u}{s_\phi}D\boldsymbol\alpha^\star_u.
$$

Here $\boldsymbol\alpha_u=(\alpha_u,\alpha_v)$ and $\tau$ is artificial
relaxation time, not physical time. At convergence,

$$
L_N\phi=D\mathbf u^\star.
$$

Since $\phi=\Delta t\,p$, there is no additional $1/\Delta t$ on this
right-hand side. If solving for $p$ directly, the right-hand side is
$D\mathbf u^\star/\Delta t$.

The constant pressure mode is undetermined. Fix it with
$\phi\leftarrow\phi-\overline\phi$. The right-hand side has zero mean
because periodic face fluxes telescope and wall-normal velocities vanish.

One may also bypass relaxation entirely and solve this scalar Poisson
equation. With the current grid, Fourier modes in $x$ and a DCT-II basis
in $y$ diagonalize $L_N$:

$$
\lambda_{k,\ell}
=-\frac4{h^2}\sin^2\!\left(\frac{\pi k}{n}\right)
-\frac4{h^2}\sin^2\!\left(\frac{\pi\ell}{2n}\right),
\qquad 0\le k,\ell<n.
$$

For nonzero modes, set
$\widehat\phi_{k,\ell}=\widehat{D\mathbf u^\star}_{k,\ell}/\lambda_{k,\ell}$;
set $\widehat\phi_{0,0}=0$. This is the direct channel pressure solve
already used by the DNS implementation.

### 7.3 Velocity correction: unit auxiliary time

Hold $\phi,c$ fixed and evolve over $0\le s\le1$:

$$
\partial_s\alpha_u=-\frac{s_\phi}{s_u}G_x\alpha_\phi,\qquad
\partial_s\alpha_v=-\frac{s_\phi}{s_u}G_y\alpha_\phi.
$$

The right-hand side is constant in this substep, so integrate it exactly:

$$
\boxed{\mathbf u^{n+1}=\mathbf u^\star-G\phi.}
$$

Consequently,

$$
D\mathbf u^{n+1}=D\mathbf u^\star-L_N\phi.
$$

The remaining divergence equals the negative Poisson residual. With an
accurate direct solve, it is limited by floating-point roundoff. This
substep has no reason to change concentration or pressure.

## 8. Conservation of concentration and the old mass correction

Let $M_c=h^2\sum_{j,i}c_{j,i}$. On the unit square it equals the mean.
Each internal scalar flux occurs once with each sign, so

$$
\sum_{j,i}N_c(\mathbf u,c)_{j,i}=0,\qquad
\sum_{j,i}(L_Nc)_{j,i}=0.
$$

Therefore

$$
\frac{dM_c}{dt}=h^2\sum_{j,i}R_c(\mathbf u,c)_{j,i}=0.
$$

This telescoping statement does not require the predictor velocity to be
exactly divergence-free; it requires matching shared fluxes and zero wall
flux. Any Runge–Kutta method preserves this linear invariant in exact
arithmetic, since every stage derivative has zero sum.

In the finite-boson MPS calculation, a uniform scalar displacement was added
after the predictor to correct accumulated mean drift. Its maximum
per-step relative correction in the completed channel run was approximately
$7.45\times10^{-6}$. That is a correction to the finite-boson numerical
representation, not a physical source term.

In direct amplitude evolution, the conservative residual should conserve
the mean to roundoff without this correction. Monitor the **raw** mass error
first. If a roundoff correction is desired in physical variables, use

$$
c_{j,i}\leftarrow c_{j,i}+(M_c(0)-M_c(t)).
$$

For equal cell volumes this is the minimum Euclidean-norm correction with
the required total sum: minimizing $\tfrac12\sum(\delta c_{j,i})^2$
subject to a prescribed sum gives a constant $\delta c_{j,i}$.
The analogous displacement in a truncated boson basis only approximates a
uniform physical-field shift.

Conservation does not imply $0\le c\le1$. Our centered scalar scheme is not
bound-preserving and can overshoot on a coarse grid. Clipping values would
generally destroy mass conservation; a conservative flux limiter would be a
separate spatial-discretization change.

## 9. Time integration: two distinct comparisons

### 9.1 Follow the existing MPS splitting

To isolate removal of TDVP, first keep its substep structure:

1. Integrate the pressure-free coupled predictor for $\Delta t$.
2. Solve $L_N\phi=D\mathbf u^\star$, or reproduce the stated relaxation
   tolerance for a closer comparison with the previous run.
3. Apply the exact velocity correction and retain the predictor scalar.

The amplitude ODEs are exact reductions, but a numerical integrator still
has timestep error. An accurate predictor alone does not make this
one-projection Chorin splitting second-order for the constrained fluid
problem. Pressure is absent during its predictor trajectory.

### 9.2 Use the existing projected midpoint DNS scheme

For a second-order array method, define the fixed discrete projector

$$
P=I-G L_N^+D,
$$

where $L_N^+$ is the inverse on zero-mean cell fields. Let
$\mathbf w=(u,v)$ with $D\mathbf w^n=0$. Then use

$$
\begin{aligned}
\mathbf w^{1/2}
&=P\left[\mathbf w^n+\frac{\Delta t}{2}R_w(\mathbf w^n)\right],\\
c^{1/2}
&=c^n+\frac{\Delta t}{2}R_c(\mathbf w^n,c^n),\\
\mathbf w^{n+1}
&=P\left[\mathbf w^n+\Delta t R_w(\mathbf w^{1/2})\right],\\
c^{n+1}
&=c^n+\Delta t R_c(\mathbf w^{1/2},c^{1/2}).
\end{aligned}
$$

Because $P$ is fixed and $P\mathbf w^n=\mathbf w^n$, this is explicit
midpoint applied to
$\dot{\mathbf w}=PR_w(\mathbf w)$ and
$\dot c=R_c(\mathbf w,c)$. It is second-order in time for these
semidiscrete variables under the usual smoothness/stability assumptions.
The pressure recovered from a stage impulse must be interpreted at that
stage; this argument does not establish second-order terminal pressure.

These stages match `advance_one_step_with_scalar` in the DNS code.
Changing from the MPS splitting to projected midpoint changes the time
algorithm as well as the state representation. Report those effects
separately when comparing errors.

For an explicit method, monitor both advective and diffusive restrictions:

$$
\Delta t\left(\frac{\|u\|_\infty}{h}+
\frac{\|v\|_\infty}{h}\right)\lesssim C_{\rm adv},\qquad
\Delta t\le
\frac{1}{2\max(\nu,\kappa)(h^{-2}+h^{-2})}.
$$

Use a safety factor; the diffusive bound is the pure-diffusion bound for
Euler or explicit midpoint, not a guarantee for the combined nonlinear
scheme. Explicit midpoint with centered advection has no nonzero interval
of absolute stability on the imaginary axis, so diffusion and timestep
convergence must also be checked.

## 10. Initial conditions for the present mixing layer

Keep the completed case's physical parameters:

| Quantity | Value |
|---|---|
| Domain and grid | $[0,1]^2$, $16\times16$ cells |
| Reynolds / Péclet numbers | $\mathrm{Re}=\mathrm{Pe}=50$ |
| Viscosity / scalar diffusivity | $\nu=\kappa=0.02$ |
| Shear center / thickness | $y_0=0.5$, $\delta=0.04$ |
| Perturbation envelope width | $w=0.12$ |
| Primary KH mode / amplitude | $m_2=2$, $A_2=2.5$ |
| Subharmonic mode / amplitude | $m_1=1$, $A_1=0.5$ |
| Relative phase | $0$ |
| Baseline timestep / final time | $\Delta t=0.0025$, $T=0.65$ |

Set $\bar u_j=\tanh((y_j-y_0)/\delta)$ with $y_j=(j-\tfrac12)h$.
Construct the base corner streamfunction discretely:

$$
\psi^0_{1,i}=0,\qquad
\psi^0_{j+1,i}=\psi^0_{j,i}+h\bar u_j.
$$

At corner coordinates $x_i=(i-1)h$, $y_j^v=(j-1)h$, add

$$
\psi'_{j,i}=E(y_j^v)
\left[
\frac{A_2}{2\pi m_2}\cos(2\pi m_2x_i)
+\frac{A_1}{2\pi m_1}\cos(2\pi m_1x_i)
\right],
$$

where $E(y)=\exp[-((y-y_0)/w)^2]$ in the interior and is set to zero
exactly at both walls. For $\psi=\psi^0+\psi'$, use the discrete curl

$$
u_{j,i}=\frac{\psi_{j+1,i}-\psi_{j,i}}h,\qquad
v_{j,i}=-\frac{\psi_{j,i+1}-\psi_{j,i}}h.
$$

Substituting these differences into $D\mathbf u$ cancels the four corner
contributions exactly, giving a divergence-free initial velocity before
roundoff. The streamfunction is constant in $x$ on each wall, so $v=0$
there.

Initialize concentration with

$$
\tilde c_j=\tfrac12[1+\tanh((y_j-y_0)/\delta)],\qquad
c_{j,i}=\frac{\tilde c_j-\min_j\tilde c_j}
{\max_j\tilde c_j-\min_j\tilde c_j}.
$$

This produces sampled plateaus of zero and one and mean $1/2$.
Set $\phi=0$ initially. Divide each field by its encoding scale if
storing amplitudes rather than physical arrays.

These are deliberately large finite-amplitude perturbations: the initial
peak velocity component is about $2.66$, despite the reference speed used
to define Re being one. At this resolution and viscosity, the case tests
finite-amplitude roller interaction. It is not evidence of a
grid-converged linear KH instability.

## 11. Why the finite-boson qualification matters

With occupation cutoff $N_b$,

$$
[a,a^\dagger]=I-(N_b+1)|N_b\rangle\langle N_b|.
$$

The truncated annihilation matrix is nilpotent, so it has no nonzero
eigenvalue. A truncated displacement state therefore cannot satisfy
$a|\alpha\rangle=\alpha|\alpha\rangle$ exactly for $\alpha\ne0$.
The proof in Section 2 does not apply exactly to that finite matrix model.

Also, a bond-dimension-one MPS is a product of local states, which need not
be coherent states. One-site TDVP starting from bond dimension one cannot
increase that dimension. Observing $\chi=1$ in the completed run therefore
does not independently prove coherent-manifold invariance of its truncated
evolution.

Direct amplitude integration implements the ideal coherent reduction.
It should agree with the finite-boson calculation as the latter's
representation and integration errors are reduced, but it need not
reproduce that run bit for bit. If instead the objective were the exact
dynamics of the finite-boson model, one amplitude per site would not in
general suffice.

Generators with multiple creation operators per monomial, such as a Kerr
term $(a^\dagger)^2a^2$, generally do not preserve ordinary coherent
states. The exact reduction here depends on the specific
$a_r^\dagger F_r(a)$ structure, not merely on choosing a coherent
initial condition.

## 12. Implementation and verification workflow

The direct implementation can keep four real arrays, compute the flux
residuals in Section 6, and use either algorithm in Section 9. For the
projected midpoint version:

```text
initialize u, v, c using the discrete streamfunction and scalar profile
record the initial fields and initial scalar mean
for each physical timestep:
    choose dt and shorten it to hit the next output time exactly
    compute Ru, Rv, Rc at (u, v, c)
    project (u + dt*Ru/2, v + dt*Rv/2) to obtain (u_half, v_half)
    c_half = c + dt*Rc/2
    compute Ru_half, Rv_half, Rc_half at (u_half, v_half, c_half)
    project (u + dt*Ru_half, v + dt*Rv_half) to obtain (u_next, v_next)
    c_next = c + dt*Rc_half
    check divergence, wall values, scalar mean, and finite values
    save scheduled fields, diagnostics, and restart arrays
```

The local stencil work and state storage scale as $O(n^2)$ per residual
evaluation. The FFT/DCT pressure solve costs $O(n^2\log n)$.
There are no boson-cutoff, MPS-bond, MPO-cache, or TDVP-sweep parameters.
Spatial resolution, timestep error, pressure accuracy, and stability
remain numerical concerns.

Verification should proceed in the following order:

1. Compare each amplitude residual, after rescaling, with its classical
   MAC residual on identical input arrays. Include wall-adjacent rows.
2. Verify $DG=L_N$, discrete adjointness, zero wall-normal velocity,
   and the telescoping scalar-flux and diffusion sums.
3. Compare a direct amplitude step with the ideal MPS substep equations
   using the same splitting and pressure tolerance.
4. Compare projected midpoint with the existing coupled DNS on identical
   initial fields and output times. In physical variables these implement
   the same method, so agreement should approach roundoff.
5. Compare with the saved finite-boson MPS trajectory, explicitly recording
   differences in initial encoding, time splitting, pressure tolerance,
   and the old scalar-mean correction.
6. Halve the timestep and assess field errors and roller trajectories.
   Refine the grid separately before claiming spatial convergence.

For the 260-step reference, save eight snapshots at step indices
$[0,37,74,111,149,186,223,260]$. Record
$u,v,\phi,c$, vorticity, kinetic energy, relative divergence, concentration
mean/range, and method/configuration metadata.

Compute vertex vorticity with the same MAC convention:

$$
\omega_{j,i}
=\frac{v_{j,i}-v_{j,i-1}}h
-\frac{u_{j,i}-u_{j-1,i}}h
\quad\text{at interior y vertices}.
$$

Its wall value is zero under these free-slip conditions. The DNS stores
both wall rows; the MPS snapshots omit the implicit top row. Align these
layouts before taking norms.

For the existing pairing diagnostic, form
$q(x_i)=\sum_j\max[-\omega_{j,i},0]$ and the discrete Fourier magnitudes
$A_m=|\sum_iq(x_i)e^{-2\pi\mathrm{i}m(i-1)/n}|$, where $\mathrm{i}^2=-1$.
Plot $A_1,A_2$, and $A_1/A_2$
alongside vorticity and concentration snapshots. A ratio crossing one
indicates subharmonic dominance, but can also reflect faster decay of
mode 2; it is not sufficient by itself to demonstrate vortex merging.
Use the spatial evolution to distinguish pairing from viscous decay.

The completed MPS/DNS comparison is available in
[the saved metrics](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/comparison_metrics.csv)
and [pairing plot](./outputs/mps_dns_16x16_re50_pe50_single_freeslip_delta004_largekh/pairing_compatibility.png).
Those are results of the previous finite-boson MPS calculation, not a new
TDVP-free run.

The proposed change is to evaluate the already-derived classical amplitude
equations directly. Coherent-state reconstruction becomes optional output,
while the physical mixing-layer model, conservative fluxes, and boundary
conditions remain explicit in the array evolution.
