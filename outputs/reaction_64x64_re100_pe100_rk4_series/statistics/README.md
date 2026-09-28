# Reacting mixing-layer statistics: MF versus DNS

64×64, Re=Pe=100, RK4, Da=1,10,100, t=0…0.65. Postprocessing only;
no trajectory was changed or rerun. All MF results were revalidated against
the stored explicit local Fock kets; DNS snapshot diagnostics, configurations,
initial fields and output times were checked. Input SHA-256 hashes are in
[summary.json](./summary.json).

## Three figures

1. [Reynolds stresses](./reynolds_stresses.png) ([PDF](./reynolds_stresses.pdf)): R11, R22 and signed R12 versus time.
2. [Vorticity thickness](./vorticity_thickness.png) ([PDF](./vorticity_thickness.pdf)): delta_omega versus time.
3. [Unmixedness](./unmixedness.png) ([PDF](./unmixedness.pdf)): signed c1/c2 covariance using both explicit averaging conventions.

Every panel has Da=1,10,100; **solid DNS, dashed MF**. Colors/marker shapes
identify Da. Marker locations are staggered *among existing saved times* to
show coincident curves; no data, coordinates or curves are offset. Only eight
field snapshots were saved per trajectory. Straight connections are visual
guides, not additional measurements or temporal interpolation claims.

## Definitions and discrete evaluation

Let overbar mean the periodic-x spatial mean at fixed y and time. Angle
brackets with xy denote the full-domain mean. These are spatial statistics
of a deterministic 2D realization, not time/ensemble statistics or quantum
connected correlations. The MF observables are measured physical fields.

MAC velocities are collocated at scalar cell centers before all stress products:

$$u^c_{j,i}=(u_{j,i}+u_{j,i+1})/2,\qquad
v^c_{j,i}=(v_{j,i}+v_{j+1,i})/2.$$

$$u'=u^c-\overline{u^c}(y,t),\quad v'=v^c-\overline{v^c}(y,t),\qquad
(R_{11},R_{22},R_{12})(t)=\langle(u'^2,v'^2,u'v')\rangle_{xy}.$$

These are kinematic Reynolds stresses (no density factor, no minus sign on
R12). Subtracting the local x mean prevents the mean shear itself from being
counted as a fluctuation. No extra velocity-jump normalization is applied;
the fields already use the simulation's nondimensional reference velocity.

$$\delta_\omega(t)=\frac{\Delta U(t)}{\max_y|\partial_y\bar u|},\qquad
\Delta U(t)=|\bar u(y_{N-1},t)-\bar u(y_0,t)|.$$

The outermost cell-center means estimate the two external velocities. The
derivative at each interior y face is (bar_u[j]-bar_u[j-1])/dy, with zero wall
derivatives for free slip; no smoothing, fitting or spectral differentiation.
This is thickness of the *mean shear*, not inverse peak instantaneous
vorticity. Finite-grid initial thickness need not equal the continuum 2*delta
for a tanh profile. The conventional gradient-based definition is described
in [Baltzer & Livescu, JFM, equation (4.13)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/variabledensity-effects-in-incompressible-nonbuoyant-sheardriven-turbulent-mixing-layers/250CE5A774864C97D4C73FB0742C2797).

The user's signed unmixedness is shown with both possible meanings of the brackets:

$$C_{12}^{xy}=\langle(c_1-\langle c_1\rangle_{xy})(c_2-\langle c_2\rangle_{xy})\rangle_{xy},$$

$$C_{12}^{x|y}=\langle\overline{(c_1-\bar c_1(y,t))(c_2-\bar c_2(y,t))}\rangle_y.$$

The first includes stratification between different y positions; the second
isolates within-x-plane fluctuations before averaging over y. Both use raw,
independently evolved c1 and c2; no clipping, absolute value, sign reversal or
division by mean concentrations is applied. Negative covariance represents
anticorrelation, not by itself a negative-concentration artifact. At t=0,
c2=1-c1: the global covariance is minus Var_xy(c1), whereas the streamwise
covariance is zero. At every snapshot the code checks the total-covariance identity

$$C_{12}^{xy}=C_{12}^{x|y}+\langle(\bar c_1-\langle c_1\rangle_{xy})
(\bar c_2-\langle c_2\rangle_{xy})\rangle_y.$$

Reaction is passive to momentum, so all three Da velocity-statistic curves
coincide within each method up to numerical roundoff. This is expected, not
a missing case. Scalar covariances do depend on Da. Agreement with DNS is
not a grid/timestep-convergence or positivity certificate.

## Final-time values

| Da | Method | R11 | R22 | R12 | Vorticity thickness | Global covariance | Streamwise covariance |
|---|---|---|---|---|---|---|---|
| 1 | DNS | 0.029215297 | 0.089632495 | -0.0084739305 | 0.37697639 | -0.11937418 | -0.012897619 |
| 1 | MF | 0.027414777 | 0.084225114 | -0.0069155262 | 0.37764756 | -0.11890009 | -0.012659341 |
| 10 | DNS | 0.029215297 | 0.089632495 | -0.0084739305 | 0.37697639 | -0.10438029 | -0.0086125434 |
| 10 | MF | 0.027414777 | 0.084225114 | -0.0069155262 | 0.37764756 | -0.1039605 | -0.0084203479 |
| 100 | DNS | 0.029215297 | 0.089632495 | -0.0084739305 | 0.37697639 | -0.091620182 | -0.0042925387 |
| 100 | MF | 0.027414777 | 0.084225114 | -0.0069155262 | 0.37764756 | -0.091150308 | -0.0041338109 |

Relative differences below are abs(MF-DNS)/abs(DNS) at the final time, not
errors against an exact solution. The shear stress changes sign over the
trajectory, so a relative error near a zero crossing would be misleading;
the exported summary also gives maximum absolute differences at saved times.

| Da | R11 difference | R22 difference | R12 difference | Thickness difference | Global covariance difference | Streamwise covariance difference |
|---|---|---|---|---|---|---|
| 1 | 6.163% | 6.033% | 18.391% | 0.178% | 0.397% | 1.847% |
| 10 | 6.163% | 6.033% | 18.391% | 0.178% | 0.402% | 2.232% |
| 100 | 6.163% | 6.033% | 18.391% | 0.178% | 0.513% | 3.698% |

## Reusable data and reproduction

- [statistics.csv](./statistics.csv): 48 rows (three Da × two methods × eight times), all unrounded values.
- [profiles.csv](./profiles.csv): 3,072 rows, including mean fields, R11(y), R22(y), R12(y), and streamwise scalar covariance(y) at every snapshot.
- [summary.json](./summary.json): provenance, MF–DNS differences, and cross-Da flow-statistic spread.

From the repository root, run `python3 plot_reaction_statistics.py` using the
same NumPy/SciPy/Matplotlib environment as the existing postprocessing.
