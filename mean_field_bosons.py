"""Explicit single-site bosonic states and self-consistent mean-field evolution.

The state is a product of local Fock vectors, not an array of amplitudes.
All three mixing-layer generators act via a_j, a_j^dagger and I matrices.
No TDVP, MPS truncation, coherent-state resetting, or classical field solve.
"""
import numpy as np


class LocalBosons:
    def __init__(self, cutoff):
        if int(cutoff) != cutoff or cutoff < 2:
            raise ValueError("boson cutoff must be an integer >= 2")
        self.cutoff = int(cutoff)
        self.dimension = self.cutoff+1
        self.annihilation = np.diag(np.sqrt(np.arange(1, self.dimension)), 1)
        self.creation = self.annihilation.T.copy()
        self.identity = np.eye(self.dimension)
        self.number = self.creation @ self.annihilation

    def coherent_states(self, alpha):
        """Initial normalized truncated coherent kets; never reset during a run."""
        states = np.ones((len(alpha), self.dimension), dtype=np.result_type(alpha, float))
        for occupation in range(1, self.dimension):
            states[:, occupation] = states[:, occupation-1]*alpha/np.sqrt(occupation)
        return self.normalize(states)

    @staticmethod
    def normalize(states):
        norms = np.sqrt(np.sum(np.abs(states)**2, axis=1))
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise FloatingPointError("invalid local bosonic state norm")
        return states/norms[:, None]

    @staticmethod
    def expectation_from_action(states, action):
        return np.sum(states.conj()*action, axis=1)/np.sum(np.abs(states)**2, axis=1)

    def amplitudes(self, states):
        return self.expectation_from_action(states, states @ self.annihilation.T)

    def derivative(self, states, generator):
        """Normalized non-Hermitian evolution by explicit local MF operators.

        K_j = f_j a_j^dagger + b_j a_j + c_j I.
        dot|psi_j> = (K_j - Re< K_j > I)|psi_j>.
        The subtraction only fixes the norm of each ket. It is not a pressure
        gauge, scalar-mass correction, or projection onto coherent states.
        """
        lowered = states @ self.annihilation.T
        alpha = self.expectation_from_action(states, lowered)
        f, b, c = generator.local_coefficients(alpha)
        action = (f[:, None]*(states @ self.creation.T)
                  + b[:, None]*lowered + c[:, None]*(states @ self.identity.T))
        growth = self.expectation_from_action(states, action).real
        return action-growth[:, None]*states

    def advance(self, states, interval, generator, substeps=1):
        """RK4 on local Fock vectors; rebuild all mean fields at every RK stage."""
        if substeps < 1 or int(substeps) != substeps:
            raise ValueError("substeps must be a positive integer")
        step = interval/substeps
        for _ in range(substeps):
            k1 = self.derivative(states, generator)
            k2 = self.derivative(states+step*k1/2, generator)
            k3 = self.derivative(states+step*k2/2, generator)
            k4 = self.derivative(states+step*k3, generator)
            states = self.normalize(states+step*(k1+2*k2+2*k3+k4)/6)
        return states

    def quality(self, states):
        lowered = states @ self.annihilation.T
        alpha = self.expectation_from_action(states, lowered)
        residual = lowered-alpha[:, None]*states
        return {
            "coherent_eigenstate_defect": float(np.max(np.linalg.norm(residual, axis=1))),
            "maximum_ceiling_probability": float(np.max(np.abs(states[:, -1])**2)),
            "maximum_norm_error": float(np.max(np.abs(np.sum(np.abs(states)**2, axis=1)-1))),
            "maximum_imaginary_amplitude": float(np.max(np.abs(alpha.imag))),
        }


def relax_pressure(states, bosons, operators, pseudo_dt, tolerance, max_steps,
                   check_every=20):
    """Pressure is relaxed by the same explicit single-site operator evolution.

    The full symmetric MF decoupling includes annihilation operators on the
    nominally frozen velocity sites. They act as scalars on ideal coherent
    kets. With finite truncation they can leak; the caller must measure this.
    We do not suppress these terms or replace evolving kets by coherent kets.
    """
    p = operators.phi_slice
    alpha = bosons.amplitudes(states)
    frozen = alpha.copy()
    frozen[p] = 0
    normalizer = max(float(np.linalg.norm(operators.pressure(frozen)[p])), 1e-12)
    for step in range(max_steps+1):
        if step % check_every == 0 or step == max_steps:
            alpha = bosons.amplitudes(states)
            residual = float(np.linalg.norm(operators.pressure(alpha)[p])/normalizer)
            if not np.isfinite(residual):
                raise FloatingPointError("non-finite single-site pressure relaxation")
            if residual <= tolerance:
                return states, {"pressure_residual": residual, "pressure_iterations": step,
                                "pressure_pseudo_time": step*pseudo_dt}
        if step < max_steps:
            states = bosons.advance(states, pseudo_dt, operators.pressure)
    raise FloatingPointError(f"single-site pressure relaxation failed after {max_steps} steps: "
                             f"residual={residual:.6e}, tolerance={tolerance:.6e}")
