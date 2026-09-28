"""Independent RK4, forward-Euler, or legacy midpoint DNS for c1+c2 -> c3.

Only benchmark code imports this module. No mean-field evolution uses it.
All species have the same diffusivity; reaction does not feed back on flow.
"""
import numpy as np

import mixing_layer_dns as dns


def species_rhs(u, v, species, config, damkohler):
    if species.shape != (3, config.ny, config.nx):
        raise ValueError("expected three cell-centered species")
    if not np.isfinite(damkohler) or damkohler < 0:
        raise ValueError("Damkohler number must be finite and nonnegative")
    transport = np.stack([dns.scalar_rhs(u, v, c, config.scalar_diffusivity,
                                        config.dx, config.dy) for c in species])
    reaction = damkohler*species[0]*species[1]
    return transport + np.array([-1., -1., 1.])[:, None, None]*reaction


def advance_one_step(u, v, species, dt, config, damkohler, method="midpoint"):
    """Coupled transport/reaction with the selected explicit integrator."""
    if method not in ("midpoint", "forward-euler", "rk4"):
        raise ValueError("unknown reacting DNS time integrator")
    if dt*damkohler*np.max(np.abs(species[0])+np.abs(species[1])) > .25*(1+1e-12):
        raise FloatingPointError("DNS timestep exceeds reaction safety bound")
    channel = config.boundary_y == "free-slip"
    momentum = dns.channel_momentum_rhs if channel else dns.momentum_rhs
    project = dns.project_channel_velocity if channel else dns.project_velocity
    if method == "rk4":
        # Classical RK4 of the coupled semi-discrete system. The linear MAC
        # Helmholtz projection acts on each momentum RHS (unit projection dt),
        # so all intermediate velocities are divergence-free. This independent
        # DNS projection is never used by the bosonic mean-field integrator.
        def rhs(uu, vv, cc):
            ru, rv = momentum(uu, vv, config.viscosity, config.dx, config.dy)
            ku, kv, pressure = project(ru, rv, 1., config.dx, config.dy)
            return ku, kv, species_rhs(uu, vv, cc, config, damkohler), pressure
        k1 = rhs(u, v, species)
        k2 = rhs(u+.5*dt*k1[0], v+.5*dt*k1[1], species+.5*dt*k1[2])
        k3 = rhs(u+.5*dt*k2[0], v+.5*dt*k2[1], species+.5*dt*k2[2])
        k4 = rhs(u+dt*k3[0], v+dt*k3[1], species+dt*k3[2])
        weighted = [(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(4)]
        # The stored pressure is the RK-stage-weighted step average, not an
        # assertion of fourth-order pointwise pressure at the endpoint.
        return u+dt*weighted[0], v+dt*weighted[1], weighted[3], species+dt*weighted[2]
    ru, rv = momentum(u, v, config.viscosity, config.dx, config.dy)
    if method == "forward-euler":
        cn = species+dt*species_rhs(u, v, species, config, damkohler)
        un, vn, pressure = project(u+dt*ru, v+dt*rv, dt, config.dx, config.dy)
        return un, vn, pressure, cn
    uh, vh, _ = project(u+.5*dt*ru, v+.5*dt*rv, .5*dt, config.dx, config.dy)
    ch = species+.5*dt*species_rhs(u, v, species, config, damkohler)
    ru, rv = momentum(uh, vh, config.viscosity, config.dx, config.dy)
    cn = species+dt*species_rhs(uh, vh, ch, config, damkohler)
    un, vn, pressure = project(u+dt*ru, v+dt*rv, dt, config.dx, config.dy)
    return un, vn, pressure, cn


def conservation_diagnostics(species, reference):
    """Independent audit of the two stoichiometric invariants and all species."""
    if not np.all(np.isfinite(species)):
        raise FloatingPointError("non-finite reactive DNS fields")
    conserved = np.array([(species[0]+species[2]).mean(), (species[1]+species[2]).mean()])
    errors = np.abs(conserved-reference)/np.maximum(np.abs(reference), 1e-12)
    result = {"scalar_mass_error": float(errors.max()), "scalar_mass": float(species.mean()),
              "scalar_minimum": float(species.min()), "scalar_maximum": float(species.max())}
    for i, label in enumerate(("13", "23")):
        result[f"invariant_{label}_mass"] = float(conserved[i])
        result[f"invariant_{label}_error"] = float(errors[i])
    for i, c in enumerate(species, 1):
        result.update({f"c{i}_mass": float(c.mean()), f"c{i}_minimum": float(c.min()),
                       f"c{i}_maximum": float(c.max())})
        # Raw sign counts include roundoff; the material-negative count uses
        # the separately stated 1e-12 reporting threshold. No state is clipped.
        positive, negative, zero = c > 0, c < 0, c == 0
        result.update({f"c{i}_positive_cells": int(positive.sum()),
                       f"c{i}_negative_cells": int(negative.sum()),
                       f"c{i}_zero_cells": int(zero.sum()),
                       f"c{i}_positive_fraction": float(positive.mean()),
                       f"c{i}_negative_fraction": float(negative.mean()),
                       f"c{i}_zero_fraction": float(zero.mean()),
                       f"c{i}_material_negative_cells": int(np.count_nonzero(c < -1e-12)),
                       f"c{i}_material_negative_fraction": float(np.mean(c < -1e-12)),
                       f"c{i}_positive_integral": float(np.where(positive, c, 0.).mean()),
                       f"c{i}_negative_integral": float(np.where(negative, c, 0.).mean())})
    if result["scalar_mass_error"] > 1e-12:
        raise FloatingPointError("reactive DNS violated stoichiometric conservation")
    return result
