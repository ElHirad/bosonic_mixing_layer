"""Normally ordered MAC generators reduced on a product of coherent states.

Each stored monomial is coefficient * a_target^dagger * a_source [* a_other].
The single-site solver uses these coefficients to construct explicit local
Fock-space generators after mean-field decoupling. No DNS routine is used.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Layout:
    n: int
    boundary_y: str

    @property
    def shapes(self):
        n = self.n
        return {"u": (n, n), "v": (n+1 if self.boundary_y == "free-slip" else n, n),
                "phi": (n, n), "c": (n, n)}

    @property
    def slices(self):
        result, offset = {}, 0
        for field, shape in self.shapes.items():
            size = int(np.prod(shape))
            result[field] = slice(offset, offset+size)
            offset += size
        return result

    @property
    def size(self):
        return self.slices["c"].stop

    def index(self, field, j, i):
        # Homogeneous wall v operators are eliminated, not evolved and clamped.
        if self.boundary_y == "free-slip":
            if field == "v" and j in (0, self.n):
                return None
            if not 0 <= j < self.shapes[field][0]:
                raise IndexError((field, j, i))
        else:
            j %= self.n
        return self.slices[field].start+j*self.n+(i % self.n)

    def pack(self, au, av, ac, aphi=None):
        if aphi is None:
            aphi = np.zeros(self.shapes["phi"])
        fields = {"u": au, "v": av, "phi": aphi, "c": ac}
        for name, value in fields.items():
            if value.shape != self.shapes[name]:
                raise ValueError(f"incorrect {name} shape: {value.shape}")
        return np.concatenate([fields[f].ravel() for f in self.shapes])

    def unpack(self, state):
        return {f: state[self.slices[f]].reshape(shape) for f, shape in self.shapes.items()}


class CoherentGenerator:
    """Sparse polynomial F(alpha), obtained from G=sum a_r^dagger F_r(a).

    `linear[r,s]` stores the coefficient of a_r^dagger a_s, NOT a
    conventional field propagator. Quadratic arrays store a_r^dagger a_s a_t.
    Evaluation makes the coherent replacements; coefficients are rebuilt only
    for a new configuration, while mean fields are recomputed at every stage.
    """

    def __init__(self, size, linear, quadratic):
        triples = [(r, s, value) for (r, s), value in linear.items() if value != 0]
        self.linear = coo_matrix(([v for _, _, v in triples],
                                  ([r for r, _, _ in triples], [s for _, s, _ in triples])),
                                 shape=(size, size)).tocsr()
        terms = [(r, s, t, v) for (r, s, t), v in quadratic.items() if v != 0]
        self.target = np.array([r for r, _, _, _ in terms], dtype=np.intp)
        self.source = np.array([s for _, s, _, _ in terms], dtype=np.intp)
        self.other = np.array([t for _, _, t, _ in terms], dtype=np.intp)
        self.coefficient = np.array([v for _, _, _, v in terms], dtype=float)
        self.size = size

    def __call__(self, alpha):
        result = self.linear @ alpha
        if self.coefficient.size:
            terms = self.coefficient*alpha[self.source]*alpha[self.other]
            # Production amplitudes are real; support complex algebra tests too.
            if np.iscomplexobj(terms):
                result = result + np.bincount(self.target, terms.real, minlength=self.size)
                result = result + 1j*np.bincount(self.target, terms.imag, minlength=self.size)
            else:
                result = result + np.bincount(self.target, terms, minlength=self.size)
        return result

    def summary(self):
        return {"linear_monomials": int(self.linear.nnz),
                "quadratic_monomials": len(self.coefficient)}

    def local_coefficients(self, alpha):
        """Return f, b, c for sum_j (f_j a_j^dagger + b_j a_j + c_j I).

        f=F(alpha), b_j=sum_r alpha_r^* dF_r/dalpha_j, c_j=-alpha_j*b_j.
        This is the full first-order decoupling, including annihilation and
        scalar terms; it is NOT the creation-only coherent-tangent shortcut.
        """
        creation = self(alpha)
        annihilation = self.linear.T @ alpha.conj()
        if self.coefficient.size:
            weights = self.coefficient*alpha[self.target].conj()
            for indices, values in ((self.source, weights*alpha[self.other]),
                                    (self.other, weights*alpha[self.source])):
                if np.iscomplexobj(values):
                    annihilation += np.bincount(indices, values.real, minlength=self.size)
                    annihilation += 1j*np.bincount(indices, values.imag, minlength=self.size)
                else:
                    annihilation += np.bincount(indices, values, minlength=self.size)
        return creation, annihilation, -alpha*annihilation


class GeneratorBuilder:
    def __init__(self, layout, scales):
        self.layout, self.scales = layout, scales
        self.linear, self.quadratic = {}, {}

    def term(self, target, coefficient, *sources):
        r = self.layout.index(*target)
        indices = [self.layout.index(*s) for s in sources]
        if r is None or None in indices:
            return
        coefficient /= self.scales[target[0]]
        for source in sources:
            coefficient *= self.scales[source[0]]
        # Annihilators commute; combining symmetric terms reduces storage.
        key = (r, *sorted(indices))
        terms = self.linear if len(sources) == 1 else self.quadratic
        terms[key] = terms.get(key, 0.0)+coefficient

    def product(self, target, coefficient, first, second):
        """Expand a product of averages into individual bosonic monomials."""
        for s in first:
            for t in second:
                self.term(target, coefficient/(len(first)*len(second)), s, t)

    def laplacian(self, target, coefficient):
        f, j, i = target
        n = self.layout.n
        for jj, ii in ((j, i-1), (j, i+1), (j-1, i), (j+1, i)):
            if self.layout.boundary_y == "free-slip" and f != "v" and not 0 <= jj < n:
                continue  # Neumann ghost equals the target; these terms cancel.
            self.term(target, coefficient, (f, jj, ii))
            self.term(target, -coefficient, target)

    def finish(self):
        return CoherentGenerator(self.layout.size, self.linear, self.quadratic)


class MeanFieldOperators:
    """Predictor, pressure-relaxation, and correction bosonic generators."""

    def __init__(self, config):
        n = config.n
        self.layout = layout = Layout(n, config.boundary_y)
        scales = {"u": config.velocity_scale, "v": config.velocity_scale,
                  "phi": config.pressure_scale, "c": config.scalar_scale}
        pred, press, corr = [GeneratorBuilder(layout, scales) for _ in range(3)]
        channel = config.boundary_y == "free-slip"
        for j in range(n):
            for i in range(n):
                u, c, p = ("u", j, i), ("c", j, i), ("phi", j, i)
                pred.laplacian(u, n*n/config.reynolds)
                pred.laplacian(c, n*n/config.peclet)
                press.laplacian(p, n*n)
                # -div(uu) at u: east/west squares, north/south uv products.
                ue, uw = [u, ("u", j, i+1)], [("u", j, i-1), u]
                pred.product(u, -n, ue, ue)
                pred.product(u, n, uw, uw)
                if not channel or j < n-1:
                    pred.product(u, -n, [u, ("u", j+1, i)],
                                 [("v", j+1, i-1), ("v", j+1, i)])
                if not channel or j > 0:
                    pred.product(u, n, [("u", j-1, i), u],
                                 [("v", j, i-1), ("v", j, i)])
                # -div(uc), with each internal flux entering adjacent cells
                # with opposite signs. No later scalar-mass repair is needed.
                pred.product(c, -n, [("u", j, i+1)], [c, ("c", j, i+1)])
                pred.product(c, n, [("u", j, i)], [("c", j, i-1), c])
                if not channel or j < n-1:
                    pred.product(c, -n, [("v", j+1, i)], [c, ("c", j+1, i)])
                if not channel or j > 0:
                    pred.product(c, n, [("v", j, i)], [("c", j-1, i), c])
                # d_tau alpha_phi = L alpha_phi - (s_u/s_phi) D alpha_vel.
                for coefficient, source in ((-n, ("u", j, i+1)), (n, u),
                                            (-n, ("v", j+1, i)), (n, ("v", j, i))):
                    press.term(p, coefficient, source)
                # d_s alpha_u = -(s_phi/s_u) G_x alpha_phi.
                corr.term(u, -n, p)
                corr.term(u, n, ("phi", j, i-1))

        for j in (range(1, n) if channel else range(n)):
            for i in range(n):
                v = ("v", j, i)
                pred.laplacian(v, n*n/config.reynolds)
                pred.product(v, -n, [("u", j-1, i+1), ("u", j, i+1)],
                             [v, ("v", j, i+1)])
                pred.product(v, n, [("u", j-1, i), ("u", j, i)],
                             [("v", j, i-1), v])
                vn, vs = [v, ("v", j+1, i)], [("v", j-1, i), v]
                pred.product(v, -n, vn, vn)
                pred.product(v, n, vs, vs)
                corr.term(v, -n, ("phi", j, i))
                corr.term(v, n, ("phi", j-1, i))

        self.predictor, self.pressure, self.correction = pred.finish(), press.finish(), corr.finish()
        self.phi_slice = layout.slices["phi"]
        # Restrict the pressure generator to the only modes that it evolves.
        # This is sparse coefficient evaluation, not inversion/diagonalization.
        self.pressure_linear = self.pressure.linear[self.phi_slice, self.phi_slice].tocsr()

    def summary(self):
        return {"amplitude_modes": self.layout.size,
                **{name: getattr(self, name).summary()
                   for name in ("predictor", "pressure", "correction")}}

