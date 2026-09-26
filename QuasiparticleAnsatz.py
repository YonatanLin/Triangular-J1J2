"""
Quasiparticle (plane-wave) excitation ansatz for a quasi-1D Hamiltonian on a cylinder, using TeNPy.

Workflow
--------
1. Define a 2D lattice model wrapped into a cylinder (periodic in y, infinite in x)
   -> this is what makes it "quasi-1D": DMRG/VUMPS treat it as an infinite MPS with a
   unit cell of Ly sites (times Lx), even though the physical lattice is 2D.
2. Run infinite DMRG (iDMRG) to get a good, well-converged ground state MPS.
3. Refine that state with VUMPS. VUMPS is the natural partner algorithm here because it
   directly produces and stores a `UniformMPS` (AL/AR/AC/C tensors), which is exactly the
   object the tangent-space / quasiparticle excitation ansatz needs.
4. Feed the resulting UniformMPS into `PlaneWaveExcitationEngine`
   (tenpy.algorithms.plane_wave_excitation) and diagonalize the effective excitation
   Hamiltonian at a series of momenta p to get the excitation dispersion E(p).

This closely follows TeNPy's own example `examples/advanced/vumps_and_plane_wave.py`,
generalized to a 2D-lattice-on-a-cylinder model instead of a plain 1D chain.

References
----------
- Vanderstraeten, Haegeman, Verstraete, "Tangent-space methods for uniform matrix product
  states", SciPost Phys. Lect. Notes 007 (2019), arXiv:1810.07006.
- Haegeman et al., PRL 2012 (single-mode / quasiparticle ansatz on infinite MPS).
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from tenpy import MomentumMPS

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.uniform_mps import UniformMPS
from tenpy.algorithms import dmrg, plane_wave_excitation
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import Config, asConfig
from tenpy.linalg.sparse import NpcLinearOperator, SumNpcLinearOperator
from tenpy.linalg.krylov_based import Arnoldi
from numpy import sin, cos, pi
from tenpy.tools.misc import setup_logging
from TyEffective import MixedGroundState, TyEffectiveOperator, ky_from_X
setup_logging(to_stdout="INFO")


# ----------------------------------------------------------------------
# 1. Define the model: a spin-1/2 XXZ / Heisenberg-like model on a cylinder
# ----------------------------------------------------------------------
class CylinderHeisenberg(CouplingMPOModel):
    """Spin-1/2 Heisenberg model on a square-lattice cylinder.

    The cylinder geometry comes entirely from the lattice parameters passed in
    model_params: 'lattice'='Square', 'bc_y'='cylinder', 'Ly'=<number of legs>,
    'bc_MPS'='infinite'. TeNPy then treats the whole cylinder circumference (Ly sites,
    times Lx if Lx>1) as the unit cell of an infinite 1D chain.
    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Sz')
        return SpinHalfSite(conserve=conserve)

    def init_terms(self, model_params):
        Jxy = model_params.get('Jxy', 1.0)
        Jz = model_params.get('Jz', 1.0)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(0.5 * Jxy, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)


def make_spin_model(Ly=4, Lx=1, Jxy=1.0, Jz=1.0, conserve='Sz', chain=False):
    lat = 'Chain' if chain else 'Square'
    model_params = dict(
        lattice=lat,
        Lx=Lx, # for 2D
        Ly=Ly, # for 2D
        L=Lx, # for 1D
        bc_MPS='infinite',
        order='default',
        Jxy=Jxy,
        Jz=Jz,
        conserve=conserve,
    )
    if not chain:
        model_params["bc_y"] = 'periodic'
        model_params["bc_x"] = 'periodic'

    M = CylinderHeisenberg(model_params)
    return M


# ----------------------------------------------------------------------
# 2. Ground state: iDMRG followed by VUMPS refinement
# ----------------------------------------------------------------------
def find_ground_state(M, chi_max=50, magz=0., return_info=False):
    assert ((-1.0 <= magz) and (magz <= 1.0)), "expect normalized magnetization"
    # Initial product state (Neel-like) on the cylinder unit cell.
    sites = M.lat.mps_sites()
    L = len(sites)
    # product_state = ['up', 'down'] * (L // 2) + (['up'] if L % 2 else [])
    N_up = int(((magz + 1.) / 2.) * L)
    N_down = L - N_up
    product_state = ['up'] * N_up + ['down'] * N_down
    psi = MPS.from_product_state(sites, product_state, bc='infinite')

    dmrg_params = {
        'mixer': True,
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10},
        'max_E_err': 1.e-10,
        'max_sweeps': 30,
    }
    eng_dmrg = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E0, psi = eng_dmrg.run()
    psi.canonical_form()
    print(f"psi total charge is {psi.get_total_charge()}")
    print(f"iDMRG ground state energy per site: {E0:.10f}")

    # `PlaneWaveExcitationEngine` needs a UniformMPS (AL/AR/AC/C tensors), not a
    # regular canonical MPS. `UniformMPS.from_MPS` does exactly this conversion
    # directly from the (already canonical) iDMRG ground state.
    uniform_psi = UniformMPS.from_MPS(psi)
    if return_info:
        stats = eng_dmrg.sweep_stats
        info = dict(psi=psi, E0=E0, max_trunc_err=stats['max_trunc_err'][-1],
                    n_sweeps=len(stats['E']))
        return uniform_psi, M, info
    return uniform_psi, M


def get_unit_cell_coords(M):
    """(x, y) coordinates of each site in the MPS unit cell, in MPS order."""
    lat = M.lat
    return [tuple(lat.position(lat.order[i])) for i in range(lat.N_sites)]


class KMomentumPWEngine(plane_wave_excitation.PlaneWaveExcitationEngine):
    """PlaneWaveExcitationEngine seeded with an explicit physical Bloch
    momentum (kx, ky), via a factor exp(i*(-kx*x_i + ky*y_i)) on the local
    excitation tensor at each unit-cell site i=(x_i, y_i).

    Generalizes the earlier Ly-only (Lx=1) seeding to Lx > 1. The relative
    sign between the kx and ky phases is NOT a typo: TeNPy's `p` (passed to
    `run()`) is defined via exp(+i*p*L) for a full unit-cell shift (see
    Unaligned_Effective_H), and matching that against the physical Bloch
    phase for shifting by Lx columns pins p = kx/Ly -- but the *internal*
    per-site phase across x within one unit cell comes out with the
    opposite sign from the y one. Verified against the exact single-magnon
    dispersion cos(kx)+cos(ky) for several (Lx, Ly) by direct
    diagonalization of the small effective-H matrix; get the sign wrong and
    it still finds exact eigenvalues (N_krylov=1) but of the WRONG (kx,ky)
    -- so this is easy to silently get backwards.

    Call `set_target_k(kx, ky, coords)` before `run(p=kx/Ly, ...)`.
    """

    def set_target_k(self, kx, ky, coords):
        self._target_kx = kx
        self._target_ky = ky
        self._coords = coords

    def initial_guess(self, qtotal_change):
        X_init = super().initial_guess(qtotal_change)
        kx = getattr(self, '_target_kx', 0.0)
        ky = getattr(self, '_target_ky', 0.0)
        coords = getattr(self, '_coords', None) or [(0, a) for a in range(len(X_init))]
        return [np.exp(1j * (-kx * x + ky * y)) * X for X, (x, y) in zip(X_init, coords)]


class _PenalisedOperator(NpcLinearOperator):
    """H_eff + coeff * T_eff acting on the list of X tensors (Eq. 27 with coeff = -alpha e^{i ky})."""

    def __init__(self, H, Ty, coeff):
        self.H, self.Ty, self.coeff = H, Ty, coeff

    def matvec(self, X):
        HX, TX = self.H.matvec(X), self.Ty.matvec(X)
        return [h + self.coeff * t for h, t in zip(HX, TX)]


class TyTargetedPWEngine(KMomentumPWEngine):
    """Plane-wave engine that *targets* the transversal momentum ky (Van Damme et al., Eq. 27):

        ( H_eff - alpha * e^{+i ky} * T_eff ) X = lambda X ,   lambda = omega - alpha ,

    with the eigenvalue of most negative real part. T_eff (see TyEffective.py) has eigenvalues
    e^{-i ky'} on an amplitude e^{+i ky' y}, so the penalty is smallest for ky' = ky. The operator
    is not Hermitian, hence TeNPy's `Arnoldi` instead of Lanczos. If `alpha` is too small the
    solver can converge to a different ky sector; `run_targeted` reports the measured ky
    (Rayleigh quotient of T_eff) so that the caller can enlarge `alpha`.

    Call `prepare_Ty(Ly)` once (it solves the mixed-transfer-matrix fixed points, independent of
    kx, ky), then `run_targeted(p, ky, alpha, ...)` for each momentum.
    """

    def prepare_Ty(self, Ly, mu_hint=1.0, N_max=60):
        self._gs = MixedGroundState(self.ALs, self.ARs, self.Cs[0], Ly, mu_hint=mu_hint, N_max=N_max)
        return self._gs

    def run_targeted(self, p, ky, alpha, qtotal_change=None, num_ev=1, sum_tol=1.e-10,
                     sum_iterations=100):
        """Returns (Es, psis, N_krylov, kys_measured, dists_from_1)."""
        self.unaligned_H = self.Unaligned_Effective_H(self, p)
        H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)
        Ty = TyEffectiveOperator(self._gs, self.VLs, p, sum_tol=sum_tol,
                                 sum_iterations=sum_iterations)
        op = _PenalisedOperator(H, Ty, -alpha * np.exp(1j * ky))
        lanczos_params = self.options.subconfig('lanczos_params')
        lanczos_params['which'] = 'SR'
        lanczos_params['num_ev'] = num_ev
        X_init = self.initial_guess(qtotal_change)
        _, Xs, N = Arnoldi(op, X_init, lanczos_params).run()
        Es, psis, kys, dists = [], [], [], []
        for X in Xs:
            HX = H.matvec(X)  # Rayleigh quotient of H_eff, shifted as in PlaneWaveExcitationEngine.run
            E = sum(npc.inner(x, h, axes='range', do_conj=True) for x, h in zip(X, HX))
            E /= sum(npc.inner(x, x, axes='range', do_conj=True) for x in X)
            Es.append(E - self.lambda_C1 - self.energy_density * self.L)
            psis.append(MomentumMPS(X, self.psi, p))
            k, d = ky_from_X(Ty, X)
            kys.append(k)
            dists.append(d)
        return np.real_if_close(Es), psis, N, np.array(kys), np.array(dists)


def measure_ky(eng, X, p, ky_seed, coords, Ly, Lx, max_unaligned=300, n_ev=6,
                tol=1e-12, shift_sign=1):
    """Same physics/algorithm as the coarse-graining version of this
    function (see its docstring for the full bug history and the Van Damme
    et al. Eq. (25) aligned+unaligned structure this implements) -- the
    ONLY thing that changed is *how* the per-column y-shift transfer step
    is computed, to fix a real scaling problem at production bond
    dimensions.

    The coarse-graining version built one dense (chi, d**Ly, chi) tensor
    per column (contracting all Ly real sites of a column into one
    physical supersite) and inserted a dense d**Ly x d**Ly permutation
    matrix between bra and ket. That is exact but EXPONENTIAL in Ly --
    fine for validating at Ly~3-4, chi~8, but infeasible at production
    scale (chi~3000): e.g. Ly=6 needs ~9GB and ~1.7e12 flops for one such
    tensor, Ly=8 needs ~37GB, and it only gets worse, regardless of
    whether the underlying MPS is dense or sparse/Sz-conserving (that
    exponential blowup is in the *coarse-graining*, before any charge
    conservation could help).

    This version replaces that with a "ring trace": the y-shift only ever
    relabels ONE physical index per real site (it's a pure permutation,
    not a generic dense operator), so it can be applied by sweeping a
    column's Ly real sites in their own natural MPS bond order, carrying
    just ONE extra d-dimensional leg (not a d**Ly one) to track the
    physical value that has to "wrap around" the periodic column. Cost
    per column becomes O(Ly * chi^3 * d^2) -- linear in Ly, and no worse
    in chi than a single ordinary MPS transfer-matrix step (the same
    chi^3 cost DMRG itself already pays per bond). Validated to reproduce
    the coarse-graining approach to machine precision for both
    shift_sign=+1/-1 and Ly=1..5, including the rectangular chi_bra !=
    chi_ket case this function actually needs (see ring_trace.py), and
    the "doubled" per-site excited-state construction below was checked
    against the direct block-triangular-matrix-product identity (a
    product of block upper triangular matrices [[L_i,B_i],[0,R_i]] has
    (0,1) block = sum_i L_1..L_{i-1} B_i R_{i+1}..R_n -- exactly Eq. 20's
    "sum over which site hosts the excitation", with no cross terms since
    the (1,0) block is always zero) rather than via Ly separate passes.

    All parameters and the return value are identical to the
    coarse-graining version -- see that docstring for the full parameter
    list, the mu/l/r degenerate-eigenvalue handling, and the meaning of
    `dist_from_1`.
    """
    import numpy as np
    from scipy.sparse.linalg import eigs, LinearOperator

    L = eng.L
    assert L == Lx * Ly
    ALs, ARs, VLs = eng.ALs, eng.ARs, eng.VLs

    def to_dense(a):
        labels = a.get_leg_labels()
        order = [labels.index('vL'), labels.index('p'), labels.index('vR')]
        return a.to_ndarray().transpose(order)

    dense_AL = [to_dense(ALs[i]) for i in range(L)]
    dense_AR = [to_dense(ARs[i]) for i in range(L)]
    dense_VL = [to_dense(VLs[i]) for i in range(L)]
    chi = dense_AL[0].shape[0]
    d = dense_AL[0].shape[1]

    for c in range(Lx):
        xs = set(coords[c * Ly + k][0] for k in range(Ly))
        assert len(xs) == 1, "columns must be contiguous in MPS order"

    def ring_step(msg, bra_sites, ket_sites, sign=None):
        """One column's mixed-transfer-matrix step, ring-trace version.
        msg: (chi_bra, chi_ket). bra_sites/ket_sites: Ly dense (chi,d,chi)
        tensors for this column in natural MPS bond order (chi_bra and
        chi_ket need not be equal to each other). Never forms a d**Ly
        object -- see module docstring / ring_trace.py for the derivation
        and validation of this exact contraction pattern. `sign` defaults
        to the function's own `shift_sign`; step_op_back passes -shift_sign
        (see its docstring for why).
        """
        if sign is None:
            sign = shift_sign
        Lyc = len(bra_sites)
        if Lyc == 1:
            return np.einsum('IJ,IPb,JPd->bd', msg, bra_sites[0], np.conj(ket_sites[0]))

        if sign == 1:
            tmp = np.einsum('IJ,IAb->JAb', msg, bra_sites[0])
            state = np.einsum('JAb,JCd->bdAC', tmp, np.conj(ket_sites[0]))
            for m in range(1, Lyc - 1):
                tmp = np.einsum('bdAC,bCe->dAe', state, bra_sites[m])
                state = np.einsum('dAe,dDf->efAD', tmp, np.conj(ket_sites[m]))
            tmp = np.einsum('bdAC,bCe->dAe', state, bra_sites[Lyc - 1])
            return np.einsum('dAe,dAf->ef', tmp, np.conj(ket_sites[Lyc - 1]))
        else:
            tmp = np.einsum('IJ,IPb->JPb', msg, bra_sites[0])
            state = np.einsum('JPb,JAd->bdPA', tmp, np.conj(ket_sites[0]))
            for m in range(1, Lyc - 1):
                tmp = np.einsum('bdPA,dPe->bAe', state, np.conj(ket_sites[m]))
                state = np.einsum('bAe,bQf->feQA', tmp, bra_sites[m])
            tmp = np.einsum('bdPA,dPe->bAe', state, np.conj(ket_sites[Lyc - 1]))
            return np.einsum('bAe,bAf->fe', tmp, bra_sites[Lyc - 1])

    def col_sites(arr, c):
        return [arr[c * Ly + k] for k in range(Ly)]

    def step_op(msg, bra_col, ket_col):
        return ring_step(msg, bra_col, ket_col)

    def step_op_back(msg_next, bra_col, ket_col):
        # Backward pass: same column, but msg enters from the RIGHT
        # boundary. Reduce to ring_step by reversing site order and
        # transposing each site's vL/vR bond legs (so the "natural bond
        # order" ring_step expects runs the other way), then transpose
        # back at the end.
        # NOTE: reversing site order and swapping each site's vL/vR flips
        # which direction "logical site j+1" sits relative to "real site
        # j+1" -- concretely, bra's real slot m maps to logical slot
        # Ly-1-m, and working through the index algebra, the shift that
        # pairs bra's logical slot j with ket's logical slot is the
        # NEGATION of shift_sign, not shift_sign itself (confirmed
        # numerically against the coarse-graining reference for Ly up to
        # 5 -- the two are indistinguishable at Ly<=2, which is why this
        # sign only shows up once Ly>=3). msg is passed through unchanged
        # (no transpose) since (chi_bra,chi_ket) axis order is preserved.
        bra_rev = [np.transpose(t, (2, 1, 0)) for t in reversed(bra_col)]
        ket_rev = [np.transpose(t, (2, 1, 0)) for t in reversed(ket_col)]
        return ring_step(msg_next, bra_rev, ket_rev, sign=-shift_sign)

    def full_pass(msg, bra_arr, ket_arr):
        for c in range(Lx):
            msg = step_op(msg, col_sites(bra_arr, c), col_sites(ket_arr, c))
        return msg

    def full_pass_back(msg, bra_arr, ket_arr):
        for c in reversed(range(Lx)):
            msg = step_op_back(msg, col_sites(bra_arr, c), col_sites(ket_arr, c))
        return msg

    # --- mu, l, r: leading eigenvalue/fixed points of the mixed (Ty-
    # inserted) transfer matrix, via a genuine eigensolve (plain power
    # iteration fails on the generic complex-conjugate-pair degeneracy of
    # a real-gauged ground state's transfer matrix -- see docstring).
    N = chi * chi

    def fwd(vec):
        return full_pass(vec.reshape(chi, chi), dense_AL, dense_AL).reshape(-1)

    def bwd(vec):
        return full_pass_back(vec.reshape(chi, chi), dense_AR, dense_AR).reshape(-1)

    if N <= max(n_ev + 2, 4):
        Mfwd = np.zeros((N, N), dtype=complex)
        Mbwd = np.zeros((N, N), dtype=complex)
        for k in range(N):
            e = np.zeros(N, dtype=complex)
            e[k] = 1.0
            Mfwd[:, k] = fwd(e)
            Mbwd[:, k] = bwd(e)
        vals_l, vecs_l = np.linalg.eig(Mfwd)
        vals_r, vecs_r = np.linalg.eig(Mbwd)
    else:
        op_fwd = LinearOperator((N, N), matvec=fwd, dtype=complex)
        op_bwd = LinearOperator((N, N), matvec=bwd, dtype=complex)
        vals_l, vecs_l = eigs(op_fwd, k=n_ev, which='LM')
        vals_r, vecs_r = eigs(op_bwd, k=n_ev, which='LM')

    best = None
    for i in range(len(vals_l)):
        j = int(np.argmin(np.abs(vals_r - vals_l[i])))
        if abs(vals_r[j] - vals_l[i]) > 1e-6 * max(abs(vals_l[i]), 1e-300):
            continue
        mu = vals_l[i]
        dphi = abs(((np.angle(mu) - ky_seed + np.pi) % (2 * np.pi)) - np.pi)
        key = (dphi, -abs(mu))
        if best is None or key < best[0]:
            best = (key, mu, vecs_l[:, i].reshape(chi, chi), vecs_r[:, j].reshape(chi, chi))
    if best is None:
        raise RuntimeError("could not match forward/backward eigenvalues of the "
                            "mixed transfer matrix")
    mu, lenv, renv = best[1], best[2], best[3]

    # --- excited per-site "doubled" tensors: block (0,0)=AL, (1,1)=AR,
    # block (0,1)=B AT EVERY SITE OF THE COLUMN SIMULTANEOUSLY. This looks
    # like it should double-count, but it doesn't: a product of block
    # upper triangular matrices [[L_i,B_i],[0,R_i]] has (1,0) block
    # identically zero at every step, so its (0,1) block is exactly
    # sum_i (L_1...L_{i-1}) B_i (R_{i+1}...R_n) -- the desired "exactly
    # one excitation, summed over which site hosts it" -- with no cross
    # (B_i B_j) terms possible. Coarse-graining this chain (as the old
    # version did explicitly) or ring-tracing it (as we do here) both
    # just contract bonds, so the same identity carries through unchanged.
    Xd = [x.to_ndarray() for x in X]
    Bs = [np.tensordot(dense_VL[i], Xd[i], axes=([2], [0])) for i in range(L)]

    def doubled_site(i):
        T = np.zeros((2 * chi, d, 2 * chi), dtype=complex)
        T[:chi, :, :chi] = dense_AL[i]
        T[chi:, :, chi:] = dense_AR[i]
        T[:chi, :, chi:] = Bs[i]
        return T

    doubled = [doubled_site(i) for i in range(L)]

    denom = np.einsum('bc,bc->', full_pass(lenv, dense_AL, dense_AL), renv)

    # aligned (k=0): bra's and ket's excitation both within this one
    # reference repeat of the unit cell -- one ring_step per column using
    # the doubled per-site tensors on both sides.
    msg_ref = np.zeros((2 * chi, 2 * chi), dtype=complex)
    msg_ref[:chi, :chi] = lenv
    for c in range(Lx):
        msg_ref = step_op(msg_ref, col_sites(doubled, c), col_sites(doubled, c))
    aligned = np.einsum('bc,bc->', msg_ref[chi:, chi:], renv)

    # unaligned (k=+-1,2,...): bra excites in the reference repeat, ket
    # excites m repeats later (or vice versa) -- explicit geometric sum,
    # mirroring infinite_sum_right/infinite_sum_left in
    # plane_wave_excitation.py, just walking the mixed transfer matrix
    # instead of an MPO Hamiltonian.
    msg10, msg01 = msg_ref[chi:, :chi].copy(), msg_ref[:chi, chi:].copy()
    total_unaligned = 0.0 + 0.0j
    phase_R, phase_L = np.exp(-1j * p * L), np.exp(1j * p * L)
    scale = max(abs(aligned), abs(denom), 1e-300)

    acc_phase, n_used_R = 1.0 + 0.0j, 0
    for m in range(1, max_unaligned + 1):
        acc_phase *= phase_R
        msg_step = np.zeros((chi, 2 * chi), dtype=complex)
        msg_step[:, :chi] = msg10
        for c in range(Lx):
            msg_step = step_op(msg_step, col_sites(dense_AR, c), col_sites(doubled, c))
        msg10 = msg_step[:, :chi]
        term = acc_phase * np.einsum('bc,bc->', msg_step[:, chi:], renv)
        total_unaligned += term
        n_used_R = m
        if abs(term) < tol * scale and m > 3:
            break

    acc_phase, n_used_L = 1.0 + 0.0j, 0
    for m in range(1, max_unaligned + 1):
        acc_phase *= phase_L
        msg_step = np.zeros((2 * chi, chi), dtype=complex)
        msg_step[:chi, :] = msg01
        for c in range(Lx):
            msg_step = step_op(msg_step, col_sites(doubled, c), col_sites(dense_AR, c))
        msg01 = msg_step[:chi, :]
        term = acc_phase * np.einsum('bc,bc->', msg_step[chi:, :], renv)
        total_unaligned += term
        n_used_L = m
        if abs(term) < tol * scale and m > 3:
            break

    ratio = (aligned + total_unaligned) / denom
    return (float(np.angle(ratio)), float(1 - np.abs(ratio)),
            dict(aligned=aligned, unaligned=total_unaligned, denom=denom, mu=mu,
                 n_used_R=n_used_R, n_used_L=n_used_L))


# ----------------------------------------------------------------------
# 3. Quasiparticle ansatz: plane-wave excitations on top of the uniform MPS
# ----------------------------------------------------------------------
def wrap_angle(a):
    """Map an angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def compute_dispersion_targeted(uniform_psi, M, momenta, qtotal_change=None, alpha=1.0,
                                ky_tol=1.e-2, dist_tol=0.05, max_retries=4, lanczos_N_max=100,
                                sum_tol=1.e-6, sum_iterations=80, verbose=True, kys=None):
    """Dispersion E(kx, ky) with ky *targeted* through Eq. 27 (see TyTargetedPWEngine).

    For every (kx, ky) the eigenproblem H_eff - alpha e^{i ky} T_eff is solved; the ky of the
    result is measured from T_eff itself (angle and modulus of X^dag T_eff X / X^dag X); if the
    angle deviates from the target by more than `ky_tol` or the modulus from 1 by more than
    `dist_tol`, `alpha` is doubled and the solve repeated (at most `max_retries` times).
    `sum_tol`, `sum_iterations` control the geometric sums over unit cells inside T_eff.

    Choice of alpha: it must exceed roughly (energy spread between ky sectors) / (1 - cos(2 pi/Ly));
    too small a value is caught and fixed by the retry loop. Larger than necessary is *not* free:
    T_eff is only approximately unitary for a finite bond dimension (1-|T_eff| ~ 1e-3 at chi=24),
    and the penalty trades energy for |T_eff| -> 1, shifting the returned energy by roughly
    6e-4 * alpha (measured for the Ly=4 XY cylinder at chi=24, ky=pi). Start small.
    `kys` restricts the transversal momenta (default: all 2 pi n / Ly).
    Returns an array of shape (len(momenta), len(kys)) with the energies (nan if a sector failed).
    """
    Ly, Lx = M.lat.Ls[1], M.lat.Ls[0]
    coords = get_unit_cell_coords(M)
    if kys is None:
        kys = 2 * np.pi * np.arange(Ly) / Ly
    kys = np.atleast_1d(kys)
    eng = TyTargetedPWEngine(uniform_psi, M, {'lanczos_params': {'N_max': lanczos_N_max}})
    gs = eng.prepare_Ty(Ly)
    if verbose:
        print(f"ground-state mixed transfer matrix: mu = {gs.mu:.10f}  |mu| = {abs(gs.mu):.10f}  "
              f"(fixed-point residual {gs.fixed_point_residual:.1e})")
    E_out = np.full((len(momenta), len(kys)), np.nan)
    for ikx, kx in enumerate(momenta):
        p = kx / Ly
        for iky, ky in enumerate(kys):
            eng.set_target_k(kx, ky, coords)
            a = alpha
            for attempt in range(max_retries + 1):
                Es, psis, N, ky_meas, dist = eng.run_targeted(
                    p, ky, a, qtotal_change=qtotal_change, sum_tol=sum_tol,
                    sum_iterations=sum_iterations)
                err = abs(wrap_angle(ky_meas[0] - ky))
                ok = err < ky_tol and abs(dist[0]) < dist_tol
                if ok:
                    break
                a *= 2
            E_out[ikx, iky] = Es[0] if ok else np.nan
            if verbose:
                print(f"    kx={kx:6.3f}  ky_target={ky:6.3f}  E={Es[0]: .6f}  ky_measured={ky_meas[0]:6.3f}"
                      f"  1-|T_eff|={dist[0]:.1e}  alpha={a:g}  krylov={N}"
                      + ("" if ok else "  ** ky NOT reached **"), flush=True)
    return E_out


def compute_dispersion(uniform_psi, M, momenta, qtotal_change=None, num_ev=1,
                       lanczos_N_max=50):
    """Diagonalize the tangent-space effective Hamiltonian on a full 2D grid
    of physical Bloch momenta (kx, ky): kx from `momenta` (should span the
    full zone, e.g. linspace(-pi, pi, ...)) and ky = 2*pi*n/Ly (n=0..Ly-1),
    quantized by the cylinder circumference. p = kx/Ly is computed
    internally per kx (see KMomentumPWEngine).

    For each computed excited state, the momentum is also measured directly
    from the wavefunction (via `measure_ky`, not just trusted from the seed)
    and printed together with the distance of the leading eigenvalue of the
    mixed transfer matrix from 1 -- a diagnostic that should be ~0 for a
    genuine y-momentum eigenstate, and is what to watch once the ground
    state is no longer the exactly-solvable chi_max=1 case.

    Parameters
    ----------
    uniform_psi : UniformMPS
        Ground state from VUMPS (or converted from a canonical infinite MPS).
    M : MPOModel
        The (infinite) model / Hamiltonian.
    momenta : array-like of float
        Physical momentum kx along the infinite direction, one full
        Brillouin zone.
    qtotal_change : list of int or None
        Charge sector of the excitation relative to the ground state, e.g. [1] to flip
        one unit of a conserved U(1)/Sz charge. Use None / [0] for the same sector.
    num_ev : int
        Number of branches to extract *per ky sector* (default 1, the lowest
        excitation in that sector). Total branches returned per kx is Ly*num_ev.
    """

    Ly = M.lat.Ls[1]
    Lx = M.lat.Ls[0]
    coords = get_unit_cell_coords(M)
    kys = 2 * np.pi * np.arange(Ly) / Ly

    print("starting dispersion calculation")
    pw_params = {'lanczos_params': {'N_max': lanczos_N_max}}
    eng_pw = KMomentumPWEngine(uniform_psi, M, pw_params)
    dispersions = []

    for kx in momenta:
        p = kx / Ly
        Es_p = []
        for ky in kys:
            eng_pw.set_target_k(kx, ky, coords)
            Es, psis, N = eng_pw.run(p, qtotal_change=qtotal_change, num_ev=num_ev)
            if len(Es) < num_ev:
                warnings.warn(
                    f"kx={kx:.4f}, ky={ky:.4f}: only {len(Es)}/{num_ev} state(s) "
                    f"found (N_krylov={N})."
                )
            for E, psi_exc in zip(Es, psis):
                ky_meas, dist, ky_info = measure_ky(eng_pw, psi_exc._X, p, ky, coords, Ly, Lx)
                print(f"    kx={kx:6.3f}  ky_target={ky:6.3f}  E={E: .6f}  "
                      f"ky_measured={ky_meas:6.3f}  dist_from_1={dist:.2e}  "
                      f"|mu_ground|={abs(ky_info['mu']):.4f}")
                print(f"ky_measured - ky_target mod 2pi is {(ky_meas - ky) % (2*pi)}")
            Es_p.extend(Es)
        dispersions.extend(Es_p)
        print(f"kx = {kx:6.3f}   E_exc(ky) = {np.array(Es_p)}")
    return np.array(dispersions)


# ----------------------------------------------------------------------
# 4. Put it all together
# ----------------------------------------------------------------------
if __name__ == '__main__':
    colors = ["blue", "red", "green"]
    # markersizes = [8, 5, 3]
    markersizes = [5, 3, 1]
    chain = False
    target_ky = True  # False: previous seed-only ky (mixes ky sectors for entangled states)
    # for ind, Lx in enumerate([4, 2, 1]):
    Ly = 4
    momenta = np.linspace(-np.pi, np.pi, 10)  # physical kx, full Brillouin zone
    # momenta = np.array([0.0, pi/2, pi, 2*pi])
    for ind, Lx in enumerate([2]):  # Lx > 1 now supported; compare against Lx=1
        color = colors[ind]
        markersize = markersizes[ind]
        chi_max = 1
        M = make_spin_model(Ly=Ly, Lx=Lx, Jxy=1.0, Jz=0.0, conserve='Sz', chain=chain)
        uniform_psi, M = find_ground_state(M, chi_max=chi_max, magz=-1.0)

        # qtotal_change=[0] keeps the excitation in the same Sz sector as the ground state
        # (e.g. magnon-like spin-flip pairs); use [1] (or [-1]) to target single-magnon-like
        # excitations that change total Sz by one unit, if your conserved charge allows it.
        num_ev = 1  # branches per ky sector; total per kx is Ly * num_ev
        if target_ky:
            # ky is imposed through H_eff - alpha e^{i ky} T_eff (Van Damme et al., Eq. 27)
            dispersion = compute_dispersion_targeted(uniform_psi, M, momenta,
                                                     qtotal_change=[2], alpha=1.0).reshape(-1)
        else:
            # old path: ky only seeded in the initial guess and measured afterwards
            dispersion = compute_dispersion(uniform_psi, M, momenta, qtotal_change=[2],
                                            num_ev=num_ev, lanczos_N_max=50)

        plt.plot(np.repeat(momenta, Ly * num_ev) / np.pi, dispersion, "o",
                 color=color, markersize=markersize)

    plt.xlabel("kx [pi]")
    plt.ylabel("E[J]")

    k_plot = np.linspace(-pi, pi, 500)
    kys = 2 * np.pi * np.arange(Ly) / Ly
    for ky in kys:
        plt.plot(k_plot/pi, cos(k_plot) + cos(ky), "r-")
        # plt.plot(k_plot/pi, cos(k_plot + pi) + cos(ky), "r-")

    plt.show()
    np.savetxt('dispersion.dat', np.column_stack([momenta, dispersion]),
               header='momentum   E_excitation')
    print("Saved dispersion to dispersion.dat")