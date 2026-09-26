"""Effective transversal-translation operator T_eff on the excitation tensors X.

Implements the object T_eff of Van Damme et al., PRB 104, 115142 (2021), Eq. (25)-(26):

    X'^dagger T_eff X = <Phi_p(B')| T_y |Phi_p(B)> / <Psi| T_y |Psi> ,   B = V_L X ,

as a block-sparse `NpcLinearOperator` that acts on the list of X tensors (one per site of the
unit cell) in exactly the same way as the effective Hamiltonian of TeNPy's
`PlaneWaveExcitationEngine`, so both can be combined in Eq. (27),

    ( H_eff - alpha * e^{+i ky} * T_eff ) X = lambda X .

Convention. T_y moves the content of site y to site y+1 inside every column of Ly sites (the
MPS order of the unit cell is column after column). A plane wave with amplitude e^{+i ky y} is
then an eigenvector of T_eff with eigenvalue e^{-i ky}; `ky_from_X` returns ky in that sense.
The kx phases follow TeNPy's `PlaneWaveExcitationEngine` (e^{-ipL} for a ket excitation one cell
to the right, e^{+ipL} to the left), so the same `p` must be used for H_eff and T_eff.

Shifted-ket trick. <Psi'|T_y|Psi> is an ordinary two-chain contraction if the ket chain is
shifted by one site against the bra chain: within a column, bra slot k is contracted with the ket
tensor A_{k-1}. The single physical leg that wraps around (slot 0 of the bra with the last ket
tensor of the column) is carried as one extra leg `c`, created at the first slot of the column
and absorbed at its last slot. Environments inside a column therefore have the legs
(vR*, vR, c*) (left) or (vL*, vL, c) (right); at column boundaries they are plain two-leg
chi x chi tensors. All labels: `*` = bra (conjugated) chain, no `*` = ket chain.
"""

import logging
import numpy as np

from tenpy.linalg import np_conserved as npc
from tenpy.linalg.sparse import NpcLinearOperator
from tenpy.linalg.krylov_based import Arnoldi

logger = logging.getLogger(__name__)

__all__ = ['MixedGroundState', 'TyEffectiveOperator', 'ky_from_X']


# ----------------------------------------------------------------------------------------------
# single-slot primitives
# ----------------------------------------------------------------------------------------------
def _add(a, b):
    """Sum of two optional npc arrays (None = zero)."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def left_step(E, bra, ket, close, first, last):
    """Extend a left environment by one bra slot.

    E     : environment before the slot, legs (vR*, vR) if `first`, else (c*, vR*, vR).
    bra   : bra tensor of the slot (conjugated inside).
    ket   : ket tensor paired with this slot (site g-1), None if `first`.
    close : ket tensor absorbed at the end of the column (site g), None unless `last`.
    Returns the environment before the next slot; two legs (vR*, vR) if `last`.
    """
    E = npc.tensordot(E, bra.conj(), axes=(['vR*'], ['vL*']))
    if first:
        E.ireplace_label('p*', 'c*')
    else:
        E = npc.tensordot(E, ket, axes=(['vR', 'p*'], ['vL', 'p']))
    if last:
        E = npc.tensordot(E, close, axes=(['vR', 'c*'], ['vL', 'p']))
    return E


def right_close(E, close):
    """Right boundary (vL*, vL) of a column -> (vL*, vL, c) with the last ket tensor absorbed."""
    E = npc.tensordot(close, E, axes=(['vR'], ['vL']))
    E.ireplace_label('p', 'c')
    return E


def right_step(E, bra, ket, first):
    """Move a right environment (vL*, vL, c) from behind a bra slot to in front of it.

    Returns (vL*, vL) at the start of a column (`first`), else again (vL*, vL, c).
    """
    E = npc.tensordot(bra.conj(), E, axes=(['vR*'], ['vL*']))
    if first:
        return npc.trace(E, 'p*', 'c')
    return npc.tensordot(ket, E, axes=(['vR', 'p'], ['vL', 'p*']))


def open_bra(Lenv, ket, Renv, VL, first):
    """Contract everything except the bra tensor V_L of one slot -> gradient w.r.t. conj(X').

    Lenv : left environment before the slot;  Renv : right environment behind it, (vL*, vL, c).
    """
    T = npc.tensordot(Lenv, VL.conj(), axes=(['vR*'], ['vL*']))
    if first:
        T = npc.tensordot(T, Renv, axes=(['vR', 'p*'], ['vL', 'c']))
    else:
        T = npc.tensordot(T, ket, axes=(['vR', 'p*'], ['vL', 'p']))
        T = npc.tensordot(T, Renv, axes=(['vR', 'c*'], ['vL', 'c']))
    T.ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
    return T.itranspose(['vL', 'vR'])


# ----------------------------------------------------------------------------------------------
# whole-unit-cell passes
# ----------------------------------------------------------------------------------------------
class _Cell:
    """Column bookkeeping for a unit cell of Lx columns with Ly sites each."""

    def __init__(self, L, Ly):
        assert L % Ly == 0
        self.L, self.Ly = L, Ly

    def first(self, g):
        return g % self.Ly == 0

    def last(self, g):
        return g % self.Ly == self.Ly - 1


def left_cell(E, bra, ket, cell):
    """One full unit cell of the mixed transfer matrix acting on a left environment."""
    for g in range(cell.L):
        f, l = cell.first(g), cell.last(g)
        E = left_step(E, bra[g], None if f else ket[g - 1], ket[g] if l else None, f, l)
    return E


def right_cell(E, bra, ket, cell):
    """One full unit cell of the mixed transfer matrix acting on a right environment."""
    for g in reversed(range(cell.L)):
        f, l = cell.first(g), cell.last(g)
        if l:
            E = right_close(E, ket[g])
        E = right_step(E, bra[g], None if f else ket[g - 1], f)
    return E


class _EnvOperator(NpcLinearOperator):
    """Wrap a cell pass as an NpcLinearOperator on a two-leg environment."""

    def __init__(self, fn, labels):
        self.fn, self.labels = fn, labels

    def matvec(self, vec):
        return self.fn(vec).itranspose(self.labels)


def _random_env(leg, labels, dtype=np.complex128):
    def rand(shape):
        return np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)

    return npc.Array.from_func(rand, [leg.conj(), leg], dtype=dtype, labels=labels)


def _leading_eigenpair(op, v0, mu_hint, N_max, num_ev=4):
    """Leading eigenvalue of `op` (largest modulus; ties resolved by phase closest to mu_hint)."""
    params = {'which': 'LM', 'num_ev': num_ev, 'N_max': N_max, 'N_min': 4, 'P_tol': 1.e-16}
    Es, vs, N = Arnoldi(op, v0, params).run()
    Es = np.asarray(Es)
    big = np.abs(Es) > 0.9 * np.max(np.abs(Es))
    phase_dist = np.abs(np.angle(Es * np.exp(-1j * np.angle(mu_hint))))
    idx = min(np.flatnonzero(big), key=lambda i: phase_dist[i])
    v = vs[idx]
    res = npc.norm(op.matvec(v) - Es[idx] * v)
    logger.info('mixed transfer matrix: mu = %s (|mu| = %.10f), residual %.2e, %d Krylov steps',
                Es[idx], abs(Es[idx]), res, N)
    return Es[idx], v, res


# ----------------------------------------------------------------------------------------------
# ground-state data (independent of X, kx, ky): computed once
# ----------------------------------------------------------------------------------------------
class MixedGroundState:
    """mu, l, r and the site-resolved mixed environments of the ground state.

    Parameters
    ----------
    ALs, ARs : list of npc.Array
        Left/right isometric tensors of the unit cell (labels vL, p, vR).
    C0 : npc.Array
        Center matrix (labels vL, vR) on the bond to the left of site 0, i.e. on the unit-cell
        boundary (`UniformMPS.get_C(0)`); AL_i C_i = C_{i-1} AR_i. It fixes the normalisation
        <Psi|T_y|Psi> = mu^N (l| C* x C |r).
    Ly : int
        Number of sites per column; the unit cell holds L/Ly columns.
    mu_hint : complex
        Selects mu among near-degenerate leading eigenvalues by its phase (default: closest to 1).
    N_max : int
        Krylov steps for the fixed-point eigensolves.
    """

    def __init__(self, ALs, ARs, C0, Ly, mu_hint=1.0, N_max=60):
        self.ALs, self.ARs = ALs, ARs
        self.L = len(ALs)
        self.cell = _Cell(self.L, Ly)
        cell = self.cell

        legR = ALs[-1].get_leg('vR')
        op_l = _EnvOperator(lambda E: left_cell(E, ALs, ALs, cell), ['vR*', 'vR'])
        mu_l, l, res_l = _leading_eigenpair(op_l, _random_env(legR, ['vR*', 'vR']), mu_hint, N_max)
        legL = ARs[0].get_leg('vL')
        op_r = _EnvOperator(lambda E: right_cell(E, ARs, ARs, cell), ['vL*', 'vL'])
        mu_r, r, res_r = _leading_eigenpair(op_r, _random_env(legL, ['vL*', 'vL']), mu_l, N_max)
        if abs(mu_l - mu_r) > 1.e-6 * abs(mu_l):
            logger.warning('left/right leading eigenvalues of the mixed transfer matrix differ: '
                           '%s vs %s', mu_l, mu_r)
        self.mu = mu_l
        self.fixed_point_residual = max(res_l, res_r)
        # ground-state expectation value per unit cell: D = (l| C* x C |r) (in the untwisted case
        # l = r = 1 and D = tr C^dag C = 1). Every unit cell touched by an excitation carries
        # 1/mu, the rest of the chain cancels against the ground-state normalisation.
        lC = npc.tensordot(l, C0.conj(), axes=(['vR*'], ['vL*']))
        lC = npc.tensordot(lC, C0, axes=(['vR'], ['vL']))
        D = npc.tensordot(lC, r, axes=(['vR*', 'vR'], ['vL*', 'vL']))
        self.l = l / D
        self.r = r

        # site-resolved environments without excitation (all AL/AL to the left, AR/AR to the right)
        self.LP = []  # LP[g]: before slot g
        E = self.l
        for g in range(self.L):
            self.LP.append(E)
            f, la = cell.first(g), cell.last(g)
            E = left_step(E, ALs[g], None if f else ALs[g - 1], ALs[g] if la else None, f, la)
        self.RP = [None] * self.L  # RP[g]: behind slot g, legs (vL*, vL, c)
        self.RPbd = [None] * self.L  # RPbd[g] (last slot of a column): two-leg boundary behind it
        E = self.r
        for g in reversed(range(self.L)):
            f, la = cell.first(g), cell.last(g)
            if la:
                self.RPbd[g] = E
                E = right_close(E, ARs[g])
            self.RP[g] = E
            E = right_step(E, ARs[g], None if f else ARs[g - 1], f)


# ----------------------------------------------------------------------------------------------
# T_eff
# ----------------------------------------------------------------------------------------------
class TyEffectiveOperator(NpcLinearOperator):
    """T_eff acting on the list of excitation tensors X (labels vL, vR), see module docstring.

    Parameters
    ----------
    gs : MixedGroundState
    VLs : list of npc.Array
        Orthogonal complements of the AL tensors (as in the PlaneWaveExcitationEngine).
    p : float
        Momentum as passed to `PlaneWaveExcitationEngine.run` (already includes the factor L).
    sum_tol, sum_iterations :
        Convergence criteria of the geometric sums over cells (same meaning as in TeNPy).
    """

    def __init__(self, gs, VLs, p, sum_tol=1.e-10, sum_iterations=100):
        self.gs, self.VLs, self.p = gs, VLs, p
        self.ALs, self.ARs = gs.ALs, gs.ARs
        self.L, self.cell = gs.L, gs.cell
        self.sum_tol, self.sum_iterations = sum_tol, sum_iterations
        self.scale = 1. / gs.mu  # one factor 1/mu for every unit cell that is touched
        self.phase_R = np.exp(-1.j * p * self.L)  # ket excitation one cell to the right
        self.phase_L = np.exp(+1.j * p * self.L)  # ... to the left
        self.n_sum_iterations = (0, 0)

    # -- sweeps that carry the "one ket excitation" track (B); the excitation-free track is the
    #    ground-state data gs.LP / gs.RP / gs.RPbd (all start from the fixed points l, r) --
    def _sweep_left(self, LB, Bs, store=False):
        """LB: left boundary that already contains a ket excitation (or None). Returns the
        environment behind the cell and, if `store`, the list of LB before every slot."""
        AL, AR, cell, gs = self.ALs, self.ARs, self.cell, self.gs
        LBs = []
        for g in range(self.L):
            if store:
                LBs.append(LB)
            LP = gs.LP[g]
            f, la = cell.first(g), cell.last(g)
            k, c = (None if f else g - 1), (g if la else None)
            kt = lambda T, s: None if s is None else T[s]
            LBn = None if LB is None else left_step(LB, AL[g], kt(AR, k), kt(AR, c), f, la)
            if k is not None:  # excitation on the ket tensor paired with this slot
                LBn = _add(LBn, left_step(LP, AL[g], Bs[k], kt(AR, c), f, la))
            if c is not None:  # excitation on the ket tensor closing the column
                LBn = _add(LBn, left_step(LP, AL[g], kt(AL, k), Bs[c], f, la))
            LB = LBn
        return LB, LBs

    def _sweep_right(self, RB, Bs):
        """RB: right boundary that already contains a ket excitation (or None). Returns the
        environment in front of the cell and the list of RB behind every slot."""
        AL, AR, cell, gs = self.ALs, self.ARs, self.cell, self.gs
        R3B = [None] * self.L
        for g in reversed(range(self.L)):
            f, la = cell.first(g), cell.last(g)
            if la:  # absorb the closing ket tensor; the excitation may sit right here
                RBn = None if RB is None else right_close(RB, AL[g])
                RB = _add(RBn, right_close(gs.RPbd[g], Bs[g]))
            R3B[g] = RB
            if f:
                RB = None if RB is None else right_step(RB, AR[g], None, True)
            else:
                RBn = None if RB is None else right_step(RB, AR[g], AL[g - 1], False)
                RB = _add(RBn, right_step(gs.RP[g], AR[g], Bs[g - 1], False))
        return RB, R3B

    def _geometric_sum(self, term, transfer, phase):
        total = term
        n = 0
        for n in range(1, self.sum_iterations + 1):
            term = (phase * self.scale) * transfer(term)
            total = total + term
            if npc.norm(term) < self.sum_tol:
                break
        else:
            logger.warning('geometric sum over unit cells not converged (|last term| = %.2e)',
                           npc.norm(term))
        return total, n

    def matvec(self, X):
        gs, L, cell = self.gs, self.L, self.cell
        AL, AR, VL = self.ALs, self.ARs, self.VLs
        Bs = [npc.tensordot(VL[g], X[g], axes=(['vR'], ['vL'])) for g in range(L)]

        # ket excitation in unit cells to the left / right of the reference cell
        LB_end, _ = self._sweep_left(None, Bs)
        LB_end = self.scale * LB_end
        L_sum, nL = self._geometric_sum(LB_end, lambda E: left_cell(E, AL, AR, cell), self.phase_L)
        L_bd = self.phase_L * L_sum

        RB_start, _ = self._sweep_right(None, Bs)
        RB_start = self.scale * RB_start
        R_sum, nR = self._geometric_sum(RB_start, lambda E: right_cell(E, AR, AL, cell),
                                        self.phase_R)
        R_bd = self.phase_R * R_sum
        self.n_sum_iterations = (nL, nR)

        # reference cell: bra excitation V_L opened at every slot, ket excitation anywhere
        _, LBs = self._sweep_left(L_bd, Bs, store=True)
        _, R3B = self._sweep_right(R_bd, Bs)

        out = []
        for g in range(L):
            f = cell.first(g)
            LPg, LBg, RPg, RBg = gs.LP[g], LBs[g], gs.RP[g], R3B[g]
            k = None if f else g - 1
            kt = lambda T: None if k is None else T[k]
            o = open_bra(LBg, kt(AR), RPg, VL[g], f)
            o = o + open_bra(LPg, kt(AL), RBg, VL[g], f)
            if k is not None:
                o = o + open_bra(LPg, Bs[k], RPg, VL[g], f)
            out.append(self.scale * o)
        return out


def ky_from_X(Teff, X):
    """(ky, 1 - |ratio|) of an excitation, from the Rayleigh quotient X^dag T_eff X / X^dag X.

    ky is defined by an amplitude e^{+i ky y} (eigenvalue e^{-i ky} of T_eff, see module doc).
    """
    TX = Teff.matvec(X)
    ratio = sum(npc.inner(x, t, axes='range', do_conj=True) for x, t in zip(X, TX))
    ratio /= sum(npc.inner(x, x, axes='range', do_conj=True) for x in X)
    return float(-np.angle(ratio)), float(1. - abs(ratio))
