"""
Tests for Gutzwiller.FreeFermionSpinCorrelations.

The function takes the (2n x 2n) correlation matrix C (mode index 2*i + spin, spin 0=up, 1=down), where the down block
is stored particle-hole transformed: C_down = 1 - C[dn, dn]^T. It returns the (n x n) matrix <S_i . S_j>.
The reference is exact diagonalisation (Jordan-Wigner) of a small Slater determinant with real orbitals.
"""
import numpy as np
import pytest

from Gutzwiller import FreeFermionSpinCorrelations

I2 = np.eye(2)
SZ_JW = np.diag([1.0, -1.0])
ANNIHILATE = np.array([[0, 1], [0, 0]], dtype=float)  # |0> <-> |1>, c|1> = |0>


def _annihilation_ops(n_modes):
    ops = []
    for m in range(n_modes):
        factors = [SZ_JW] * m + [ANNIHILATE] + [I2] * (n_modes - m - 1)
        op = factors[0]
        for f in factors[1:]:
            op = np.kron(op, f)
        ops.append(op)
    return ops


def _random_orbitals(rng, n, n_particles):
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return q[:, :n_particles]


def _slater_state(c_ops, orbitals_up, orbitals_dn):
    n = orbitals_up.shape[0]
    dim = c_ops[0].shape[0]
    state = np.zeros(dim)
    state[0] = 1.0  # vacuum is the all-|0> state
    for spin, orbitals in enumerate([orbitals_up, orbitals_dn]):
        for k in range(orbitals.shape[1]):
            creator = sum(orbitals[i, k] * c_ops[2 * i + spin].T for i in range(n))
            state = creator @ state
    return state / np.linalg.norm(state)


def _exact_spin_correlations(orbitals_up, orbitals_dn):
    n = orbitals_up.shape[0]
    c = _annihilation_ops(2 * n)
    psi = _slater_state(c, orbitals_up, orbitals_dn)

    def spin_ops(i):
        cu, cd = c[2 * i], c[2 * i + 1]
        sz = 0.5 * (cu.T @ cu - cd.T @ cd)
        sp = cu.T @ cd
        return sp, sz

    corr = np.zeros((n, n))
    for i in range(n):
        spi, szi = spin_ops(i)
        for j in range(n):
            spj, szj = spin_ops(j)
            # S_i . S_j = Sz Sz + 1/2 (S+ S- + S- S+)
            op = szi @ szj + 0.5 * (spi @ spj.T + spi.T @ spj)
            corr[i, j] = psi @ op @ psi
    return corr


def _build_C(orbitals_up, orbitals_dn):
    """Correlation matrix in the convention expected by FreeFermionSpinCorrelations."""
    n = orbitals_up.shape[0]
    P_up = orbitals_up @ orbitals_up.T
    P_dn = orbitals_dn @ orbitals_dn.T
    C = np.zeros((2 * n, 2 * n))
    C[0::2, 0::2] = P_up
    C[1::2, 1::2] = (np.eye(n) - P_dn).T  # so that I - C_dd^T = P_dn
    return C


@pytest.mark.parametrize("n, n_up, n_dn, seed", [(3, 1, 1, 0), (4, 2, 2, 1), (4, 2, 1, 2), (4, 3, 1, 3), (3, 0, 2, 4)])
def test_matches_exact_diagonalisation(n, n_up, n_dn, seed):
    rng = np.random.default_rng(seed)
    orb_up = _random_orbitals(rng, n, n_up)
    orb_dn = _random_orbitals(rng, n, n_dn)
    result = FreeFermionSpinCorrelations(_build_C(orb_up, orb_dn))
    expected = _exact_spin_correlations(orb_up, orb_dn)
    assert result.shape == (n, n)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_result_is_symmetric_and_real():
    rng = np.random.default_rng(10)
    orb_up = _random_orbitals(rng, 4, 2)
    orb_dn = _random_orbitals(rng, 4, 2)
    result = FreeFermionSpinCorrelations(_build_C(orb_up, orb_dn))
    np.testing.assert_allclose(result.imag, 0, atol=1e-12)
    np.testing.assert_allclose(result, result.T, atol=1e-12)


def test_singly_occupied_sites_onsite_value():
    # one electron per site (sites 0,1 up; sites 2,3 down), a product state: <S_i.S_i> = 3/4, <S_i.S_j> = Sz_i Sz_j.
    n = 4
    orb_up = np.eye(n)[:, :2]
    orb_dn = np.eye(n)[:, 2:]
    result = FreeFermionSpinCorrelations(_build_C(orb_up, orb_dn))
    sz = np.array([0.5, 0.5, -0.5, -0.5])
    expected = np.outer(sz, sz) + np.diag(0.75 - sz ** 2)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_non_hermitian_blocks_rejected():
    C = np.zeros((4, 4))
    C[0, 2] = 1.0  # C_up not hermitian
    with pytest.raises(AssertionError):
        FreeFermionSpinCorrelations(C)
