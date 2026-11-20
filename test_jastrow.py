"""Tests for the nearest-neighbor SzSz and XY Jastrow factors (Gutzwiller.ApplyNNSzSzJastrow, ApplyNNXYJastrow)."""
import itertools

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import FermionSite, SpinHalfSite

from Gutzwiller import (ApplyNNSzSzJastrow, ApplyNNXYJastrow, GetPiFluxTriangularLattice, GroupBondsIntoMPOLayers,
                        NearestNeighborMPSBonds, StrangLayerSequence, XYJastrowLayerMPO)
from Main import BuildTriangularLattice

EXACT_TRUNC_PAR = {"chi_max": 4096, "svd_min": 1e-14}


def _spin_lattice(Lx, Ly, bc_MPS, geometry, conserve="Sz"):
    bc = ("open" if bc_MPS == "finite" else "periodic", "periodic")
    return BuildTriangularLattice(Lx, Ly, SpinHalfSite(conserve), bc_MPS, bc=bc, geometry=geometry)


def _brute_force_nn_bonds(lat):
    """Pairs at real-space distance 1 on the cylinder, from the site positions only."""
    L = lat.N_sites
    finite = lat.bc_MPS == "finite"
    circumference = lat.Ls[1] * np.asarray(lat.basis[1])
    x_period = lat.Ls[0] * np.asarray(lat.basis[0])

    def position(i):
        # mps2lat_idx folds i into the unit cell for infinite MPS, so add the unit cell shift by hand
        return lat.position(lat.mps2lat_idx(i % L)) + (i // L) * x_period

    bonds = set()
    for i in range(L):
        for j in range(i + 1, L if finite else i + L + 1):
            d = min(np.linalg.norm(position(j) + n * circumference - position(i)) for n in (-1, 0, 1))
            if abs(d - 1) < 1e-8:
                bonds.add((i, j))
    return bonds


@pytest.mark.parametrize("geometry,Lx,Ly", [("YC", 2, 4), ("YC", 4, 6), ("XC", 2, 3), ("XC", 2, 4)])
@pytest.mark.parametrize("bc_MPS", ["finite", "infinite"])
def test_nn_bonds_match_geometry(geometry, Lx, Ly, bc_MPS):
    lat = _spin_lattice(Lx, Ly, bc_MPS, geometry)
    bonds = NearestNeighborMPSBonds(lat)
    assert len(bonds) == len(set(bonds))
    assert set(bonds) == _brute_force_nn_bonds(lat)
    if bc_MPS == "infinite":
        L = lat.N_sites
        assert len(bonds) == 3 * L
        degree = np.bincount([k % L for bond in bonds for k in bond], minlength=L)
        assert np.all(degree == 6)


@pytest.mark.parametrize("geometry,Lx,Ly", [("YC", 2, 6), ("YC", 6, 4), ("XC", 2, 4)])
@pytest.mark.parametrize("bc_MPS", ["finite", "infinite"])
def test_spin_lattice_matches_projected_fermion_lattice(geometry, Lx, Ly, bc_MPS):
    """The Abrikosov projection groups fermion sites (2k, 2k+1) into spin site k; the spin lattice used for the
    Jastrow factor must have the same bonds and site positions."""
    fermion_lat = GetPiFluxTriangularLattice(FermionSite("N"), Lx, Ly, True, bc_MPS, geometry)
    spin_lat = BuildTriangularLattice(Lx, Ly, SpinHalfSite("Sz"), bc_MPS, bc=fermion_lat.boundary_conditions,
                                      geometry=geometry)
    assert 2 * spin_lat.N_sites == fermion_lat.N_sites

    fermion_bonds = NearestNeighborMPSBonds(fermion_lat)
    assert all((i - j) % 2 == 0 for i, j in fermion_bonds), "NN hoppings should not mix flavors"
    assert {(i // 2, j // 2) for i, j in fermion_bonds} == set(NearestNeighborMPSBonds(spin_lat))

    for k in range(spin_lat.N_sites):
        fermion_positions = [fermion_lat.position(fermion_lat.mps2lat_idx(2 * k + s)) for s in range(2)]
        assert np.allclose(np.mean(fermion_positions, axis=0), spin_lat.position(spin_lat.mps2lat_idx(k)))


def test_layers_have_disjoint_strings():
    for bc_MPS in ["finite", "infinite"]:
        lat = _spin_lattice(2, 6, bc_MPS, "YC")
        L = lat.N_sites
        bonds = NearestNeighborMPSBonds(lat)
        layers = GroupBondsIntoMPOLayers(bonds, L, bc_MPS == "finite")
        assert sorted(b for layer in layers for b in layer) == bonds
        for layer in layers:
            cuts = [c % L for i, j in layer for c in range(i, j)]
            assert len(cuts) == len(set(cuts))


def test_bond_longer_than_unit_cell_raises():
    # YC Lx=1: the wrapping diagonal bond has range 2Ly-1 > L=Ly
    lat = _spin_lattice(1, 4, "infinite", "YC")
    with pytest.raises(ValueError):
        GroupBondsIntoMPOLayers(NearestNeighborMPSBonds(lat), lat.N_sites, False)


def _random_finite_mps(lat, conserve, seed):
    site = SpinHalfSite(conserve)
    N = lat.N_sites
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    if conserve == "Sz":
        sz_total = np.sum(_basis_sz(site, N), axis=1)
        vec[np.abs(sz_total) > 1e-12] = 0
    vec /= np.linalg.norm(vec)
    theta = npc.Array.from_ndarray(vec.reshape([2] * N), [site.leg] * N, labels=[f"p{i}" for i in range(N)])
    return MPS.from_full([site] * N, theta, form="B"), vec


def _basis_sz(site, N):
    """Sz of every site for each basis state of the dense vector, in the site's own basis order."""
    sz_diag = np.real(np.diag(site.get_op("Sz").to_ndarray()))
    return sz_diag[np.array(list(itertools.product(range(2), repeat=N)))]


def _dense_jastrow(vec, bonds, site, N, v):
    sz = _basis_sz(site, N)
    E = sum(sz[:, i] * sz[:, j] for i, j in bonds)
    out = np.exp(-v * E) * vec
    return out / np.linalg.norm(out)


def _full_vector(psi):
    return psi.get_theta(0, psi.L).to_ndarray().ravel() * psi.norm


@pytest.mark.parametrize("geometry,Lx,Ly", [("YC", 4, 3), ("XC", 2, 3)])
@pytest.mark.parametrize("conserve", ["Sz", None])
@pytest.mark.parametrize("v", [0.8, -1.3, 3.0])
def test_finite_matches_dense(geometry, Lx, Ly, conserve, v):
    lat = _spin_lattice(Lx, Ly, "finite", geometry, conserve)
    psi, vec = _random_finite_mps(lat, conserve, seed=7)
    trunc_err = ApplyNNSzSzJastrow(psi, lat, v, EXACT_TRUNC_PAR)
    expected = _dense_jastrow(vec, NearestNeighborMPSBonds(lat), psi.sites[0], lat.N_sites, v)
    psi.test_sanity()
    assert trunc_err.eps < 1e-12
    assert abs(psi.norm - 1) < 1e-12
    assert abs(abs(np.vdot(expected, _full_vector(psi))) - 1) < 1e-10


def test_finite_zero_and_inverse():
    lat = _spin_lattice(4, 3, "finite", "YC")
    psi, _ = _random_finite_mps(lat, "Sz", seed=3)
    phi = psi.copy()
    ApplyNNSzSzJastrow(phi, lat, 0.0, EXACT_TRUNC_PAR)
    assert abs(abs(phi.overlap(psi)) - 1) < 1e-12
    ApplyNNSzSzJastrow(phi, lat, 1.1, EXACT_TRUNC_PAR)
    assert abs(phi.overlap(psi)) < 1 - 1e-3
    ApplyNNSzSzJastrow(phi, lat, -1.1, EXACT_TRUNC_PAR)
    assert abs(abs(phi.overlap(psi)) - 1) < 1e-10


def _classical_ising_yc_nn_correlations(Ly, K):
    """
        <s_i s_j> (s = +-1) on the infinite YC triangular cylinder with weight exp(-K sum_<ij> s_i s_j), by column
        transfer matrix T[s, s'] (s left column, s' right column). Returns (x-bond, y-bond, diagonal [-1,1] bond).
    """
    cols = 1 - 2 * np.array(list(itertools.product([0, 1], repeat=Ly)))
    s, sp = cols[:, None, :], cols[None, :, :]
    E = np.sum(s * sp, axis=2) + np.sum(sp * np.roll(s, -1, axis=2), axis=2) \
        + np.sum(sp * np.roll(sp, -1, axis=2), axis=2)
    T = np.exp(-K * E)
    w, vr = np.linalg.eig(T)
    wl, vl = np.linalg.eig(T.T)
    r = np.real(vr[:, np.argmax(np.real(w))])
    l = np.real(vl[:, np.argmax(np.real(wl))])
    lam = np.max(np.real(w))

    def expect(f):
        return l @ (T * f) @ r / (lam * (l @ r))

    x_bond = expect(s[..., 0] * sp[..., 0])
    y_bond = expect(sp[..., 0] * sp[..., 1])
    diag_bond = expect(sp[..., 0] * s[..., 1])  # (x, y) - (x-1, y+1)
    return x_bond, y_bond, diag_bond


@pytest.mark.parametrize("Ly,v", [(3, 0.7), (3, -0.4), (4, 0.7), (4, -0.4), (4, 3.0)])
def test_infinite_x_polarized_matches_classical_ising(Ly, v):
    """Jastrow on the x-polarized product state: |psi_J(sigma)|^2 ~ exp(-2v sum Sz Sz) is a classical Ising
    weight with K = v/2 for s = 2Sz, solvable exactly on the infinite cylinder by a transfer matrix.
    (v < 0 stays above the ferromagnetic K_c = -ln(3)/4: deeper in, the Z2-symmetric state is a cat state with a
    huge correlation length and the canonical form converges very slowly.)"""
    Lx = 2
    lat = _spin_lattice(Lx, Ly, "infinite", "YC", conserve=None)
    psi = MPS.from_product_state(lat.mps_sites(), [np.array([1., 1.]) / np.sqrt(2)] * lat.N_sites, bc="infinite")
    trunc_err = ApplyNNSzSzJastrow(psi, lat, v, EXACT_TRUNC_PAR)
    psi.test_sanity()
    assert trunc_err.eps < 1e-12

    def szsz(i, j):
        return np.real(psi.correlation_function("Sz", "Sz", sites1=[i], sites2=[j])[0, 0])

    x_bond, y_bond, diag_bond = _classical_ising_yc_nn_correlations(Ly, v / 2)
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            assert abs(4 * szsz(i, i + Ly) - x_bond) < 1e-9
            assert abs(4 * szsz(i, x * Ly + (y + 1) % Ly) - y_bond) < 1e-9
            assert abs(4 * szsz(x * Ly + (y + 1) % Ly, i + Ly) - diag_bond) < 1e-9


def _drop_charges(psi):
    """Same state on SpinHalfSite(None), whose physical basis is (up, down), while the charge-sorted
    SpinHalfSite('Sz') basis is (down, up)."""
    site = SpinHalfSite(None)
    sites = [site] * psi.L
    perm = [psi.sites[0].state_labels[label] for label in ("up", "down")]
    assert [site.state_labels[label] for label in ("up", "down")] == [0, 1]
    Bs = [npc.Array.from_ndarray_trivial(psi.get_B(i, "B").transpose(["vL", "p", "vR"]).to_ndarray()[:, perm, :],
                                         labels=["vL", "p", "vR"]) for i in range(psi.L)]
    return MPS(sites, Bs, [psi.get_SL(i) for i in range(psi.L)] + [psi.get_SR(psi.L - 1)], bc=psi.bc, form="B")


def test_infinite_charge_conservation_consistent():
    """Same random iMPS with and without Sz conservation gives the same Jastrow state, so the charges of the
    layer MPOs (including the legs crossing the unit cell boundary) are consistent."""
    lat = _spin_lattice(2, 3, "infinite", "YC")
    np.random.seed(5)
    psi = MPS.from_random_unitary_evolution(lat.mps_sites(), 6, ["up", "down"] * (lat.N_sites // 2),
                                            bc="infinite")
    psi_no_charge = _drop_charges(psi)
    v = 1.2
    ApplyNNSzSzJastrow(psi, lat, v, EXACT_TRUNC_PAR)
    ApplyNNSzSzJastrow(psi_no_charge, _spin_lattice(2, 3, "infinite", "YC", conserve=None), v, EXACT_TRUNC_PAR)
    psi.test_sanity()
    for op in ["Sz", "Sp"]:
        op2 = "Sz" if op == "Sz" else "Sm"
        c = psi.correlation_function(op, op2, sites1=range(6), sites2=range(12))
        c_no_charge = psi_no_charge.correlation_function(op, op2, sites1=range(6), sites2=range(12))
        assert np.allclose(c, c_no_charge, atol=1e-9)


# ---------------------------------------------------------------- XY Jastrow factor

XY_TRUNC_PAR = {"chi_max": 48, "svd_min": 1e-10}  # the iMPS tests are slow, chi saturates the cap


def _xy_bond_op(site):
    """Sx Sx + Sy Sy on two sites, as a (4, 4) matrix in the site's basis order."""
    Sp, Sm = site.get_op("Sp").to_ndarray(), site.get_op("Sm").to_ndarray()
    return 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp))


def _apply_two_site(vec, op, i, j, N):
    """(4, 4) operator op on sites i, j of the dense vector."""
    t = np.tensordot(op.reshape(2, 2, 2, 2), vec.reshape([2] * N), axes=([2, 3], [i, j]))
    return np.moveaxis(t, [0, 1], [i, j]).ravel()


def _dense_xy_hamiltonian(bonds, site, N):
    Sp = scipy.sparse.csr_matrix(site.get_op("Sp").to_ndarray())
    Sm = scipy.sparse.csr_matrix(site.get_op("Sm").to_ndarray())

    def embed(ops):
        out = scipy.sparse.identity(1, format="csr")
        for k in range(N):
            out = scipy.sparse.kron(out, ops.get(k, scipy.sparse.identity(2)), format="csr")
        return out

    return sum(0.5 * (embed({i: Sp, j: Sm}) + embed({i: Sm, j: Sp})) for i, j in bonds)


def _dense_xy_strang(vec, lat, v, n_steps):
    """The same Strang product as ApplyNNXYJastrow, applied gate by gate to the dense vector."""
    N = lat.N_sites
    site = lat.mps_sites()[0]
    layers = GroupBondsIntoMPOLayers(NearestNeighborMPSBonds(lat), N, True, site_disjoint=True)
    bond_op = _xy_bond_op(site)
    for layer, weight in StrangLayerSequence(len(layers), n_steps):
        gate = scipy.linalg.expm(-weight * v * bond_op)
        for i, j in layers[layer]:
            vec = _apply_two_site(vec, gate, i, j, N)
    return vec / np.linalg.norm(vec)


def _dense_xy_exact(vec, lat, v):
    H = _dense_xy_hamiltonian(NearestNeighborMPSBonds(lat), lat.mps_sites()[0], lat.N_sites)
    out = scipy.sparse.linalg.expm_multiply(-v * H, vec)
    return out / np.linalg.norm(out)


@pytest.mark.parametrize("conserve", ["Sz", None])
@pytest.mark.parametrize("bond", [(0, 1), (0, 2), (1, 2)])
def test_xy_gate_matches_expm(conserve, bond):
    """The MPO of a single bond, with or without an identity string, is exp(-v (Sx Sx + Sy Sy))."""
    site = SpinHalfSite(conserve)
    v = -1.9
    mpo = XYJastrowLayerMPO([site] * 3, [bond], "finite", v)
    mpo.test_sanity()
    W0, W1, W2 = [mpo.get_W(k).transpose(["wL", "wR", "p", "p*"]).to_ndarray() for k in range(3)]
    full = np.einsum("xaAB,abCD,byEF->ACEBDF", W0, W1, W2).reshape(8, 8)
    gate = scipy.linalg.expm(-v * _xy_bond_op(site))
    expected = np.column_stack([_apply_two_site(e, gate, *bond, 3) for e in np.eye(8)])
    assert np.allclose(full, expected, atol=1e-12)


@pytest.mark.parametrize("geometry,Lx,Ly", [("YC", 2, 4), ("YC", 2, 6), ("XC", 2, 4)])
@pytest.mark.parametrize("bc_MPS", ["finite", "infinite"])
def test_site_disjoint_layers(geometry, Lx, Ly, bc_MPS):
    lat = _spin_lattice(Lx, Ly, bc_MPS, geometry)
    L = lat.N_sites
    bonds = NearestNeighborMPSBonds(lat)
    layers = GroupBondsIntoMPOLayers(bonds, L, bc_MPS == "finite", site_disjoint=True)
    assert sorted(b for layer in layers for b in layer) == bonds
    for layer in layers:
        cuts = [c % L for i, j in layer for c in range(i, j)]
        assert len(cuts) == len(set(cuts))
        sites = [k % L for bond in layer for k in bond]
        assert len(sites) == len(set(sites))


@pytest.mark.parametrize("n_layers,n_steps", [(1, 3), (2, 1), (5, 1), (5, 4)])
def test_strang_sequence(n_layers, n_steps):
    sequence = StrangLayerSequence(n_layers, n_steps)
    assert [layer for layer, _ in sequence] == [layer for layer, _ in reversed(sequence)]
    assert all(a[0] != b[0] for a, b in zip(sequence, sequence[1:]))
    for layer in range(n_layers):
        assert abs(sum(w for l, w in sequence if l == layer) - 1) < 1e-12
    if n_layers > 1:
        assert len(sequence) == n_steps * (2 * n_layers - 2) + 1


@pytest.mark.parametrize("geometry,Lx,Ly", [("YC", 4, 3), ("XC", 2, 3)])
@pytest.mark.parametrize("conserve", ["Sz", None])
@pytest.mark.parametrize("v,n_steps", [(0.8, 1), (-1.3, 3)])
def test_xy_finite_matches_dense_strang(geometry, Lx, Ly, conserve, v, n_steps):
    lat = _spin_lattice(Lx, Ly, "finite", geometry, conserve)
    psi, vec = _random_finite_mps(lat, conserve, seed=11)
    trunc_err = ApplyNNXYJastrow(psi, lat, v, n_steps, EXACT_TRUNC_PAR)
    expected = _dense_xy_strang(vec, lat, v, n_steps)
    psi.test_sanity()
    assert trunc_err.eps < 1e-12
    assert abs(psi.norm - 1) < 1e-12
    assert abs(abs(np.vdot(expected, _full_vector(psi))) - 1) < 1e-10


def test_xy_trotter_error_second_order():
    """The distance between the Strang and the exact Jastrow states vanishes as 1/N^2."""
    lat = _spin_lattice(4, 3, "finite", "YC")
    _, vec = _random_finite_mps(lat, "Sz", seed=2)
    v = 0.6
    exact = _dense_xy_exact(vec, lat, v)
    errors = [np.linalg.norm(_dense_xy_strang(vec, lat, v, n) - exact) for n in [2, 4, 8]]
    assert errors[0] > 1e-8, "Trotter error unexpectedly small, the test is meaningless"
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.5 < coarse / fine < 4.5


def test_xy_finite_zero_and_inverse():
    """The Strang product is palindromic, so the -v product inverts the v product exactly."""
    lat = _spin_lattice(4, 3, "finite", "YC")
    psi, _ = _random_finite_mps(lat, "Sz", seed=3)
    phi = psi.copy()
    ApplyNNXYJastrow(phi, lat, 0.0, 2, EXACT_TRUNC_PAR)
    assert abs(abs(phi.overlap(psi)) - 1) < 1e-12
    ApplyNNXYJastrow(phi, lat, 1.1, 2, EXACT_TRUNC_PAR)
    assert abs(phi.overlap(psi)) < 1 - 1e-3
    ApplyNNXYJastrow(phi, lat, -1.1, 2, EXACT_TRUNC_PAR)
    assert abs(abs(phi.overlap(psi)) - 1) < 1e-10


def _random_imps(lat, seed):
    np.random.seed(seed)
    return MPS.from_random_unitary_evolution(lat.mps_sites(), 6, ["up", "down"] * (lat.N_sites // 2),
                                             bc="infinite")


def _nn_correlations(psi):
    c = [psi.correlation_function(op1, op2, sites1=range(psi.L), sites2=range(2 * psi.L))
         for op1, op2 in [("Sz", "Sz"), ("Sp", "Sm")]]
    return np.array(c)


def test_xy_infinite_charge_conservation_consistent():
    """Same random iMPS with and without Sz conservation gives the same XY Jastrow state, so the charges of the
    channels carried by the identity strings (including across the unit cell boundary) are consistent."""
    lat = _spin_lattice(2, 3, "infinite", "YC")
    psi = _random_imps(lat, seed=5)
    psi_no_charge = _drop_charges(psi)
    v, n_steps = 0.3, 2
    ApplyNNXYJastrow(psi, lat, v, n_steps, XY_TRUNC_PAR)
    ApplyNNXYJastrow(psi_no_charge, _spin_lattice(2, 3, "infinite", "YC", conserve=None), v, n_steps, XY_TRUNC_PAR)
    psi.test_sanity()
    assert np.allclose(_nn_correlations(psi), _nn_correlations(psi_no_charge), atol=1e-7)


def test_xy_infinite_inverse():
    """v followed by -v gives back the original iMPS (the Strang product is palindromic), up to the truncation
    errors: the chi of a generic iMPS grows without bound, so it saturates the cap (round trip error ~6e-5 at
    chi 48, ~3e-6 at chi 96)."""
    lat = _spin_lattice(2, 3, "infinite", "YC")
    psi = _random_imps(lat, seed=8)
    phi = psi.copy()
    ApplyNNXYJastrow(phi, lat, 0.3, 2, XY_TRUNC_PAR)
    change = np.max(np.abs(_nn_correlations(phi) - _nn_correlations(psi)))
    assert change > 1e-2
    ApplyNNXYJastrow(phi, lat, -0.3, 2, XY_TRUNC_PAR)
    assert np.max(np.abs(_nn_correlations(phi) - _nn_correlations(psi))) < 2e-3 * change

if __name__ == "__main__":
    test_xy_trotter_error_second_order()