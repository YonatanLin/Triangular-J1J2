"""Tests for the Kagome Dirac pi flux ansatz (KagomeMonopoleCondensatePiFluxModel), gauge of Fig. 1b of
He et al., PRX 7, 031020 (2017)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from tenpy.networks.site import FermionSite

from Gutzwiller import (KagomeMonopoleCondensatePiFluxModel, GetPiFluxKagomeLattice, KagomePiFluxBondSigns,
                        CreateHamiltonianMatrixFromCouplingsList, PlotModelHoppingsByPhase,
                        KagomeGutzwillerSpec, KagomeSpecialBzPoints,
                        CalculateExactCMatrixForMeanFieldModel, SymmetricSpinBlockSpectrum)
from Main import model_type_dirac, getTriangularLatticeSpecielBzPoints, KagomeGutzwillerCaseDirName
from Noninteracting import KagomePiFluxBlochEnergies, KagomePiFluxBandStructure

plot_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KagomePiFluxTestOutput")

KAGOME_NN = [(0, 1, (0, 0)), (0, 2, (0, 0)), (1, 2, (0, 0)), (1, 0, (1, 0)), (2, 0, (0, 1)), (2, 1, (-1, 1))]


def _model(Lx, Ly, spinfull=False, particle_hole=False, flux=0.0, bc=None, monopole_Q=0):
    site = FermionSite(conserve='N')
    lat = GetPiFluxKagomeLattice(site, Lx, Ly, spinfull, "finite", bc=bc)
    model = KagomeMonopoleCondensatePiFluxModel({"lattice": lat, "init_H_MPO": False, "monopole_Q": monopole_Q,
                                                 "flux": flux, "particle_hole": particle_hole})
    return model, lat


def _H(model, lat):
    return CreateHamiltonianMatrixFromCouplingsList(model, lat.N_sites, dtype=np.complex128)


def _loop_product(t, lat, loop):
    """ product of the hoppings t_ij = -H_ij around a closed loop of (x, y, sublattice), y periodic """
    Ly = lat.Ls[1]
    idx = [lat.lat2mps_idx([x, y % Ly, u]) for x, y, u in loop]
    prod = 1.0 + 0j
    for a, b in zip(idx, idx[1:] + idx[:1]):
        prod *= t[a, b]
    return prod


def _bloch_spectrum(Lx, Ly):
    """ bulk Bloch spectrum of the ansatz (with the chemical potential) on the Lx x Ly torus: the momenta with
    k.a_x = 2 pi n / Lx and k.a_y = 2 pi m / Ly, a_x = (2, 0), a_y = (1, sqrt(3)), in the magnetic Brillouin zone """
    n, m = np.meshgrid(np.arange(Lx), np.arange(Ly // 2), indexing="ij")
    kx = np.pi * n / Lx
    ky = (2 * np.pi * m / Ly - kx) / np.sqrt(3)
    return np.sort(KagomePiFluxBlochEnergies(kx, ky).ravel())


def test_fluxes_through_triangles_and_hexagons():
    Lx, Ly = 4, 4
    model, lat = _model(Lx, Ly)
    t = -_H(model, lat)
    for x in range(Lx):
        for y in range(Ly):
            up = [(x, y, 0), (x, y, 1), (x, y, 2)]
            np.testing.assert_allclose(_loop_product(t, lat, up), 1.0, atol=1e-14)
            if x + 1 < Lx:
                down = [(x, y, 1), (x + 1, y, 0), (x + 1, y - 1, 2)]
                hexagon = [(x, y, 1), (x + 1, y, 0), (x + 1, y, 2), (x, y + 1, 1), (x, y + 1, 0), (x, y, 2)]
                np.testing.assert_allclose(_loop_product(t, lat, down), 1.0, atol=1e-14)
                np.testing.assert_allclose(_loop_product(t, lat, hexagon), -1.0, atol=1e-14)


def test_gauge_of_fig_1b():
    """ red bonds (s = -1): left and right sides of the up triangles for even y, left side and base for odd y """
    y = np.arange(4)
    np.testing.assert_array_equal(KagomePiFluxBondSigns(0, 2, (0, 0), y), [-1, -1, -1, -1])
    np.testing.assert_array_equal(KagomePiFluxBondSigns(1, 2, (0, 0), y), [-1, 1, -1, 1])
    np.testing.assert_array_equal(KagomePiFluxBondSigns(0, 1, (0, 0), y), [1, -1, 1, -1])
    for u1, u2, dr in KAGOME_NN[3:]:
        np.testing.assert_array_equal(KagomePiFluxBondSigns(u1, u2, dr, y), np.ones(4))


def test_torus_spectrum_has_dirac_points_at_zero_energy():
    Lx, Ly = 4, 4  # k = (+-pi/2, pi/2) allowed, so the Dirac points are sampled
    model, lat = _model(Lx, Ly, bc=['periodic', 'periodic'])
    H = _H(model, lat)
    e = np.linalg.eigvalsh(H)
    np.testing.assert_allclose(e, _bloch_spectrum(Lx, Ly), atol=1e-12)
    n_half = lat.N_sites // 2
    # the chemical potential puts the half filling Fermi level (the Dirac points) at zero energy
    np.testing.assert_allclose(e[n_half - 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(e[n_half], 0.0, atol=1e-12)


def test_bloch_bands_dirac_points():
    """ Dirac points at k.a_y = k.a_x = +-pi/2 (Q of He et al., a_1 = a_y, a_2 = a_x), at zero energy """
    Q = np.array([np.pi / 4, np.pi / (4 * np.sqrt(3))])
    for k in [Q, -Q]:
        e = KagomePiFluxBlochEnergies(k[0], k[1])
        np.testing.assert_allclose(e[2:4], 0.0, atol=1e-12)
    # elsewhere the bands 3 and 4 are gapped and the occupied bands are below zero
    rng = np.random.default_rng(0)
    k = rng.uniform(-np.pi, np.pi, size=(2, 2000))
    e = KagomePiFluxBlochEnergies(k[0], k[1])
    assert np.all(e[..., 2] <= 1e-12) and np.all(e[..., 3] >= -1e-12)


def test_cylinder_band_structure_approaches_2D():
    """ the energy per occupied mode on a wide cylinder approaches the 2D average over the magnetic Brillouin zone """
    n = np.arange(200) / 200
    n1, n2 = np.meshgrid(n, n)
    b1 = 2 * np.pi * np.array([0.5, -1.0 / (2 * np.sqrt(3))])
    b2 = 2 * np.pi * np.array([0.0, 1.0 / (2 * np.sqrt(3))])
    E_2D = np.mean(KagomePiFluxBlochEnergies(n1 * b1[0] + n2 * b2[0], n1 * b1[1] + n2 * b2[1])[..., 0:3])
    E_cylinder = KagomePiFluxBandStructure(Ly=40, plot=False)
    assert abs(E_cylinder - E_2D) < 2e-3, (E_cylinder, E_2D)


def test_spinfull_particle_hole_blocks():
    model, lat = _model(3, 4, spinfull=True, particle_hole=True, flux=0.5)
    H = _H(model, lat)
    H_up, H_down = H[0::2, 0::2], H[1::2, 1::2]
    np.testing.assert_allclose(H_down, -np.conj(H_up), atol=1e-14)
    np.testing.assert_allclose(H[0::2, 1::2], 0.0, atol=1e-14)

    model_spinless, lat_spinless = _model(3, 4, flux=0.5)
    np.testing.assert_allclose(H_up, _H(model_spinless, lat_spinless), atol=1e-14)


def test_external_flux_only_changes_seam_bonds():
    Lx, Ly = 3, 4
    model_0, lat = _model(Lx, Ly)
    model_pi, _ = _model(Lx, Ly, flux=1.0)
    H_0, H_pi = _H(model_0, lat), _H(model_pi, lat)
    for i in range(lat.N_sites):
        for j in range(lat.N_sites):
            if abs(H_0[i, j]) < 1e-14:
                continue
            y_i, y_j = lat.mps2lat_idx(i)[1], lat.mps2lat_idx(j)[1]
            seam = abs(y_i - y_j) > 1
            np.testing.assert_allclose(H_pi[i, j], -H_0[i, j] if seam else H_0[i, j], atol=1e-14)


@pytest.mark.parametrize("Lx, Ly, flux", [(4, 4, 0.0), (4, 4, 1.0), (5, 4, 0.0), (4, 6, 0.0)])
def test_open_cylinder_correlation_matrix_is_half_filled_singlet(Lx, Ly, flux):
    """ the open cylinder spectrum isn't symmetric about 0 (in gap edge states), so each spin block is half filled """
    model, lat = _model(Lx, Ly, spinfull=True, particle_hole=True, flux=flux)
    assert not SymmetricSpinBlockSpectrum(_H(model, lat), 1e-9)
    C, _ = CalculateExactCMatrixForMeanFieldModel(model, model_type_dirac, 0, True)
    n_sites = lat.N_sites // 2
    C_up, C_hole = C[0::2, 0::2], C[1::2, 1::2]
    np.testing.assert_allclose(np.trace(C_up).real, n_sites // 2, atol=1e-10)
    np.testing.assert_allclose(np.trace(C_hole).real, n_sites // 2, atol=1e-10)
    np.testing.assert_allclose(C[0::2, 1::2], 0.0, atol=1e-14)
    # the down spinons occupy the same orbitals as the up spinons (a singlet): h = f_down^dag gives
    # <h_j^dag h_i> = delta_ij - <f_down,j^dag f_down,i>^*
    np.testing.assert_allclose(C_hole, np.eye(n_sites) - np.conj(C_up), atol=1e-10)
    np.testing.assert_allclose(C @ C, C, atol=1e-10)


def test_kagome_spec():
    spec = KagomeGutzwillerSpec()
    site = FermionSite(conserve='N')
    fermion_lat = spec.fermion_lattice(site, 3, 4, True, "infinite")
    assert fermion_lat.boundary_conditions == ['periodic', 'periodic']
    from tenpy.networks.site import SpinHalfSite
    spin_lat = spec.spin_lattice(SpinHalfSite(conserve='Sz'), 3, 4, "finite", ['open', 'periodic'])
    # grouping the (up, down) pairs of the fermion lattice gives the spin lattice ordering
    fermion_lat_finite = spec.fermion_lattice(site, 3, 4, True, "finite")
    for i in range(spin_lat.N_sites):
        x, y, u = spin_lat.mps2lat_idx(i)
        assert tuple(fermion_lat_finite.mps2lat_idx(2 * i)) == (x, y, 2 * u)
        assert tuple(fermion_lat_finite.mps2lat_idx(2 * i + 1)) == (x, y, 2 * u + 1)
        np.testing.assert_allclose(fermion_lat_finite.position([x, y, 2 * u]) + [0.08, 0.0],
                                   spin_lat.position([x, y, u]), atol=1e-14)
    spec.check_dimensions(5, 4)
    with pytest.raises(AssertionError):
        spec.check_dimensions(4, 3)

    points = KagomeSpecialBzPoints()
    for key, k in getTriangularLatticeSpecielBzPoints().items():
        np.testing.assert_allclose(points[key], k / 2)
        np.testing.assert_allclose(points[key + "_ext"], k)
    # M points of the Kagome Brillouin zone are half reciprocal lattice vectors: exp(i M.a) = +-1
    for a in spin_lat.basis:
        assert abs(abs(np.cos(np.dot(points["M1"], a))) - 1) < 1e-12

    assert (KagomeGutzwillerCaseDirName("r/", 4, 6, 100, 0.5, "finite", 0, 0.0, 0, jastrow_nn_zz=0.1)
            == "r/Kagome_finite_Lx_4_Ly_6_chi_100_flux_0.5_gsindex_0_monQ_0_jasNNzz_0.1/")


@pytest.mark.parametrize("geometry", ["Kagome", "YC", "XC"])
def test_structure_factor_period_lattice(geometry):
    """ S(k) is periodic under the basis used to tile its plot, so the tiled copies of a momentum agree """
    from tenpy.models.lattice import Kagome
    from tenpy.networks.site import SpinHalfSite
    from Main import BuildTriangularLattice
    from WaveFunctionProperties import (_structure_factor_reciprocal_basis, compute_structure_factor_grid,
                                        _tile_kpoints)
    spin_site = SpinHalfSite(conserve='Sz')
    if geometry == "Kagome":
        lat = Kagome(4, 4, spin_site, bc=['open', 'periodic'], bc_MPS='finite')
    else:
        lat = BuildTriangularLattice(4, 4, spin_site, "finite", bc=['open', 'periodic'], geometry=geometry)
    rng = np.random.default_rng(1)
    C = rng.normal(size=(lat.N_sites, lat.N_sites))
    C = C + C.T
    G1, G2 = _structure_factor_reciprocal_basis(lat)
    k = rng.normal(size=(5, 2))
    S = lambda ks: compute_structure_factor_grid(C, lat, ks=ks)[1]
    np.testing.assert_allclose(S(k + G1), S(k), atol=1e-10)
    np.testing.assert_allclose(S(k + G2), S(k), atol=1e-10)

    ks, Sk = compute_structure_factor_grid(C, lat, n1=2 * lat.Ls[0], n2=2 * lat.Ls[1])
    k_tiled, S_tiled, _ = _tile_kpoints(ks, Sk, lat, 1)
    _, groups = np.unique(np.round(k_tiled, 8), axis=0, return_inverse=True)
    groups = groups.ravel()
    for g in np.unique(groups):
        assert np.ptp(S_tiled[groups == g]) < 1e-10
    if geometry == "Kagome":
        # the extended Brillouin zone: the period lattice is spanned by twice the Bravais reciprocal vectors
        b = 2 * np.asarray(lat.reciprocal_basis)
        coefficients = np.linalg.solve(b.T, np.array([G1, G2]).T)
        np.testing.assert_allclose(coefficients, np.round(coefficients), atol=1e-12)
        assert abs(round(abs(np.linalg.det(coefficients)))) == 1


def test_monopole_not_implemented():
    with pytest.raises(NotImplementedError):
        _model(3, 4, monopole_Q=1)


def test_plot_hoppings_small_lattice():
    """ saves the ansatz on a small cylinder to KagomePiFluxTestOutput/kagome_pi_flux_hoppings.png """
    model, lat = _model(3, 4)
    fig, ax = plt.subplots(figsize=(8, 7))
    # seam bonds (around the cylinder) are longer than 1 and skipped
    PlotModelHoppingsByPhase(model, ax, max_bond_length=1.5, plot_order=True)
    for x in range(lat.Ls[0] - 1):
        for y in range(lat.Ls[1] - 1):
            center = lat.position([x, y, 0]) + np.array([1.5, 0.5 * np.sqrt(3)])
            ax.text(*center, r"$\pi$", ha="center", va="center", fontsize=11)
    ax.set_title("Kagome Dirac $\\pi$ flux, $H_{ij} = -s_{ij}$\n"
                 "phase 0: $s=-1$ (red in Fig. 1b)\nphase $\\pi$: $s=1$ (black)")
    os.makedirs(plot_output_dir, exist_ok=True)
    path = os.path.join(plot_output_dir, "kagome_pi_flux_hoppings.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    assert os.path.isfile(path)

if __name__ == "__main__":
    test_plot_hoppings_small_lattice()
