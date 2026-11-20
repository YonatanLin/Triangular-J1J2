"""Checks of the shifted YC cylinder (circumference C = Ly a_y + shift a_x, tenpy's bc_shift) in Noninteracting.py.

- shift = 0 reproduces the energies per mode of Noninteracting.py before the shift option was added.
- The momentum quantization of TriangularPiFluxCutQuantization is exact: on an Lx x Ly torus glued with the shift, the
  Bloch energies at the allowed momenta are the spectrum of the real space hopping matrix, for even and odd shifts.
- shift = -1 agrees with the helical cylinder of Gutzwiller.py (helical_bc_shift, TriangularCylinderBC): the same
  couplings (up to the gauge HelicalSeamGaugeSigns) and the same spectrum.
"""
import numpy as np
import pytest
from numpy import pi, sqrt

from Noninteracting import (PiFluxBandStructure, KagomePiFluxBandStructure, PiFluxSquaredEnergy,
                            KagomePiFluxBlochEnergies, ShiftedCylinderCircumference, TriangularPiFluxCutQuantization,
                            YCName)

TRI_A_X, TRI_A_Y = np.array([1.0, 0.0]), np.array([0.5, sqrt(3) / 2])
KAG_A_X, KAG_A_Y = np.array([2.0, 0.0]), np.array([1.0, sqrt(3)])


# --- shift = 0 reproduces the previous results (values of Noninteracting.py before the change)
BASELINE = [
    ("YC", 4, 0.0, -2.3136991937797373), ("YC", 4, 1.0, -2.4135609864892054), ("YC", 4, 0.5, -2.375924624208477),
    ("YC", 5, 0.0, -2.37257539927262), ("YC", 5, 1.0, -2.3730653640483697), ("YC", 5, 0.5, -2.3673190973060314),
    ("YC", 6, 0.0, -2.381794180062701), ("YC", 6, 1.0, -2.3542413446607897), ("YC", 6, 0.5, -2.3724257135397657),
    ("XC", 6, 0.0, -2.3735778179684073), ("XC", 6, 1.0, -2.3735778179684077),
    ("XC", 8, 0.0, -2.3614840914014965), ("XC", 8, 1.0, -2.378843152831557),
]


@pytest.mark.parametrize("geometry, Ly, tet, E", BASELINE)
def test_no_shift_reproduces_previous_results(geometry, Ly, tet, E):
    assert np.isclose(PiFluxBandStructure(Ly=Ly, plot=False, tet=tet, geometry=geometry), E, rtol=0, atol=1e-13)
    if geometry == "YC":
        assert np.isclose(PiFluxBandStructure(Ly=Ly, plot=False, tet=tet, geometry=geometry, shift=0), E,
                          rtol=0, atol=1e-13)


@pytest.mark.parametrize("tet, E", [(0.0, -2.4980640788354256), (1.0, -2.5070909979059057)])
def test_kagome_no_shift_reproduces_previous_results(tet, E):
    assert np.isclose(KagomePiFluxBandStructure(Ly=4, plot=False, tet=tet, shift=0), E, rtol=0, atol=1e-13)


def test_names():
    assert YCName(4) == "YC4" and YCName(4, -1) == "YC4-1" and YCName(6, 2) == "YC6+2"


# --- exact spectra on a torus glued with a shift
def _torus_hopping_matrix(Lx, Ly, shift, tet, bonds):
    """
        Real space hopping matrix on the Lx x Ly torus, site (x, y) -> x Ly + y, glued as tenpy does with bc_shift:
        (x, y + Ly) is (x - shift, y), x periodic. The hopping c^dag_i c_j to a site j across the y boundary gets the
        phase e^{-i pi tet}, as tenpy's coupling_strength_add_ext_flux with the phase pi tet (Gutzwiller.py). The
        spectrum doesn't depend on the sign of tet (time reversal, E(k) = E(-k)).
        bonds: list of (dx, dy, sign(x, y)), the strength of the bond indexed by its first site.
    """
    H = np.zeros((Lx * Ly, Lx * Ly), dtype=np.complex128)
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy, sign in bonds:
                x2, y2 = x + dx, y + dy
                winding = y2 // Ly
                x2, y2 = (x2 - winding * shift) % Lx, y2 - winding * Ly
                t = sign(x, y) * np.exp(-1j * pi * tet * winding)
                H[x * Ly + y, x2 * Ly + y2] += t
                H[x2 * Ly + y2, x * Ly + y] += np.conj(t)
    return H


def _triangular_pi_flux_bonds():
    """ the signs of Gutzwiller.MonopoleCondensatePiFluxModel (YC, no monopole flux) for the bond starting at (x, y):
    x bonds 1, y bonds s_y(x) = (-1)^(x + 1), diagonal bonds (-1, 1) s_y(x - 1) (tenpy indexes the strength array of a
    bond with dr_x = -1 by x - 1, Lattice.coupling_shape). Pi flux per unit cell, the magnetic unit cell (2 a_x, a_y) """
    s_y = lambda x, y: 1 - 2 * ((x + 1) % 2)
    return [(1, 0, lambda x, y: 1.0), (0, 1, s_y), (-1, 1, lambda x, y: s_y(x - 1, y))]


def _torus_momenta(C, Lx, a_x, A1, A2, cut_spacing, cut_offset, tet):
    """ the distinct momenta (modulo the magnetic reciprocal lattice of A1, A2) with k.C = cut_offset + pi tet +
    cut_spacing m and k.(Lx a_x) = 2 pi j """
    ks, keys = [], set()
    T = Lx * a_x
    for j in range(-4 * Lx, 4 * Lx):
        for m in range(-200, 200):
            k = np.linalg.solve(np.array([C, T]), np.array([cut_offset + pi * tet + cut_spacing * m, 2 * pi * j]))
            f = np.round((np.array([k @ A1, k @ A2]) / (2 * pi)) % 1.0, 8) % 1.0
            key = tuple(f)
            if key not in keys:
                keys.add(key)
                ks.append(k)
    return np.array(ks)


@pytest.mark.parametrize("Ly", [3, 4, 5, 6])
@pytest.mark.parametrize("shift", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize("tet", [0.0, 0.5, 1.0])
def test_triangular_torus_spectrum_matches_quantization(Ly, shift, tet):
    Lx = 6 # even, a multiple of the magnetic unit cell 2 a_x
    E_real_space = np.linalg.eigvalsh(_torus_hopping_matrix(Lx, Ly, shift, tet, _triangular_pi_flux_bonds()))

    C = ShiftedCylinderCircumference(TRI_A_X, TRI_A_Y, Ly, shift)
    cut_spacing, cut_offset = TriangularPiFluxCutQuantization(Ly, shift)
    ks = _torus_momenta(C, Lx, TRI_A_X, 2 * TRI_A_X, TRI_A_Y, cut_spacing, cut_offset, tet)
    E_k = sqrt(PiFluxSquaredEnergy(ks[:, 0], ks[:, 1]))
    E_bloch = np.sort(np.concatenate([E_k, -E_k]))
    # odd shift: k and k + Q are both on the cuts and carry one cylinder state per band together
    multiplicity = 1 if shift % 2 == 0 else 2
    assert len(E_bloch) == multiplicity * Lx * Ly
    assert np.allclose(E_bloch, np.sort(np.repeat(E_real_space, multiplicity)), atol=1e-10)


def test_odd_shift_needs_the_modified_quantization():
    """ the ordinary quantization k.C = pi tet + 2 pi m fails for an odd shift (sanity check of the test) """
    Lx, Ly, shift, tet = 6, 4, -1, 0.0
    E_real_space = np.linalg.eigvalsh(_torus_hopping_matrix(Lx, Ly, shift, tet, _triangular_pi_flux_bonds()))
    C = ShiftedCylinderCircumference(TRI_A_X, TRI_A_Y, Ly, shift)
    ks = _torus_momenta(C, Lx, TRI_A_X, 2 * TRI_A_X, TRI_A_Y, 2 * pi, 0.0, tet)
    E_k = sqrt(PiFluxSquaredEnergy(ks[:, 0], ks[:, 1]))
    E_bloch = np.sort(np.concatenate([E_k, -E_k]))
    assert len(E_bloch) != len(E_real_space) or not np.allclose(E_bloch, np.sort(E_real_space), atol=1e-6)


@pytest.mark.parametrize("Ly", [4, 5])
@pytest.mark.parametrize("shift", [-1, 1, 3])
def test_odd_shift_flux_0_and_pi_are_degenerate(Ly, shift):
    """ for an odd shift the one site translation along y maps the flux tet to tet + 1 (Gutzwiller.helical_bc_shift):
    the cuts k.C = pi tet - pi Ly / 2 + pi m are the same for tet and tet + 1 """
    E0 = PiFluxBandStructure(Ly=Ly, plot=False, tet=0.0, shift=shift)
    E1 = PiFluxBandStructure(Ly=Ly, plot=False, tet=1.0, shift=shift)
    assert np.isclose(E0, E1, rtol=0, atol=1e-12)


@pytest.mark.parametrize("Ly", [2, 4])
@pytest.mark.parametrize("shift", [-2, -1, 0, 1])
@pytest.mark.parametrize("tet", [0.0, 1.0])
def test_kagome_torus_spectrum_matches_quantization(Ly, shift, tet):
    from Gutzwiller import KagomePiFluxBondSigns, kagome_pi_flux_dirac_energy
    Lx = 3
    kagome_NN = [(0, 1, (0, 0)), (0, 2, (0, 0)), (1, 2, (0, 0)), (1, 0, (1, 0)), (2, 0, (0, 1)), (2, 1, (-1, 1))]
    N = Lx * Ly * 3
    H = np.zeros((N, N), dtype=np.complex128)
    idx = lambda x, y, u: (x * Ly + y) * 3 + u
    for x in range(Lx):
        for y in range(Ly):
            for u1, u2, dr in kagome_NN:
                x2, y2 = x + dr[0], y + dr[1]
                winding = y2 // Ly
                x2, y2 = (x2 - winding * shift) % Lx, y2 - winding * Ly
                t = -KagomePiFluxBondSigns(u1, u2, dr, np.array([y]))[0] * np.exp(-1j * pi * tet * winding)
                H[idx(x, y, u1), idx(x2, y2, u2)] += t
                H[idx(x2, y2, u2), idx(x, y, u1)] += np.conj(t)
    E_real_space = np.sort(np.linalg.eigvalsh(H)) - kagome_pi_flux_dirac_energy

    C = ShiftedCylinderCircumference(KAG_A_X, KAG_A_Y, Ly, shift)
    ks = _torus_momenta(C, Lx, KAG_A_X, KAG_A_X, 2 * KAG_A_Y, 2 * pi, 0.0, tet)
    E_bloch = np.sort(KagomePiFluxBlochEnergies(ks[:, 0], ks[:, 1]).ravel())
    assert len(E_bloch) == N
    assert np.allclose(E_bloch, E_real_space, atol=1e-10)


# --- shift = -1 is the helical cylinder of Gutzwiller.py
def _gutzwiller_helical_torus(Lx, Ly, tet):
    from tenpy.networks.site import FermionSite
    from Gutzwiller import MonopoleCondensatePiFluxModel, TriangularCylinderBC, helical_bc_shift, \
        CreateHamiltonianMatrixFromCouplingsList
    from Main import BuildTriangularLattice
    assert helical_bc_shift == -1
    site = FermionSite(conserve='N')
    # the bc of the helical iMPS, periodic along the cylinder: here a finite torus
    bc = TriangularCylinderBC("infinite", "YC", helical=True)
    lat = BuildTriangularLattice(Lx, Ly, site, "finite", bc=bc, geometry="YC")
    model = MonopoleCondensatePiFluxModel({"lattice": lat, "init_H_MPO": False, "monopole_Q": 0, "flux": tet,
                                           "particle_hole": False})
    H = CreateHamiltonianMatrixFromCouplingsList(model, lat.N_sites, dtype=np.complex128)
    order = [lat.lat2mps_idx([x, y, 0]) for x in range(Lx) for y in range(Ly)]
    return H[np.ix_(order, order)]


@pytest.mark.parametrize("Ly", [3, 4, 5, 6])
@pytest.mark.parametrize("tet", [0.0, 0.5, 1.0])
def test_helical_gutzwiller_couplings_and_spectrum(Ly, tet):
    Lx = 6
    H_gutzwiller = _gutzwiller_helical_torus(Lx, Ly, tet)
    H_shift = _torus_hopping_matrix(Lx, Ly, -1, tet, _triangular_pi_flux_bonds())
    # the same couplings, up to the gauge transformation of Gutzwiller.HelicalSeamGaugeSigns: (-1)^x on the last row
    phi = np.array([(1 - 2 * (x % 2)) if y == Ly - 1 else 1 for x in range(Lx) for y in range(Ly)])
    assert np.allclose(H_gutzwiller, phi[:, None] * H_shift * phi[None, :], atol=1e-12)
    # and the same spectrum, which is that of the band structure at the momenta allowed with shift = -1
    C = ShiftedCylinderCircumference(TRI_A_X, TRI_A_Y, Ly, -1)
    cut_spacing, cut_offset = TriangularPiFluxCutQuantization(Ly, -1)
    ks = _torus_momenta(C, Lx, TRI_A_X, 2 * TRI_A_X, TRI_A_Y, cut_spacing, cut_offset, tet)
    E_k = sqrt(PiFluxSquaredEnergy(ks[:, 0], ks[:, 1]))
    E_bloch = np.sort(np.concatenate([E_k, -E_k]))
    assert np.allclose(E_bloch, np.sort(np.repeat(np.linalg.eigvalsh(H_gutzwiller), 2)), atol=1e-10)


@pytest.mark.parametrize("Ly", [4, 5])
@pytest.mark.parametrize("tet", [0.0, 1.0])
def test_helical_cylinder_energy_per_mode(Ly, tet):
    """ the energy per mode of the infinite helical cylinder (PiFluxBandStructure, shift=-1) is the limit of long
    helical tori of Gutzwiller.py """
    Lx = 200
    E_gutzwiller = np.linalg.eigvalsh(_gutzwiller_helical_torus(Lx, Ly, tet))
    E_per_mode_torus = np.mean(E_gutzwiller[E_gutzwiller < 0])
    E_per_mode = PiFluxBandStructure(Ly=Ly, plot=False, tet=tet, geometry="YC", shift=-1)
    assert abs(E_per_mode - E_per_mode_torus) < 2e-3


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for Ly, tet in [(4, 0.0), (4, 1.0), (5, 0.0), (5, 1.0), (6, 0.0), (6, 1.0)]:
        print(f"Ly={Ly}, tet={tet}: E per mode YC{Ly}: {PiFluxBandStructure(Ly, plot=False, tet=tet)}, "
              f"YC{Ly}-1: {PiFluxBandStructure(Ly, plot=False, tet=tet, shift=-1)}")
