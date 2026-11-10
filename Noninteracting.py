import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, sqrt
from scipy.spatial import Voronoi

from TryingTemfpy import rc_params
plt.rcParams.update(rc_params)

def rotate_vector(vec, tet):
    c, s = np.cos(tet), np.sin(tet)
    rotation_matrix = np.array([[c, -s], [s, c]])

    # Perform the rotation
    return np.dot(rotation_matrix, vec)

def plot_bz1(ax, b1, b2):
    """
    Plots the 1st Brillouin Zone given two reciprocal lattice vectors.
    """
    # 1. Generate reciprocal lattice points (3x3 grid around origin is enough)
    points = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            points.append(i * b1 + j * b2)
    points = np.array(points)

    vor = Voronoi(points)

    center_idx = 4
    region_idx = vor.point_region[center_idx]
    region = vor.regions[region_idx]

    if -1 not in region:  # Ensure it's a closed cell
        verts = vor.vertices[region]
        # Reorder vertices for a clean polygon plot
        centroid = np.mean(verts, axis=0)
        angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
        verts = verts[np.argsort(angles)]

        # Close the loop
        verts = np.vstack([verts, verts[0]])

        # 5. Plot on the provided axis
        ax.plot(verts[:, 0], verts[:, 1], 'k--', linewidth=1.5, label='BZ1')


def PiFluxSquaredEnergy(kx, ky):
    a1 = np.array([1. / 2., sqrt(3) / 2.0])
    a2 = np.array([3. / 2., -sqrt(3) / 2.0])
    k1 = kx * a1[0] + ky * a1[1]
    k2 = kx * a2[0] + ky * a2[1]
    k3 = k1 + k2
    E_sq = 4 * cos(k1) ** 2 + (1 + cos(k3) + cos(k2) - cos(k1)) ** 2 + (
            sin(k3) + sin(k2) - sin(k1)) ** 2
    return E_sq

def InFirstBZ(k, b1, b2, tol=1e-6, half_open=False):
    """
        whether k is in the first Brillouin zone (Wigner-Seitz cell) of the reciprocal lattice spanned by b1, b2.
        half_open: of every pair of opposite zone edges (k.G = |G|^2 / 2 and its image k.G = -|G|^2 / 2) keep only one,
        so momenta on the zone boundary, which are equivalent to their image on the opposite edge, are counted once.
    """
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if (i, j) == (0, 0):
                continue
            G = i * b1 + j * b2
            G_norm = sqrt(np.dot(G, G))
            open_edge = half_open and (i > 0 or (i == 0 and j > 0))
            if np.dot(k, G) / G_norm > G_norm / 2 + (-tol if open_edge else tol):
                return False
    return True


def YCName(Ly, shift=0):
    """ YC{Ly}, and YC{Ly}{shift:+d} for the shifted YC cylinder (YC{Ly}-1 for the helical one, as in
    Main.GutzwillerCaseDirName) """
    return f"YC{Ly}" if shift == 0 else f"YC{Ly}{shift:+d}"


def ShiftedCylinderCircumference(a_x, a_y, Ly, shift=0):
    """
        Circumference vector C = Ly a_y + shift a_x of a YC cylinder with Ly sites around it, glued with an integer shift
        along the cylinder: (x, y + Ly) is identified with (x - shift, y), tenpy's bc_shift convention
        (JiangAnalysis._WrapAroundCylinder). shift = Gutzwiller.helical_bc_shift = -1 is the helical cylinder YC{Ly}-1,
        shift = 0 the ordinary YC cylinder.
    """
    return Ly * np.asarray(a_y, dtype=float) + shift * np.asarray(a_x, dtype=float)


def TriangularPiFluxCutQuantization(Ly, shift=0):
    """
        (cut_spacing, cut_offset) of the momenta allowed on the triangular YC cylinder of the pi flux ansatz with
        circumference C = Ly a_y + shift a_x (ShiftedCylinderCircumference): k.C = cut_offset + pi tet + cut_spacing m.

        The gauge of Gutzwiller.MonopoleCondensatePiFluxModel has the magnetic unit cell (2 a_x, a_y) (the y and
        diagonal bond signs alternate with x). For an even shift C is a magnetic lattice vector, the cylinder is the
        plane modulo C and psi(r + C) = e^{i pi tet} psi(r): k.C = pi tet + 2 pi m, the usual quantization (shift = 0:
        k.(Ly a_y) = pi tet + 2 pi m).
        For an odd shift C is not a magnetic lattice vector: unfolding the cylinder (whose seam bonds carry the signs of
        their first site) into the plane gives the planar gauge only after the gauge transformation G(y) with
        G(y) G(y + 1) = (-1)^floor(y / Ly), and the boundary condition becomes psi(r + C) = e^{i pi tet} (-1)^y psi(r).
        The translation by a_x is a symmetry only with the gauge transformation (-1)^y = e^{i Q.r}, Q = (0, 2 pi/sqrt 3)
        (Q.a_x = 0, Q.a_y = pi), so this condition connects the Bloch states at k and k + Q (which have the same
        energies). Applying it twice gives e^{i (2 k + Q).C} = e^{2 i pi tet}, i.e.
            k.C = pi tet - pi Ly / 2 + pi m,
        cuts twice as dense as for an even shift, on which k and k + Q (Q.C = pi Ly) are both allowed and together carry
        one cylinder state per band (the condition picks one of the two combinations). The momenta on the cuts thus
        count every cylinder state twice, which doesn't change averages such as the energy per mode.
        Physically the cylinder translation by one site along y maps the flux tet to tet + 1 for an odd shift (see
        Gutzwiller.helical_bc_shift), which the offset -pi Ly / 2 and the doubled cuts encode.
    """
    if shift % 2 == 0:
        return 2 * pi, 0.0
    return pi, -pi * Ly / 2


def CylinderCutMomenta(C, tet, b1, b2, k_max=2 * pi, n_k=500, cut_spacing=2 * pi, cut_offset=0.0):
    """
        Momenta in the first Brillouin zone of b1, b2 allowed on a cylinder of circumference vector C with flux tet
        (in units of pi), k.C = cut_offset + pi tet + cut_spacing m, sampled on a grid of n_k kx values in
        [-k_max, k_max] (the defaults are the ordinary cylinder, k.C = 2 pi m + pi tet; for a shifted cylinder see
        TriangularPiFluxCutQuantization). All cuts crossing the zone are generated and the zone is half open, so a cut
        lying on a zone edge is counted once, independently of which m labels it.
        Returns an array (n_momenta, 2).
    """
    C = np.asarray(C, dtype=float)
    # |k| <= k_zone for k in the zone, so the cuts crossing it have |k.C| <= k_zone |C|
    k_zone = max(sqrt(np.dot(G, G)) for G in (b1, b2, b1 + b2, b1 - b2))
    m_max = int(np.ceil((k_zone * sqrt(np.dot(C, C)) + abs(cut_offset) + pi * abs(tet)) / cut_spacing)) + 1
    ks = []
    for kx in np.linspace(-k_max, k_max, n_k):
        for m in range(-m_max, m_max + 1):
            ky = (cut_offset + pi * tet + cut_spacing * m - kx * C[0]) / C[1]
            k_vec = np.array([kx, ky])
            if InFirstBZ(k_vec, b1, b2, half_open=True):
                ks.append(k_vec)
    return np.array(ks)


def CylinderBandStructure(occupied_energies, C, tet, b1, b2, title, file_name, plot=True, k_max=2 * pi,
                          cut_spacing=2 * pi, cut_offset=0.0):
    """
        Mean field band structure on an infinite cylinder of circumference vector C with flux tet (in units of pi),
        momenta k.C = cut_offset + pi tet + cut_spacing m (CylinderCutMomenta).
        occupied_energies(kx, ky): energies of the occupied bands (relative to the Fermi level), with shape
        kx.shape + (n_occupied_bands,). b1, b2: reciprocal vectors of the (magnetic) unit cell.
        Plots the highest occupied band in the k plane (zero at the Dirac points), the momenta allowed on the cylinder
        and the Brillouin zone. Returns the energy per occupied mode.
    """
    ks = CylinderCutMomenta(C, tet, b1, b2, k_max=k_max, cut_spacing=cut_spacing, cut_offset=cut_offset)
    E_per_mode = np.mean(occupied_energies(ks[:, 0], ks[:, 1]))

    if plot:
        print(f"energy per mode: {E_per_mode}")
        fig, ax = plt.subplots()
        ax.plot(ks[:, 0] / pi, ks[:, 1] / pi, "ko", markersize=1)
        plot_bz1(ax, b1 / pi, b2 / pi)
        Kx, Ky = np.meshgrid(np.linspace(-k_max, k_max, 1000), np.linspace(-k_max, k_max, 1000))
        im = ax.imshow(np.max(occupied_energies(Kx, Ky), axis=-1), origin="lower",
                       extent=(-k_max / pi, k_max / pi, -k_max / pi, k_max / pi), cmap='RdBu')
        ax.set_xlabel(r"$k_x[\pi]$")
        ax.set_ylabel(r"$k_y[\pi]$")
        if abs(tet) < 1e-15:
            ax.set_title(fr"{title}, $\theta$=0")
        else:
            ax.set_title(fr"{title}, $\theta$={tet}$\pi$")
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        ax.legend()
        fig.savefig(file_name, bbox_inches='tight')
    return E_per_mode


def PiFluxBandStructure(Ly=4, plot=True, tet=0., geometry="YC", shift=0):
    """
        Lower band of the pi-flux model on an infinite cylinder, returns the energy per occupied mode (one per magnetic
        unit cell). tet is the flux through the cylinder in units of pi (tet=1.0 is flux=1.0 in Gutzwiller.py).
        Ly is the number of sites around the cylinder: for XC this is the number of rows, i.e. twice the Ly of
        Gutzwiller.py / TriangularXC.
        shift: YC only, the shifted YC cylinder of circumference C = Ly a_y + shift a_x (ShiftedCylinderCircumference,
        tenpy's bc_shift), shift = -1 is the helical cylinder YC{Ly}-1 of Gutzwiller.py (helical=True). The allowed
        momenta are k.C = pi tet + 2 pi m for an even shift and k.C = pi tet - pi Ly / 2 + pi m for an odd shift (see
        TriangularPiFluxCutQuantization).
    """
    assert shift == round(shift), "the shift must be an integer"
    shift = int(shift)
    assert shift == 0 or geometry == "YC", "the shifted cylinder is implemented for YC only"
    if geometry == "XC":
        assert Ly % 2 == 0, "XC cylinder needs an even number of rows"
    cut_spacing, cut_offset = 2 * pi, 0.0
    if geometry == "YC":
        k1_bz = 2 * pi * np.array([-0.5, -sqrt(3) / 2])
        k2_bz = 2 * pi * np.array([0.5, -1.0 / (2 * sqrt(3))])
        a_x, a_y = np.array([1.0, 0.0]), np.array([0.5, sqrt(3) / 2]) # Main.BuildTriangularLattice
        C = ShiftedCylinderCircumference(a_x, a_y, Ly, shift)
        cut_spacing, cut_offset = TriangularPiFluxCutQuantization(Ly, shift)
        name = YCName(Ly, shift)
    else:
        k1_bz = 2 * pi * np.array([1.0, 0.0])
        k2_bz = 2 * pi * np.array([0.0, 1./sqrt(3)])
        C = Ly * np.array([0.0, sqrt(3) / 2]) # half the XC circumference vector per row
        name = f"{geometry}{Ly}"

    E_per_mode = CylinderBandStructure(lambda kx, ky: (-1) * sqrt(PiFluxSquaredEnergy(kx, ky))[..., None], C, tet,
                                       k1_bz, k2_bz, name,
                                       f"noninteracting_band_structure_{name if shift != 0 else geometry}_Ly_{Ly}"
                                       f"_flux_{tet}.pdf", plot=plot, cut_spacing=cut_spacing, cut_offset=cut_offset)

    if plot:
        Kx, Ky = np.meshgrid(np.linspace(-2 * pi, 2 * pi, 1000), np.linspace(-2 * pi, 2 * pi, 1000))
        E_sq = PiFluxSquaredEnergy(Kx, Ky)
        E_sq_theory = 2 * (3 + cos(2*Kx) - cos(Kx - sqrt(3)*Ky) + cos(Kx + sqrt(3)*Ky))
        print("diff from theory: ", np.max(np.abs(E_sq - E_sq_theory)))
        print("E_sq minimum: ", np.min(E_sq))
    return E_per_mode


def KagomePiFluxBlochEnergies(kx, ky):
    """
        Bands of the Kagome Dirac pi flux ansatz (KagomeMonopoleCondensatePiFluxModel, including the chemical potential
        that puts the Dirac points at zero energy), with shape kx.shape + (6,), sorted. Magnetic unit cell
        (a_x, 2 a_y), a_x = (2, 0), a_y = (1, sqrt(3)), orbitals 3 * (y mod 2) + sublattice.
    """
    from Gutzwiller import KagomePiFluxBondSigns, kagome_pi_flux_dirac_energy

    a_x, a_y = np.array([2.0, 0.0]), np.array([1.0, sqrt(3)])
    kagome_NN = [(0, 1, (0, 0)), (0, 2, (0, 0)), (1, 2, (0, 0)), (1, 0, (1, 0)), (2, 0, (0, 1)), (2, 1, (-1, 1))]
    kx, ky = np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)
    h = np.zeros(kx.shape + (6, 6), dtype=np.complex128)
    for y in range(2):
        for u1, u2, dr in kagome_NN:
            y2 = y + dr[1]
            R = dr[0] * a_x + (y2 // 2) * 2 * a_y # magnetic unit cell displacement
            phase = np.exp(1j * (kx * R[0] + ky * R[1]))
            t = -KagomePiFluxBondSigns(u1, u2, dr, np.array([y]))[0]
            h[..., 3 * y + u1, 3 * (y2 % 2) + u2] += t * phase
            h[..., 3 * (y2 % 2) + u2, 3 * y + u1] += t * np.conj(phase)
    return np.linalg.eigvalsh(h) - kagome_pi_flux_dirac_energy


def KagomePiFluxBandStructure(Ly=4, plot=True, tet=0., shift=0):
    """
        Kagome Dirac pi flux ansatz on an infinite YC(2 Ly) cylinder of circumference Ly * a_y (Ly as in Gutzwiller.py)
        with flux tet (in units of pi). Plots the highest occupied band (3 of the 6 bands of the magnetic unit cell are
        occupied at half filling) and returns the energy per occupied mode, relative to the Dirac energy.
        shift: the shifted YC cylinder of circumference C = Ly a_y + shift a_x (ShiftedCylinderCircumference, tenpy's
        bc_shift). The magnetic unit cell (a_x, 2 a_y) contains a_x and the bond signs depend on y only, so for the even
        Ly C is a magnetic lattice vector for every shift and the momenta are k.C = pi tet + 2 pi m.
    """
    assert Ly % 2 == 0, "the pi flux magnetic unit cell is doubled along a_y, Ly must be even"
    assert shift == round(shift), "the shift must be an integer"
    shift = int(shift)

    # reciprocal vectors of the magnetic unit cell (a_x, 2 a_y)
    b1 = 2 * pi * np.array([0.5, -1.0 / (2 * sqrt(3))])
    b2 = 2 * pi * np.array([0.0, 1.0 / (2 * sqrt(3))])
    C = ShiftedCylinderCircumference(np.array([2.0, 0.0]), np.array([1.0, sqrt(3)]), Ly, shift)
    name = YCName(2 * Ly, shift)
    return CylinderBandStructure(lambda kx, ky: KagomePiFluxBlochEnergies(kx, ky)[..., 0:3], C, tet, b1, b2,
                                 f"Kagome {name}", f"noninteracting_band_structure_Kagome_{name}_flux_{tet}.pdf",
                                 plot=plot, k_max=pi)


def ParamagnonSpectrum():
    kx = np.linspace(-pi, pi, 1000)
    ky = np.linspace(-pi, pi, 1000)
    Kx, Ky = np.meshgrid(kx, ky)
    delta = 0.5
    K = np.array([pi, pi])

    spectrum = lambda _kx, _ky: cos(_kx) + cos(_ky)
    eps_k = spectrum(Kx, Ky)
    eps_kpK = spectrum(Kx + K[0], Ky + K[1])
    asym_term = np.sqrt(0.25*((eps_k - eps_kpK)**2) + delta**2)
    sym_term = 0.5 * (eps_k + eps_kpK)
    Ep = sym_term + asym_term
    Em = sym_term - asym_term

    neg_energy_indicator = (Ep < 0) + (Em < 0)
    fig, ax = plt.subplots(figsize=(5, 6))
    #image = ax.imshow(neg_energy_indicator, origin="lower", extent=(np.min(Kx), np.max(Kx), np.min(Ky), np.max(Ky)),
    #                  cmap='RdBu')
    image = ax.imshow(Ep, origin="lower", extent=(np.min(Kx), np.max(Kx), np.min(Ky), np.max(Ky)),
                      cmap='RdBu')
    fig.colorbar(image, ax=ax, pad=0.02)
    plt.show()


if __name__ == "__main__":
    # ParamagnonSpectrum()
    #PiFluxBandStructure(Ly=10, plot=True, tet=0, geometry="XC")
    #PiFluxBandStructure(Ly=10, plot=True, tet=1.0, geometry="XC")

    PiFluxBandStructure(Ly=6, plot=True, tet=1.0, geometry="YC", shift=-1)
    #PiFluxBandStructure(Ly=5, plot=True, tet=0.0, geometry="YC")
    #PiFluxBandStructure(Ly=5, plot=True, tet=1.0, geometry="YC")
    #PiFluxBandStructure(Ly=6, plot=True, tet=0.0, geometry="XC")
    #PiFluxBandStructure(Ly=6, plot=True, tet=1.0, geometry="XC")
    #PiFluxBandStructure(Ly=8, plot=True, tet=0.0, geometry="XC")
    #PiFluxBandStructure(Ly=8, plot=True, tet=1.0, geometry="XC")
    #PiFluxBandStructure(Ly=10, plot=True, tet=0.0, geometry="XC")
    #PiFluxBandStructure(Ly=10, plot=True, tet=1.0, geometry="XC")

    #PiFluxBandStructure(Ly=6, plot=True, tet=0.0, geometry="XC")
    #PiFluxBandStructure(Ly=6, plot=True, tet=1.0, geometry="XC")
    #PiFluxBandStructure(Ly=8, plot=True, tet=0.0, geometry="XC")
    #PiFluxBandStructure(Ly=8, plot=True, tet=1.0, geometry="XC")
    #PiFluxBandStructure(Ly=9, plot=True, tet=0.0, geometry="YC")
    #PiFluxBandStructure(Ly=9, plot=True, tet=1.0, geometry="YC")

    #KagomePiFluxBandStructure(Ly=5)
    #KagomePiFluxBandStructure(Ly=5, tet=1.0)

    #PiFluxBandStructure(Ly=6, plot=True, tet=0.5, geometry="XC")
    #PiFluxBandStructure(Ly=6, plot=True, tet=1.0, geometry="XC")
    #PiFluxBandStructure(Ly=6, plot=True, tet=1.5, geometry="XC")
    #Es = []
    #tets = np.linspace(0.0, 2.0, 25)
    #for tet in tets:
    #    E = PiFluxBandStructure(Ly=6, plot=False, tet=tet, geometry="XC")
    #    Es.append(E)
    #plt.plot(tets, Es, "o")
    plt.show()
    # PiFluxBandStructure(Ly=8, plot=True, tet
    # =1.0, geometry="YC")
    #PiFluxBandStructure(Ly=9, plot=True, tet=0, geometry="YC")
    #PiFluxBandStructure(Ly=9, plot=True, tet=1.0, geometry="YC")
    # PiFluxBandStructure(Ly=5, plot=True, tet=1.0)
