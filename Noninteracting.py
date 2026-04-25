import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, sqrt
from scipy.spatial import Voronoi

fontsize = 18
rc_params = {
    "font.family": "serif",
    "figure.dpi": 300,
    'text.usetex': True,
    #"axes.labelsize": 50,
    #"axes.titlesize": 50,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "figure.titlesize": fontsize,
    "legend.loc": "upper left",
}
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

def PiFluxBandStructure(Ly=4, plot=False):
    pi_factor = 1.5
    Kx, Ky = np.meshgrid(np.linspace(-pi_factor*pi, pi_factor*pi, 1000),
                         np.linspace(-pi_factor*pi, pi_factor*pi, 1000))

    k1_bz = 2 * pi * np.array([-0.5, -sqrt(3) / 2])
    k2_bz = 2 * pi * np.array([0.5, -1.0 / (2 * sqrt(3))])

    debug = True
    if plot:
        fig, ax = plt.subplots()

    kxs = np.linspace(-2 * pi, 2 * pi, 200)
    E_tot = 0.0
    N_states = 0
    for i_kx, kx in enumerate(kxs):
        for j in range(Ly):
            m = j - Ly // 2
            ky = 2. / sqrt(3) * ((2 * pi * m / Ly) - 0.5 * kx)

            k_vec = np.array([kx, ky])
            k1_bz_unit = k1_bz / sqrt(np.dot(k1_bz, k1_bz))
            k2_bz_unit = k2_bz / sqrt(np.dot(k2_bz, k2_bz))
            kvec_k1bz = np.dot(k_vec, k1_bz_unit)
            kvec_k2bz = np.dot(k_vec, k2_bz_unit)
            if ((-pi) - 1e-6) < kvec_k1bz < (pi + 1e-6) and (-pi / sqrt(3) - 1e-6) < kvec_k2bz < (
                    (pi / sqrt(3)) + 1e-6):
                E_sq = PiFluxSquaredEnergy(kx, ky)
                E_tot -= sqrt(E_sq)
                N_states += 1
                if plot:
                    ax.plot(kx / pi, ky / pi, "ko", markersize=1)
    E_per_mode = E_tot / N_states

    E_sq = PiFluxSquaredEnergy(Kx, Ky)
    E_sq_theory = 2 * (3 + cos(2*Kx) - cos(Kx - sqrt(3)*Ky) + cos(Kx + sqrt(3)*Ky))

    if plot:
        plot_bz1(ax, k1_bz/ pi, k2_bz / pi)

    if plot:
        print(f"energy per site: {E_per_mode}")
        print("diff from theory: ", np.max(np.abs(E_sq - E_sq_theory)))
        print("E_sq minimum: ", np.min(E_sq))
        im = ax.imshow((-1)*sqrt(E_sq), origin="lower", extent = (-pi_factor, pi_factor, -pi_factor, pi_factor),
                       cmap='RdBu')
        ax.set_xlabel("$k_x[\pi]$")
        ax.set_ylabel("$k_y[\pi]$")
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        ax.legend()
        fig.savefig("noninteracting_band_structure" + ".pdf", bbox_inches='tight')
        plt.show()
    return E_per_mode

if __name__ == "__main__":
    PiFluxBandStructure()
