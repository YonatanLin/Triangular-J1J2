import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, sqrt

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

def bandStructure():
    a1 = np.array([1./2., sqrt(3)/2.0])
    a2 = np.array([3./2., -sqrt(3)/2.0])
    a3 = a1 + a2
    Kx, Ky = np.meshgrid(np.linspace(-3*pi/4, 3*pi/4, 2000), np.linspace(-3*pi/4, 3*pi/4, 2000))
    k1 = Kx * a1[0] + Ky * a1[1]
    k2 = Kx * a2[0] + Ky * a2[1]
    k3 = Kx * a3[0] + Ky * a3[1]

    E_sq = 4 * cos(k1) ** 2 + (1 + cos(k3) + cos(k2) - cos(k1)) ** 2 + (sin(k3) - sin(k2) + sin(k1)) ** 2
    print("E_sq minimum: ", np.min(E_sq))
    fig, ax = plt.subplots()
    im = ax.imshow((-1)*sqrt(E_sq), origin="lower", extent = [-0.75, 0.75, -0.75, 0.75], cmap='RdBu')
    ax.set_xlabel("$k_x[\pi]$")
    ax.set_ylabel("$k_y[\pi]$")
    fig.savefig("noninteracting_band_structure" + ".pdf", bbox_inches='tight')
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    plt.show()

bandStructure()
