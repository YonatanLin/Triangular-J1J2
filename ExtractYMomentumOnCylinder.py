import numpy as np
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Square


def getYMomentumPermutation(lat):
    Lx, Ly, Lu = lat.shape
    perm = np.arange(lat.N_sites)
    for x in range(Lx):
        for u in range(Lu):
            for y in range(Ly):
                perm[lat.lat2mps_idx([x, y, u])] = lat.lat2mps_idx([x, (y + 1) % Ly, u])
    return perm

def ring_sz(psi, lat):
    Ly = lat.Ls[1]
    sz = psi.expectation_value('Sz')
    # return np.array([sz[lat.lat2mps_idx([0, y, 0])] for y in range(Ly)])
    return sz

def TestYPermutation():
    Ly = 4
    site = SpinHalfSite(conserve='Sz')
    lat = Square(2, Ly, site, bc=['periodic', 'periodic'], bc_MPS='infinite')

    # product state: a single up spin at y=0 on each ring, rest down
    p_state = ['down'] * lat.N_sites

    p_state[lat.lat2mps_idx([0, 0, 0])] = 'up'
    p_state[lat.lat2mps_idx([0, 3, 0])] = 'up'
    p_state[lat.lat2mps_idx([1, 3, 0])] = 'up'

    psi = MPS.from_product_state([site] * lat.N_sites, p_state, bc='infinite', unit_cell_width=Ly)

    # perm[i] = new position of the site currently at i (here: y -> y+1)
    Lx, _, Lu = lat.shape
    perm = getYMomentumPermutation(lat)
    print("before:", ring_sz(psi, lat))
    psi_T = psi.copy()
    psi_T.permute_sites(perm, trunc_par={'chi_max': 50})
    print("after :", ring_sz(psi_T, lat))
    print("|<psi|T psi>| (expect 0):", abs(psi.overlap(psi_T)))

    psi_n = psi.copy()
    for _ in range(Ly):
        psi_n.permute_sites(perm, trunc_par={'chi_max': 50})
    print("after Ly translations:", ring_sz(psi_n, lat))
    print("|<psi|T^Ly psi>| (expect 1):", abs(psi.overlap(psi_n)))

if __name__ == "__main__":
    TestYPermutation()