import pickle

import numpy as np
from numpy import sqrt, pi
from numpy.linalg import eigh
import matplotlib.pyplot as plt

import tenpy
from tenpy import networks
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite, SpinHalfSite

from temfpy import slater
import temfpy.gutzwiller as gutz
from temfpy.utils import HT

from TryingTemfpy import local
from Noninteracting import PiFluxBandStructure
from WaveFunctionProperties import (plot_structure_factor, ComputeMomentumSpaceStructureFactor,
                                    CalculateSpinSpinCorrelations)
from Main import (AbsMagzFromNormMagz, BuildTriangularLattice, TriangularXC, LxInfiniteMPSCorrelations,
                  CreateGutzwillerCaseDir, PlotLattice, PrintCouplings, ImshowMatrix, SaveSimulationOutput,
                  calculateStructureFactorAtSpecialPoints, getSpecielBzPoints, glob_results_dir, code_dir,
                  model_type_dirac, model_type_Z2, pauli_x, pauli_y, pauli_z)

svd_min_slater_default = 5e-7

Lx_short_factor_temfpy_iMPS = 50

paulis = np.asarray([pauli_x, pauli_y, pauli_z])

def parity_mask(leg, parity=0):
    mask = (leg.to_qflat() % 2 == parity % 2).ravel()
    return mask

def SetZeroTensorChargesInGutzwillerWavefunction(psi, abs_magz):
    charges = np.array([psi.get_B(i).qtotal[0] for i in range(psi.L)])
    print("charges: ", charges)
    print("last tensor charges: ", psi.get_B(psi.L-1).qtotal)
    assert (np.all(charges[:-1] == 0) and charges[-1] == psi.L + 2 * abs_magz) #factor of two over magnetization becasue a site with up spin has occupation of 2
    spin_site = networks.SpinHalfSite("Sz")
    spin_leg = spin_site.leg
    chinfo_s = spin_leg.chinfo
    for idx, B in enumerate(psi._B):
        leg_vL, leg_p, leg_vR = [B.get_leg(label) for label in ["vL", "p", "vR"]]
        leg_p.chinfo = chinfo_s
        leg_p.charges = spin_leg.charges
        new_qtot = 0
        leg_vL.chinfo = chinfo_s
        if idx == 0:
            leg_vL.charges -= idx
        if idx == len(psi._B) - 1:
            new_qtot = 2 * abs_magz
        leg_vR.chinfo = chinfo_s
        if idx != len(psi._B) - 1:
            leg_vR.charges -= idx + 1

        B_modified_charge = tenpy.Array.from_ndarray(B.to_ndarray(), [leg_vL, leg_p, leg_vR], qtotal=[new_qtot],
                                                     labels=["vL", "p", "vR"])
        psi._B[idx] = B_modified_charge



def iMPSAbrikosov(mps, norm_magz):
    conserved_fermion = mps.sites[0].conserve
    assert(conserved_fermion == "N")
    conserved_spin = "Sz"

    spin_site = networks.SpinHalfSite(conserved_spin)

    # need to change charges to keep original virtual legs charges, otherwise for infinite mps the charges on the leftmost
    # and rightmost legs don't agree, breaking unitcell periodicity
    #chinfo = npc.ChargeInfo([1])
    #modified_spin_leg = npc.LegCharge.from_qflat(chinfo, [[0], [2]])
    #spin_site.change_charge(modified_spin_leg)

    spin_leg = spin_site.leg
    chinfo_s = spin_leg.chinfo
    mps.group_sites(2)
    abs_magz = AbsMagzFromNormMagz(norm_magz, mps.L)

    # can take _B[0] because the physical charge is the same on every tensor
    mask_p = parity_mask(mps._B[0].get_leg("p"))

    for idx, B in enumerate(mps._B):
        B.legs[B.get_leg_index("p")] = B.get_leg("p").to_LegCharge()

        # mask_vL = parity_mask(B.get_leg("vL"))
        # passing B.total to parity max for iMPS
        # mask_vR = parity_mask(B.get_leg("vR"), B.qtotal)

        # B.iproject([mask_vL, mask_p, mask_vR], ["vL", "p", "vR"])
        B.iproject([mask_p], ["p"])

        if conserved_spin == "Sz":
            B.chinfo = chinfo_s

            leg_vL, leg_p, leg_vR = [B.get_leg(label) for label in ["vL", "p", "vR"]]

            leg_p.chinfo = chinfo_s
            leg_p.charges = spin_leg.charges

            leg_vL.chinfo = chinfo_s
            #leg_vL.charges -= idx

            leg_vR.chinfo = chinfo_s
            # leg_vR.charges -= idx + 1

        else:  # None
            B = B.drop_charge(charge="parity_N", chinfo=chinfo_s)

    SetZeroTensorChargesInGutzwillerWavefunction(mps, abs_magz)

    mps.chinfo = chinfo_s
    mps.grouped = 1
    mps.sites = [spin_site] * mps.L

    mps.form = [None] * mps.L

    # need to define mps._S for canonical form, taking a vector of ones since canonical form shouldn't use the values anyway,
    # just the shapes (?)
    # mps._S = [None] * (mps.L + 1)
    S = []
    for i in range(mps.L):
        B = mps._B[i]
        S += [np.ones(B.to_ndarray().shape[0])]
    mps._S = S

    debug = False
    if debug:
        for i in range(len(mps._B)):
            B0_array = mps._B[i].to_ndarray()
            mps_vL_qflat = mps._B[i].get_leg("vL").to_qflat()
            mps_p_qflat = mps._B[i].get_leg("p").to_qflat()
            mps_vR_qflat = mps._B[i].get_leg("vR").to_qflat()
            B0_charge = mps._B[i].qtotal[i]
            for i, ch_i in enumerate(mps_vL_qflat):
                for j, ch_j in enumerate(mps_p_qflat):
                    for k, ch_k in enumerate(mps_vR_qflat):
                        if abs(B0_array[i, j, k]) > 1e-15:
                            assert (B0_charge == (ch_i[0] + ch_j[0] - ch_k[0]))

    # Transform into right canoncial form
    mps.canonical_form()


def isEdgeMode(abs_state, side, eps=0.01):
    n_sites = abs_state.shape[0]
    assert(abs(np.sum(abs_state ** 2) - 1) < 2e-14)

    if side == "L":
        abs_state_right = abs_state[(n_sites // 2):] ** 2
        return (sum(abs_state_right) < eps)
    else:
        abs_state_left = abs_state[0:(n_sites // 2)] ** 2
        return (sum(abs_state_left) < eps)



def getZeroModeSupportSide(zm):
    site_with_max_norm = np.argmax(np.abs(zm))
    side = "L" if site_with_max_norm < zm.shape[0]//2 else "R"
    abs_zm = np.abs(zm)
    zm_sum = np.sum(abs_zm)
    n_sites = zm.shape[0]
    assert(isEdgeMode(abs_zm, side))

    return side


def GetEigenvectorsForCorrelationMatrixArbitraryOccupation(H, N, psi_support_per_spin, zero_energy_tol, num_zero_modes,
                                                           indices_to_remove = None):
    H_up = H[0::2, 0::2]
    H_down = H[1::2, 1::2]
    e_up, v_up = eigh(H_up)
    e_down, v_down = eigh(H_down)
    v_up, e_up = v_up[:, 0:N // 2], e_up[0:N // 2]
    v_down, e_down = v_down[:, 0:N // 2], e_down[0:N // 2]
    eigendata_per_spin = {"up": (v_up, e_up), "down": (v_down, e_down)}
    if indices_to_remove is None:
        indices_to_remove = {}
        # assert (np.all(np.abs(e_up[-2:]) < zero_energy_tol) and np.all(np.abs(e_down[-2:]) < zero_energy_tol))
        for spin in ["up", "down"]:
            indices_to_remove_spin = []
            for i in range(-1, -1 - num_zero_modes//2, -1):
                v_spin_i, e_spin_i = eigendata_per_spin[spin][0][:, i], eigendata_per_spin[spin][1][i]
                assert(np.abs(e_spin_i) < zero_energy_tol)
                zm_i = v_spin_i
                support_side = getZeroModeSupportSide(zm_i)
                if support_side == psi_support_per_spin[spin]:
                    # indices_to_remove[spin] = i
                    indices_to_remove_spin.append(i)
            indices_to_remove[spin] = indices_to_remove_spin
    else:
        assert psi_support_per_spin is None, "over specification of orbitals to remove"

    for spin in ["up", "down"]:
        v_spin, e_spin = eigendata_per_spin[spin]
        v_spin = np.delete(v_spin, indices_to_remove[spin], axis=1)
        e_spin = np.delete(e_spin, indices_to_remove[spin])
        eigendata_per_spin[spin] = (v_spin, e_spin)

    v = np.zeros((H.shape[0], eigendata_per_spin[spin][1].shape[0] + eigendata_per_spin[spin][1].shape[0]),
                 dtype=v_up.dtype)
    e = np.zeros(v.shape[1])
    for parity, spin in [(0, "up"), (1, "down")]:
        v[parity::2, parity::2] = eigendata_per_spin[spin][0]
        e[parity::2] = eigendata_per_spin[spin][1]
    return v, e


def TestGetEigenvectorsForCorrelationMatrixArbitraryOccupation(load=False):
    if not load:
        random_mat = np.random.rand(10, 10)
        H_up = random_mat + np.transpose(random_mat)
        H_down = (-1) * H_up
        H = np.zeros((20, 20))
        H[0::2, 0::2] = H_up
        H[1::2, 1::2] = H_down
    else:
        H = np.loadtxt("test_mat.csv", dtype=np.float64)

    filling = 12
    v_block_diag, e_block_diag = (
        GetEigenvectorsForCorrelationMatrixArbitraryOccupation(H, filling, None, 0,
                                                 0, indices_to_remove={"up":-1, "down":-2}))
    assert(v_block_diag.shape[1] == filling - 2)
    e_full_diag, v_full_diag = eigh(H)
    for i, ei in enumerate(e_block_diag):
        vi = v_block_diag[:,i]
        Hvi = H @ vi
        eigenvalue_condition = np.min(np.abs(e_full_diag - e_block_diag[i])) < 5e-14 # ei is an eigenvalue
        eigenvector_condition = np.max(np.abs(Hvi - ei * vi)) < 5e-14 # vi is an eigenvector with eigenvalue ei
        if not eigenvalue_condition or not eigenvector_condition: # failed
            np.savetxt('test_mat.csv', H)
        assert(eigenvector_condition), "bad eigenvectors from block diagonalization"
        assert(eigenvalue_condition), "bad eigenstates from block diagonalization"
    print("success")
    return


def CorrelationMatrixArbitraryOccupation(H, N, psi_support_per_spin, zero_energy_tol, num_zero_modes,
                                         indices_to_remove=None):
    assert(N % 2 == 0)
    v, e = GetEigenvectorsForCorrelationMatrixArbitraryOccupation(H, N, psi_support_per_spin, zero_energy_tol,
                                                                  num_zero_modes, indices_to_remove=indices_to_remove)
    if indices_to_remove is None:
        assert(v.shape[1] == N - num_zero_modes // 2 and e.shape[0] == N - num_zero_modes // 2), "wrong occupation"
        assert(np.sum(np.abs(e) < zero_energy_tol) == num_zero_modes // 2)
    C = v @ HT(v)

    if np.iscomplexobj(C) and np.allclose(C.imag, 0.0, rtol=0, atol=1e-14):
        C = C.real  # eliminate zero imaginary parts
    return C, N


def CreateHamiltonianMatrixFromCouplingsList(model, N_sites, dtype=np.float64):
    couplings_list = model.all_coupling_terms().to_TermList()
    onsite_list = model.all_onsite_terms().to_TermList()
    H = np.zeros((N_sites, N_sites), dtype=dtype)
    for coupling in couplings_list:
        strength = coupling[1]
        site1 = coupling[0][0][1]
        site2 = coupling[0][1][1]
        if "Cd" in coupling[0][0][0]:
            H[site1, site2] = strength
        else:
            H[site2, site1] = strength
    for onsite_term in onsite_list:
        site = onsite_term[0][0][1]
        op = onsite_term[0][0][0]
        assert(op == 'N')
        strength = onsite_term[1]
        H[site, site] = strength
    assert (np.abs(H - np.conj(np.transpose(H))) < 1e-15).all()
    return H


def FreeFermionSpinCorrelations(C):
    """
    input: two-fermions spatial correlation matrix C, corresponding to a quadratic Hamiltonian
    :return: spin-spin correlations, where the spin operator is expressed as a fermion bilinear contracted with Pauli
    matrices
    """
    C_up = C[0::2, 0::2]
    C_down = np.eye(C_up.shape[0]) - np.transpose(C[1::2, 1::2])

    assert(np.max(np.abs(C_up - HT(C_up))) < 1e-14), "C_up should be hermitian"
    assert (np.max(np.abs(C_down - HT(C_down))) < 1e-14), "C_down should be hermitian"
    assert(C_up.shape == C_down.shape)

    G = np.zeros((C_up.shape[0], C_up.shape[1], 2), dtype=np.complex128)
    G[:, :, 0] = C_up
    G[:, :, 1] = C_down

    n = G.shape[0]

    if G.shape != (n, n, 2):
        raise ValueError(f"G must have shape (n, n, 2), got {G.shape}")

    # --- Mean (disconnected) part: <S^a_i> <S^b_j> ---
    # <S^a_i> = (1/2) sum_sigma sigma^a_{sigma,sigma} G^sigma_{ii}
    diag_pauli = np.einsum('aii->ai', paulis)  # (3, 2):  sigma^a_{s,s}
    G_ii = np.einsum('iis->si', G)  # (2, n):  G^s_{ii}
    S_mean = 0.5 * np.einsum('as,si->ai', diag_pauli, G_ii)  # (3, n)

    disconnected = np.einsum('ai,bj->abij', S_mean, S_mean)  # (3,3,n,n)

    # --- Connected (exchange) part ---
    # term2[i,j,s'] = delta_ij - G^{s'}_{ji}
    delta_ij = np.eye(n)[:, :, None]  # (n, n, 1)
    G_ji = np.transpose(G, (1, 0, 2))  # G_ji[i,j,s'] = G[j,i,s']
    term2 = delta_ij - G_ji  # (n, n, 2)

    # connected[a,b,i,j] = 1/4 * sum_{s,s'} P[a,s,s'] P[b,s',s] G[i,j,s] term2[i,j,s']
    connected = 0.25 * np.einsum(
        'ast,bts,ijs,ijt->abij', paulis, paulis, G, term2
    )

    total_spin_spin_tensor = disconnected + connected
    spin_spin_correlations = np.einsum('iiab->ab', total_spin_spin_tensor) #sum_{s,s} C[s,s,i,j]
    return spin_spin_correlations


def PlotModelHoppingsByPhase(model, ax, linewidth_min=0.4, linewidth_max=3.0, plot_sites=True,
                             plot_order=False, add_colorbar=True):
    couplings_list = model.all_coupling_terms().to_TermList()
    lat = model.lat
    basis_vectors = np.asarray(lat.basis, dtype=float)
    unit_cell_positions = np.asarray(lat.unit_cell_positions, dtype=float)

    def site_position(site_mps_index):
        lat_idx = lat.mps2lat_idx(site_mps_index)
        return np.dot(np.asarray(lat_idx[:2], dtype=float), basis_vectors) + unit_cell_positions[lat_idx[-1], :]

    # Build unique undirected hoppings using Hamiltonian matrix elements H_ij.
    hoppings = {}
    for coupling in couplings_list:
        op1 = coupling[0][0][0]
        i = coupling[0][0][1]
        j = coupling[0][1][1]
        t = coupling[1]
        if "Cd" not in op1:
            i, j = j, i
        key = (min(i, j), max(i, j))
        if key in hoppings:
            continue
        hoppings[key] = (i, j, t)

    if len(hoppings) == 0:
        return

    abs_vals = np.asarray([np.abs(hoppings[key][2]) for key in hoppings], dtype=float)
    abs_min = float(np.min(abs_vals))
    abs_max = float(np.max(abs_vals))

    #color1 = "#000033"  # Very Deep Navy
    #color2 = "#FFFF00"  # Bright Yellow
    cyclic_colors = ["#440154", "#22a884", "#fde725", "#440154"]
    line_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("color_map", cyclic_colors)

    for key in hoppings:
        i, j, t = hoppings[key]
        p1 = site_position(i)
        p2 = site_position(j)

        theta = np.mod(np.angle(t), 2.0 * np.pi)
        if(abs(theta - 2*pi) < 1e-12):
            theta = 0
        phase01 = theta / (2.0 * np.pi)
        color = line_cmap(phase01)  # 0 -> blue, 2pi -> red

        if abs(abs_max - abs_min) > 1e-15:
            width = linewidth_min + (np.abs(t) - abs_min) * (linewidth_max - linewidth_min) / (abs_max - abs_min)
        else:
            width = 0.5 * (linewidth_min + linewidth_max)

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=width, solid_capstyle='round')

    if plot_sites:
        lat.plot_sites(ax)
    if plot_order:
        lat.plot_order(ax)
    lat.plot_basis(ax, origin=-0.5 * (lat.basis[0] + lat.basis[1]))
    ax.set_aspect('equal')

    if add_colorbar:
        norm = plt.Normalize(vmin=0.0, vmax=2.0 * np.pi)
        sm = plt.cm.ScalarMappable(cmap=line_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"phase $\theta$")
        cbar.set_ticks([0.0, np.pi, 2.0 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])


def DeterminePiFluxCoupling(x, y, dx, dy, basis_vectors):
    assert((basis_vectors[0][:] == [1.0, 0.0]).all())
    assert ((np.abs(basis_vectors[1][:] - [0.5, sqrt(3) / 2.]) < 1e-15).all())
    if dx == 1 and dy == 0:
        return 1.0
    elif dx == 0 and dy == 1:
        if x % 2 == 0:
            return -1.0
        return 1.0
    else:
        assert(dx == -1 and dy == 1)
        if x % 2 == 0:
            return 1.0
        return -1.0


def getPhysicalVectorFromLatticeVector(lat, u1, u2, dr):
    basis_vectors = np.asarray(lat.basis, dtype=float)
    dy = (lat.unit_cell_positions[u2][1] - lat.unit_cell_positions[u1][1]) + (np.dot(dr, basis_vectors))[1]
    dx = (lat.unit_cell_positions[u2][0] - lat.unit_cell_positions[u1][0]) + (np.dot(dr, basis_vectors))[0]
    return dx, dy


def getParticleHoleHoppingSign(ind):
    return 1 - 2*(ind % 2)


def PeriodicLandauGaugePhases(lat, Lx_cell, monopole_Q, particle_hole):
    """
    Phases g_j = exp(-i theta_j) of the U(1) gauge transformation c_j -> g_j c_j that maps the open-cylinder
    Landau gauge of MonopoleCondensatePiFluxModel (y-bond phase exp(-i dphi x), not Lx_cell-periodic) to a gauge
    that is periodic under x -> x + Lx_cell (the magnetic unit cell). theta_j = s_j * dphi * Lx_cell * y * floor(x / Lx_cell),
    with s_j = +1 for up spinons and -1 for particle-hole transformed down spinons. This is needed for the iMPS
    construction: without it, the right environments of the short and long chains differ by a large gauge transformation
    (a k_y boost by 2 pi Q / Ly), making their Schmidt vectors orthogonal. After Gutzwiller projection it is a pure
    on-site phase (both f_up and f_down pick up exp(i theta)), so the physical spin state is unchanged.
    Use as C -> g[:, None] * C * g.conj()[None, :].
    """
    Ly = lat.Ls[1]
    dphi = monopole_Q * 2 * pi / (Lx_cell * Ly)
    theta = np.zeros(lat.N_sites)
    for j in range(lat.N_sites):
        x, y, u = lat.mps2lat_idx(j)
        s = getParticleHoleHoppingSign(u) if particle_hole else 1
        theta[j] = s * dphi * Lx_cell * y * (x // Lx_cell)
    return np.exp(-1j * theta)


class MeanFieldSpinonModel(CouplingMPOModel):
    def init_H_from_terms(self):
        if(self.init_MPO):
            super().init_H_from_terms()


class MonopoleCondensatePiFluxModel(MeanFieldSpinonModel):
    def init_terms(self, model_params):
        plus_hc = True
        init_MPO = model_params["init_H_MPO"]
        monopole_Q = model_params["monopole_Q"]
        flux = model_params["flux"] * pi
        particle_hole = model_params["particle_hole"]
        self.init_MPO = init_MPO
        lat = self.lat
        bc = lat.boundary_conditions
        dphi = monopole_Q * 2 * pi / (lat.N_sites / len(lat.unit_cell_positions))
        geometry = "XC" if isinstance(lat, TriangularXC) else "YC"
        YC = (geometry == "YC")
        if particle_hole:
           if YC:
               assert(len(lat.unit_cell_positions) == 2), "wrong unit cell size for spinfull model"
           else:
               assert (len(lat.unit_cell_positions) == 4), "wrong unit cell size for spinfull model"


        assert(monopole_Q == round(monopole_Q))
        assert(bc[1] == "periodic")
        if not YC:
            assert (abs(dphi) < 1e-15)

        Lx, Ly = lat.Ls[0], lat.Ls[1]
        bc_x, bc_y = bc[0], bc[1]

        nys, nxs = np.arange(0, Ly), np.arange(0, Lx)
        y_coors, x_coors = np.meshgrid(nys, nxs) # first coordinate of the matrix (row) is the x coordinate
        strength_x = np.ones((Lx, Ly), dtype=np.complex128)
        if YC:
            strength_y = np.exp(-1j * dphi * x_coors) * (1 - 2 * ((x_coors + 1) % 2))
            strength_diag = strength_y * np.exp(-1j * (dphi / 2))
        else:
            strength_y = 1 - 2 * ((x_coors + y_coors) % 2)
            strength_diag = (-1) * strength_y

        if bc_x == "periodic":
            if YC:
                strength_x[-1, :] *= np.exp(1j * dphi * y_coors[-1, :] * Lx) # x bonds on last column
                strength_diag[-1, :] *= np.exp(-1j * dphi * y_coors[-1, :] * Lx) # diag bonds on last column
            else:
                assert(abs(dphi) < 1e-15)

        for u1, u2, dr in lat.pairs["nearest_neighbors"]:
            hole_transformed = False
            if particle_hole:
                assert (u1 % 2 == u2 % 2)
                hole_transformed = getParticleHoleHoppingSign(u1) < 0

            dx, dy = getPhysicalVectorFromLatticeVector(lat, u1, u2, dr)
            if abs(dy) < 1e-15:
                strength = strength_x
            elif np.sign(dx) == np.sign(dy):
                strength = strength_y
            else:
                XC_sgn = -1 if (geometry == "XC" and dr[0] >= 1) else 1
                strength = XC_sgn * strength_diag

            couplings_shape = self.lat.coupling_shape(dr)[0]
            strength = strength[0:couplings_shape[0], 0:couplings_shape[1]]
            strength_with_flux = self.coupling_strength_add_ext_flux(strength, dr, [0, flux])
            if hole_transformed:
                # f_down -> h_down^\dagger maps t_ij f_i^\dagger f_j to
                # -conj(t_ij) h_i^\dagger h_j, up to the dropped constant.
                strength_with_flux = -np.conj(strength_with_flux)
            self.add_coupling(strength_with_flux, u1, "Cd", u2, "C", dr, plus_hc=plus_hc)


class Z2MeanFieldModel(MeanFieldSpinonModel):
    def init_terms(self, model_params):
        mu = model_params["mu"] # chemical potential
        zeta = model_params["zeta"] # onsite pairing
        hoppings = model_params["hoppings"] # dict with hopping per direction
        pairings = model_params["pairings"] # dict with pairing per direction
        init_MPO = model_params["init_H_MPO"]
        self.init_MPO = init_MPO
        lat = self.lat
        bc = lat.boundary_conditions
        geometry = "XC" if isinstance(lat, TriangularXC) else "YC"
        YC = (geometry == "YC")
        assert(YC)
        Lx, Ly = lat.Ls[0], lat.Ls[1]
        nys, nxs = np.arange(0, Ly), np.arange(0, Lx)
        y_coors, x_coors = np.meshgrid(nys, nxs) # first coordinate of the matrix (row) is the x coordinate

        y_parity_signs = 1 - 2 * (y_coors % 2)
        constant_signs = np.ones(x_coors.shape)

        unitcell_length = len(lat.unit_cell_positions)
        # on site terms - chemical potential and pairing
        for i in range(unitcell_length // 2):
            ind1 = 2 * i
            ind2 = 2 * i + 1
            ph_sgn_1 = getParticleHoleHoppingSign(ind1)
            ph_sgn_2 = getParticleHoleHoppingSign(ind2)
            self.add_onsite(ph_sgn_1 * mu, ind1, "N")
            self.add_onsite(ph_sgn_2 * mu, ind2, "N")
            self.add_coupling(zeta, 2*i, "Cd", 2*i + 1, "C", [0, 0], plus_hc=True)

        neighbor_ranges = ["nearest_neighbors", "next_nearest_neighbors"]
        for neighbor_range in neighbor_ranges:
            for u1, u2, dr in (lat.pairs[neighbor_range]):
                assert(u1 % 2 == u2 % 2)
                ph_sgn = getParticleHoleHoppingSign(u1)
                dr_tuple = (int(dr[0]), int(dr[1]))
                dx, dy = getPhysicalVectorFromLatticeVector(lat, u1, u2, dr)

                if neighbor_range == "nearest_neighbors":
                    if (abs(dy) > 1e-15) and (np.sign(dx) == np.sign(dy)):
                        signs = y_parity_signs
                    else:
                        signs = (-1) * constant_signs
                else:
                    if abs(dx) < 1e-15:
                        signs = y_parity_signs
                    else:
                        signs = (-1) * constant_signs

                couplings_shape = self.lat.coupling_shape(dr)[0]
                signs = signs[0:couplings_shape[0], 0:couplings_shape[1]]

                if dr_tuple in hoppings:
                    hopping = hoppings[dr_tuple]
                    self.add_coupling(ph_sgn * signs * hopping, u1, "Cd", u2, "C", dr, plus_hc=True)
                if dr_tuple in pairings:
                    pairing = pairings[dr_tuple]
                    self.add_coupling(signs * pairing, u1, "Cd", (u2 + 1)%2, "C", dr, plus_hc=True)


def AddTermToFermionCouplingsDict(couplings_dict, i, j, strength):
    assert(i < j)
    couplings_dict[(i, j, 'Cd JW', 'C')] = strength
    couplings_dict[(i, j, 'JW C', 'Cd')] = strength


def AddCouplingsToModelDict(test_sites, coupled_sites):
    couplings = {}
    assert (len(test_sites) == 2 and test_sites[0] % 2 != test_sites[1] % 2)
    ph_signs = [getParticleHoleHoppingSign(site % 2) for site in test_sites]
    for ind, center_site in enumerate(test_sites):
        ph_sign = ph_signs[ind]
        for site, strength in coupled_sites:
            coupling_sign_1 = ph_sign if (site % 2 == center_site % 2) else 1
            coupling_sign_2 = ph_sign if (site % 2 != center_site % 2) else 1
            if site < center_site:
                AddTermToFermionCouplingsDict(couplings, site, center_site,
                                              coupling_sign_1 * strength)
                AddTermToFermionCouplingsDict(couplings, site - 1, center_site,
                                              coupling_sign_2 * strength)
            else:
                AddTermToFermionCouplingsDict(couplings, center_site, site,
                                              coupling_sign_1 * strength)
                AddTermToFermionCouplingsDict(couplings, center_site, site - 1,
                                              coupling_sign_2 * strength)
    return couplings


def AddCouplingsToZ2ModelDict(test_sites, coupled_sites, zeta):
    couplings = AddCouplingsToModelDict(test_sites, coupled_sites)
    AddTermToFermionCouplingsDict(couplings, test_sites[0], test_sites[1], zeta)
    return couplings


def TestDictsAreCompatible(couplings_dict, expected_couplings_dict):
    assert (len(couplings_dict.keys()) == len(expected_couplings_dict.keys()))
    for key in couplings_dict.keys():
        if (couplings_dict[key] != expected_couplings_dict[key]):
            print(
                f"test failed - unexpected coupling for {key}: expected {expected_couplings_dict[key]}, got {couplings_dict[key]}")
            exit(1)


def GetZ2CouplingDictFromStrengths(x_nn_strength, y_nn_strength, nnn_strength_y,
                                   nnn_strength_diag):
    return {(1, 0): x_nn_strength, (0, 1): y_nn_strength, (-1, 1): x_nn_strength,
     (-1, 2): nnn_strength_y, (1, 1): nnn_strength_diag}



def Z2MeanFieldModelOptimalQSL():
    zeta = -0.8
    mu = 0.8

    x_nn_hopping = 1.0
    y_nn_hopping = -2.6
    nnn_hopping_y = 0.25
    nnn_hopping_diag = -0.1

    x_nn_pairing = 1.25
    y_nn_pairing = 1.5
    nnn_pairing_y = 0.0
    nnn_pairing_diag = 0.0

    model_params = {"mu": mu, "zeta": zeta, "init_H_MPO": False}
    model_params["hoppings"] = GetZ2CouplingDictFromStrengths(x_nn_hopping, y_nn_hopping, nnn_hopping_y,
                                                              nnn_hopping_diag)
    model_params["pairings"] = GetZ2CouplingDictFromStrengths(x_nn_pairing, y_nn_pairing, nnn_pairing_y,
                                                              nnn_pairing_diag)
    return model_params


def getPiFluxLatticeOrdering(Lx, Ly, unit_cell_size):
    ordering = []
    if unit_cell_size == 4:
        for i in range(Lx):
            for j in range(Ly):
                ordering.append((i, j, 0))
                ordering.append((i, j, 1))
            for j in range(Ly):
                ordering.append((i, j, 2))
                ordering.append((i, j, 3))
    else:
        for i in range(Lx):
            for j in range(Ly):
                ordering.append((i, j, 0))
            for j in range(Ly):
                ordering.append((i, j, 1))

    ordering = np.array(ordering)
    return ordering


def GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, bc_MPS, geometry):

    bc = ['open', 'periodic']

    if bc_MPS == "infinite":
        bc[0] = 'periodic'

    nearest_neighbors = []
    if spinfull:
        if geometry == "YC":
            dxs = ([1, 0], [0, 1], [-1, 1])
            for i in range(2):
                for dx in dxs:
                    nearest_neighbors += [[i, i, dx]]

    return BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc=bc, geometry=geometry,
                                  spinfull_fermions=spinfull)


def getZeroModesDict(gs_manifold_index):
    if gs_manifold_index == 0:
        psi_support_per_spin = {"up": "L", "down": "L"}
    elif gs_manifold_index == 1:
        psi_support_per_spin = {"up": "R", "down": "L"}
    elif gs_manifold_index == 2:
        psi_support_per_spin = {"up": "L", "down": "R"}
    else:
        psi_support_per_spin = {"up": "R", "down": "R"}

    return psi_support_per_spin


def CalculateExactCMatrixForPiFlux(gs_manifold_index, model_params, model_type,
                                   zero_energy_tol=1e-9, plot_lattice=False, abs_magz=0,
                                   results_dir=None, debug=False):
    particle_hole = model_params["particle_hole"]
    triangular_lat = model_params["lattice"]
    if model_type == model_type_dirac:
        model = MonopoleCondensatePiFluxModel(model_params)
    elif model_type == model_type_Z2:
        model = Z2MeanFieldModel(model_params)
    else:
        raise ValueError("unrecognized model type")


    if plot_lattice:
        fig, ax = plt.subplots(figsize=(6, 5))
        PlotLattice(triangular_lat, ax, additional_couplings_to_plot=model.pos_hoppings_list)
        PlotLattice(triangular_lat, ax, additional_couplings_to_plot=model.neg_hoppings_list,
                    nnn_line_style="--", )
        if local:
            plt.show()

    H = CreateHamiltonianMatrixFromCouplingsList(model, triangular_lat.N_sites, dtype=np.complex128)
    e, v = eigh(H)

    Lx, Ly = triangular_lat.Ls

    if debug:
        np.savetxt(f"debug_magnetized_iMPS/e_Lx_{Lx}.txt", e)
        np.savetxt(f"debug_magnetized_iMPS/v_Lx_{Lx}.txt", v)

    zero_modes_indices = np.where(np.abs(e) < zero_energy_tol)
    num_zero_modes = zero_modes_indices[0].shape[0]
    assert (num_zero_modes % 4 == 0), "number of zero modes should be a multiple of 4"
    if num_zero_modes not in [0, 4]:
        assert (gs_manifold_index == 0), "don't support gs_manifold_index for more than 4 zero modes"
    print(f"number of zero modes in pi flux model: {num_zero_modes}")
    zero_modes = num_zero_modes > 0
    finite_magz = abs(abs_magz) > 0
    if model_type == model_type_Z2:
        assert not zero_modes, "shouldn't have zero modes in the Z2 gapped ansatz"

    if finite_magz:
        assert (particle_hole), "expect particle hole for magnetized state"
        assert (abs_magz <= triangular_lat.N_sites // 4), "absolute magnetization cannot exceed the number of sites"
        N_up = (triangular_lat.N_sites // 4 + abs_magz)
        N_filling = 2 * N_up

        n_edge_modes_L_up = 0
        n_edge_modes_L_down = 0
        n_edge_modes_R_up = 0
        n_edge_modes_R_down = 0
        for i in range(N_filling):
            abs_vi = np.abs(v[:, i])
            if (isEdgeMode(abs_vi, "L", 1e-9)):
                if (np.max(np.abs(abs_vi[0::2])) > 1e-6):
                    n_edge_modes_L_up += 1
                else:
                    n_edge_modes_L_down += 1
            if isEdgeMode(abs_vi, "R", 1e-9):
                if (np.max(np.abs(abs_vi[0::2])) > 1e-6):
                    n_edge_modes_R_up += 1
                else:
                    n_edge_modes_R_down += 1

        print(f"number of left edge modes for up spins: {n_edge_modes_L_up}")
        print(f"number of left edge modes for down spins: {n_edge_modes_L_down}")
        print(f"number of right edge modes for up spins: {n_edge_modes_R_up}")
        print(f"number of right edge modes for down spins: {n_edge_modes_R_down}")

        if results_dir is not None:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(e[0::2], "bo")
            ax.plot([N_up, N_up], [np.min(e), np.max(e)], "r--")
            ax.set_xlabel("state index")
            ax.set_ylabel("e")
            fig.savefig(results_dir + "up_spin_filling.png", bbox_inches='tight')

    elif model_type == model_type_dirac:
        if particle_hole and not zero_modes:
            #TODO: is it okay that we don't enter here if there are zero modes?
            N_filling_no_ph = triangular_lat.N_sites // 2
            N_up, N_down_holes = DetermineSpinsOccupation(N_filling_no_ph, H, e)
            print(f"N_up={N_up}, N_down_holes={N_down_holes}")
            N_filling = 2*N_up
        else:
            N_filling = (np.max(zero_modes_indices) + 1) if zero_modes else (triangular_lat.N_sites // 2)
    else:
        N_filling = triangular_lat.N_sites // 2

    print("pi-flux energy from exact diagonalization: ", np.sum(e[0:N_filling])/N_filling)

    if (zero_modes and not finite_magz): #need to carefully determine zero modes occupation
        psi_support_per_spin = getZeroModesDict(gs_manifold_index)
        C, _ = CorrelationMatrixArbitraryOccupation(H, N_filling, psi_support_per_spin, zero_energy_tol,
                                                    num_zero_modes)
    else:
        if N_filling < e.shape[0]:
            assert(abs(e[N_filling] - e[N_filling - 1]) > zero_energy_tol), \
                "Fermi level should sit inside a gap for magnetized state"
        C, _ = slater.correlation_matrix(H, N_filling)

    return C, triangular_lat


def GetTriangularFluxSlaterMPS(Lx, Ly, spinfull, site, geometry, slater_trunc_par, unitcell_width,
                               bc_MPS, gs_manifold_index, model_type, flux=0.0, particle_hole=True,
                               norm_magz=0., monopole_Q=0, iMPS_Lx_factor=Lx_short_factor_temfpy_iMPS, results_dir=None):
    zero_energy_tol = 1e3 * slater_trunc_par["degeneracy_tol"]
    assert(Lx % 2 == 0), "pi-flux model requires even-sized unitcell"
    imps_unitcell = unitcell_width * Lx * Ly

    C = None
    finite = (bc_MPS == "finite")
    finite_bc_MPS = "finite"

    if model_type == model_type_dirac:
        model_params = {"init_H_MPO": False, "monopole_Q": monopole_Q, "flux": flux,
                        "particle_hole": particle_hole}
    elif model_type == model_type_Z2:
        model_params = Z2MeanFieldModelOptimalQSL()
    else:
        raise ValueError("inrecognized model type")
        model_params = None

    triangular_lat_finite = GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, finite_bc_MPS, geometry)
    if finite:
        abs_magz = AbsMagzFromNormMagz(norm_magz, triangular_lat_finite.N_sites // 2)
        model_params["lattice"] = triangular_lat_finite
        C, triangular_lattice = CalculateExactCMatrixForPiFlux(gs_manifold_index, model_params, model_type,
                                                               zero_energy_tol = zero_energy_tol,
                                                               plot_lattice=False, abs_magz=abs_magz,
                                                               results_dir=results_dir)

        psi_from_slater = slater.C_to_MPS(C, trunc_par=slater_trunc_par)
    else:
        abs_magz_unitcell = AbsMagzFromNormMagz(norm_magz, triangular_lat_finite.N_sites // 2)
        Q_unitcell = model_params["monopole_Q"]

        Lx_short, Lx_long = iMPS_Lx_factor * Lx, (iMPS_Lx_factor + 1) * Lx

        model_params_short = model_params.copy()
        triangular_lat_short = GetPiFluxTriangularLattice(site, Lx_short, Ly, spinfull, finite_bc_MPS, geometry)
        model_params_short["lattice"] = triangular_lat_short
        model_params_short["monopole_Q"] = Q_unitcell * (Lx_short // Lx)
        abs_magz_short = AbsMagzFromNormMagz(norm_magz, triangular_lat_short.N_sites // 2)

        C_short, triangular_lat_short = CalculateExactCMatrixForPiFlux(gs_manifold_index, model_params_short, model_type,
                                                                       zero_energy_tol=zero_energy_tol,
                                                                       abs_magz=abs_magz_short)

        model_params_long = model_params.copy()
        triangular_lat_long = GetPiFluxTriangularLattice(site, Lx_long, Ly, spinfull, finite_bc_MPS, geometry)
        model_params_long["lattice"] = triangular_lat_long
        model_params_long["monopole_Q"] = Q_unitcell * (Lx_long // Lx)
        abs_magz_long = AbsMagzFromNormMagz(norm_magz, triangular_lat_long.N_sites // 2)

        assert (abs_magz_short + abs_magz_unitcell == abs_magz_long), \
                "magnetization should fit in unitcell for iMPS temfpy calculation"

        C_long, triangular_lat_long = CalculateExactCMatrixForPiFlux(gs_manifold_index, model_params_long, model_type,
                                                                     zero_energy_tol=zero_energy_tol,
                                                                     abs_magz=abs_magz_long)
        middle_site_mps_ind_short = triangular_lat_short.lat2mps_idx([Lx_short // 2, 0, 0])
        # exit(0)

        if model_type == model_type_dirac and Q_unitcell != 0:
            assert geometry == "YC", "monopole flux only implemented for YC"
            for lat_i, name in [(triangular_lat_short, "short"), (triangular_lat_long, "long")]:
                g = PeriodicLandauGaugePhases(lat_i, Lx, Q_unitcell, particle_hole)
                if name == "short":
                    C_short = g[:, None] * C_short * np.conj(g)[None, :]
                else:
                    C_long = g[:, None] * C_long * np.conj(g)[None, :]

        offset = "auto"
        if(C_short.dtype == np.complex128):
            # bug in temfpy, complex correlations are unhandled when they determine offset...
            offset = 0

        psi_from_slater, error = slater.C_to_iMPS(C_short, C_long, slater_trunc_par, sites_per_cell=imps_unitcell,
                                                  cut=middle_site_mps_ind_short, offset=offset)

        infinite_bc = ("periodic", "periodic")
        triangular_lattice = BuildTriangularLattice(Lx, Ly, site, bc_MPS, infinite_bc,
                                                    geometry, spinfull)

    psi_from_slater.unit_cell_width = triangular_lattice.mps_unit_cell_width

    return psi_from_slater, C, triangular_lattice


def TriangularPiFluxAnsatz(Lx=2, Ly=3, spinfull=True, bc_MPS="finite",
                           chi_max_temfpy = 1000, flux=0.0, geometry="YC", particle_hole=True,
                           gs_manifold_index=0, norm_magz=0, monopole_Q=0):
    assert(Lx > 1)
    assert(norm_magz == 0)
    if not spinfull:
        particle_hole = False
    main_results_dir = "PiFluxAnsatzResults/"
    finite = (bc_MPS == "finite")
    model_type = model_type_dirac

    site = FermionSite(conserve='N')
    triangular_lat = GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, bc_MPS, geometry)

    results_dir = CreateGutzwillerCaseDir(main_results_dir, Lx, Ly, chi_max_temfpy, flux, geometry, bc_MPS,
                                          gs_manifold_index, model_type, norm_magz, monopole_Q=monopole_Q)

    pi_flux_model_params = {"lattice": triangular_lat, "flux":flux, "init_H_MPO": True, "monopole_Q":monopole_Q,
                            "particle_hole":particle_hole}
    pi_flux_model = MonopoleCondensatePiFluxModel(pi_flux_model_params)

    plot_lattice = False
    if plot_lattice:
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
        PlotModelHoppingsByPhase(pi_flux_model, ax_lat)
        fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')
        if local:
            plt.show()

    slater_trunc_par = {"chi_max": chi_max_temfpy, "svd_min": svd_min_slater_default, "degeneracy_tol": 1e-12}
    Lx_exact_C_infinite = 10
    assert(geometry == "XC" or geometry == "YC")
    flavors = 2 if spinfull else 1
    unitcell_width = flavors if geometry == "YC" else 2 * flavors
    psi_from_slater, C, lat = GetTriangularFluxSlaterMPS(Lx, Ly, spinfull, site, geometry,
                                                         slater_trunc_par, unitcell_width, bc_MPS,
                                                         gs_manifold_index, model_type,
                                                         flux=flux, particle_hole=particle_hole)
    print("psi slater normalization before canonization: ", psi_from_slater.overlap(psi_from_slater))
    psi_from_slater.canonical_form()
    if not finite:
        finite_bc_MPS = "finite"
        triangular_lattice_for_corr = GetPiFluxTriangularLattice(site, 2 * Lx_exact_C_infinite, Ly, spinfull,
                                                                 finite_bc_MPS, geometry)
        model_params = {"init_H_MPO": False, "monopole_Q": 0, "flux": flux,
                        "particle_hole": particle_hole, "lattice":triangular_lattice_for_corr}
        abs_magz = AbsMagzFromNormMagz(norm_magz, triangular_lattice_for_corr.N_sites // 2)
        C, _ = CalculateExactCMatrixForPiFlux(gs_manifold_index, model_params, model_type, abs_magz=abs_magz)

    sites1 = None
    sites2 = None
    if bc_MPS == "infinite":
        sites1 = np.arange(0, C.shape[0])
        sites2 = np.arange(0, C.shape[0])
    print("psi slater normalization: ", psi_from_slater.overlap(psi_from_slater))
    E_slater_mps = pi_flux_model.H_MPO.expectation_value(psi_from_slater)
    E_slater_mps_per_site = E_slater_mps
    avg_occupation = 0.5
    if bc_MPS == "finite":
        E_slater_mps_per_site /= (triangular_lat.N_sites * avg_occupation)
    else:
        E_slater_mps_per_site /= avg_occupation
    print("Energy per mode for mps-slater:", E_slater_mps_per_site)
    print("Energy per mode exact: ", PiFluxBandStructure((unitcell_width // flavors) * Ly, geometry=geometry,
                                                         tet=pi*flux))
    mps_slater_corr = psi_from_slater.correlation_function("Cd", "C", sites1=sites1, sites2=sites2)

    np.savetxt(results_dir + "correlations.csv", mps_slater_corr)

    assert(C.shape[0] == C.shape[1])
    X,Y = np.meshgrid(np.arange(0,C.shape[0]),np.arange(0,C.shape[0]))
    fig_slater_corr, ax_slater_corr = plt.subplots(figsize=(6, 5))
    fig_mps_slater_corr, ax_mps_slater_corr = plt.subplots(figsize=(6, 5))

    ImshowMatrix(ax_slater_corr, fig_slater_corr, X, Y, C, "i", "j")
    ImshowMatrix(ax_mps_slater_corr, fig_mps_slater_corr, X, Y, mps_slater_corr, "i", "j")

    with open(results_dir + 'psi_slater' + ".pkl", 'wb') as f:
        pickle.dump(psi_from_slater, f)

    fig_slater_corr.savefig(results_dir + "slater_exact_correlations.png", bbox_inches='tight')
    fig_mps_slater_corr.savefig(results_dir + "slater_mps_correlations.png", bbox_inches='tight')

    print("correlations max distance between slater-mps and exact slater: ", np.max(np.abs(mps_slater_corr - C)))
    print("slater norm: ", psi_from_slater.norm)


def RescaleMPSForGutzwiller(psi):
    # Since the norm may be below rounding error for large system, canonization can run into problems,
    # as the singular values get very close to zero in the QR decomposition. We rescale the matrices here,
    # to avoid this problem.
    for i in range(len(psi._B)):
        Bi = psi._B[i]
        physical_axis_shape = Bi.to_ndarray().shape[1]
        psi._B[i] = Bi.scale_axis(1.5 * np.ones(physical_axis_shape), axis=1)


def SpinonTriangularLatticeMeanFieldGutzwillerProjection(Ly, geometry, bc_MPS, gs_manifold_index, model_type,
                                                         Lx=6, chi_max=3000, flux=0.0, norm_magz=0.0, monopole_Q=0,
                                                         show_transverse_correlations=False,
                                                         iMPS_Lx_factor=Lx_short_factor_temfpy_iMPS,
                                                         svd_min=None):
    
    print(f"norm_magz: {norm_magz}")
    site = FermionSite(conserve='N')
    spin_site = SpinHalfSite(conserve='Sz')
    gutzwiller_results_dir = "MonopoleCondensateGutzwiller/"
    if local:
        gutzwiller_results_dir = glob_results_dir + gutzwiller_results_dir

    spinfull = True
    particle_hole = spinfull
    debug = False
    if local:
        results_dir = CreateGutzwillerCaseDir(gutzwiller_results_dir, Lx, Ly, chi_max, flux, geometry, bc_MPS,
                                              gs_manifold_index, model_type, norm_magz, monopole_Q, svd_min=svd_min)
    else:
        results_dir = "./"
    assert((bc_MPS == "finite") or (bc_MPS == "infinite"))
    finite = (bc_MPS == "finite")
    #fig_lat, ax_lat = plt.subplots()
    #lat = GetPiFluxTriangularLattice(site, Lx, Ly, True, bc_MPS, geometry)
    #PlotLattice(lat, ax_lat)
    #plt.show()

    if svd_min is None:
        svd_min = svd_min_slater_default
    slater_trunc_par = {"chi_max": chi_max, "svd_min": svd_min, "degeneracy_tol": 1e-12}
    assert (geometry == "XC" or geometry == "YC")
    flavors = 2 if spinfull else 1
    unitcell_width = flavors if geometry == "YC" else 2 * flavors

    psi_pi_flux, C, triangular_lat = GetTriangularFluxSlaterMPS(Lx, Ly, spinfull, site, geometry, slater_trunc_par,
                                                                unitcell_width, bc_MPS, gs_manifold_index,
                                                                model_type, flux=flux, particle_hole=particle_hole,
                                                                norm_magz=norm_magz, monopole_Q=monopole_Q,
                                                                iMPS_Lx_factor=iMPS_Lx_factor, results_dir=results_dir)
    # np.savetxt(results_dir + "C_slater.csv", C)

    if debug and finite:
        _triangular_lat = GetPiFluxTriangularLattice(site, Lx, Ly, spinfull, bc_MPS, geometry)
        pi_flux_model = MonopoleCondensatePiFluxModel({"lattice" : _triangular_lat,
                                                       "init_H_MPO" : True, "flux" : flux, "monopole_Q" : monopole_Q,
                                                       "particle_hole": particle_hole})

        fig, ax_lat = plt.subplots(figsize=(6,5))
        PlotModelHoppingsByPhase(pi_flux_model, ax_lat, plot_order=False)
        if local:
            plt.show()

        print(f"energy per mode of triangular pi flux gs = "
              f"{pi_flux_model.H_MPO.expectation_value(psi_pi_flux) / (0.5 * _triangular_lat.N_sites)}")

    psi_pi_flux.canonical_form()
        
    RescaleMPSForGutzwiller(psi_pi_flux)

    if particle_hole:
        if finite:
            gutz.abrikosov_ph(psi_pi_flux, inplace=True)
        else:
            iMPSAbrikosov(psi_pi_flux, norm_magz)

    else:
        assert(finite)
        gutz.abrikosov(psi_pi_flux, inplace=True)

    with open(results_dir + 'psi_gutzwiller' + ".pkl", 'wb') as f:
        pickle.dump(psi_pi_flux, f)

    assert(abs(psi_pi_flux.overlap(psi_pi_flux) - 1.0) < 1e-7)

    Lx, Ly = triangular_lat.Ls
    Lx_for_corr = Lx if finite else LxInfiniteMPSCorrelations(Lx, Ly)
    Ly_for_corr = Ly
    Nsites_for_iMPS_corr = Lx_for_corr * Ly_for_corr * (unitcell_width // flavors)
    if not particle_hole:
        return
    if finite:
        spin_corr_x = CalculateSpinSpinCorrelations(psi_pi_flux, transverse_correlations=show_transverse_correlations)
    else:
        spin_corr_x = CalculateSpinSpinCorrelations(psi_pi_flux, np.arange(0, Nsites_for_iMPS_corr),
                                                    np.arange(0, Nsites_for_iMPS_corr),
                                                    transverse_correlations=show_transverse_correlations)

    spin_lat = BuildTriangularLattice(Lx_for_corr, Ly_for_corr, spin_site, bc_MPS, geometry=geometry)
    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    PlotLattice(spin_lat, ax_lat)

    ks, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, spin_lat, n1=6, n2=6)

    fig_corr_k, ax_corr_k = plt.subplots(figsize=(6, 5))
    plot_structure_factor(ks, spin_corr_k, triangular_lat, ax_corr_k, mode='voronoi')
    YC_triangular_lat = BuildTriangularLattice(1, 1, spin_site, bc_MPS, geometry="YC")
    YC_triangular_lat.plot_brillouin_zone(ax_corr_k)
    ax_corr_k.set_title("Spin Correlations")

    special_points_structure_factor = calculateStructureFactorAtSpecialPoints(spin_lat, spin_corr_x)

    SaveSimulationOutput(results_dir, spin_corr_x, ks, spin_corr_k, fig_corr_k, fig_lat,
                         special_points_structure_factor=special_points_structure_factor, lattice=spin_lat)


def TryMonopoleModelHofstadter(output_dir, Lx, Ly, plot=True,
                               bc=('open', 'periodic'), bc_MPS="finite", flux=0.0):
    """
        spectrum of the monopole condensate model vs. monopole charge
    """
    fermion_site = FermionSite(conserve='N')
    particle_hole = False
    lat = BuildTriangularLattice(Lx, Ly, fermion_site, bc_MPS, bc=bc)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
    debug = False
    if debug:
        monopole_model = MonopoleCondensatePiFluxModel({"init_H_MPO": False, "lattice": lat, "flux":flux,
                                                        "monopole_Q": 1, "particle_hole": particle_hole})
        PrintCouplings(monopole_model)
        fig, ax = plt.subplots()
        #PlotLattice(lat, ax)
        PlotModelHoppingsByPhase(monopole_model, ax, plot_order=True)
        if local:
            plt.show()
        exit(1)

    monopole_Qs = np.arange(0, Lx*Ly)
    energies = np.zeros((Lx * Ly, monopole_Qs.shape[0]))
    for monopole_Q in monopole_Qs:
    # for monopole_Q in range(2):
        pi_flux_model_params = {"init_H_MPO": False, "lattice": lat,
                                "monopole_Q": monopole_Q, "flux":flux, "particle_hole": particle_hole}
        monopole_model = MonopoleCondensatePiFluxModel(pi_flux_model_params)
        H = CreateHamiltonianMatrixFromCouplingsList(monopole_model, lat.N_sites, dtype=np.complex128)
        if(len(lat.unit_cell_positions) == 2): # spinfull
            H_up = H[0::2, 0::2]
            H_down = H[1::2, 1::2]
            e_up, _ = eigh(H_up)
            e_down, _ = eigh(H_down)
            energies[:, monopole_Q] = e_up
            e = np.concatenate([e_up, e_down])
        else:
            e, _ = eigh(H)
            energies[:, monopole_Q] = e
        if plot:
            ax.plot(monopole_Q / (Lx*Ly) * np.ones(e.shape), e, "bo", markersize=1)

    if plot:
        ax.set_xlabel(r"$\phi / 2\pi$")
        ax.set_ylabel(r"$\epsilon$")
        fig.savefig(f"{output_dir}Hofstadter_Lx_{Lx}_Ly_{Ly}_x_{bc[0]}.png", bbox_inches='tight')
        if local:
            plt.show()

    return energies


def CheckMegnatizedPiFluxEnergyVsMonopoleDensity(Lx, Ly, norm_magz, bc, ax, fig, color="b", plot=True, save_dir=None):
    """
        Calculate the single particle spectrum of the spinless pi-flux model with finite monopole density,
        then determine occupation of energy levels from magnetization (the spinfull spectrum is just the spinless
        spectrum twice) and calculate the total energy per monopole charge in the given magnetization.
    """
    single_particle_energies = TryMonopoleModelHofstadter(None, Lx, Ly, plot=False, bc=bc)
    energies = np.zeros(single_particle_energies.shape[0])
    N_filling = Lx * Ly
    abs_magz = AbsMagzFromNormMagz(norm_magz, N_filling)
    N_up = int(np.ceil(N_filling // 2 + abs_magz))
    assert(N_up <= N_filling)
    N_down = N_filling - N_up
    assert((N_up + N_down) == N_filling)

    for monopole_Q in range(single_particle_energies.shape[1]):
        e_Q = single_particle_energies[:, monopole_Q]
        energy_up = np.sum(e_Q[0:N_up])
        energy_down = np.sum(e_Q[0:N_down])
        energies[monopole_Q] = (energy_up + energy_down) / N_filling

    if plot:
        ax.axvline(norm_magz / 2, linestyle="--", color='black', linewidth=0.6, alpha=0.75)
        ax.axvline(norm_magz, linestyle="--", color='black', linewidth=0.6, alpha=0.75)
        ax.plot(np.arange(0, energies.shape[0]) / energies.shape[0], energies, color+"o",
                label=f"x {bc[0]}")
        ax.set_xlabel(r"$\phi / 2\pi$")
        ax.set_ylabel(r"$e$")
        ax.set_title(f"Noninteracting Total Energy vs. Monopole Flux for Lx,Ly={Lx,Ly}")
        ax.legend()
        if save_dir is not None:
            fig.savefig(save_dir + f"/noninteracting_mon_energies_magz_{norm_magz:.4f}.png", bbox_inches='tight')
    return np.min(energies)


def CheckOptimalMonopoleStateEnergyVsMagnetization(Lx, Ly, bc=("open", "periodic")):
    norm_magzs = np.array([0., 2., 4., 6., Lx*Ly/6, Lx*Ly/3., Lx*Ly/2]) / (Lx * Ly)
    es = []
    for norm_magz in norm_magzs:
        e = CheckMegnatizedPiFluxEnergyVsMonopoleDensity(Lx, Ly, norm_magz, bc, None, None, plot=False)
        es.append(e)

    fig, ax = plt.subplots(figsize=(5,6))
    ax.plot(norm_magzs, es, "o")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$min_{Q}\{E\}$")
    if local:
        plt.show()


def DetermineSpinsOccupation(N_spins, H, e):
    eps_round = 1e-14
    Ef = e[N_spins - 1] + eps_round

    H_up = H[0::2, 0::2]
    H_down = H[1::2, 1::2]
    e_up, _ = eigh(H_up)
    e_down, _ = eigh(H_down)
    for i in range(e_up.shape[0]):
        assert (np.min(np.abs(e_up[i] - e)) < 3e-14)
        assert (np.min(np.abs(e_down[i] - e)) < 3e-14), f"e_down doesn't match full Hamiltonian e, err={np.min(np.abs(e_down[i] - e))} for i = {i}, e_down={e_down[i]}"

    N_up = int(np.argmax(e_up > Ef))
    N_down = int(np.argmax(e_down > Ef))
    assert((N_up + N_down) == N_spins)
    return N_up, N_down


def checkPiFluxFreeSpinCorrelations():
    Lx, Ly = 10, 10
    site = FermionSite('N')
    bc_exact = ("periodic", "periodic")
    geometry = "XC"
    flux = 1.0
    abs_magz = 0
    Q = 0
    spinfull_fermions = True
    gs_manifold_index = 0
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite",
                                            bc_exact, geometry, spinfull_fermions)
    pi_flux_parameters = {"init_H_MPO": False, "monopole_Q": Q, "flux": flux,
                          "particle_hole": spinfull_fermions, "lattice": triangular_lat}
    C_x_exact, _ = CalculateExactCMatrixForPiFlux(gs_manifold_index, pi_flux_parameters,
                                                  model_type_dirac, abs_magz=abs_magz)
    fig_exact, ax_exact = plt.subplots(figsize=(5, 6))
    spin_spin_corr = FreeFermionSpinCorrelations(C_x_exact)
    spin_triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite", bc_exact, geometry)

    special_bz_points = getSpecielBzPoints()
    M1, M2, M3 = special_bz_points["M1"], special_bz_points["M2"], special_bz_points["M3"]
    Ms = (M1, M2, M3)
    for i_M, M in enumerate(Ms):
        S_M = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat, Kx=np.array([M[0]]),
                                                   Ky=np.array([M[1]]))[-1]
        print(f"spin structure factor at M{i_M+1}: {S_M}")


    Kx, Ky, C_k = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat, new_implementation=False)
    Ky0_ind = np.argmin(np.abs(Ky[:, 0] - M1[1]))
    C_ky0_slice = C_k[Ky0_ind, :]
    fig_slice, ax_slice = plt.subplots(figsize=(5, 6))
    ax_slice.plot(Kx[0, :], np.abs(C_ky0_slice), "o")

    ImshowMatrix(ax_exact, fig_exact, Kx, Ky, np.abs(C_k))
    lat_for_bz = BuildTriangularLattice(1, 1, site, "finite", bc_exact, "YC")
    lat_for_bz.plot_brillouin_zone(ax_exact)
    plt.show()


def checkXC8SlaterCorrelations():
    geometry = "XC"
    flux = 0.0
    gs_manifold_index = 0
    Ly = 4
    temfpy_dir = (glob_results_dir + f"LocalGutzwillerResults/Dirac_infinite_Lx_2_Ly_{Ly}_" +
                  "chi_25000_flux_{flux}_{geometry}_gsindex_{gs_manifold_index}/")
    spinfull_fermions = True

    Lx = 10
    site = FermionSite('N')
    bc_exact = ("open", "periodic")
    abs_magz = 0
    Q = 0

    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite",
                                            bc_exact, geometry, spinfull_fermions)
    pi_flux_parameters = {"init_H_MPO": False, "monopole_Q": Q, "flux": flux,
                          "particle_hole": spinfull_fermions, "lattice":triangular_lat}
    C_x_exact, _ = CalculateExactCMatrixForPiFlux(gs_manifold_index, pi_flux_parameters,
                                                  model_type_dirac, abs_magz=abs_magz)

    fig_exact, ax_exact = plt.subplots(figsize=(5, 6))
    plot_spatial = False
    if plot_spatial:
        x, y = np.arange(0, C_x_exact.shape[0]), np.arange(0, C_x_exact.shape[1])
        X, Y = np.meshgrid(x, y)
        ImshowMatrix(ax_exact, fig_exact, X, Y, C_x_exact)
        C_x_temfpy = np.loadtxt(temfpy_dir + "slater_correlations.csv")
        fig_temfpy, ax_temfpy = plt.subplots(figsize=(5, 6))
        ImshowMatrix(ax_temfpy, fig_temfpy, X, Y, C_x_temfpy)
        #rel_diff = (C_x_exact - C_x_temfpy) / (np.abs(C_x_exact) + np.abs(C_x_temfpy))
        #rel_diff[np.abs(C_x_exact) < 1e-12] = 0.0
        #rel_diff[np.abs(C_x_temfpy) < 1e-12] = 0.0
        #ImshowMatrix(ax_diff, fig_diff, X, Y, rel_diff)
    else:
        spin_spin_corr = FreeFermionSpinCorrelations(C_x_exact)
        spin_triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite", bc_exact, geometry)
        ks, C_k = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat)
        plot_structure_factor(ks, C_k, triangular_lat, ax_exact)
        lat_for_bz = BuildTriangularLattice(1, 1, site, "finite", bc_exact, "YC")
        lat_for_bz.plot_brillouin_zone(ax_exact)

    plt.show()


def DebugMagnetizedIMPS():
    Lx_unitcell = 4
    Ly = 4
    # Lx_short, Lx_long = 400, 404
    Lx_short, Lx_long = 80, 84
    spin_degeneracy = 2
    # norm_magz = 2. / 16.
    norm_magz = 2. / (Lx_unitcell * Ly)
    N_short, N_long = spin_degeneracy * Lx_short * Ly, spin_degeneracy * Lx_long * Ly
    abs_magz_short, abs_magz_long = (AbsMagzFromNormMagz(norm_magz, N_short // 2),
                                     AbsMagzFromNormMagz(norm_magz, N_long // 2))
    N_up_short, N_up_long = N_short // 4 + abs_magz_short, N_long // 4 + abs_magz_long
    N_filling_short, N_filling_long = 2*N_up_short, 2*N_up_long
    # N_filling_short, N_filling_long = 1796, 1814

    parent_dir = code_dir + "debug_magnetized_iMPS/"
    e_short = np.loadtxt(parent_dir + f"e_Lx_{Lx_short}.txt")
    e_long = np.loadtxt(parent_dir + f"e_Lx_{Lx_long}.txt")
    v_short = np.loadtxt(parent_dir + f"v_Lx_{Lx_short}.txt", dtype=np.complex128)
    v_long = np.loadtxt(parent_dir + f"v_Lx_{Lx_long}.txt", dtype=np.complex128)

    v_short_filled = v_short[:, :N_filling_short]
    C_short =  v_short_filled @ HT(v_short_filled)

    v_long_filled = v_long[:, :N_filling_long]
    C_long = v_long_filled @ HT(v_long_filled)

    triangular_lat_short = BuildTriangularLattice(Lx_short, Ly, FermionSite('N'), "finite", spinfull_fermions=True)
    middle_site_mps_ind_short = triangular_lat_short.lat2mps_idx([Lx_short // 2, 0, 0])
    imps_unitcell = 4 * Ly * spin_degeneracy

    slater_trunc_par = {"chi_max": 1500, "svd_min": svd_min_slater_default, "degeneracy_tol": 1e-12}

    n_short = C_short.shape[0]
    n_long = C_long.shape[0]
    xy_arr = np.array([0.0,1.0])
    fig, ax = plt.subplots()
    ImshowMatrix(ax, fig, xy_arr, xy_arr, np.abs(C_short), xlabel="x", ylabel="y", title="Short System Correlations")
    fig, ax = plt.subplots()
    ImshowMatrix(ax, fig, xy_arr, xy_arr, np.abs(C_short), xlabel="x", ylabel="y", title="Long System Correlations")
    fig, ax = plt.subplots()
    subblock = "Right"
    if subblock == "Right":
        C_long = C_long[::-1, ::-1]
        C_short = C_short[::-1, ::-1]

    #ImshowMatrix(ax, fig, xy_arr, xy_arr,
    #             np.abs(C_long[0:n_short, 0:n_short] - C_short[0:n_short, 0:n_short]),
    #             xlabel="x", ylabel="y", title=f"Correlations Difference {subblock} Subblock")
    ImshowMatrix(ax, fig, xy_arr, xy_arr,
                 np.abs(C_long[0:n_short // 2, 0:n_short // 2]) - np.abs(C_short[0:n_short // 2, 0:n_short // 2]),
                 xlabel="x", ylabel="y", title=f"Correlations Difference {subblock} Subblock")

    # else:
    #     ImshowMatrix(ax, fig, xy_arr, xy_arr,
    #                  np.abs(C_short[(n_short - n_short//2):n_short, (n_short - n_short//2):n_short] -
    #                  C_long[(n_long - n_short//2):n_long, (n_long - n_short // 2):n_long]),
    #                  xlabel="x", ylabel="y", title=f"Correlations Difference {subblock} Subblock")
    plt.show()

    exit(0)
    psi_from_slater, error = slater.C_to_iMPS(C_short, C_long, slater_trunc_par,
                                              sites_per_cell=imps_unitcell,
                                              cut=middle_site_mps_ind_short)

    print("Here")

    #plt.plot(np.ones(400), e_short[e_short.shape[0]//2-190:e_short.shape[0]//2+210], "o", markersize=1)
    #plt.plot(2*np.ones(450), e_long[e_long.shape[0]//2-200:e_long.shape[0]//2+250], "o", markersize=1)

    # plt.show()
    exit(0)



if __name__ == "__main__":
    #SpinonTriangularLatticeMeanFieldGutzwillerProjection(6, "YC", "infinite", 0, model_type_dirac,
    #                                                     Lx=12, chi_max=100, flux=0.001, norm_magz=0.0277,
    #                                                     monopole_Q=0, show_transverse_correlations=True)
    #exit(0)

    # TestFreeFermionsSpinCorrelations()
    # checkXC8SlaterCorrelations()
    # checkPiFluxFreeSpinCorrelations()

    # DebugMagnetizedIMPS()

    # checkXC8SlaterCorrelations()

    # TryMonopoleModelHofstadter("./", 18, 18, bc=("open", "periodic"))

    #Lx = 6
    #Ly = 6
    #magz = 1./3.
    #magz = 0.0
    #monopole_Q = round(magz * (Lx * Ly / 2))
    #TryMonopoleModelHofstadter(output_dir, 18, 18, bc=("periodic", "periodic"))
    #TryMonopoleModelHofstadter(output_dir, 18, 18, bc=("open", "periodic"))

    # TriangularPiFluxAnsatz(2, 4, True, "infinite", 2000, 0.0, "XC")
    # TriangularPiFluxAnsatz(4, 3, False, "infinite", 100, 2.0, "XC")

    # TriangularPiFluxAnsatz(2, 2, True, "infinite", 4000, 1.0, "XC")

    #fig, ax = plt.subplots(figsize=(6, 5))
    #Lx = 24
    #Ly = 6
    #k = Lx * Ly / 12
    #norm_magz = 2. * k / (Lx * Ly)
    #CheckMegnatizedPiFluxEnergyVsMonopoleDensity(Lx, Ly, norm_magz, ("periodic", "periodic"), ax, fig, color="b")
    #CheckMegnatizedPiFluxEnergyVsMonopoleDensity(Lx, Ly, norm_magz, ("open", "periodic"), ax, fig, color="r",
    #                                             save_dir = project_dir + "Meetings/20_6_2026/")

    # fig.savefig(f"{output_dir}ener_vs_mon_dens_magz_{magz}.png", bbox_inches='tight')
    # plt.show()

    #CheckOptimalMonopoleStateEnergyVsMagnetization(24, 6)

    norm_magz_fac = 2.0
    Lx, Ly = 2, 4
    norm_magz = norm_magz_fac / (Lx * Ly)
    iMPS_Lx_factor = 100
    chi_max = 125
    abs_magz = AbsMagzFromNormMagz(norm_magz, Lx * Ly)

    monopole_Q_opt = int(norm_magz_fac // 2)
    monopole_Q = 1

    flux = 0.0
    bc_MPS = "infinite"
    SpinonTriangularLatticeMeanFieldGutzwillerProjection(Ly, "YC", bc_MPS, 0, model_type_dirac,
                                                         Lx=Lx, chi_max=chi_max, flux=flux, norm_magz=norm_magz,
                                                         monopole_Q=monopole_Q, show_transverse_correlations=True,
                                                         iMPS_Lx_factor=iMPS_Lx_factor)

    #####################
    # i = 1
    #for i in range(4):
    #    SpinonTriangularLatticeMeanFieldGutzwillerProjection(2, "XC", "finite", i, model_type_dirac,
    #                                                         Lx=40, chi_max=1000, flux=0.0)
    # TriangularPiFluxGutzwiller(3, "XC", "infinite", 0, Lx=2, chi_max=1000, flux=1.0)

    # TestZ2MeanFieldModel()
    #TriangularPiFluxGutzwiller(8, "YC", "infinite", 0, Lx=2, chi_max=2500, flux=0.0)

    # TriangularPiFluxGutzwiller(3, "XC", "finite", 0, Lx=200, chi_max=1000, flux=1.0)
    # TriangularPiFluxGutzwiller(3, "XC", "infinite", 0, Lx=2, chi_max=1000, flux=1.0)
