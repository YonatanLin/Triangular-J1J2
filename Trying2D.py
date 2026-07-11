from temfpy.slater import C_to_iMPS

from TryingTemfpy import local
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from TryingTemfpy import rc_params
from Noninteracting import PiFluxBandStructure
plt.rcParams.update(rc_params)

from temfpy import slater
import temfpy.gutzwiller as gutz
import tenpy
import numpy as np
from numpy import sin, cos, sqrt, pi
from tenpy.models.model import CouplingModel, CouplingMPOModel
from tenpy.networks.mps import MPS
import tenpy.linalg.np_conserved as npc
from tenpy.models import lattice
from tenpy.algorithms import dmrg
from numpy.linalg import eigh, det, inv
from tenpy.tools.misc import setup_logging
import pickle
from tenpy.networks.site import FermionSite, SpinHalfSite
from tenpy.models.spins import SpinModel
from pathlib import Path
import json
from tenpy import networks
from temfpy.utils import HT
from tenpy import MPSEnvironment
import threading
import time
if local:
    from SpinChirality import plot_scalar_spin_chirality

setup_logging(to_stdout="INFO")
if not local:
    print(f"num threads: {tenpy.tools.process.mkl_get_nthreads()}")


svd_min_slater_default = 5e-7
default_chi_max = 3000
default_dmrg_params = {'mixer': True, 'max_E_err': 1.0e-10, 'trunc_params': {'chi_max': default_chi_max, 'svd_min': 1.0e-7},
                    'combine': True, 'chi_list': {0: 50, 3: 100, 7: default_chi_max}, 'min_sweeps': 7, 'max_sweeps': 8,
                   'N_sweeps_check': 1}
code_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Code/"
meetings_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Meetings/"

Lx_short_factor_temfpy_iMPS = 50

model_type_dirac = "Dirac"
model_type_Z2 = "Z2"

pauli_x = np.array([[0., 1.], [1., 0.]])
pauli_y = np.array([[0., -1j], [1j, 0.]])
pauli_z = np.array([[1., 0.0], [0.0, -1.0]])
paulis = np.asarray([pauli_x, pauli_y, pauli_z])

def AbsMagzFromNormMagz(norm_magz, N_sites):
    print(f"norm_magz: {norm_magz}, N_sites: {N_sites}")
    magz_tot_doubled = int(round(norm_magz * N_sites))
    assert(magz_tot_doubled % 2 == 0)
    return magz_tot_doubled // 2


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


def ChangeChiInDMRGParams(dmrg_params, chi_max):
    dmrg_params["trunc_params"]["chi_max"] = chi_max
    dmrg_params["chi_list"] = {0: 50, 3: 100, 7: chi_max}


def CreateGutzwillerCaseDir(main_results_dir, Lx, Ly, chi_max, flux, geometry, bc_MPS,
                            gs_manifold_index, model_type, norm_magz, monopole_Q):
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    case_name = f"{bc_MPS}_Lx_{Lx}_Ly_{Ly}_chi_{chi_max}_flux_{flux}_{geometry}_gsindex_{gs_manifold_index}"

    if model_type is not None:
        case_name = f"{model_type}_" + case_name
    if float(norm_magz) > 1e-15:
        norm_magz_float = float(norm_magz)
        case_name += f"_magz_{norm_magz_float:.4f}"
    if monopole_Q is not None:
        case_name += f"_monQ_{monopole_Q}"

    case_name += "/"
    gutz_dir = main_results_dir + case_name
    Path(gutz_dir).mkdir(parents=True, exist_ok=True)
    return gutz_dir


def AddAndTrackCoupling(model, strength, u1, op1, u2, op2, dx, couplings_list, plus_hc=False,
                        flux=0.0):
    if abs(flux) > 1e-15:
        assert(plus_hc)
        assert(model.lat.boundary_conditions[1] == "periodic") #should be a cylinder which is periodic in the y direction
        strength_with_flux = model.coupling_strength_add_ext_flux(strength, dx, [0, flux])
        model.add_coupling(strength_with_flux, u1, op1, u2, op2, dx, plus_hc=plus_hc)
    else:
        model.add_coupling(strength, u1, op1, u2, op2, dx, plus_hc=plus_hc)

    couplings_list.append((u1, u2, dx))


def PrintCouplings(model, include_sites=None):
    couplings_list = model.all_coupling_terms().to_TermList()
    couplings_dict = {}
    for coupling in couplings_list:
        site_i = coupling[0][0][1]
        site_j = coupling[0][1][1]
        coupling_strength = coupling[1]
        operators = (coupling[0][0][0], coupling[0][1][0])

        if include_sites is None:
            print(f"i={site_i}, j={site_j}: tet={np.angle(coupling_strength) / pi}, {operators[0]}, {operators[1]}")
        else:
            if(site_i in include_sites or site_j in include_sites):
                dict_key = (site_i, site_j, operators[0], operators[1])
                assert (dict_key not in couplings_dict.keys())
                couplings_dict[dict_key] = coupling_strength
                # print(f"i={site_i}, j={site_j}: {coupling_strength}, {operators[0]}, {operators[1]}")
    return couplings_dict


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


def CalculateSpinSpinCorrelations(psi, sites1=None, sites2=None, inf_mps_unitcell_fac=3, transverse_correlations=False):
    if psi.bc == "infinite" and sites1 is None and sites2 is None:
        L = psi.L
        sites1 = np.arange(0, inf_mps_unitcell_fac*L)
        sites2 = np.arange(0, inf_mps_unitcell_fac*L)

    pm_corr = psi.correlation_function("Sp", "Sm", sites1=sites1, sites2=sites2)
    mp_corr = psi.correlation_function("Sm", "Sp", sites1=sites1, sites2=sites2)
    spin_corr_transverse = 0.5 * (pm_corr + mp_corr)
    if transverse_correlations:
        return spin_corr_transverse
    zz_corr = psi.correlation_function("Sz", "Sz", sites1=sites1, sites2=sites2)
    spin_corr = spin_corr_transverse + zz_corr
    return spin_corr


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


def _nearest_neighbor_dimer_bonds_by_site(psi, lat, sites, pair_key="nearest_neighbors"):
    bonds_by_site = {int(site): [] for site in sites}
    base_bonds = []
    for u1, u2, dx in lat.pairs[pair_key]:
        mps_sites_1, mps_sites_2, _, _ = lat.possible_couplings(u1, u2, dx)
        for site1, site2 in zip(mps_sites_1, mps_sites_2):
            site1 = int(site1)
            site2 = int(site2)
            base_bonds.append((site1, site2))

    if psi.bc == "infinite":
        min_requested_site = min(bonds_by_site)
        max_requested_site = max(bonds_by_site)
        min_base_site = min(min(bond) for bond in base_bonds)
        max_base_site = max(max(bond) for bond in base_bonds)
        min_shift = (min_requested_site - max_base_site) // psi.L - 1
        max_shift = (max_requested_site - min_base_site) // psi.L + 2
        shifts = [shift * psi.L for shift in range(min_shift, max_shift)]
    else:
        shifts = [0]

    for shift in shifts:
        for site1, site2 in base_bonds:
            shifted_bond = (site1 + shift, site2 + shift)
            if shifted_bond[0] in bonds_by_site:
                bonds_by_site[shifted_bond[0]].append(shifted_bond)
            if shifted_bond[1] in bonds_by_site:
                bonds_by_site[shifted_bond[1]].append(shifted_bond)
    return bonds_by_site


def _spin_dot_component_terms(site1, site2):
    return [
        (0.5, [("Sp", site1), ("Sm", site2)]),
        (0.5, [("Sm", site1), ("Sp", site2)]),
        (1.0, [("Sz", site1), ("Sz", site2)]),
    ]


def _site_array_for_correlations(psi, lat, sites, inf_mps_unitcell_fac):
    if sites is not None:
        return np.asarray(sites, dtype=int)
    if psi.bc == "infinite":
        return np.arange(0, inf_mps_unitcell_fac * psi.L)
    return np.arange(0, lat.N_sites)


def CalculateDimerDimerCorrelations(psi, lat, sites1=None, sites2=None, inf_mps_unitcell_fac=3,
                                    pair_key="nearest_neighbors"):
    """
    Calculate <D_i D_j>, where D_i = sum_{k nearest neighbor of i} S_i . S_k.

    The result is not connected, matching CalculateSpinSpinCorrelations.
    """
    sites1 = _site_array_for_correlations(psi, lat, sites1, inf_mps_unitcell_fac)
    sites2 = _site_array_for_correlations(psi, lat, sites2, inf_mps_unitcell_fac)
    if len(sites1) == 0 or len(sites2) == 0:
        return np.zeros((len(sites1), len(sites2)), dtype=float)

    min_requested_site = min(np.min(sites1), np.min(sites2))
    max_requested_site = max(np.max(sites1), np.max(sites2))
    if psi.bc != "infinite" and (min_requested_site < 0 or max_requested_site >= lat.N_sites):
        raise ValueError("lat.N_sites must cover all requested correlation sites")

    sites = np.unique(np.concatenate([sites1, sites2]))
    bonds_by_site = _nearest_neighbor_dimer_bonds_by_site(psi, lat, sites, pair_key)
    dimer_corr = np.zeros((len(sites1), len(sites2)), dtype=complex)
    for ind1, site1 in enumerate(sites1):
        for ind2, site2 in enumerate(sites2):
            if ind2 < ind1: # only consider unique pairs without reagrding order
                continue
            corr = 0.0
            for bond1 in bonds_by_site[int(site1)]:
                for strength1, term1 in _spin_dot_component_terms(*bond1):
                    for bond2 in bonds_by_site[int(site2)]:
                        for strength2, term2 in _spin_dot_component_terms(*bond2):
                            term1_expec = psi.expectation_value_term(term1)
                            term2_expec = psi.expectation_value_term(term2)
                            corr += strength1 * strength2 * (psi.expectation_value_term(term1 + term2))
            dimer_corr[ind1, ind2] = corr
            dimer_corr[ind2, ind1] = corr #symmetrize

    if np.max(np.abs(np.imag(dimer_corr))) < 1e-13:
        return np.real(dimer_corr)
    return dimer_corr


def TestDimerDimerCorrelations():
    site = SpinHalfSite(conserve=None)
    lat = BuildTriangularLattice(6, 6, site, "finite", bc=("open", "open"), geometry="YC")
    psi = MPS.from_product_state(
        lat.mps_sites(),
        ["up"] * lat.N_sites,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )

    dimer_corr = CalculateDimerDimerCorrelations(psi, lat)
    expected_dimer_corr = np.zeros((lat.N_sites, lat.N_sites))
    assert(dimer_corr.shape == expected_dimer_corr.shape)
    assert(np.max(np.abs(dimer_corr - expected_dimer_corr)) < 1e-13)

    partial_corr = CalculateDimerDimerCorrelations(psi, lat, sites1=[0, 3], sites2=[0, 3])
    assert(partial_corr.shape == (2, 2))
    assert(np.max(np.abs(partial_corr)) < 1e-13)
    print("success")


def PlotLattice(lat, ax, additional_couplings_to_plot=None, plot_nn_couplings=True, nnn_color="green",
                nnn_line_style="-", plot_order=True):
    #if add_nn_explicitly:
    #    lat.plot_coupling(ax, coupling=nn_couplings_list, linewidth=1.0)
    if plot_nn_couplings:
        lat.plot_coupling(ax, linewidth=1.0)

    if additional_couplings_to_plot is not None:
        lat.plot_coupling(ax, coupling=additional_couplings_to_plot, linewidth=0.5,
                      color="green", linestyle=nnn_line_style, wrap=True)
    if plot_order:
        lat.plot_order(ax)
    lat.plot_sites(ax)
    lat.plot_basis(ax, origin=-0.5 * (lat.basis[0] + lat.basis[1]))
    ax.set_aspect('equal')
    ax.set_xlim(-1)
    ax.set_ylim(-1)


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


def ImshowMatrix(ax, fig, X, Y, spin_corr_k, xlabel = r"$k_x$",
                ylabel=r"$k_y$", title=None, label=""):
    # if xlabel == r"$k_x / \pi$":
    #    norm = 1. / pi
    image = ax.imshow(
        np.real(spin_corr_k),
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='RdBu',
        aspect='auto', label=label
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(r"$S(\mathbf{k})$")
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0.0, color='white', linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color='white', linewidth=0.6, alpha=0.5)


def getShortestDistanceOnLatticeAxis(ax_coor_site1, ax_coor_site2, ax_bc, ax_L, finite_axis):
    coor_diff = ax_coor_site1 - ax_coor_site2
    coor_dist = abs(coor_diff)
    if ax_bc == "periodic" and finite_axis:
        assert(coor_dist < ax_L)
        dist_orig = coor_dist
        dist_opposite = ax_L - coor_dist
        if dist_orig < dist_opposite:
            return np.sign(coor_diff) * dist_orig
        else:
            return (-1) * np.sign(coor_diff) * dist_opposite
    return coor_diff


def StructureFactorPairPhases(coor_i, coor_j, bc_MPS, bcs, Ls, basis_vectors, unit_cell_pos, Kx, Ky):
    finite = (bc_MPS != "infinite")
    coor_diff_x = getShortestDistanceOnLatticeAxis(coor_i[0], coor_j[0], bcs[0], Ls[0], finite)
    coor_diff_y = getShortestDistanceOnLatticeAxis(coor_i[1], coor_j[1], bcs[1], Ls[1], True)

    coor_diff = np.array([coor_diff_x, coor_diff_y])
    r_ij = np.dot(coor_diff, basis_vectors) + (unit_cell_pos[coor_i[-1], :] - unit_cell_pos[coor_j[-1], :])

    phases = np.exp(-1j * (Kx * r_ij[0] + Ky * r_ij[1]))
    return phases


def ComputeMomentumSpaceStructureFactor(corr_x, lat, assert_realness=True, transform_expectation_value=False,
                                        Kx=None, Ky=None):
    if transform_expectation_value:
        assert(corr_x.ndim == 1)
    elif lat.bc_MPS != "infinite":
        assert (lat.N_sites == corr_x.shape[0] and lat.N_sites==corr_x.shape[1])

    corr_x_shape = corr_x.shape
    if Kx is None:
        assert(Ky is None), "need to specify momentum along both axes"
        kx = ky = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        Kx, Ky = np.meshgrid(kx, ky)
    bcs = lat.boundary_conditions
    Ls = lat.Ls
    corr_k = np.zeros(Kx.shape, dtype=complex)
    bc_MPS = lat.bc_MPS
    unit_cell_pos = lat.unit_cell_positions
    basis_vectors = np.asarray(lat.basis, dtype=float)

    for i in range(corr_x_shape[0]):
        coor_i = lat.mps2lat_idx(i)
        if transform_expectation_value:
                coor_center = [(Ls[0]-1)/2., (Ls[1]-1)/2., len(unit_cell_pos)//2]
                phases = StructureFactorPairPhases(coor_i, coor_center, bc_MPS, bcs, Ls, basis_vectors,
                                                   unit_cell_pos, Kx, Ky)
                corr_k += corr_x[i] * phases
        else:
            for j in range(corr_x_shape[1]):
                coor_j = lat.mps2lat_idx(j)
                phases = StructureFactorPairPhases(coor_i, coor_j, bc_MPS, bcs, Ls, basis_vectors,
                                                   unit_cell_pos, Kx, Ky)
                corr_k += corr_x[i, j] * phases
                
    corr_k = corr_k / lat.N_sites
    if assert_realness:
        assert(np.max(np.abs(np.imag(corr_k))) < 1e-13)

    return Kx, Ky, corr_k


def PlotSquareLatticeStructureFactor(Lx=6, Ly=6):
    site = SpinHalfSite(conserve=None)
    square_lat = lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=['open', 'open'])

    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    square_lat.plot_order(ax_lat)
    square_lat.plot_coupling(ax_lat)
    square_lat.plot_sites(ax_lat)
    square_lat.plot_basis(ax_lat)

    product_state = []
    for i in range(square_lat.N_sites):
            lat_ind = square_lat.mps2lat_idx(i)
            x = lat_ind[0]
            y = lat_ind[1]
            product_state.append("up" if (x + y) % 2 == 0 else "down")

    psi = MPS.from_product_state(
        square_lat.mps_sites(),
        product_state,
        bc=square_lat.bc_MPS,
        unit_cell_width=square_lat.mps_unit_cell_width,
    )
    psi.canonical_form()

    spin_corr_x = CalculateSpinSpinCorrelations(psi)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, square_lat)

    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
    ImshowMatrix(ax, fig, Kx, Ky, spin_corr_k, title=title)

    square_lat.plot_brillouin_zone(ax)

    fig.tight_layout()
    if local:
            plt.show()


def Generate120DegOrderedState(lat=None, Lx=None, Ly=None, plot=False):
    if lat == None:
        site = SpinHalfSite(conserve=None)
        Lx = 9
        Ly = 9
        triangular_lat = lattice.Triangular(Lx=Lx, Ly=Ly, site=site, bc=['periodic', 'open'])
        lat = triangular_lat
    else:
        site = lat.mps_sites()[0]
    basis = lat.basis
    unitcell_pos = lat.unit_cell_positions

    aligned_with_x = (basis[0][1] == 0.0 or basis[1][1] == 0.0)
    aligned_with_y = (basis[0][0] == 0.0 or basis[1][0] == 0.0)
    assert(aligned_with_x or aligned_with_y)

    psi = MPS.from_product_state(
        lat.mps_sites(),
        ["up"] * lat.N_sites,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )

    rot_120_angle = 2 * pi / 3
    debug = abs(rot_120_angle - pi) < 1e-15
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    I2 = np.eye(2)
    rotation_op_120 = cos(rot_120_angle / 2.) * I2 - 1j * sin(rot_120_angle / 2.) * pauli_x
    rotation_op_240 = cos(rot_120_angle) * I2 - 1j * sin(rot_120_angle) * pauli_x
    site.add_op("rot_once", rotation_op_120)
    site.add_op("rot_twice", rotation_op_240)

    for i in range(lat.N_sites):
        lat_ind = lat.mps2lat_idx(i)
        if aligned_with_x:
            column_cor = lat_ind[0] * basis[0][0] + lat_ind[1] * basis[1][0] + unitcell_pos[lat_ind[2], :][0]
        else:
            column_cor = lat_ind[0] * basis[0][1] + lat_ind[1] * basis[1][1] + unitcell_pos[lat_ind[2], :][1]
        sublattice_ind = int((2 * column_cor) % 3)
        if sublattice_ind == 1:
            psi.apply_local_op(i, "rot_once")
        elif sublattice_ind == 2:
            psi.apply_local_op(i, "rot_twice")

    psi.canonical_form()

    if debug and Lx == 4 and Ly == 4:
        magz = psi.expectation_value("Sz")
        magz_expected = 0.5 * np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1])
        assert ((magz - magz_expected) == 0.0).all()
    if plot:
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        spin_corr_x = CalculateSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat)
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k)
        lat.plot_brillouin_zone(ax_corr)
        if local:
            plt.show()

    return psi


def GenerateStripeOrderedState(lat, plot=False):
    basis = lat.basis
    aligned_with_x = (basis[0][1] == 0.0 or basis[1][1] == 0.0)
    aligned_with_y = (basis[0][0] == 0.0 or basis[1][0] == 0.0)
    assert(aligned_with_x or aligned_with_y)

    product_state = []
    for i in range(lat.N_sites):
        lat_ind = lat.mps2lat_idx(i)
        if aligned_with_x:
            column_ind = lat_ind[1]
        else:
            column_ind = lat_ind[0]
        product_state.append("up" if column_ind % 2 == 1 else "down")

    psi = MPS.from_product_state(
        lat.mps_sites(),
        product_state,
        bc=lat.bc_MPS,
        unit_cell_width=lat.mps_unit_cell_width,
    )
    psi.canonical_form()

    if plot:
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
        lat.plot_order(ax_lat)
        lat.plot_coupling(ax_lat)
        lat.plot_sites(ax_lat)
        if local:
            plt.show()

    return psi


def TestCorrelationsWithNontrivialUnitCell(Lx, Ly, state="120", geometry="YC"):
    site = SpinHalfSite(conserve=None)

    unit_cell_spin_lat = None
    basis = None

    if geometry == "YC":
        unit_cell_spin_lat = [[0.0, 0.0], [1.0, 0.0]]
        basis = [[2.0, 0.0], [0.5, sqrt(3) / 2.]]
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite", unit_cell=unit_cell_spin_lat,
                                            basis=basis, geometry=geometry)

    if state == "120":
        spin_state = Generate120DegOrderedState(triangular_lat, Lx, Ly, False)
    elif state == "stripe":
        spin_state = GenerateStripeOrderedState(triangular_lat, False)
    else:
        ValueError("Illegal spin state option")
        return

    print(spin_state.expectation_value("Sz"))

    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    PlotLattice(triangular_lat, ax_lat, plot_order=True)
    if local:
            plt.show()

    spin_corr_x = CalculateSpinSpinCorrelations(spin_state)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lat)
    fig, ax = plt.subplots(figsize=(6, 5))
    ImshowMatrix(ax, fig, Kx, Ky, np.abs(spin_corr_k))
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite")
    triangular_lat.plot_brillouin_zone(ax)
    if local:
            plt.show()
    return spin_state


def RunDMRG(model, psi_init, dmrg_params=default_dmrg_params,
            plot_convergence=True, print_final_results=False,
            expected_energy=None, results_dir="", energies_fig_title=None):
    E_initial = model.H_MPO.expectation_value(psi_init)
    print("initial energy before dmrg: ", E_initial)
    info = dmrg.run(psi_init, model, dmrg_params)
    E_final = info['E']
    stats = info['sweep_statistics']
    energies = stats['E']
    sweeps = stats['sweep']
    if print_final_results:
        print(f'E = {E_final:.13f}')
        print('final bond dimensions: ', psi_init.chi)
        np.savetxt("Energies.txt", np.array([E_initial] + energies))

    if plot_convergence:
        fig,ax = plt.subplots(figsize=(6, 5))
        ax.plot([-1] + sweeps, [E_initial] + energies, "o")
        ax.set_title("DMRG Sweeps Energies")
        ax.set_xlabel("sweep")
        ax.set_ylabel("E")
        if expected_energy is not None:
            ax.axhline(expected_energy, color="red", linewidth=0.6, alpha=0.5, linestyle="dashed")
        if energies_fig_title is not None:
            fig.savefig(results_dir + energies_fig_title, bbox_inches='tight')

        if local:
            plt.show()
    return E_initial, E_final, sweeps, energies, info


def TestSquareLattice(Lx=5, Ly=5, bc=('open', 'open'), J2s=[0.0],
                      bc_MPS="finite"):
    for J2 in J2s:
        site = SpinHalfSite(conserve='Sz')
        square_lat = lattice.Square(Lx=Lx, Ly=Ly, site=site, bc=list(bc), bc_MPS=bc_MPS)
        J = 1.0

        J1J2_model = SpinModel({"lattice": square_lat, "Jx": J, "Jy": J, "Jz": J})

        nnn_couplings_list = []
        if abs(J2) > 0.0:
            for basis_vec in ([-1, -1], [-1, 1], [1, -1], [1, 1]):
                AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sp", 0, "Sm", basis_vec,
                                    nnn_couplings_list)
                AddAndTrackCoupling(J1J2_model, 0.5 * J2, 0, "Sm", 0, "Sp", basis_vec,
                                    nnn_couplings_list)
                AddAndTrackCoupling(J1J2_model, J2, 0, "Sz", 0, "Sz", basis_vec,
                                    nnn_couplings_list)

        J1J2_model.init_H_from_terms()

        results_dir = "SquareLatticeJ1J2/"
        fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
        PlotLattice(square_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)
        fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')

        psi = MPS.from_product_state(
            square_lat.mps_sites(),
            ["up"] * (square_lat.N_sites // 2) + ["down"] * (square_lat.N_sites - square_lat.N_sites // 2),
            bc=square_lat.bc_MPS,
            unit_cell_width=square_lat.mps_unit_cell_width,
        )
        psi.canonical_form()

        initial_energy = J1J2_model.H_MPO.expectation_value(psi)
        initial_energy_per_site = initial_energy
        if bc_MPS == "finite":
            initial_energy_per_site /= square_lat.N_sites
        #if J2 == 0.0:
        #    assert(initial_energy_per_site == 0.5 * (1. - 1. / L))
        print("Initial energy per site: ", initial_energy_per_site)
        energies_fig_title = "energy_convergence_J2_" + str(J2) + ".png"
        E_initial, E_gs, sweeps, energies, info = RunDMRG(J1J2_model, psi,
                                                          print_final_results=True, results_dir=results_dir,
                                                          energies_fig_title=energies_fig_title)
        energy_per_site = E_gs

        with open("Energy_J2_"+str(J2)+".txt", "w") as f:
            f.write(f"Energy per site = {energy_per_site:.13f}")

        if bc_MPS == "finite":
            energy_per_site /= square_lat.N_sites
        print(f"Energy per site = {energy_per_site:.13f}")

        with open(results_dir + 'psi_gs_J2_'+str(J2)+".pkl", 'wb') as f:
            pickle.dump(psi, f)

        spin_corr = CalculateSpinSpinCorrelations(psi)
        Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr, square_lat,
                                                                  assert_realness=False)

        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        title = f"Spin structure factor"
        ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k, title=title)
        square_lat.plot_brillouin_zone(ax_corr)
        fig_corr.tight_layout()

        fig_corr.savefig("SquareLatticeJ1J2/spin_correlations_J2_"+str(J2)+".png", bbox_inches='tight')

        if local:
            plt.show()
    # return energy_per_site, psi, square_lat


class TriangularXC(tenpy.models.lattice.Lattice):
    dim = 2
    Lu = 2

    def __init__(self, Lx, Ly, sites, spinfull_fermions, **kwargs):

        expected_number = 4 if spinfull_fermions else 2

        try:
            iter(sites)
        except TypeError:
            sites = [sites] * expected_number
        if len(sites) != expected_number:
            raise ValueError(
                "need to specify a single site or exactly {0:d}, got {1:d}".format(expected_number, len(sites)))

        basis = np.array(([1.0, 0.], [0., np.sqrt(3)]))
        delta = np.array([0.5, np.sqrt(3)/2.])

        if spinfull_fermions:
            pos = (-delta / 2 - 0.1 * delta, -delta / 2 + 0.1 * delta, delta / 2 - 0.1 * delta, delta / 2 + 0.1 * delta)
        else:
            pos = (-delta / 2., delta / 2)

        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        NN = []
        nNN = []
        for i in range(len(pos)//2):
            site1 = i
            site2 = i + len(pos)//2
            NN += [(site1, site2, np.array([0, 0])), (site2, site1, np.array([0, 1])), (site1, site1, np.array([1, 0])),
                  (site2, site2, np.array([1, 0])), (site2, site1, np.array([1, 0])), (site2, site1, np.array([1, 1]))]
            nNN += [(site1, site1, np.array([0, 1])), (site2, site2, np.array([0, 1])), (site1, site2, np.array([1, -1])),
                   (site2, site1, np.array([2, 0])), (site1, site2, np.array([1, 0])), (site2, site1, np.array([2, 1]))]
        kwargs.setdefault('pairs', {})
        kwargs['pairs'].setdefault('nearest_neighbors', NN)
        kwargs['pairs'].setdefault('next_nearest_neighbors', nNN)
        kwargs.setdefault('bc', ("open", "periodic")) #cylinder
        tenpy.models.lattice.Lattice.__init__(self, [Lx, Ly], sites, **kwargs)


def BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc = ('periodic', 'periodic'), geometry="YC",
                           spinfull_fermions=False):
    if geometry == "YC":
        basis = [[1.0, 0.0], [0.5, sqrt(3) / 2.]]

        unit_cell = [[0.0, 0.0]]
        if spinfull_fermions:
            unit_cell = [[-0.1, 0.0], [0.1, 0.0]]

        nearest_neighbors = []
        next_nearest_neighbors = []
        num_flavors = 2 if spinfull_fermions else 1
        for i in range(num_flavors):
            nearest_neighbors += [[i, i, [1, 0]], [i, i, [0, 1]],
                                 [i, i, [-1, 1]]]
            next_nearest_neighbors += [[i, i, [1, 1]], [i, i, [-1, 2]],
                                       [i, i, [-2, 1]]]

        triangular_lat = lattice.Lattice([Lx, Ly], [site]*len(unit_cell), basis=basis,
                                        positions=unit_cell, bc=bc, pairs={'nearest_neighbors': nearest_neighbors,
                                                                           'next_nearest_neighbors': next_nearest_neighbors},
                                        bc_MPS=bc_MPS)

        return triangular_lat
    elif geometry == "XC":
        triangular_lat = TriangularXC(Lx, Ly, site, spinfull_fermions, bc_MPS=bc_MPS, bc=bc)
        return triangular_lat
    else:
        raise ValueError("unrecognized geometry")


def initialStateFromFile(initial_state):
    return "from_file" in initial_state


def GetTriangularLatticeInitialState(initial_state, triangular_lat, initial_psi_dir, abs_magz):
    Lx = triangular_lat.Ls[0]
    Ly = triangular_lat.Ls[1]
    N_sites = triangular_lat.N_sites
    assert(abs_magz <= N_sites // 2), "normalized magnetization cannot exceed 1"
    if abs_magz > 0:
        assert(initial_state == "Random")
    if initial_state == "Random":
        N_up = N_sites // 2 + abs_magz
        N_down = N_sites - N_up
        down_indices = np.random.choice(np.arange(0, N_sites), N_down, replace=False)
        print(down_indices)
        down_indices = np.array(down_indices)
        product_state = ["up"] * N_sites
        for down_ind in down_indices:
            product_state[down_ind] = "down"
        psi = MPS.from_product_state(
            triangular_lat.mps_sites(),
            product_state,
            bc=triangular_lat.bc_MPS,
            unit_cell_width=triangular_lat.mps_unit_cell_width,
        )

    elif initial_state == "120":
        psi = Generate120DegOrderedState(lat=triangular_lat, Lx=Lx, Ly=Ly)

    elif initial_state == "stripe":
        psi = GenerateStripeOrderedState(lat=triangular_lat)

    elif initialStateFromFile(initial_state):
        initial_psi_path = initial_psi_dir + 'psi_gs.pkl'
        print(f"loading initial state: {initial_psi_path}")
        with open(initial_psi_path, 'rb') as psi_load:
            psi = pickle.load(psi_load)
        #psi.canonical_form()

    else:
        raise ValueError("unrecognized initial state")
        psi = None
    return psi


def TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, initial_state, conserve, J2, geometry, chi, max_sweeps, norm_magz):
    bc_string = ""
    for bc_ax in bc:
        if bc_ax == "periodic":
            bc_string += "p"
        else:
            assert(bc_ax == "open")
            bc_string += "o"

    geometry_dir = f"Lx_{Lx}_Ly_{Ly}_bc_{bc_string}_{geometry}/"
    params_dir = f"{bc_MPS}_init_{initial_state}_conserve_{conserve}_J2_{J2}"
    if chi is not None:
        params_dir += f"_chi_{chi}"
    if max_sweeps is not None:
        params_dir += f"_maxsweeps_{max_sweeps}"
    if float(norm_magz) > 1e-15:
        norm_magz_float = float(norm_magz)
        params_dir += f"_magz_{norm_magz_float:.4f}"
    return geometry_dir, params_dir + "/"


def CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc, bc_MPS, initial_state, conserve, J2, geometry,
                            chi_max=None, max_sweeps=None, norm_magz=0.0):
    Path(main_results_dir).mkdir(parents=True, exist_ok=True)
    geometry_dir, params_dir = TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, initial_state, conserve, J2, geometry,
                                                         chi_max, max_sweeps, norm_magz)
    results_dir = main_results_dir + geometry_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_dir += params_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir


def GenerateJ1J2SpinTriangularModel(J2, triangular_lat):
    J1 = 1.0
    assert(len(triangular_lat.pairs["nearest_neighbors"]) > 0 and
           len(triangular_lat.pairs["next_nearest_neighbors"]) > 0)
    J1J2_model = SpinModel({"lattice": triangular_lat, "Jx": J1, "Jy": J1, "Jz": J1})
    J1J2_model.manually_call_init_H = True

    nnn_couplings_list = []
    if abs(J2) > 0.0:
        for u1, u2, dx in triangular_lat.pairs['next_nearest_neighbors']:
            AddAndTrackCoupling(J1J2_model, 0.5 * J2, u1, "Sp", u2, "Sm", dx,
                                nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, 0.5 * J2, u1, "Sm", u2, "Sp", dx,
                                nnn_couplings_list)
            AddAndTrackCoupling(J1J2_model, J2, u1, "Sz", u2, "Sz", dx,
                                nnn_couplings_list)
    J1J2_model.init_H_from_terms()
    return J1J2_model, nnn_couplings_list


def calculateGutzwillerEnergyTriangularJ1J2(gutz_results_dir, Lx, Ly, chi, flux, bc_MPS, J2, bc, geometry,
                                            gs_manifold_index, norm_magz, monopole_Q, reorder_lattice=False, model_type=None):
    psi_path = CreateGutzwillerCaseDir(gutz_results_dir, Lx, Ly, chi, flux, geometry,
                                       bc_MPS, gs_manifold_index, model_type, norm_magz, monopole_Q) + "/psi_gutzwiller.pkl"

    print(f"calculating energy for MPS in path {psi_path} with triangular J1J2 model for J2={J2}")
    site = SpinHalfSite(conserve="Sz")

    with open(psi_path, 'rb') as f:
        psi = pickle.load(f)
    
    finite = (bc_MPS == "finite")
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc, geometry=geometry)
    J1J2_model, _ = GenerateJ1J2SpinTriangularModel(J2, triangular_lat)
    if reorder_lattice:
        exit("reorder lattice not supported anymore")

    print(triangular_lat.N_sites)
    E = J1J2_model.H_MPO.expectation_value(psi)
    if bc_MPS == "finite":
        E /= triangular_lat.N_sites
    print("Energy: ", E)
    return E


def SaveSimulationOutput(results_dir, spin_corr_x, Kx, Ky, spin_corr_k, fig_corr_k, fig_lat):
    np.savetxt(results_dir + "spin_corr_x.csv", spin_corr_x)
    np.savetxt(results_dir + "spin_corr_k.csv", spin_corr_k)
    np.savetxt(results_dir + "Kx.csv", Kx)
    np.savetxt(results_dir + "Ky.csv", Ky)
    fig_corr_k.savefig(results_dir + "momentum_space_correlations.png", bbox_inches='tight')
    fig_lat.savefig(results_dir + "lattice.png", bbox_inches='tight')


def TriangularJ1J2DMRG(Lx, Ly, bc, bc_MPS, conserve=True, initial_state="Random", J2=0.0, geometry="YC",
                       chi_max=None, initial_psi_dir=None, max_sweeps=None, norm_magz=0.0):
    if isinstance(bc, str):
        bc_parsed = bc.split("-")
        bc = (bc_parsed[0], bc_parsed[1])

    if local:
        main_results_dir = "LocalJ1J2TriangularDMRGResults/"
        results_dir = CreateTriangularCaseDir(main_results_dir, Lx, Ly, bc, bc_MPS, initial_state, conserve, J2,
                                              geometry)
    else:
        results_dir = "./"

    if conserve:
        site = SpinHalfSite(conserve='Sz')
    else:
        site = SpinHalfSite(conserve=None)

    triangular_lat = BuildTriangularLattice(Lx, Ly, site, bc_MPS, bc=bc, geometry=geometry)
    abs_magz = AbsMagzFromNormMagz(norm_magz, triangular_lat.N_sites)
    center_site_mps_index = triangular_lat.lat2mps_idx([Lx // 2, Ly // 2, 0])
    print("center site mps index: ", center_site_mps_index)

    J1J2_model, nnn_couplings_list = GenerateJ1J2SpinTriangularModel(J2, triangular_lat)

    fig_lat, ax_lat = plt.subplots(figsize=(6, 5))
    PlotLattice(triangular_lat, ax_lat, additional_couplings_to_plot=nnn_couplings_list)

    print_couplings = False
    if print_couplings:
        PrintCouplings(J1J2_model)
        if local:
            plt.show()
        exit(1)

    if local:
            plt.show()

    psi = GetTriangularLatticeInitialState(initial_state, triangular_lat, initial_psi_dir, abs_magz)
   
    dmrg_params = default_dmrg_params

    if chi_max is not None:
        ChangeChiInDMRGParams(dmrg_params, chi_max)
    chi_max = dmrg_params['trunc_params']['chi_max']

    if max_sweeps is not None:
        dmrg_params['max_sweeps'] = max_sweeps

    if initialStateFromFile(initial_state):
        chi_max_psi = np.max(psi.chi)
        chi_max = int(max(chi_max_psi, chi_max))

        ChangeChiInDMRGParams(dmrg_params, chi_max)
        dmrg_params['chi_list'] = {0:chi_max}
        dmrg_params['min_sweeps'] = 3

    with open(results_dir + "dmrg_params.json", "w") as f:
        json.dump(dmrg_params, f, indent=4)

    if initialStateFromFile(initial_state):
        with open("psi_initial_dir.txt", 'w') as f:
            f.write(initial_psi_dir)
    else:
        with open(results_dir + 'psi_initial' + ".pkl", 'wb') as f:
            pickle.dump(psi, f)

    RunDMRG(J1J2_model, psi, dmrg_params=dmrg_params, print_final_results=True, results_dir=results_dir,
            energies_fig_title="energies.png")

    psi.canonical_form()
    E_final = J1J2_model.H_MPO.expectation_value(psi)
    print(f"Energy calculated from full MPO: {E_final}")
    with open(results_dir + 'psi_gs' + ".pkl", 'wb') as f:
        pickle.dump(psi, f)

    sites1, sites2 = None, None
    lat_for_corr = triangular_lat
    if bc_MPS == "infinite":
        Lx_large = 10 * Lx
        sites1 = np.arange(0, Ly * Lx_large)
        sites2 = np.arange(0, Ly * Lx_large)
        lat_for_corr = BuildTriangularLattice(Lx_large, Ly, site, bc_MPS, bc=bc, geometry=geometry)

    spin_corr_x = CalculateSpinSpinCorrelations(psi, sites1, sites2)

    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, lat_for_corr,
                                                              assert_realness=False)

    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor on a {Lx}x{Ly} square lattice"
    ImshowMatrix(ax_corr, fig_corr, Kx, Ky, spin_corr_k, title=title)
    lat_for_corr.plot_brillouin_zone(ax_corr)
    SaveSimulationOutput(results_dir, spin_corr_x, Kx, Ky, spin_corr_k, fig_corr, fig_lat)


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


def TestZ2MeanFieldModel():
    triangular_lat = BuildTriangularLattice(3, 5, FermionSite(conserve="N"), "finite",
                                            bc=("open", "open"), spinfull_fermions=True)
    zeta = 5.0
    x_nn_hopping = 1.0
    y_nn_hopping = 2.0
    nnn_hopping = 3.0
    model_params = {"mu": 1.0, "zeta": 5.0, "init_H_MPO": False, "lattice": triangular_lat}
    model_params["hoppings"] = {(1, 0) : x_nn_hopping, (0, 1) : y_nn_hopping, (-1, 1) : x_nn_hopping,
                                (-1, 2): nnn_hopping, (1, 1): nnn_hopping}
    model_params["pairings"] = model_params["hoppings"]
    z2_model = Z2MeanFieldModel(model_params)

    center_sites = [14, 15]
    couplings_dict = PrintCouplings(z2_model, include_sites=center_sites)
    spindown_sites_coupled_to_spindown_center = [(13, (-1)*y_nn_hopping), (17, y_nn_hopping),
                                                 (5, -1.0*x_nn_hopping), (25, -1.0*x_nn_hopping), (7, -1.0*x_nn_hopping),
                                                 (23, -1.0*x_nn_hopping), (9, nnn_hopping), (21, nnn_hopping),
                                                 (3, (-1) * nnn_hopping), (27, (-1)*nnn_hopping)]

    expected_couplings_dict = AddCouplingsToZ2ModelDict(center_sites, spindown_sites_coupled_to_spindown_center, zeta)
    TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    tests_sites = [22, 23]
    couplings_dict = PrintCouplings(z2_model, include_sites=tests_sites)
    couplings_to_tests_sites = [(13, (-1) * x_nn_hopping), (15, (-1) * x_nn_hopping),
                                (25, (-1)*y_nn_hopping), (21, y_nn_hopping), (11, (-1)*nnn_hopping),
                                (17, (-1)*nnn_hopping)]
    expected_couplings_dict = AddCouplingsToZ2ModelDict(tests_sites, couplings_to_tests_sites, zeta)
    TestDictsAreCompatible(couplings_dict, expected_couplings_dict)

    print(z2_model.all_onsite_terms().to_TermList())
    CreateHamiltonianMatrixFromCouplingsList(z2_model, 30)
    fig, ax = plt.subplots()
    # PlotModelHoppingsByPhase(z2_model, ax)
    PlotLattice(triangular_lat, ax)
    plt.show()


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

        psi_from_slater, error = slater.C_to_iMPS(C_short, C_long, slater_trunc_par,
                                                  sites_per_cell=imps_unitcell,
                                                  cut=middle_site_mps_ind_short)

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


def ExplicitMPSNorm(mps):
    tensors = mps._B if (mps.bc == "finite") else [mps.get_B(i) for i in range(mps.L)]
    first_tensor = tensors[0].to_ndarray()
    norm_transfer = np.eye(first_tensor.shape[0], dtype=np.result_type(first_tensor.dtype, np.complex128))
    for B in tensors:
        B_arr = B.to_ndarray()
        ket_contracted = np.tensordot(norm_transfer, B_arr, axes=([1], [0]))
        norm_transfer = np.tensordot(B_arr.conj(), ket_contracted, axes=([0, 1], [0, 1]))

    if mps.bc == "finite":
        assert norm_transfer.shape == (1, 1)
        return norm_transfer[0, 0]
    return norm_transfer


def calculateOverlapBetweenGutzwillerAndDMRG(dmrg_dir, gutzwiller_dir,
                                             psi_gutz_fname='psi_gutzwiller.pkl'):

    with open(dmrg_dir + 'psi_gs.pkl', 'rb') as f_dmrg:
        psi_dmrg = pickle.load(f_dmrg)
    with open(gutzwiller_dir + psi_gutz_fname, 'rb') as f_gutz:
        psi_gutz = pickle.load(f_gutz)
    
    #compressed_chi = 2000
    #max_trunc_err_dmrg = psi_dmrg.compress({"compression_method":'SVD', "trunc_params":{"chi_min":compressed_chi, "chi_max":compressed_chi}})
    #max_trunc_err_gutz = psi_gutz.compress({"compression_method":'SVD', "trunc_params":{"chi_min":compressed_chi, "chi_max":compressed_chi}})
    #print(f"compressed dmrg psi with max truncation error {max_trunc_err_dmrg} and gutzwiller psi with max truncation error {max_trunc_err_gutz}")

    #overlap = abs(psi_dmrg.overlap(psi_gutz, num_ev=4))
    print(f"calculating overlap between dmrg wavefunction in {dmrg_dir} and gutzwiller wavefunction in {gutzwiller_dir}")
    overlap = abs(psi_dmrg.overlap(psi_gutz))
    print("overlap between wavefunctions: ", overlap)
    return overlap


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
                                                         show_transverse_correlations=False, iMPS_Lx_factor=Lx_short_factor_temfpy_iMPS):
    
    print(f"norm_magz: {norm_magz}")
    site = FermionSite(conserve='N')
    spin_site = SpinHalfSite(conserve='Sz')
    gutzwiller_results_dir = "MonopoleCondensateGutzwiller/"

    spinfull = True
    particle_hole = spinfull
    debug = False
    if local:
        results_dir = CreateGutzwillerCaseDir(gutzwiller_results_dir, Lx, Ly, chi_max, flux, geometry, bc_MPS,
                                              gs_manifold_index, model_type, norm_magz, monopole_Q)
    else:
        results_dir = "./"
    assert((bc_MPS == "finite") or (bc_MPS == "infinite"))
    finite = (bc_MPS == "finite")
    #fig_lat, ax_lat = plt.subplots()
    #lat = GetPiFluxTriangularLattice(site, Lx, Ly, True, bc_MPS, geometry)
    #PlotLattice(lat, ax_lat)
    #plt.show()

    slater_trunc_par = {"chi_max": chi_max, "svd_min": svd_min_slater_default, "degeneracy_tol": 1e-12}
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

    Lx_for_corr = triangular_lat.Ls[0] if finite else 20
    Ly_for_corr = triangular_lat.Ls[1]
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

    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, spin_lat)

    fig_corr_k, ax_corr_k = plt.subplots(figsize=(6, 5))
    ImshowMatrix(ax_corr_k, fig_corr_k, Kx, Ky, spin_corr_k)
    spin_lat_singlesite_unitcell = BuildTriangularLattice(1, 1, spin_site, bc_MPS, geometry=geometry)
    spin_lat_singlesite_unitcell.plot_brillouin_zone(ax_corr_k)
    ax_corr_k.set_title("Spin Correlations")

    SaveSimulationOutput(results_dir, spin_corr_x, Kx, Ky, spin_corr_k, fig_corr_k, fig_lat)


def ComputeCorrelationsFromMPSFile(parent_results_path, Lx, Ly, bc, bc_MPS,
                                   conserve=True, geometry=None, psi_fname="psi_gs.pkl", sort_charge=False,
                                   Lx_for_infinite_bc_MPS=None, psi_dir=None, transverse_correlations=False):
    psi_dir = parent_results_path + psi_dir

    with open(psi_dir + psi_fname, 'rb') as f:
        psi = pickle.load(f)

    site = SpinHalfSite(conserve=('Sz' if conserve else None),sort_charge=sort_charge)
    Lx_correlations = Lx
    if bc_MPS == "infinite" or Lx_for_infinite_bc_MPS is not None:
        assert(Lx_for_infinite_bc_MPS is not None)
        Lx_correlations = Lx_for_infinite_bc_MPS

    triangular_lattice = BuildTriangularLattice(Lx_correlations, Ly, site, bc_MPS, bc=bc,
                                                geometry=geometry)

    spin_corr_x = CalculateSpinSpinCorrelations(psi, sites1=np.arange(0, triangular_lattice.N_sites),
                                                sites2=np.arange(0, triangular_lattice.N_sites),
                                                transverse_correlations=transverse_correlations)
    Kx, Ky, spin_corr_k = ComputeMomentumSpaceStructureFactor(spin_corr_x, triangular_lattice,
                                                              assert_realness=True)

    # spin_corr_k_from_file = np.loadtxt(psi_dir + "spin_corr_k.csv", dtype=complex)
    # print("largest diff: ", np.max(np.abs(spin_corr_k - spin_corr_k_from_file)))
    fig, ax = plt.subplots(figsize=(6, 5))
    title = f"Spin structure factor"
    ImshowMatrix(ax, fig, Kx, Ky, spin_corr_k, title=title)
    triangular_lattice.plot_brillouin_zone(ax)
    fig.savefig(psi_dir + "momentum_space_correlations_local", bbox_inches='tight')
    if local:
            plt.show()


def PlotCorrelationsFromFiles(results_dir, energy_ax=None, initial_state="",
                              show_energies = True, psi_fname="psi_gs.pkl", output_dir=None,
                              fig_title="", k_space=True):
    if show_energies:
        if energy_ax is None:
            fig, energy_ax = plt.subplots()
        energies = np.loadtxt(results_dir + "Energies.txt")
        energy_ax.plot(energies, "o", label=initial_state)

    # with open(results_dir + psi_fname, 'rb') as f:
    #     psi = pickle.load(f)
    #print("Magentization of gs: ", psi.get_total_charge(only_physical_legs=True))
    #print("Magentization of gs: ", np.sum(psi.expectation_value("Sz")))

    fig, ax = plt.subplots(figsize=(6, 5))
    if k_space:
        Kx = np.loadtxt(results_dir + "Kx.csv")
        Ky = np.loadtxt(results_dir + "Ky.csv")
        corr_k = np.loadtxt(results_dir + "spin_corr_k.csv", dtype=np.complex128)

        triangular_lat = BuildTriangularLattice(2, 2, SpinHalfSite(conserve='Sz'), "finite")
        triangular_lat.plot_brillouin_zone(ax)
        assert (np.max(np.abs(np.imag(corr_k))) < 1e-14)
        ImshowMatrix(ax, fig, Kx, Ky, np.real(corr_k), title="Spin Correlations")
    else:
        corr_x = np.loadtxt(results_dir + "spin_corr_x.csv", dtype=np.complex128)
        ImshowMatrix(ax, fig, np.array([0,1]), np.array([0,1]), corr_x, xlabel="x/L",
                     ylabel="y/L")

    if output_dir is not None:
        fig.savefig(output_dir + f"correlations_{fig_title}.png", bbox_inches='tight')


def PlotRealSpaceCorrelations(results_dir):
    corr_x = np.loadtxt(results_dir + "spin_corr_x.csv", dtype=np.complex128)
    print(corr_x.shape)
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])
    fig, ax = plt.subplots(figsize=(6, 5))
    ImshowMatrix(ax, fig, X, Y, np.real(corr_x), label="real part")
    ax.legend()


def GutzwillerDMRGOverlaps(J2s, gutz_parent_dir, Lx, Ly, gutz_chi_max, gutz_flux, gutz_mon_Q,
                           output_dir, dmrg_initial_state, dmrg_parent_dir, geometry, bc_MPS, gutz_gs_manifold_index,
                           dmrg_chi_max, dmrg_max_sweeps, dmrg_conserve, model_type, norm_magz):
    overlaps = []
    dmrg_energies = []
    gutz_energies = []
    gutz_case_dir = CreateGutzwillerCaseDir(gutz_parent_dir, Lx, Ly, gutz_chi_max, gutz_flux, geometry,
                                            bc_MPS, gutz_gs_manifold_index, model_type=model_type, norm_magz=norm_magz,
                                            monopole_Q=gutz_mon_Q)
    finite = (bc_MPS == "finite")
    bc = ("open", "periodic") if finite else ("periodic", "periodic")
    
    with open(output_dir + "parent_directories.txt", 'w') as f:
        f.write(f"dmrg parent dir: {dmrg_parent_dir}\n")
        f.write(f"gutzwiller parent dir: {gutz_parent_dir}\n")

    for J2 in J2s:
        dmrg_geom_dir, dmrg_params_dir = (
            TriangularJ1J2CaseDirName(Lx, Ly, bc, bc_MPS, dmrg_initial_state, dmrg_conserve, J2, geometry, dmrg_chi_max, dmrg_max_sweeps, norm_magz))
        dmrg_dir = dmrg_parent_dir + dmrg_geom_dir + dmrg_params_dir
        
        unitcell_width = 2 if geometry == "XC" else 1
        
        sweep_energies = np.loadtxt(dmrg_dir + "Energies.txt", dtype=np.float64)
        dmrg_energy = sweep_energies[-1]
        if finite:
            dmrg_energy /= (Lx * Ly * unitcell_width)
        dmrg_energies.append(dmrg_energy)

        gutz_energy = calculateGutzwillerEnergyTriangularJ1J2(gutz_parent_dir, Lx, Ly, gutz_chi_max, gutz_flux, bc_MPS, J2, bc,
                                                              geometry, gutz_gs_manifold_index, norm_magz, gutz_mon_Q, model_type=model_type)
        gutz_energies.append(gutz_energy)

        PlotCorrelationsFromFiles(dmrg_dir, show_energies=False, output_dir=output_dir,
                                  fig_title=f"dmrg_J2_{J2}")

        overlap_J2 = calculateOverlapBetweenGutzwillerAndDMRG(dmrg_dir, gutz_case_dir)
        overlaps.append(overlap_J2)


    J2s = np.array(J2s)
    overlaps = np.array(overlaps)
    dmrg_energies = np.array(dmrg_energies)
    gutz_energies = np.array(gutz_energies)
    data = np.column_stack((J2s, overlaps, dmrg_energies, gutz_energies))
    np.savetxt(output_dir + "data.txt", data, header='J2 overlap E_DMRG E_Gutzwiller')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(J2s, overlaps, "o")
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel("overlap")
    fig.savefig(output_dir + f"overlaps_initial_state_{dmrg_initial_state}.png", bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(J2s, dmrg_energies, "ro", label="dmrg")
    ax.plot(J2s, gutz_energies, "bo", label="Gutzwiller")
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$E$")
    ax.legend()
    fig.savefig(output_dir + f"energies_initial_state_{dmrg_initial_state}.png", bbox_inches='tight')


def linear_model(x, m, b):
    return m * x + b


def FitLinearModel(x, y):
    popt, pcov = curve_fit(linear_model, x, y)
    m = popt[0]
    b = popt[1]
    x0 = (-1) * b / m
    dm = np.sqrt(pcov[0, 0])
    db = np.sqrt(pcov[1, 1])
    dx0 = np.sqrt((db / m) ** 2 + (dm * b / (m ** 2)) ** 2)

    return {"m": m, "b": b, "db": db, "dm": dm}


def GutzwillerBondDimensionScaling(gutz_results_dir, Lx, Ly, chis, flux,
                                   output_dir):
    Es = []
    inv_chis = np.array(1. / np.array(chis))
    bc_MPS = "finite"
    J2 = 0.125
    bc = ("open", "periodic")
    geometry = "YC"
    gs_manifold_index = 0
    for chi in chis:
        E = calculateGutzwillerEnergyTriangularJ1J2(gutz_results_dir, Lx, Ly, chi, flux, bc_MPS, J2, bc, geometry,
                                                    gs_manifold_index, None, None)
        Es.append(E)

    fit_params = FitLinearModel(inv_chis, Es)
    m, b, db, dm = fit_params["m"], fit_params["b"], fit_params["db"], fit_params["dm"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(inv_chis, Es, "bo")

    inv_chis_plot_fit = np.linspace(0.0, np.max(inv_chis), 2)
    ax.plot(inv_chis_plot_fit, m*inv_chis_plot_fit + b, "b--")
    ax.errorbar([0.0], [b], yerr=[db], fmt="ro")

    ax.set_xlabel(r"$1 / \chi$")
    ax.set_ylabel(r"$E$")
    print(f"Energy for chi=inf: {b} +- {db}")
    fig.savefig(output_dir + "Gutzwiller_E_chi_scaling.png", bbox_inches="tight")
    if local:
            plt.show()


def DMRGCorrelations():
    main_results_dir = "NewTenpyTriangularLatticeResults/"
    case_dir = f"Lx_6_Ly_6_bc_op/finite_init_Random_conserve_1_J2_{0.0}/"
    PlotCorrelationsFromFiles(code_dir + main_results_dir + case_dir)
    if local:
            plt.show()

    for J2 in [0.1, 0.11, 0.12, 0.125, 0.13, 0.14, 0.15]:
        fig, energy_ax = plt.subplots(figsize=(6, 5))
        for initial_state in ["stripe", "Random"]:
            case_dir = f"Lx_6_Ly_6_bc_op/finite_0.0_init_{initial_state}_conserve_1_J2_{J2}/"
            PlotCorrelationsFromFiles(code_dir + main_results_dir + case_dir, energy_ax=energy_ax,
                                      initial_state=initial_state)
        energy_ax.legend()
        if local:
            plt.show()



def TryCylinderFlux():
    lat = BuildTriangularLattice(2, 3, FermionSite(), "finite")
    fermion_model = CouplingModel(lat)
    for u1, u2, dx in lat.pairs['nearest_neighbors']:
        strength_with_flux = fermion_model.coupling_strength_add_ext_flux(1.0, dx, [0, pi / 4])
        fermion_model.add_coupling(strength_with_flux, u1, 'Cd', u2, 'C', dx)
        fermion_model.add_coupling(np.conj(strength_with_flux), u2, 'Cd', u1, 'C', -np.array(dx))
    fig, ax = plt.subplots(figsize=(6, 5))
    PlotLattice(lat, ax)
    if local:
            plt.show()


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


def CompareGutzwillerGroundStateSectorsXC():
    gutz_dir = code_dir + "LocalGutzwillerResults/Dirac_finite_Lx_40_Ly_2_chi_1000_flux_0.0_XC_gsindex_"
    for i in range(4):
        i_dir = gutz_dir + f"{i}/"
        PlotCorrelationsFromFiles(i_dir, show_energies=False, output_dir=i_dir)
        for j in range(i + 1, 4):
            psi1_path = i_dir + "psi_gutzwiller.pkl"
            psi2_path = gutz_dir + f"{j}/" + "psi_gutzwiller.pkl"
            with open(psi1_path, 'rb') as psi_load:
                psi1 = pickle.load(psi_load)
            with open(psi2_path, 'rb') as psi_load:
                psi2 = pickle.load(psi_load)
            print(f"overlap {i},{j}: ", psi1.overlap(psi2))

            C_i = np.loadtxt(gutz_dir + f"{i}/C_slater.csv")
            C_j = np.loadtxt(gutz_dir + f"{j}/C_slater.csv")
            fig, ax = plt.subplots(figsize=(6, 5))
            ImshowMatrix(ax, fig, np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.abs(C_i - C_j),
                         xlabel="x/L", ylabel="y/L")
            fig.savefig(f"C_diff_{i}{j}.png", bbox_inches='tight')
            plt.show()



def calculateZ2EntanglementEntropy():
    parent_dir = code_dir + "Z2_Topological_EE/"
    cases = []
    central_bonds = []
    chis = [2000, 3000, 4000, 6000]
    Lx = 2
    Lys = [4, 5, 6, 7]
    for i_Ly, Ly in enumerate(Lys):
        cases.append(parent_dir + f"Z2_infinite_Lx_{Lx}_Ly_{Ly}_chi_{chis[i_Ly]}_flux_0.0_YC_gsindex_0/")
        central_bonds.append(Lx * Ly // 2)

    EEs = []
    for i_case, case in enumerate(cases):
        with open(case + "psi_gutzwiller.pkl", 'rb') as f:
            psi = pickle.load(f)
        EE_central_bond = psi.entanglement_entropy(bonds=[central_bonds[i_case]])[0]
        EEs.append(EE_central_bond)

    Lys = np.array(Lys)
    EEs = np.array(EEs)
    plt.plot(Lys, EEs, "o")

    fit_params = FitLinearModel(Lys, EEs)
    m, b, db, dm = fit_params["m"], fit_params["b"], fit_params["db"], fit_params["dm"]
    print(f"Constant EE contribution: {b} +- {db}")
    Ly_linplot = np.array([0, np.max(Lys)])
    plt.plot(Ly_linplot, m*Ly_linplot + b, "-")

    plt.show()


def getEnergyDifferenceBetweenSectors(dir1, dir2, title, dmrg, fig_name):
    data1 = np.loadtxt(dir1 + "data.txt")
    data2 = np.loadtxt(dir2 + "data.txt")
    J2 = data1[:, 0]
    assert(np.max(np.abs(J2 - data2[:,0])) <= 0.0)
    E_dmrg_1 = data1[:,2]
    E_dmrg_2 = data2[:,2]
    E_gutz_1 = data1[:, 3]
    E_gutz_2 = data2[:, 3]
    fig, ax = plt.subplots(figsize=(6, 5))
    if dmrg:
        ax.plot(J2, E_dmrg_1 - E_dmrg_2, "ro", label="DMRG")
    else:
        ax.plot(J2, E_gutz_1 - E_gutz_2, "bo", label="Gutzwiller")
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$\Delta$E")
    ax.set_title(title)
    fig.savefig(fig_name, bbox_inches='tight')
    plt.show()


def TestFreeFermionsSpinCorrelations():
    Lx, Ly = 6, 6
    spinfull_fermions = True

    site = FermionSite('N')
    triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite",
                                            ("open", "periodic"), "XC", spinfull_fermions)
    pi_flux_parameters = {"init_H_MPO": False, "monopole_Q": 0, "flux": 0.0,
                          "particle_hole": spinfull_fermions, "lattice": triangular_lat}
    C_x_exact, _ = CalculateExactCMatrixForPiFlux(0, pi_flux_parameters, model_type_dirac,
                                                  abs_magz=Lx*Ly)

    spin_spin_corr = FreeFermionSpinCorrelations(C_x_exact)
    spin_triangular_lat = BuildTriangularLattice(Lx, Ly, site, "finite", ("open", "periodic"),
                                                 "XC")
    momentum_space = True
    fig, ax = plt.subplots(figsize=(6, 5))
    if momentum_space:
        Kx, Ky, C_k = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat)
        ImshowMatrix(ax, fig, Kx, Ky, C_k)
        lat_for_bz = BuildTriangularLattice(1, 1, site, "finite", ("open", "open"), "YC")
        lat_for_bz.plot_brillouin_zone(ax)
    else:
        ImshowMatrix(ax, fig, np.array([0.,1.]), np.array([0.,1.0]), spin_spin_corr, "X", "Y")

    plt.show()

    for i in range(spin_spin_corr.shape[0]):
        for j in range(spin_spin_corr.shape[1]):
            if i != j:
                assert(np.abs(spin_spin_corr[i,j] - 0.25) < 1e-14), "test failed - incorrect off diagonal term"
            else:
                assert(np.abs(spin_spin_corr[i,j] - 0.75) < 1e-14), "test failed - incorrect diagonal term"


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

    M1 = np.array([0., 2 * pi / sqrt(3)])
    M2 = np.array([pi, (-1) * pi / sqrt(3)])
    M3 = np.array([pi, pi / sqrt(3)])
    Ms = (M1, M2, M3)
    for i_M, M in enumerate(Ms):
        S_M = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat, Kx=np.array([M[0]]),
                                                   Ky=np.array([M[1]]))[-1]
        print(f"spin structure factor at M{i_M+1}: {S_M}")


    Kx, Ky, C_k = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat)
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
    temfpy_dir = (code_dir + f"LocalGutzwillerResults/Dirac_infinite_Lx_2_Ly_{Ly}_" +
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
        Kx, Ky, C_k = ComputeMomentumSpaceStructureFactor(spin_spin_corr, spin_triangular_lat)
        ImshowMatrix(ax_exact, fig_exact, Kx, Ky, np.abs(C_k))
        lat_for_bz = BuildTriangularLattice(1, 1, site, "finite", bc_exact, "YC")
        lat_for_bz.plot_brillouin_zone(ax_exact)

    plt.show()


def calculateMonopoleEnergies(parent_dir, norm_magz, mon_Qs):
    assert(mon_Qs[0] == 0)
    Es = []
    fig, ax = plt.subplots(figsize=(6,5))
    Lx, Ly = 6, 6
    flux = 0.0
    chi = 2000
    J2 = 0.125
    for monopole_Q in mon_Qs:
        E = calculateGutzwillerEnergyTriangularJ1J2(parent_dir, Lx, Ly, chi, flux,
                                                    "finite", J2, ("open", "periodic"),
                                                    "YC", 0, norm_magz, mon_Q, model_type=model_type_dirac,
                                                    norm_magz=norm_magz, monopole_Q=monopole_Q)
        Es.append(E)

    Es = np.array(Es)
    mon_Qs = np.array(mon_Qs)
    E_fermi_pocket = Es[0]
    e_diff = (Es[1:] - E_fermi_pocket) / abs(E_fermi_pocket)
    ax.plot(mon_Qs[1:], e_diff, "o")
    ax.set_xlabel(r"$Q[2\pi/N]$")
    ax.set_ylabel(r"$\delta E[J_1] / E_{fp}$")
    ax.set_title("Energy vs. Monopole Flux for J2/J1=1/8")
    return e_diff, ax, fig


def calculateOverlapsFastLocal(Lx, Ly, Q, magz):
    dir_gutz = monopole_dir + f"Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz}_monQ_{Q}/"
    dir_dmrg = code_dir + (f"TriangularJ1J2DMRG_6_15_e476229/Lx_{Lx}_Ly_{Ly}_bc_op_YC/"
                       f"finite_init_Random_conserve_1_J2_0.1_chi_3500_maxsweeps_50_magz_{magz}/")

    with open(dir_gutz + "psi_gutzwiller.pkl", 'rb') as f:
        psi_gutz = pickle.load(f)
    with open(dir_dmrg + "psi_gs.pkl", 'rb') as f:
        psi_dmrg = pickle.load(f)

    # psi_gutz.unit_cell_width = psi_dmrg.unit_cell_width

    print(f"overlap with magz={magz}, Q={Q}: {np.abs(psi_dmrg.overlap(psi_gutz))}")


def plotMonopoleOrderParameter():
    Lx, Ly = 6,6

    lattice = BuildTriangularLattice(Lx, Ly, SpinHalfSite("Sz"), "finite", bc=("open", "periodic"))

    magz1 = 0.056
    Q1 = 0
    magz2 = 0.111
    Q2 = 0

    dir1 = monopole_dir + f"Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz1}_monQ_{Q1}/"
    dir2 = monopole_dir + f"Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz2}_monQ_{Q2}/"

    with open(dir1 + "psi_gutzwiller.pkl", 'rb') as f:
        psi1 = pickle.load(f)
    with open(dir2 + "psi_gutzwiller.pkl", 'rb') as f:
        psi2 = pickle.load(f)

    psi1.unit_cell_width = lattice.mps_unit_cell_width
    psi2.unit_cell_width = lattice.mps_unit_cell_width
    env = MPSEnvironment(bra=psi2, ket=psi1)
    local_vals = env.expectation_value("Sp")

    Kx, Ky, monopole_op_k = ComputeMomentumSpaceStructureFactor(local_vals, lattice, assert_realness=False,
                                                                transform_expectation_value=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ImshowMatrix(ax, fig, Kx, Ky, np.abs(monopole_op_k), title=r"Fourier Transform of $<n+1|S^+|n>$")
    lattice.plot_brillouin_zone(ax)
    fig.savefig(meetings_dir + f"20_6_2026/monopole_OP/LX_{Lx}_Ly_{Ly}_magzlow_{magz1}_Q1_{Q1}_Q2_{Q2}.png", bbox_inches='tight')
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
    output_dir = "C:/Users/yonli/Desktop/Thesis/Triangular J1J2/Meetings/4_5_2026/"
    monopole_dir = code_dir + "MonopoleCondensateGutzwiller/"

    # TestFreeFermionsSpinCorrelations()
    # checkXC8SlaterCorrelations()
    # checkPiFluxFreeSpinCorrelations()
    #exit(1)

    # DebugMagnetizedIMPS()

    # checkXC8SlaterCorrelations()

    # TryMonopoleModelHofstadter("./", 18, 18, bc=("open", "periodic"))

    #getEnergyDifferenceBetweenSectors(code_dir + "../Meetings/1_6_2026/XC8/Flux0_Random/",
    #                                  code_dir + "../Meetings/1_6_2026/XC8/Flux1_Random/", r"Flux 0 vs. Flux $\pi$",
    #                                  False, "ener_diff_XC8_gutz.png")

    #case_dir = "../Meetings/1_6_2026/XC8/Flux0_Random/"
    #PlotCorrelationsFromFiles(code_dir + case_dir, show_energies=False,
    #                          output_dir=code_dir + case_dir, fig_title="gutz")
    #case_dir = "../Meetings/1_6_2026/XC8/Flux1_Random/"
    #PlotCorrelationsFromFiles(code_dir + case_dir, show_energies=False,
    #                          output_dir=code_dir + case_dir, fig_title="gutz")
    # calculateZ2EntanglementEntropy()
    # TestDimerDimerCorrelations()
    # PlotSquareLatticeStructureFactor(Lx=3, Ly=3)

    #dir = code_dir + "/LocalGutzwillerResults/Z2_infinite_Lx_2_Ly_4_chi_500_flux_0.0_YC_gsindex_0_magz_1/"
    #with open(dir + "psi_gutzwiller.pkl", 'rb') as f:
    #   psi = pickle.load(f)
    #print(np.sum(psi.expectation_value("Sz")))

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
    #                                             save_dir = code_dir + "../Meetings/20_6_2026/")

    # fig.savefig(f"{output_dir}ener_vs_mon_dens_magz_{magz}.png", bbox_inches='tight')
    # plt.show()

    #CheckOptimalMonopoleStateEnergyVsMagnetization(24, 6)

    #gutz_dir = code_dir + "LocalGutzwillerResults/"
    #Lx, Ly = 2, 6
    #chi_gutz = 2000
    #flux_gutz = 0.0
    #geometry = "YC"
    #J2 = 0.125
    #calculateGutzwillerEnergyTriangularJ1J2(gutz_dir, Lx, Ly, chi_gutz, flux_gutz, geometry=geometry,
    #                                        J2=J2, bc_MPS="infinite", bc=("periodic", "periodic"))
    #

    #arr = np.array([[1+1j, 1-1j], [1-1j, 1-1j]])
    #print(det(arr))
    #exit(0)

    # C_magnetized = np.loadtxt("debug_magnetized_iMPS/C_Lx_80_m_0.0833.txt", dtype=np.complex128)
    # C_unmagnetized = np.loadtxt("debug_magnetized_iMPS/C_Lx_80_m_0.0.txt", dtype=np.complex128)
    # fig, ax = plt.subplots(figsize=(5,6))
    # # ImshowMatrix(ax, fig, np.array([0.0,1.0]), np.array([0.0,1.0]), np.abs(C), xlabel="X", ylabel="Y")
    # middle_site = C_magnetized.shape[0]//2
    # ax.plot(C_magnetized[middle_site, (middle_site)::2], "bo", markersize=2, label="with magz")
    # ax.plot(C_unmagnetized[middle_site, (middle_site)::2], "ro", markersize=2, label="without magz")
    # ax.legend()
    # plt.show()

    for norm_magz_fac in [0.]:
        Lx, Ly = 2, 6
        norm_magz = norm_magz_fac / (Lx * Ly)
        iMPS_Lx_factor = 20
        chi_max = 1000
        abs_magz = AbsMagzFromNormMagz(norm_magz, Lx * Ly)
        monopole_Q_opt = int(norm_magz_fac//2)
        flux = 0.0
        bc_MPS = "infinite"
        SpinonTriangularLatticeMeanFieldGutzwillerProjection(Ly, "YC", bc_MPS, 0,
                                                             model_type_dirac, Lx=Lx, chi_max=chi_max, flux=flux,
                                                             norm_magz=norm_magz, monopole_Q=0,
                                                             show_transverse_correlations=True,
                                                             iMPS_Lx_factor=iMPS_Lx_factor)


    # calculateOverlapsFastLocal(6, 6, 2, 0.056)

    #ComputeCorrelationsFromMPSFile(code_dir, 6, 6, ("open", "periodic"), "finite", geometry="YC",
    #                               psi_dir="LocalGutzwillerResults/Dirac_finite_Lx_6_Ly_6_chi_1000_flux_0.0_YC_gsindex_0_magz_6_monQ_8/",
    #                               psi_fname="psi_gutzwiller.pkl", transverse_correlations=True)
    #plt.show()

    #plotMonopoleOrderParameter()
    #plt.show()

    #Lx, Ly = 6, 6
    # magz = 0.111
    # for Q in [0,2]:
    #     dir = monopole_dir + f"/Dirac_finite_Lx_{Lx}_Ly_{Ly}_chi_2000_flux_0.0_YC_gsindex_0_magz_{magz}_monQ_{Q}/"
    #bc = ("open", "periodic")
    #site = SpinHalfSite('Sz')
    #lat =  BuildTriangularLattice(Lx, Ly, site, "finite", bc=bc)
    #     with open(dir + "psi_gutzwiller.pkl", 'rb') as f:
    #         psi = pickle.load(f)
    #_, fig, ax = plot_scalar_spin_chirality(psi3, lat)
    #fig.savefig(meetings_dir + f"20_6_2026/chirality_magz_{0.056}_J2_0.1_dmrg.png", bbox_inches='tight')
    #plt.show()

    #dir = code_dir + "LocalGutzwillerResults/Dirac_finite_Lx_6_Ly_6_chi_1000_flux_0. 0_YC_gsindex_0_magz_6_monQ_6/"
    #with open(dir + "psi_gutzwiller.pkl", 'rb') as f:
    #  psi = pickle.load(f)

    #####################
    # i = 1
    #for i in range(4):
    #    SpinonTriangularLatticeMeanFieldGutzwillerProjection(2, "XC", "finite", i, model_type_dirac,
    #                                                         Lx=40, chi_max=1000, flux=0.0)
    # TriangularPiFluxGutzwiller(3, "XC", "infinite", 0, Lx=2, chi_max=1000, flux=1.0)

    # TestZ2MeanFieldModel()
    #TriangularPiFluxGutzwiller(8, "YC", "infinite", 0, Lx=2, chi_max=2500, flux=0.0)

    #calculateGutzwillerEnergyTriangularJ1J2("LocalGutzwillerResults/", 2, 8, 2500, 0.0,
    #                                        "infinite", 0.125, ("periodic", "periodic"), "YC",
    #                                        0)

    # TriangularPiFluxGutzwiller(3, "XC", "finite", 0, Lx=200, chi_max=1000, flux=1.0)
    # TriangularPiFluxGutzwiller(3, "XC", "infinite", 0, Lx=2, chi_max=1000, flux=1.0)

    # spin_corr_x1 = np.loadtxt(psi1_dir + "spin_corr_x.csv")
    # spin_corr_x2 = np.loadtxt(psi2_dir + "spin_corr_x.csv")
    # plt.imshow(spin_corr_x1 - spin_corr_x2)
    # plt.show()
    #
    # psi_fname = "psi_gutzwiller.pkl"
    # with open(psi1_dir + psi_fname, 'rb') as f:
    #     psi1 = pickle.load(f)
    # with open(psi2_dir + psi_fname, 'rb') as f:
    #     psi2 = pickle.load(f)
    # print(psi1.overlap(psi2))
