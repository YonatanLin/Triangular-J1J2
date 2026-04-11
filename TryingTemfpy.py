local = False
if local:
   from temfpy import slater
   import temfpy.gutzwiller as gutz
   from temfpy.gutzwiller import number_mask, parity_mask

import tenpy
import numpy as np
from numpy import log, sin, cos, sqrt
import matplotlib.pyplot as plt
from tenpy.models import lattice
from numpy.linalg import norm, eigh
import tenpy.linalg.np_conserved as npc
from tenpy import networks

fontsize=18
rc_params = {
    "font.family": "serif",
    "figure.dpi": 200,
    #'text.usetex': True,
    #"axes.labelsize": 50,
    #"axes.titlesize": 50,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": 14,
    "figure.titlesize": fontsize,
    "legend.loc": "upper left",
}

plt.rcParams.update(rc_params)
colors = ["blue", "orange", "green", "red", "purple", "pink",
              "magenta"]

def EntanglementFromCorrelationMatrix(C, right_site):
    C_block = C[0:right_site, 0:right_site]
    e, v = eigh(C_block)
    e[e >= 1.0] = 1-1e-6
    e[e <= 0.0] = 1e-6
    if len(e) > 0 and (np.min(e) <= 0.0 or np.max(e) >= 1.0):
        print("Illegal eigenvalues in EntanglementFromCorrelationMatrix")
        exit(1)
    return (-1) * (np.sum(e * log(e)) + np.sum((1-e) * log(1 - e)))


def Hopping1D_MPS(ax_entropy, ax_expectation_value, n = 20, color="red",
                  chi_max=2000, svd_min=1e-6, label_svd=False, show_entanglement_diff=False, show_n_expect=False,
                  results_dir=None):
    H = -(np.eye(n, k=1) + np.eye(n, k=-1))
    H[0, n-1] = H[n-1, 0] = -1
    C, _ = slater.correlation_matrix(H)
    trunc_par = {"chi_max": chi_max, "svd_min": svd_min, "degeneracy_tol": 1e-12}

    # modes = slater.SchmidtModes.from_correlation_matrix(C[0], int(n/2), trunc_par)
    # schmidt = slater.SchmidtVectors.from_schmidt_modes(modes, trunc_par)
    # schmidt_R = slater.SchmidtVectors.from_correlation_matrix(C[0], int(n/2)+1, trunc_par)
    # A = slater.MPSTensorData.from_schmidt_vectors(schmidt, schmidt_R, "left")
    mps = slater.C_to_MPS(C, trunc_par)

    entanglement_from_mps = mps.entanglement_entropy()
    x = np.linspace(0, n-2 , n-1) / (n-2)
    label = rf"$n={n}$"
    if label_svd:
        label = rf"$n={n},\chi={chi_max}$"

    if not show_entanglement_diff:
        ax_entropy.plot(x, entanglement_from_mps, "o", label=label, color=color)
    entanglements_from_corr_mat = np.zeros(n-1)
    for right_site in range(1, n):
        entanglements_from_corr_mat[right_site-1] = EntanglementFromCorrelationMatrix(C, right_site)
    if show_entanglement_diff:
        diff_ent = entanglement_from_mps[1:n-2] - entanglements_from_corr_mat[1:n-2]
        avg_diff_ent = np.average(diff_ent)
        diff_ent[diff_ent == 0.0] = 1e-15
        ax_entropy.semilogy(x[1:n-2], np.abs(diff_ent), "o", color=color, label=label)
    else:
        ax_entropy.plot(x, entanglements_from_corr_mat, "--", color=color)

    ax_entropy.set_xlabel(r"$r$")
    ax_entropy.set_ylabel(r"$S$")
    entanglement_title = "Entanglement Entropy"
    if show_entanglement_diff:
        entanglement_title += " Relative to Exact"
    ax_entropy.set_title(entanglement_title)
    ax_entropy.legend(loc="lower right")

    if show_n_expect:
        n_expect = mps.expectation_value("Cd C")
        print("N_occ / N: ", np.sum(n_expect) / n)
        print("MPS total charge: ", mps.get_total_charge(only_physical_legs=True) / n)
        ax_expectation_value.plot(np.linspace(0, n-1 , n) / (n-1), n_expect, label=label, color=color)

    mps_correaltion_function = mps.correlation_function("Cd", "C")
    fig_correlations, ax_correlations = plt.subplots()

    x = np.arange(0,n) / (n-1)
    ax_correlations.plot(x, C[n//2, :], "ro", label="from C matrix")
    ax_correlations.plot(x,mps_correaltion_function[n//2,:], "b^", label="from MPS")
    ax_correlations.set_xlabel(r"$r$")
    ax_correlations.set_ylabel(r"C[L/2,r]")
    ax_correlations.legend(loc='lower right')
    if results_dir is not None:
        fig_correlations.savefig(results_dir + f"correlations_n_{n}_chi_{chi_max}.png", bbox_inches='tight')


def Hopping1D_iMPS(results_dir):
    def H(L, t1, t2, periodic=False):
        M = t1 * np.ones(L - 1)
        M[1::2] = t2
        M = np.diag(M, 1)
        H = M + M.T
        if periodic:
            H[0,L-1] = t1 if (abs(H[0,1]-t2) < 1e-15) else t2
            H[L-1, 0] = H[0,L-1]
        return H

    trunc_par = {"chi_max": 700, "svd_min": 1e-6}  # cf. Listing 1
    L_short = 100
    t1 = -1.0
    t2 = t1 - 0.1
    try_small_unitcell = True
    cell = 2 if t2 != t1 else 1  # cf. periodicity of H
    if try_small_unitcell:
        cell = 1
    H_short = H(L_short, t1, t2)
    C_short, _ = slater.correlation_matrix(H_short)
    H_long = H(L_short + cell, t1, t2)
    C_long, _ = slater.correlation_matrix(H_long)
    tight_binding_iMPS, error = slater.C_to_iMPS(C_short, C_long, trunc_par, sites_per_cell = cell, cut = L_short // 2)
    tight_binding_MPS = slater.C_to_MPS(C_long, trunc_par)
    L = tight_binding_MPS.L
    iMPS_correlation_func = tight_binding_iMPS.correlation_function("Cd", "C", sites1=np.arange(L//2,L//2+cell),
                                                      sites2=np.arange(0,L))
    MPS_correlation_func = tight_binding_MPS.correlation_function("Cd", "C")
    fig,axs = plt.subplots(1, cell)
    for unit_cell_site in range(cell):
        ax = axs if cell == 1 else axs[unit_cell_site]
        center_site = L//2+unit_cell_site
        distance_range = L//2-unit_cell_site
        x = np.linspace(-1.0, 1.0, 2*distance_range)
        ax.plot(x, MPS_correlation_func[center_site,(center_site-distance_range):(center_site+distance_range)],
                "ro", label="MPS")
        ax.plot(x, iMPS_correlation_func[unit_cell_site,(center_site-distance_range):(center_site+distance_range)],
                "b^", label="iMPS")
        ax.set_xlabel(r"$r$")
        if unit_cell_site == 0:
            ax.set_ylabel(r"C[L/2,r]")
        ax.set_title(f"Unit Cell Site {unit_cell_site+1}")
        ax.legend(loc='lower right')

    fig.savefig(results_dir + "correlations.png", bbox_inches='tight')
    plt.show()


def ContractMPS(mps):
    # mps_tensors_list = [mps.get_B(i) for i in range(mps.L)]
    mps_tensors_list = mps._B
    contracted_mps = mps_tensors_list[0]
    for B in mps_tensors_list[1:]:
        contracted_mps = tenpy.linalg.np_conserved.tensordot(
            contracted_mps, B, axes=(['vR'], ['vL'])
        )
    contracted_mps = np.squeeze(contracted_mps.to_ndarray())
    return contracted_mps


def MyAbrikosov(mps):
        conserved_fermion = mps.sites[0].conserve
        if conserved_fermion == "N":
            mask = number_mask
        elif conserved_fermion == "parity":
            mask = parity_mask
        else:
            raise ValueError(
                f"FermionSite must conserve either 'N' or 'parity', found {conserved_fermion}"
            )

        # TeNPy bindings
        spin_site = networks.SpinHalfSite(None)
        spin_leg = spin_site.leg
        chinfo_s = spin_leg.chinfo

        # We start by grouping neighboring sites
        # This will result in LegPipe objects for all physical legs
        mps.group_sites(2)

        # The mask for the physical leg is independent of the site
        mask_p = mask(mps._B[0].get_leg("p"), 1)
        min_charge = np.min(mps._B[0].get_leg("vL").to_qflat())
        for idx, B in enumerate(mps._B):
            # Remove LegPipe structure
            B.legs[B.get_leg_index("p")] = B.get_leg("p").to_LegCharge()

            mask_vL = mask(B.get_leg("vL"), idx + min_charge)
            mask_vR = mask(B.get_leg("vR"), idx + 1 + min_charge)
            if(idx == len(mps._B) - 1):
                mask_vR = mask(B.get_leg("vR"), min_charge)

            # Change the occupation number leg charges to spin charges
            # --------------------------------------------------------
            #print("left leg", B.get_leg("vL").to_qflat())
            #print("right leg", B.get_leg("vR").to_qflat())
            #print("mask left: ", mask_vL)
            #print("mask right: ", mask_vR)

            print("B shape before: ", B.to_ndarray().shape)
            B.iproject([mask_vL, mask_p, mask_vR], ["vL", "p", "vR"])
            print("B shape after: ", B.to_ndarray().shape)
            B = B.drop_charge(chinfo=chinfo_s)

        mps.chinfo = chinfo_s
        mps.grouped = 1
        mps.sites = [spin_site] * mps.L
        mps.unit_cell_width = mps.L

        return mps

def InitializeInfiniteEqualWeightTwoFermionState():
    site = tenpy.networks.site.FermionSite(conserve='N')
    psi_nd_arr = np.zeros((2, 2, 2, 2))

    psi_nd_arr[1, 0, 0, 1] = 1 / 2.
    psi_nd_arr[1, 1, 0, 0] = 1 / 2.
    psi_nd_arr[0, 0, 1, 1] = 1 / 2.
    psi_nd_arr[0, 1, 1, 0] = -1 / 2.
    #psi_nd_arr[0, 0, 1, 1] = 1 / 2.
    psi = npc.Array.from_ndarray(psi_nd_arr, [site.leg] * 4, labels=["p0", "p1", "p2", "p3"])
    sites = [site] * 4
    psi_mps = MPS.from_full(sites, psi)
    psi_mps.canonical_form()

    SVs = [np.array(psi_mps.get_SL(i), copy=True) for i in range(psi_mps.L)]
    SVs.append(np.array(psi_mps.get_SL(0), copy=True))
    psi_inf = MPS(psi_mps.sites, psi_mps._B, SVs, bc='infinite')
    for i in range(len(psi_mps.sites)):
        assert np.max(np.abs(psi_inf.get_B(i).to_ndarray() - psi_mps.get_B(i).to_ndarray())) < 1e-15
    psi_inf.canonical_form()
    # psi_gutzwiller = gutz.abrikosov(psi_mps, return_canonical=False)

    # psi_gutzwiller = gutz.abrikosov(psi_inf, return_canonical=False)
    psi_gutzwiller = MyAbrikosov(psi_inf)
    psi_gutzwiller.canonical_form()
    contracted_mps_gutz = ContractMPS(psi_gutzwiller)

    for i in range(2):
        for j in range(2):
            print(f"|{i},{j}>:{contracted_mps_gutz[i, j]}")


def TryGutzwiller():
    H = np.array([[0, -1, 0, -2], [-1, 0, -2, 0], [0, -2, 0, -1], [-2, 0, -1, 0]])
    C, _ = slater.correlation_matrix(H)
    trunc_par = {"chi_max": 100, "svd_min": 1e-12, "degeneracy_tol": 1e-12}
    mps = slater.C_to_MPS(C, trunc_par)
    contracted_mps = ContractMPS(mps)
    print("The fermion mps gives the state: ")
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if(abs(contracted_mps[i,j,k,l]) > 1e-15):
                        print(f"|{i},{j},{k},{l}>:{contracted_mps[i,j,k,l]}")

    print("The spin mps gives the state: ")
    mps_gutz = gutz.abrikosov(mps)
    contracted_mps_gutz = ContractMPS(mps_gutz)
    for i in range(2):
        for j in range(2):
            print(f"|{i},{j}>:{contracted_mps_gutz[i,j]}")


if __name__ == "__main__":
    try_iMPS = True
    try_MPS = False
    try_Gutzwiller = False
    results_dir = "TemfpyGames/"
    # InitializeInfiniteEqualWeightTwoFermionState()
    if try_Gutzwiller:
        TryGutzwiller()
    if try_iMPS:
        iMPS_dir = results_dir + "iMPS/"
        Hopping1D_iMPS(iMPS_dir)

    if try_MPS:
        MPS_results_dir = results_dir + "MPS/"
        show_n_expect = False
        ax_n_expect = None
        if show_n_expect:
            fig_n_expect, ax_n_expect = plt.subplots()
        show_size_dependence = True
        show_trunc_dependence = False

        if show_size_dependence:
            fig_entanglement, ax_entanglement = plt.subplots()
            for i_n, n in enumerate(range(30, 80, 20)):
                Hopping1D_MPS(ax_entanglement, ax_n_expect, n=n, color = colors[i_n], show_n_expect=show_n_expect,
                              results_dir=MPS_results_dir)

            fig_entanglement.savefig(MPS_results_dir + "entanglement.png", bbox_inches='tight')

            plt.legend()
            plt.show()

        if show_trunc_dependence:
            # svd_mins = [1e-1, 1e-2, 1e-3, 1e-4]
            chis = [10, 25, 50, 250]
            n = 80
            fig_entanglement, ax_entanglement = plt.subplots()
            for ind, chi in enumerate(chis):
                Hopping1D_MPS(ax_entanglement, ax_n_expect, n=n, color=colors[ind], chi_max=chi,
                              label_svd=True, show_entanglement_diff=True, show_n_expect=show_n_expect,
                              results_dir=MPS_results_dir)
            fig_entanglement.savefig(MPS_results_dir + "entanglement.png", bbox_inches='tight')
            ax_entanglement.legend(loc='lower right')
            plt.show()

