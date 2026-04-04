from temfpy import slater
import tenpy
import numpy as np
from numpy import log, sin, cos, sqrt
import matplotlib.pyplot as plt
from tenpy import TransferMatrix
from tenpy.models import lattice
from numpy.linalg import norm, eigh
import temfpy.gutzwiller as gutz
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
                  chi_max=2000, svd_min=1e-6, label_svd=False, show_entanglement_diff=False, show_n_expect=False):
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
    label = "n=" + str(n)
    if label_svd:
        label += ", svd_trunc = " + str(svd_min)

    if not show_entanglement_diff:
        ax_entropy.plot(x, entanglement_from_mps, "o", label=label, color=color)
    entanglements_from_corr_mat = np.zeros(n-1)
    for right_site in range(1, n):
        entanglements_from_corr_mat[right_site-1] = EntanglementFromCorrelationMatrix(C, right_site)
    if show_entanglement_diff:
        diff_ent = entanglement_from_mps[1:n-2] - entanglements_from_corr_mat[1:n-2]
        avg_diff_ent = np.average(diff_ent)
        ax_entropy.plot(x[1:n-2], np.abs(diff_ent),
                        "o", color=color, label=label + ", diff_ent = " + str(round(avg_diff_ent,4)))
    else:
        ax_entropy.plot(x, entanglements_from_corr_mat, "--", color=color)

    if show_n_expect:
        n_expect = mps.expectation_value("Cd C")
        print("N_occ / N: ", np.sum(n_expect) / n)
        print("MPS total charge: ", mps.get_total_charge(only_physical_legs=True) / n)
        ax_expectation_value.plot(np.linspace(0, n-1 , n) / (n-1), n_expect, label=label, color=color)

    mps_correaltion_function = mps.correlation_function("Cd", "C")
    fig_correlations, ax_correlations = plt.subplots()
    ax_correlations.plot(C[n//2, :], "ro", label="corr from correlation matrix")
    ax_correlations.plot(mps_correaltion_function[n//2,:], "b^", label="corr from mps")

def Hopping1D_iMPS():
    def H(L, t1, t2, periodic=False):
        M = t1 * np.ones(L - 1)
        M[1::2] = t2
        M = np.diag(M, 1)
        H = M + M.T
        if periodic:
            H[0,L-1] = t1 if (abs(H[0,1]-t2) < 1e-15) else t2
            H[L-1, 0] = H[0,L-1]
        return H
    trunc_par = {"chi_max": 500, "svd_min": 1e-6}  # cf. Listing 1
    L_short = 50
    t1 = -1.0
    t2 = t1
    try_large_unitcell = True
    cell = 2 if t2 != t1 else 1  # cf. periodicity of H
    if try_large_unitcell:
        cell = 4
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
        ax.plot(MPS_correlation_func[center_site,(center_site-distance_range):(center_site+distance_range)],
                                "ro")
        ax.plot(iMPS_correlation_func[unit_cell_site,(center_site-distance_range):(center_site+distance_range)],
                                "b^")
    plt.show()

def ContractMPS(mps):
    mps_tensors_list = [mps.get_B(i) for i in range(mps.L)]
    contracted_mps = mps_tensors_list[0]
    for B in mps_tensors_list[1:]:
        contracted_mps = tenpy.linalg.np_conserved.tensordot(
            contracted_mps, B, axes=(['vR'], ['vL'])
        )
    contracted_mps = np.squeeze(contracted_mps.to_ndarray())
    return contracted_mps

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
    try_iMPS = False
    try_MPS = False
    try_Gutzwiller = True

    if try_Gutzwiller:
        TryGutzwiller()
    if try_iMPS:
        Hopping1D_iMPS()

    if try_MPS:
        show_n_expect = False
        ax_n_expect = None
        if show_n_expect:
            fig_n_expect, ax_n_expect = plt.subplots()
        show_size_dependence = True
        show_trunc_dependence = False

        if show_size_dependence:
            fig_entanglement,ax_entanglement = plt.subplots()
            for i_n, n in enumerate(range(50, 70, 20)):
                Hopping1D_MPS(ax_entanglement, ax_n_expect, n=n, color = colors[i_n], show_n_expect=show_n_expect)
            plt.legend()
            plt.show()

        if show_trunc_dependence:
            svd_mins = [1e-1, 1e-2, 1e-3, 1e-4]
            n = 80
            fig_entanglement, ax_entanglement = plt.subplots()
            for ind, svd_min in enumerate(svd_mins):
                Hopping1D_MPS(ax_entanglement, ax_n_expect, n=n, color=colors[ind], svd_min=svd_min,
                              label_svd=True, show_entanglement_diff=True, show_n_expect=show_n_expect)
                plt.legend()
            plt.show()

