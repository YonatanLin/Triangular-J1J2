
from temfpy import slater
import tenpy
import numpy as np
from numpy import log, sin, cos, sqrt, pi
import matplotlib.pyplot as plt
from tenpy.models import lattice
from tenpy import MPS
import tenpy.linalg.np_conserved as npc
import temfpy.gutzwiller as gutz
from tenpy import NearestNeighborModel, CouplingModel, SpinHalfSite, Chain, TwoSiteDMRGEngine, MPS, CouplingMPOModel, \
    SpinModel, FermionSite, FermionModel, Lattice, SpinHalfFermionSite
from Trying2D import RunDMRG, GetSpinSpinCorrelations
from temfpy.gutzwiller import number_mask, parity_mask
from tenpy import networks


class HaldaneShastryModel(CouplingMPOModel):
    def init_terms(self, model_params):
        lattice = model_params["lattice"]
        L = lattice.N_sites
        for i in range(L):
            for j in range(i+1,L):
                r_chord = sin((pi / L) * (i-j))
                strength = ((2*pi / L) ** 2) / (r_chord**2)
                self.add_local_term(strength, [("Sz", (i,0)), ("Sz", (j,0))])
                self.add_local_term(0.5 * strength, [("Sm", (i, 0)), ("Sp", (j, 0))])
                self.add_local_term(0.5 * strength, [("Sp", (i, 0)), ("Sm", (j, 0))])


def GutzwillerState(L, finite=True):
    def tight_binding(_L):
        _L_spinfull = 2*_L
        H = -(np.eye(_L_spinfull, k=2) + np.eye(_L_spinfull, k=-2))
        for i in range(_L_spinfull):
            if i%2 == 1:
                H[i,:] *= -1
        H[0, _L_spinfull-2] = H[_L_spinfull-2, 0] = -1
        H[1, _L_spinfull-1] = H[_L_spinfull-1, 1] = 1
        return H

    chi_max = 1000
    svd_min = 1e-6
    trunc_par = {"chi_max": chi_max, "svd_min": svd_min, "degeneracy_tol": 1e-12}
    if finite:
        H = tight_binding(L)
        C, _ = slater.correlation_matrix(H, N=L)
        mps = slater.C_to_MPS(C, trunc_par)
    else:
        L_short, L_long = L, L+1
        H_short = tight_binding(L_short)
        C_short, _ = slater.correlation_matrix(H_short, N=L_short)
        H_long = tight_binding(L_long)
        C_long, _ = slater.correlation_matrix(H_long, N=L_long)
        mps, error = slater.C_to_iMPS(C_short, C_long, trunc_par, sites_per_cell=2,
                                      cut=L_short // 2)

    # mps_gutz = gutz.abrikosov(mps)
    mps_gutz = gutz.abrikosov_ph(mps, return_canonical=False)
    mps_gutz.canonical_form()
    # for i_B, B in enumerate(mps_gutz._B):
    #    mps_gutz._B[i_B] = B.drop_charge()
    if finite:
        mps_gutz.unit_cell_width = L
    else:
        mps_gutz.unit_cell_width = 1
    if not finite:
        mps_gutz.canonical_form()

    return mps_gutz


def HaldaneShastry(L=20):
    site = SpinHalfSite(conserve="Sz")
    chain = Chain(L, site, bc='periodic')
    haldane_shastry_model = HaldaneShastryModel({"lattice":chain})
    print(haldane_shastry_model.all_coupling_terms().to_TermList())
    psi_dmrg = MPS.from_product_state(chain.mps_sites(), ["up"] * (L // 2) + ["down"] * (L - L//2))
    RunDMRG(haldane_shastry_model, psi_dmrg, plot_convergence=True, print_final_results=True)
    psi_gutz = GutzwillerState(L)
    energy_dmrg = haldane_shastry_model.H_MPO.expectation_value(psi_dmrg)
    energy_gutz = haldane_shastry_model.H_MPO.expectation_value(psi_gutz)
    # expected_energy = (-1) * ((L / (2*pi)) ** 2) * (pi ** 2 / 24) * (L + 5. / L)
    expected_energy = (-pi**2 / 24.) * (L + 5. / L)
    # expected_energy = (-1 / 24.) * ((2*pi/L) ** 2) * ((L**2 - 1)) * (L / 2)
    print(f"Energy per-site, dmrg: {energy_dmrg / L}")
    print(f"Energy per-site, Gutzwiller state: {energy_gutz / L}")
    print(f"Expected energy per-site: {expected_energy / L}")
    print("overlap between dmrg wavefunction and Gutzwiller wavefunction: ", np.abs(psi_dmrg.overlap(psi_gutz)))

    spin_correlations_dmrg = GetSpinSpinCorrelations(psi_dmrg)
    spin_correlations_gutz = GetSpinSpinCorrelations(psi_gutz)
    print(spin_correlations_gutz.shape)
    fig,ax = plt.subplots()
    ax.plot(spin_correlations_dmrg[L//2, :], "ro", label="dmrg")
    ax.plot(spin_correlations_gutz[L//2, :], "b^", label="Gutzwiller")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    L = 18
    # GutzwillerState(L, False)
    HaldaneShastry(L)