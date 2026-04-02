import tenpy
from tenpy import MPS, FermionSite
import tenpy.linalg.np_conserved as npc
import numpy as np
from numpy import log, sin, cos, sqrt
from numpy.linalg import matmul

def TestSpinSystem():
    chinfo = npc.ChargeInfo([1])  # just a U(1) charge
    # charges for up, down state
    p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [-1]])
    Sz = npc.Array.from_ndarray([[0.5, 0.0], [0.0, -0.5]], [p_leg, p_leg.conj()])
    Sp = npc.Array.from_ndarray([[0.0, 1.0], [0.0, 0.0]], [p_leg, p_leg.conj()])
    Sm = npc.Array.from_ndarray([[0.0, 0.0], [1.0, 0.0]], [p_leg, p_leg.conj()])

    # print(Sm.legs)
    print(Sz._data)
    print(Sz._qdata)
    print(Sm._data)
    print(Sm._qdata)

    Hxy = 0.5 * (npc.outer(Sp, Sm) + npc.outer(Sm, Sp))
    Hz = npc.outer(Sz, Sz)
    H = Hxy + Hz

    H.iset_leg_labels(['s1', 't1', 's2', 't2'])
    H = H.combine_legs([['s1', 's2'], ['t1', 't2']], qconj=[+1, -1])
    print(H.legs[0].to_qflat().flatten())
    print(H.qtotal)

def TestFermionSystem(product=True):
    site1 = FermionSite(conserve='N')
    site1.leg.label = "p0"
    site2 = FermionSite(conserve='N')
    site2.leg.label = "p1"
    sites = [site1, site2]
    # p_leg1 = npc.LegCharge.from_qflat(chinfo, [[1], [0]])
    # p_leg2 = npc.LegCharge.from_qflat(chinfo, [[1], [0]])

    if product:
        product_state = ["full", "full"]
        psi_mps = MPS.from_product_state(sites, product_state)

    else:
        psi_nd_arr = np.zeros((2, 2))
        psi_nd_arr[0, 1] = 1 / sqrt(2)
        psi_nd_arr[1, 0] = 1 / sqrt(2)
        psi = npc.Array.from_ndarray(psi_nd_arr, [site1.leg, site2.leg], labels=["p0", "p1"])
        psi_mps = MPS.from_full(sites, psi)

        A1_empty = npc.Array.to_ndarray(psi_mps.get_B(0))[:,0,:]
        A1_filled = npc.Array.to_ndarray(psi_mps.get_B(0))[:,1,:]
        A2_empty = npc.Array.to_ndarray(psi_mps.get_B(1))[:, 0, :]
        A2_filled = npc.Array.to_ndarray(psi_mps.get_B(1))[:, 1, :]
        print("00 magnitude: ", matmul(A1_empty, A2_empty))
        print("01 magnitude: ", matmul(A1_empty, A2_filled))
        print("10 magnitude: ", matmul(A1_filled, A2_empty))
        print("11 magnitude: ", matmul(A1_filled, A2_filled))

        print("correlation matrix: ", psi_mps.correlation_function("Cd", "C"))

    print(f"Total sites: {psi_mps.L}")
    print(f"Charge names: {psi_mps.chinfo.names}")
    print(f"Total charge (particle number): {psi_mps.get_total_charge(only_physical_legs=True)}")

    #for i in range(psi.L):
    #    print(f"Site {i} state: {product_state[i]}")

if __name__ == "__main__":
    TestFermionSystem(False)
