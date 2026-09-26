"""Tests for TyEffective.py (T_eff, Eq. 25-27 of Van Damme et al., PRB 104, 115142).

Run:  python TestTyEffective.py            (all fast tests, ~2 min)
      python TestTyEffective.py product    (a single test by name)
      python TestTyEffective.py afm        (slow: square-lattice Heisenberg AFM, energy at M)
      python TestTyEffective.py afm_disp   (slow: same model, Ly=6, dispersion Gamma -> M, saves a figure)
"""
import sys
import numpy as np

from tenpy.linalg import np_conserved as npc
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.uniform_mps import UniformMPS
from tenpy.algorithms.plane_wave_excitation import construct_orthogonal

from TyEffective import MixedGroundState, TyEffectiveOperator, ky_from_X

np.random.seed(1234)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def tangent_data(psi):
    """AL, AR, VL of a canonical infinite MPS, as the PlaneWaveExcitationEngine builds them."""
    u = UniformMPS.from_MPS(psi)
    L = psi.L
    ALs = [u.get_AL(i) for i in range(L)]
    ARs = [u.get_AR(i) for i in range(L)]
    VLs = [construct_orthogonal(AL) for AL in ALs]
    return ALs, ARs, VLs, u.get_C(0)


def random_X(ALs, VLs, qtotal):
    L = len(ALs)
    X = []
    for i in range(L):
        vL = VLs[i].get_leg('vR').conj()
        vR = ALs[(i + 1) % L].get_leg('vL').conj()

        def rand(shape):
            return np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)

        if vL.ind_len == 0:  # V_L has an empty complement (bond dimension = d * chi_left)
            X.append(npc.zeros([vL, vR], dtype=np.complex128, qtotal=qtotal, labels=['vL', 'vR']))
            continue
        X.append(npc.Array.from_func(rand, [vL, vR], dtype=np.complex128, qtotal=qtotal,
                                     labels=['vL', 'vR']))
    return X


def X_basis(ALs, VLs, qtotal):
    """Orthonormal basis of the X-space (list of X-lists), by exploding a random X into unit vectors."""
    L = len(ALs)
    basis = []
    template = random_X(ALs, VLs, qtotal)
    for i in range(L):
        dense = template[i].to_ndarray()
        for idx in np.ndindex(*dense.shape):
            arr = np.zeros_like(dense)
            arr[idx] = 1.
            cand = npc.Array.from_ndarray(arr, template[i].legs, dtype=np.complex128,
                                          qtotal=template[i].qtotal, labels=['vL', 'vR'],
                                          cutoff=0.)
            if abs(npc.norm(cand) - 1.) > 1.e-12:
                continue  # forbidden by charge conservation
            vec = [npc.Array.zeros_like(t) for t in template]
            vec[i] = cand
            basis.append(vec)
    return basis


def dense_matrix(op, basis):
    n = len(basis)
    M = np.zeros((n, n), dtype=complex)
    for b, e in enumerate(basis):
        Te = op.matvec(e)
        for a, f in enumerate(basis):
            M[a, b] = sum(npc.inner(fi, ti, axes='range', do_conj=True) for fi, ti in zip(f, Te))
    return M


def product_state_psi(Ly, conserve='Sz', state='down'):
    site = SpinHalfSite(conserve=conserve)
    psi = MPS.from_product_state([site] * Ly, [state] * Ly, bc='infinite')
    return psi


# ----------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------
def test_product():
    """chi = 1 product state: T_eff is the cyclic shift of the Ly numbers X_y, for any p."""
    for Ly in [1, 2, 3, 4]:
        psi = product_state_psi(Ly)
        ALs, ARs, VLs, C0 = tangent_data(psi)
        gs = MixedGroundState(ALs, ARs, C0, Ly)
        assert abs(gs.mu - 1) < 1.e-10, gs.mu
        for p in [0., 0.7]:
            T = TyEffectiveOperator(gs, VLs, p)
            X = random_X(ALs, VLs, qtotal=[2])
            TX = T.matvec(X)
            x = np.array([xi.to_ndarray().item() for xi in X])
            tx = np.array([ti.to_ndarray().item() for ti in TX])
            # content moves y -> y+1: (T X)_{y+1} = X_y
            assert np.allclose(tx, np.roll(x, 1)), (Ly, p, x, tx)
            ky, dist = ky_from_X(T, [npc.Array.from_ndarray(
                np.eye(1) * np.exp(1j * 2 * np.pi * 1 * y / Ly), X[y].legs, dtype=complex,
                qtotal=[2], labels=['vL', 'vR']) for y in range(Ly)])
            assert abs(((ky - 2 * np.pi / Ly + np.pi) % (2 * np.pi)) - np.pi) < 1.e-10 and dist < 1.e-10
        print(f'product state Ly={Ly}: T_eff = cyclic shift  [ok]')


def _trivial(arr, qconj, labels):
    """npc array with trivial charges and the given leg directions."""
    legs = [npc.LegCharge.from_trivial(n, qconj=q) for n, q in zip(arr.shape, qconj)]
    return npc.Array.from_ndarray(arr, legs, dtype=complex, labels=labels)


def _random_chain(Ly, chi, d=2):
    """Random (non-canonical) tensors A_g(chi, d, chi) as npc arrays with trivial charges."""
    arrs = [np.random.standard_normal((chi, d, chi)) + 1j * np.random.standard_normal((chi, d, chi))
            for _ in range(Ly)]
    npcs = [_trivial(a, [+1, +1, -1], ['vL', 'p', 'vR']) for a in arrs]
    return arrs, npcs


def test_cell_vs_dense():
    """left_cell / right_cell (shifted ket + carry leg) against <psi|T_y psi> from a dense state."""
    from TyEffective import left_cell, right_cell, _Cell
    for Ly, Lx, chi, ncell in [(1, 1, 3, 2), (2, 1, 3, 2), (3, 1, 3, 2), (2, 2, 2, 2), (3, 2, 2, 1)]:
        L = Ly * Lx
        arrs, npcs = _random_chain(L, chi)
        vL = np.random.standard_normal(chi) + 1j * np.random.standard_normal(chi)
        vR = np.random.standard_normal(chi) + 1j * np.random.standard_normal(chi)
        n = L * ncell
        # dense state
        psi = np.zeros((2,) * n, dtype=complex)
        for s in np.ndindex(*psi.shape):
            v = vL
            for g in range(n):
                v = v @ arrs[g % L][:, s[g], :]
            psi[s] = v @ vR
        axes = []
        for col in range(n // Ly):
            axes += [col * Ly + (k - 1) % Ly for k in range(Ly)]
        Tpsi = psi.transpose(axes)
        exact = np.vdot(psi, Tpsi)
        cell = _Cell(L, Ly)
        E = _trivial(np.outer(vL.conj(), vL), [+1, -1], ['vR*', 'vR'])
        F = _trivial(np.outer(vR.conj(), vR), [-1, +1], ['vL*', 'vL'])
        El = E
        for _ in range(ncell):
            El = left_cell(El, npcs, npcs, cell)
        val_l = npc.tensordot(El, F, axes=(['vR*', 'vR'], ['vL*', 'vL']))
        Fr = F
        for _ in range(ncell):
            Fr = right_cell(Fr, npcs, npcs, cell)
        val_r = npc.tensordot(E, Fr, axes=(['vR*', 'vR'], ['vL*', 'vL']))
        assert abs(val_l - exact) < 1.e-10 * abs(exact) and abs(val_r - exact) < 1.e-10 * abs(exact), \
            (Ly, Lx, val_l, val_r, exact)
        print(f'cell passes vs dense <psi|T_y psi>: Ly={Ly} Lx={Lx} chi={chi} cells={ncell}  [ok]')


def random_infinite_mps(L, chi, conserve=None, d_state='up'):
    site = SpinHalfSite(conserve=conserve)
    psi = MPS.from_random_unitary_evolution([site] * L, chi, [d_state, 'down'] * (L // 2) + ['up'] * (L % 2),
                                            bc='infinite')
    psi.canonical_form()
    return psi


def brute_force_F(gs, VLs, Xp, X, p, M):
    """<Phi_p(B')|T_y|Phi_p(B)> / <Psi|T_y|Psi>, explicit sum over the cell offset m of the ket
    excitation (|m| <= M), site pair (i, j); every chain is contracted from scratch."""
    from TyEffective import left_cell
    L, cell = gs.L, gs.cell
    AL, AR = gs.ALs, gs.ARs
    Bp = [npc.tensordot(VLs[g], Xp[g], axes=(['vR'], ['vL'])) for g in range(L)]
    B = [npc.tensordot(VLs[g], X[g], axes=(['vR'], ['vL'])) for g in range(L)]

    def cell_tensors(n, exc_cell, exc_site, Bexc):
        if n < exc_cell:
            return AL
        if n > exc_cell:
            return AR
        return [AL[g] if g < exc_site else (Bexc[g] if g == exc_site else AR[g]) for g in range(L)]

    total = 0.
    for m in range(-M, M + 1):
        for i in range(L):
            for j in range(L):
                E = gs.l
                n0, n1 = min(0, m), max(0, m)
                for n in range(n0, n1 + 1):
                    bra = cell_tensors(n, 0, i, Bp)
                    ket = cell_tensors(n, m, j, B)
                    E = (1. / gs.mu) * left_cell(E, bra, ket, cell)
                val = npc.tensordot(E, gs.r, axes=(['vR*', 'vR'], ['vL*', 'vL']))
                total += np.exp(-1j * p * L * m) * val
    return total


def test_bruteforce(chi=3, M=6):
    """X'^dag T_eff X against an independent explicit chain summation, random (non-symmetric) MPS,
    charge-conserving and not, Lx = 1 and 2, momenta p != 0."""
    from tenpy.algorithms.plane_wave_excitation import PlaneWaveExcitationEngine  # noqa
    for Ly, Lx, conserve, qtot in [(2, 1, None, None), (3, 1, 'Sz', [2]), (2, 2, None, None),
                                   (3, 2, None, None)]:
        L = Ly * Lx
        psi = random_infinite_mps(L, chi, conserve)
        ALs, ARs, VLs, C0 = tangent_data(psi)
        gs = MixedGroundState(ALs, ARs, C0, Ly)
        p = 0.37
        T = TyEffectiveOperator(gs, VLs, p, sum_tol=0., sum_iterations=M - 1)
        Xp = random_X(ALs, VLs, qtot)
        X = random_X(ALs, VLs, qtot)
        TX = T.matvec(X)
        val = sum(npc.inner(a, b, axes='range', do_conj=True) for a, b in zip(Xp, TX))
        ref = brute_force_F(gs, VLs, Xp, X, p, M)
        print(f'Ly={Ly} Lx={Lx} conserve={conserve}: X\'^dag T X = {val:.10f}   brute force = {ref:.10f}')
        assert abs(val - ref) < 1.e-9 * max(1., abs(ref)), (val, ref)
    print('operator matches brute-force chain sums  [ok]')


def ghz_column_psi(Ly):
    """Infinite MPS with unit cell Ly: product over columns of (|0..0> + |1..1>)/sqrt2 of spin-1
    sites (d = 3, so that no V_L complement is empty). Bond dimension 1 between columns and 2
    inside a column; entangled and exactly invariant under T_y."""
    from tenpy.networks.site import SpinSite
    site = SpinSite(S=1, conserve=None)
    Bflat = []
    for k in range(Ly):
        chiL = 1 if k == 0 else 2
        chiR = 1 if k == Ly - 1 else 2
        B = np.zeros((3, chiL, chiR))
        for s in range(2):
            a = 0 if chiL == 1 else s
            b = 0 if chiR == 1 else s
            B[s, a, b] = 1.
        Bflat.append(B)
    psi = MPS.from_Bflat([site] * Ly, Bflat, bc='infinite', form=None)
    psi.canonical_form()
    return psi


def test_ghz():
    """Exactly T_y-symmetric, entangled state with non-uniform bond dimensions: |mu| = 1 and T_eff
    is a partial isometry (T_y maps some tangent directions out of the tangent space, those give
    singular value 0), whose non-zero eigenvalues are exactly the Ly-th roots of unity, for any p."""
    for Ly in [2, 3, 4]:
        psi = ghz_column_psi(Ly)
        ALs, ARs, VLs, C0 = tangent_data(psi)
        gs = MixedGroundState(ALs, ARs, C0, Ly)
        assert abs(abs(gs.mu) - 1) < 1.e-10, gs.mu
        basis = X_basis(ALs, VLs, None)
        for p in [0., 1.1]:
            T = TyEffectiveOperator(gs, VLs, p)
            M = dense_matrix(T, basis)
            sv = np.linalg.svd(M, compute_uv=False)
            ev = np.linalg.eigvals(M)
            nz = ev[np.abs(ev) > 1.e-8]
            roots = np.angle(nz) / (2 * np.pi) * Ly
            print(f'GHZ Ly={Ly} p={p}: dim X = {len(basis)}, |mu| = {abs(gs.mu):.12f}, '
                  f'singular values in {{0,1}}: {np.allclose(sv * (1 - sv), 0, atol=1e-8)}, '
                  f'#nonzero eigenvalues = {len(nz)}, roots of unity: {np.allclose(roots, np.round(roots), atol=1e-8)}')
            assert np.all(sv < 1 + 1.e-8) and np.allclose(sv * (1 - sv), 0, atol=1.e-8)
            assert np.allclose(np.abs(nz), 1., atol=1.e-8)
            assert np.allclose(roots, np.round(roots), atol=1.e-8)
    print('T_eff on exactly symmetric states: partial isometry with Ly-th-root-of-unity spectrum  [ok]')


def _polarized_setup(Ly, Lx):
    """All-down product state of the XY model on a Ly x Lx cylinder: a single spin flip is an exact
    magnon with E(kx, ky) = cos(kx) + cos(ky)."""
    import QuasiparticleAnsatz as QA
    M = QA.make_spin_model(Ly=Ly, Lx=Lx, Jxy=1.0, Jz=0.0, conserve='Sz', chain=False)
    sites = M.lat.mps_sites()
    psi = MPS.from_product_state(sites, ['down'] * len(sites), bc='infinite')
    uni = UniformMPS.from_MPS(psi)
    return QA, M, uni


def test_polarized_dispersion():
    """End-to-end: targeted-ky engine reproduces cos(kx)+cos(ky) exactly in every ky sector, also
    when alpha starts far too small (the retry loop must enlarge it)."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    for Ly, Lx in [(4, 1), (3, 1), (4, 2)]:
        QA, M, uni = _polarized_setup(Ly, Lx)
        momenta = np.array([0.0, 0.9, 2.5])
        kys = 2 * np.pi * np.arange(Ly) / Ly
        for alpha in [5.0, 0.05]:
            E = QA.compute_dispersion_targeted(uni, M, momenta, qtotal_change=[2], alpha=alpha,
                                               lanczos_N_max=30, verbose=False)
            exact = np.cos(momenta)[:, None] + np.cos(kys)[None, :]
            err = np.nanmax(np.abs(E - exact))
            print(f'polarized XY Ly={Ly} Lx={Lx} alpha0={alpha}: max |E - (cos kx + cos ky)| = {err:.2e},'
                  f' sectors missed: {int(np.isnan(E).sum())}')
            assert not np.isnan(E).any() and err < 1.e-8


def test_vs_measure_ky(chi=8):
    """ky_from_X (new, T_eff) against the old measure_ky (dense ring trace) on the same excitation."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import QuasiparticleAnsatz as QA
    Ly = 4
    M = QA.make_spin_model(Ly=Ly, Lx=1, Jxy=1.0, Jz=0.0, conserve='Sz', chain=False)
    uni, M = QA.find_ground_state(M, chi_max=chi, magz=0.0)
    eng = QA.TyTargetedPWEngine(uni, M, {'lanczos_params': {'N_max': 30}})
    gs = eng.prepare_Ty(Ly)
    coords = QA.get_unit_cell_coords(M)
    for kx in [0.0, 1.3]:
        p = kx / Ly
        eng.set_target_k(kx, 0.0, coords)
        Es, psis, N = eng.run(p, qtotal_change=[2], num_ev=1)
        X = psis[0]._X
        T = TyEffectiveOperator(gs, eng.VLs, p, sum_tol=1.e-12, sum_iterations=300)
        k_new, d_new = ky_from_X(T, X)
        k_old, d_old, _ = QA.measure_ky(eng, X, p, 0.0, coords, Ly, 1)
        print(f'kx={kx}: new (ky, 1-|t|) = ({k_new:.6f}, {d_new:.6f})   old measure_ky = ({k_old:.6f}, {d_old:.6f})')
        # The phases agree up to the (small) phase of D/(l|r); the moduli differ by design:
        # measure_ky divides by (l|r) instead of (l| C* x C |r) (fixed points in different gauges),
        # so its |ratio| is O(1/chi) for a genuine eigenvector, whereas T_eff gives |ratio| ~ 1.
        assert abs(QA.wrap_angle(k_new - k_old)) < 1.e-2
        assert abs(d_new) < 0.05, d_new


def test_targeting_selects_sector():
    """Start from a random superposition of all ky sectors: only the Eq. 27 penalty can select the
    sector, in particular ky = +pi/2 versus 3pi/2 (equal energies, opposite T_eff phase)."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    Ly, Lx = 4, 1
    QA, M, uni = _polarized_setup(Ly, Lx)

    class RandomStart(QA.TyTargetedPWEngine):
        def initial_guess(self, qtotal_change):
            X = super(QA.KMomentumPWEngine, self).initial_guess(qtotal_change)
            return [(np.random.standard_normal() + 1j * np.random.standard_normal()) * x for x in X]

    eng = RandomStart(uni, M, {'lanczos_params': {'N_max': 30}})
    eng.prepare_Ty(Ly)
    for kx in [0.0, 0.9]:
        p = kx / Ly
        for n in range(Ly):
            ky = 2 * np.pi * n / Ly
            Es, psis, N, ky_meas, dist = eng.run_targeted(p, ky, alpha=5.0, qtotal_change=[2])
            exact = np.cos(kx) + np.cos(ky)
            print(f'kx={kx} ky_target={ky:.4f}: ky_measured={ky_meas[0]:.4f} E={Es[0]:.10f} exact={exact:.10f}')
            assert abs(QA.wrap_angle(ky_meas[0] - ky)) < 1.e-6 and abs(Es[0] - exact) < 1.e-8
    print('penalty selects the requested ky sector (sign of e^{i ky} T_eff consistent)  [ok]')


def _afm_neel_ground_state(Ly, chi, nwin=12, trunc_tol=1.e-3):
    """Square-lattice J1 Heisenberg AFM, Sz_tot = 0, one-column unit cell. Returns (uni, M) after
    checking that the DMRG truncation error is small and that the static structure factor
    (WaveFunctionProperties) has its Neel peak at M = (pi, pi)."""
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import QuasiparticleAnsatz as QA
    from WaveFunctionProperties import CalculateSpinSpinCorrelations, ComputeMomentumSpaceStructureFactor

    M = QA.make_spin_model(Ly=Ly, Lx=1, Jxy=1.0, Jz=1.0, conserve='Sz', chain=False)
    uni, M, info = QA.find_ground_state(M, chi_max=chi, magz=0.0, return_info=True)
    print(f'ground state: Ly={Ly} chi={chi}  E0/site = {info["E0"]:.8f}, '
          f'max DMRG truncation error = {info["max_trunc_err"]:.1e}')
    assert info['max_trunc_err'] < trunc_tol, 'ground state not converged for this bond dimension'
    C = CalculateSpinSpinCorrelations(info['psi'], inf_mps_unitcell_fac=nwin)
    ks, Sk = ComputeMomentumSpaceStructureFactor(C, M.lat, n1=4, n2=Ly)  # grid contains kx = pi
    kfold = (ks[:, :2] + np.pi) % (2 * np.pi) - np.pi
    at_M = np.all(np.isclose(np.abs(kfold), np.pi, atol=1.e-6), axis=1)
    print(f'S(M) = {Sk[at_M].max():.2f}; largest S(k) elsewhere = {Sk[~at_M].max():.2f}; '
          f'argmax S(k) = ({kfold[np.argmax(Sk)][0]:+.3f}, {kfold[np.argmax(Sk)][1]:+.3f})')
    assert at_M[np.argmax(Sk)], 'static structure factor does not peak at (pi, pi)'
    assert Sk[at_M].max() > 3 * Sk[~at_M].max(), 'Neel peak at M not pronounced'
    return uni, M


def _magnon_energy(uni, M, kx, ky, alpha=1.0):
    """Sz = +1 excitation energy at (kx, ky), ky imposed through Eq. 27."""
    import QuasiparticleAnsatz as QA
    E = QA.compute_dispersion_targeted(uni, M, np.array([kx]), qtotal_change=[2], alpha=alpha,
                                       verbose=False, kys=[ky])
    return float(E[0, 0])


def afm_excitation_energy(Ly, chi, kx, ky, trunc_tol=5.e-3):
    """Sz = +1 excitation energy at (kx, ky) of the square-lattice Heisenberg AFM on a
    circumference-Ly cylinder with bond dimension chi (ground state recomputed on every call)."""
    uni, M = _afm_neel_ground_state(Ly, chi, trunc_tol=trunc_tol)
    return _magnon_energy(uni, M, kx, ky)


def test_afm_square_goldstone(Ly=4, chi=30, nwin=12, trunc_tol=1.e-3, gap_tol=0.3):
    """Square-lattice nearest-neighbour Heisenberg AFM (J1 only), Sz_tot = 0 sector, one-column
    unit cell (Lx = 1), Ly even so that M = (pi, pi) is an allowed momentum. Benchmark: Fig. 4 of
    Sherman, Dupont, Moore, PRB 107, 165146 (2023).

    1. ground state sanity: small DMRG truncation error and a Neel peak of the static structure
       factor S(k) (WaveFunctionProperties) at M = (pi, pi);
    2. spin-1 (Sz = +1) magnon energy at M with kx = ky = pi imposed via Eq. 27: this is the
       Goldstone point, where linear spin-wave theory has omega = 0;
    3. the same at X = (pi, 0), where the reference dispersion is at its maximum (~2.4 in Fig. 4d).

    The energy at M is *not* exactly 0 on a finite-circumference cylinder: the paper itself notes
    that SWT is gapless at M but "any finite system will always have a gap" (its own eps(M) is
    ~0.1 at chi = 512). Here the assertion is that E(M) is small (below `gap_tol`, i.e. a few
    percent of the bandwidth) and far below E(X); the numbers are printed. Takes a few minutes.
    """
    uni, M = _afm_neel_ground_state(Ly, chi, nwin, trunc_tol)  # (1) truncation error, Neel peak of S(k)

    # (2), (3) excitation energies, ky imposed through T_eff
    E_M, E_X = _magnon_energy(uni, M, np.pi, np.pi), _magnon_energy(uni, M, np.pi, 0.0)
    print(f'magnon energy: E(M=(pi,pi)) = {E_M:.4f}   E(X=(pi,0)) = {E_X:.4f}   (SWT / QMC at X ~ 2.4)')
    assert -0.05 < E_M < gap_tol, E_M
    assert 2.0 < E_X < 3.0, E_X
    assert E_M < 0.2 * E_X
    print('Goldstone-like minimum at M, Neel peak of S(k) at M, correct scale at X  [ok]')


def test_afm_dispersion_gamma_M(Ly=6, chi=80, savepath=None, trunc_tol=5.e-3):
    """Magnon dispersion of the square-lattice Heisenberg AFM on a circumference-Ly cylinder along
    the diagonal Gamma=(0,0) -> M=(pi,pi), i.e. (k, k), compared with linear spin-wave theory
    (the SWT curve of Fig. 4d of Sherman, Dupont, Moore, PRB 107, 165146: 2 Z_c |sin k| with
    Z_c = 1.18, maximum ~2.4). Saves the figure.

    On a circumference-Ly cylinder ky is quantised to 2 pi n / Ly, so only the points
    k = 2 pi n / Ly (n = 0 .. Ly/2) lie exactly on the diagonal: 4 points for Ly = 6.
    Weak assertions: the two interior points are far above the two end points (Gamma, M),
    and lie within 25% of the SWT value. Values at Gamma and M are finite-size / finite-chi
    dependent (see test_afm_square_goldstone) and are only printed.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    uni, M = _afm_neel_ground_state(Ly, chi, trunc_tol=trunc_tol)
    ks = 2 * np.pi * np.arange(Ly // 2 + 1) / Ly
    E = np.array([_magnon_energy(uni, M, k, k) for k in ks])
    Zc = 1.18
    E_swt = 2 * Zc * np.abs(np.sin(ks))
    for k, e, e0 in zip(ks, E, E_swt):
        print(f'  k = ({k / np.pi:.3f} pi, {k / np.pi:.3f} pi):  E = {e:.4f}   (SWT {e0:.4f})')

    savepath = savepath or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        f'afm_dispersion_Gamma_M_Ly{Ly}_chi{chi}.png')
    kk = np.linspace(0, np.pi, 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(kk / np.pi, 2 * Zc * np.abs(np.sin(kk)), 'k-', lw=1, label=r'linear SWT ($Z_c=1.18$)')
    ax.plot(ks / np.pi, E, 'o', color='tab:red', label=rf'MPS quasiparticle, $L_y={Ly}$, $\chi={chi}$')
    ax.set_xlabel(r'$k/\pi$  along $(k,k)$:  $\Gamma \to M$')
    ax.set_ylabel(r'$\epsilon(k,k)/J$')
    ax.set_xlim(-0.03, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    np.savetxt(savepath.replace('.png', '.dat'), np.column_stack([ks / np.pi, E, E_swt]),
               header='k/pi (kx=ky=k)   E_MPS   E_SWT')
    print(f'figure saved to {savepath}')

    assert not np.isnan(E).any()
    interior = (ks > 0) & (ks < np.pi)
    assert np.all(E[interior] > 3 * np.maximum(E[0], E[-1]))
    assert np.all(np.abs(E[interior] / E_swt[interior] - 1) < 0.25), (E, E_swt)
    print('dispersion along Gamma -> M: arch with minima at Gamma and M, SWT scale  [ok]')


SLOW = {'afm', 'afm_disp'}  # run only when requested by name: python TestTyEffective.py afm
TESTS = {'product': test_product, 'cells': test_cell_vs_dense, 'brute': test_bruteforce, 'ghz': test_ghz,
         'polarized': test_polarized_dispersion, 'vs_old': test_vs_measure_ky,
         'sector': test_targeting_selects_sector,
         'afm': test_afm_square_goldstone,
         'afm_disp': test_afm_dispersion_gamma_M}

if __name__ == '__main__':
    names = sys.argv[1:] or [n for n in TESTS if n not in SLOW]
    for name in names:
        print(f'--- {name} ---')
        TESTS[name]()
    print('all requested tests passed')