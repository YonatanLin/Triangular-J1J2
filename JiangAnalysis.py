"""
Per-case analysis of a ground state along the lines of S. Jiang, White, Kivelson and H.-C. Jiang, PRL 137, 056703
(2026) (the LE / HE states of the J1-J2 model on the triangular lattice): the sign of the real-space correlations and
its Fourier transform (their Fig. 5), the nearest neighbor correlations by bond type (their Fig. 1), the J1-J2
energies and the Schmidt spectrum. Run by the "jiang_analysis" post-process (PostProcess.py); the comparisons between
states are in Analysis.py.
"""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tenpy.networks.site import SpinHalfSite

from Main import BuildTriangularLattice, calculateGutzwillerEnergyTriangularJ1J2
from WaveFunctionProperties import (plot_structure_factor, minimum_image_displacements, _reference_site_index,
                                    k_grid)

KS_FILES = {"xx": "ks_xx.csv", "zz": "ks_zz.csv"}
CORR_FILES = {"xx": "spin_corr_xx_x.csv", "zz": "spin_corr_zz_x.csv"}
JIANG_J2S = (0.1, 0.11, 0.125, 0.13, 0.14)
SIGN_AMPLITUDE_CUTOFF = 5e-5  # Fig. 5 of Jiang et al.: sites with smaller |<S_r0.S_r>| have no definite sign

def _case_title(results_dir):
    """Short title from the directory name, e.g. 'J2=0.08, Delz=0.7' (falls back to the directory name)."""
    name = Path(results_dir).name
    j2 = re.search(r"J2_([0-9.]+?)(?:_|$)", name)
    delz = re.search(r"Delz_([0-9.]+?)_", name)
    if j2:  # no Delz in the name: an isotropic (Heisenberg) run
        return rf"$J_2$={j2.group(1)}, $\Delta$={delz.group(1) if delz else 1}"

    vz = re.search(r"jasNNzz_([0-9.]+?)_", name)
    if vz:
        return r"$v_{z}=$" + rf"${vz.group(1)}$"
    vxy = re.search(r"jasNNxy_([0-9.]+?)_", name)
    if vxy:
        return r"$v_{xy}=$" + rf"${vxy.group(1)}$"
    if name.startswith("Dirac"):
        return "no Jastrow"
    return name


def _reference_lattice():
    """1x1 YC triangular lattice (lattice spacing 1). plot_structure_factor only needs the lattice for its
    reciprocal basis (to tile the BZ), which is the same for all triangular runs, so the lattice.pkl of each
    run is not needed (it is not copied from the cluster)."""
    return BuildTriangularLattice(1, 1, SpinHalfSite(None), "finite", ("open", "open"), "YC")


def _state_label(results_dir):
    """e.g. 'flux=1.0' (+ ', monQ=1' / ', gs=1' when not 0), falls back to _case_title."""
    name = Path(results_dir).name
    flux = re.search(r"flux_([0-9.]+?)_", name)
    if flux is None:
        return _case_title(results_dir)
    label = f"flux={flux.group(1)}"
    for tag, key in (("monQ", "monQ"), ("gs", "gsindex")):
        value = re.search(rf"{key}_(-?\d+)", name)
        if value and value.group(1) != "0":
            label += f", {tag}={value.group(1)}"
    jastrow = _case_title(results_dir)
    return label if jastrow in ("no Jastrow", name) else f"{label}, {jastrow}"


def _LoadCorrelationsOnCylinder(results_dir):
    """
        <S_i.S_j> = 2 <SxSx> + <SzSz> (SxSx = SySy by U(1)) from the spin_components post-process
        (spin_corr_xx_x.csv, spin_corr_zz_x.csv), and the lattice of its sites (lattice.pkl). Not spin_corr_x.csv:
        in older runs it is on a different block of sites than lattice.pkl.
    """
    results_dir = Path(results_dir)
    C_xx, C_zz = (np.loadtxt(results_dir / CORR_FILES[comp]) for comp in ("xx", "zz"))
    with open(results_dir / "lattice.pkl", "rb") as f:
        lat = pickle.load(f)
    assert C_xx.shape == C_zz.shape == (lat.N_sites, lat.N_sites), \
        f"correlations of shape {C_xx.shape} aren't on the {lat.N_sites} sites of lattice.pkl"
    return 2 * C_xx + C_zz, lat


def _LoadTotalStructureFactor(results_dir):
    """ks and S(k) = 2 S_xx(k) + S_zz(k) from ks_xx.csv, ks_zz.csv."""
    xx, zz = (np.loadtxt(Path(results_dir) / KS_FILES[comp]) for comp in ("xx", "zz"))
    assert np.allclose(xx[:, :2], zz[:, :2]), "xx and zz structure factors are on different k grids"
    return xx[:, :2], 2 * xx[:, 2] + zz[:, 2]


def _Geometry(results_dir):
    """'YC' or 'XC', from the directory name."""
    geometry = re.search(r"_(YC|XC)(?:_|$)", Path(results_dir).name)
    assert geometry is not None, f"no geometry (YC / XC) in the directory name {Path(results_dir).name}"
    return geometry.group(1)


def _LoadLongCorrelationsBlock(results_dir):
    """
        <S_i.S_j> of spin_corr_x.csv (older runs: between the first N sites of the infinite MPS, more columns than
        lattice.pkl) and the cylinder of its N / (sites per column) columns, with the geometry of the directory name
        and the circumference and boundary conditions of lattice.pkl.
    """
    results_dir = Path(results_dir)
    C = np.real_if_close(np.loadtxt(results_dir / "spin_corr_x.csv", dtype=np.complex128), tol=1e6)
    with open(results_dir / "lattice.pkl", "rb") as f:
        lat_pkl = pickle.load(f)
    Ly = lat_pkl.Ls[1]
    sites_per_column = Ly * len(lat_pkl.unit_cell)
    assert C.shape[0] % sites_per_column == 0, \
        f"{C.shape[0]} correlated sites are not whole columns of {sites_per_column} sites"
    bc = tuple("open" if is_open else "periodic" for is_open in lat_pkl.bc)
    lat = BuildTriangularLattice(C.shape[0] // sites_per_column, Ly, SpinHalfSite(None), lat_pkl.bc_MPS, bc,
                                 _Geometry(results_dir))
    # the correlations are between the first sites of the MPS, which are the same sites in both lattices
    n_common = min(lat.N_sites, lat_pkl.N_sites)
    assert np.allclose([lat.position(lat.mps2lat_idx(i)) for i in range(n_common)],
                       [lat_pkl.position(lat_pkl.mps2lat_idx(i)) for i in range(n_common)]), \
        f"the rebuilt lattice doesn't match lattice.pkl of {results_dir.name}"
    return C, lat


def _NearestNeighborBonds(C, lat):
    """
        The nearest neighbor bonds of lat inside the correlated block (each counted once): a list of
        (column of the first site, r_i, r_j, <S_i.S_j>). r_j is not wrapped around the cylinder, so r_j - r_i is the
        bond vector.
    """
    Lx, Ly = lat.Ls
    bonds = []
    for u1, u2, dx in lat.pairs["nearest_neighbors"]:
        dx = np.asarray(dx)
        for i in range(lat.N_sites):
            x, y, u = lat.mps2lat_idx(i)
            if u != u1 or not 0 <= x + dx[0] < Lx:
                continue
            j = lat.lat2mps_idx([x + dx[0], (y + dx[1]) % Ly, u2])
            bonds.append((x, lat.position([x, y, u1]), lat.position([x + dx[0], y + dx[1], u2]), C[i, j]))
    return bonds


def _GutzwillerPsiPath(results_dir, psi_fname="psi_gutzwiller.pkl"):
    """psi_fname in results_dir, or else in its sibling pklFiles_<results_dir name> (None if in neither)."""
    results_dir = Path(results_dir)
    for d in (results_dir, results_dir.parent / f"pklFiles_{results_dir.name}"):
        if (d / psi_fname).exists():
            return d / psi_fname
    return None


CORR_ROW_FILE = "spin_corr_row_x.csv"
ENERGIES_FILE = "energies_vs_J2.txt"
DEFAULT_ROW_COLUMNS_PER_LY = 4  # columns of the window of CalculateCorrelationRow, in units of Ly


def _RowLattice(lat_pkl, geometry, n_columns):
    """The lattice of the first n_columns columns of the MPS of lattice.pkl (lat_pkl itself if finite)."""
    if lat_pkl.bc_MPS == "finite":
        return lat_pkl
    bc = tuple("open" if is_open else "periodic" for is_open in lat_pkl.bc)
    return BuildTriangularLattice(n_columns, lat_pkl.Ls[1], SpinHalfSite(None), lat_pkl.bc_MPS, bc, geometry)


def CalculateCorrelationRow(psi, results_dir, n_columns=None):
    """
        <S_r0.S_r> between a reference site r0 (the closest to the center) and all the sites of a window of
        n_columns columns (default DEFAULT_ROW_COLUMNS_PER_LY * Ly; for a finite MPS, all of lattice.pkl), longer
        than the window of lattice.pkl, for the sign Fourier transform of Fig. 5 of Jiang et al. (which only needs
        the correlations with r0). Saves CORR_ROW_FILE in results_dir: site, x, y, <S_r0.S_r>.
    """
    from WaveFunctionProperties import CalculateSpinSpinCorrelations
    results_dir = Path(results_dir)
    with open(results_dir / "lattice.pkl", "rb") as f:
        lat_pkl = pickle.load(f)
    if n_columns is None:
        n_columns = DEFAULT_ROW_COLUMNS_PER_LY * lat_pkl.Ls[1]
    lat = _RowLattice(lat_pkl, _Geometry(results_dir), n_columns)
    ref = _reference_site_index(lat, lat.N_sites)
    print(f"correlations of site {ref} with the {lat.N_sites} sites of {lat.Ls[0]} columns")
    row = np.real_if_close(CalculateSpinSpinCorrelations(psi, sites1=[ref], sites2=np.arange(lat.N_sites))[0],
                           tol=1e6)
    positions = np.array([lat.position(lat.mps2lat_idx(i)) for i in range(lat.N_sites)])
    np.savetxt(results_dir / CORR_ROW_FILE, np.column_stack([np.arange(lat.N_sites), positions, np.real(row)]),
               header=f"reference_site={ref} n_columns={lat.Ls[0]}; columns: site x y <S_r0.S_r>")
    return row, lat, ref


def _LoadCorrelationRow(results_dir):
    """(row, lattice, reference site) from CORR_ROW_FILE, the lattice rebuilt from lattice.pkl and the header."""
    results_dir = Path(results_dir)
    path = results_dir / CORR_ROW_FILE
    with open(path) as f:
        header = f.readline()
    ref = int(re.search(r"reference_site=(\d+)", header).group(1))
    n_columns = int(re.search(r"n_columns=(\d+)", header).group(1))
    data = np.loadtxt(path, ndmin=2)
    with open(results_dir / "lattice.pkl", "rb") as f:
        lat_pkl = pickle.load(f)
    lat = _RowLattice(lat_pkl, _Geometry(results_dir), n_columns)
    positions = np.array([lat.position(lat.mps2lat_idx(i)) for i in range(lat.N_sites)])
    assert np.allclose(positions, data[:, 1:3]), f"the rebuilt lattice doesn't match {path}"
    return data[:, 3], lat, ref


def _SignStructureFactor(row, lat, ks, ref, amplitude_cutoff=SIGN_AMPLITUDE_CUTOFF):
    """
        Fourier transform of the sign of the correlations row = <S_r0.S_r> with the reference site r0 = ref
        (Fig. 5 of Jiang et al.): S~(q) = sum_r e^{-iq.(r - r0)} <S_r0.S_r> / |<S_r0.S_r>|, sites with
        |<S_r0.S_r>| < amplitude_cutoff excluded. Returns (signs, S~(q)), signs is 0 for the excluded sites.
    """
    signs = np.sign(row)
    signs[np.abs(row) < amplitude_cutoff] = 0
    rdisp = minimum_image_displacements(lat, len(row))[:, ref]  # r - r0, wrapped around the cylinder
    return signs, np.exp(-1j * ks @ rdisp.T) @ signs


def PlotSignStructureFactor(results_dir, amplitude_cutoff=SIGN_AMPLITUDE_CUTOFF, lower_cutoff=0.2, save=True):
    """
        As Fig. 5 of Jiang et al.: left, the sign of <S_r0.S_r> on the correlated block of the cylinder (green +,
        yellow -, crosses for |<S_r0.S_r>| < amplitude_cutoff, r0 in a red circle); right, |S~(q)| normalized to its
        maximum, with a lower cutoff of the color scale, on the allowed ky of the cylinder. Uses the correlations of
        CalculateCorrelationRow (CORR_ROW_FILE), or, for older runs without it, the row of the site closest to the
        center of spin_corr_x.csv (whose window is longer than that of lattice.pkl).
        Saves sign_structure_factor.pdf in results_dir.
    """
    results_dir = Path(results_dir)
    if (results_dir / CORR_ROW_FILE).exists():
        row, lat, ref = _LoadCorrelationRow(results_dir)
    else:
        C, lat = _LoadLongCorrelationsBlock(results_dir)
        ref = _reference_site_index(lat, lat.N_sites)
        row = C[ref]
    ks = k_grid(lat, *lat.Ls, *lat.reciprocal_basis)
    signs, S_sign = _SignStructureFactor(row, lat, ks, ref, amplitude_cutoff)
    S_sign = np.abs(S_sign) / np.max(np.abs(S_sign))

    fig, (ax_x, ax_k) = plt.subplots(1, 2, figsize=(17, 5), width_ratios=[2.6, 1], constrained_layout=True)
    positions = np.array([lat.position(lat.mps2lat_idx(i)) for i in range(lat.N_sites)])
    for value, style in ((1, dict(color="tab:green", marker="o", label="+")),
                         (-1, dict(color="gold", marker="o", label="-")),
                         (0, dict(color="gray", marker="x", label=rf"$|C| < {amplitude_cutoff:g}$"))):
        on = signs == value
        if on.any():
            ax_x.scatter(*positions[on].T, s=40, edgecolors="k" if value else None, linewidths=0.5, **style)
    ax_x.scatter(*positions[ref], s=200, facecolors="none", edgecolors="red", linewidths=2)
    ax_x.set_aspect("equal")
    ax_x.set_xlabel("x")
    ax_x.set_ylabel("y")
    ax_x.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    ax_x.set_title(r"sign of $\langle \mathbf{S}_{r_0}\cdot\mathbf{S}_r\rangle$")

    _, artist = plot_structure_factor(ks, S_sign, lat, ax=ax_k, cmap="magma", show_kpoints=False)
    artist.set_clim(lower_cutoff, 1)
    fig.axes[-1].set_ylabel(r"$|\tilde S(\mathbf{q})| / \max|\tilde S|$")
    _reference_lattice().plot_brillouin_zone(ax_k)  # not lat's: for XC that is the BZ of the two site unit cell
    ax_k.set_title(r"$\tilde S(\mathbf{q}) = \sum_r e^{-i\mathbf{q}\cdot\mathbf{r}}\,"
                   r"\mathrm{sgn}\langle \mathbf{S}_{r_0}\cdot\mathbf{S}_r\rangle$", fontsize=11)
    fig.suptitle(_state_label(results_dir))
    if save:
        fig.savefig(results_dir / "sign_structure_factor.pdf", dpi=200)
    return fig


def PlotNearestNeighborCorrelations(results_dir, min_width=0.5, max_width=8.0, save=True):
    """
        <S_i.S_j> on the nearest neighbor bonds of the post-process correlations (lattice.pkl), each bond colored by
        its sign (blue < 0, red > 0) with a width growing linearly with |<S_i.S_j>| from min_width (smallest |C|) to
        max_width (largest |C|), so that the differences between the bonds are visible; bonds that wrap around the
        cylinder are drawn from the lower site, beyond the top row. Saves nearest_neighbor_correlations.pdf in
        results_dir.
    """
    results_dir = Path(results_dir)
    C, lat = _LoadCorrelationsOnCylinder(results_dir)
    bonds = [(r_i, r_j, c) for _, r_i, r_j, c in _NearestNeighborBonds(C, lat)]
    values = np.array([c for _, _, c in bonds])
    abs_min, abs_max = np.abs(values).min(), np.abs(values).max()

    def width(c):
        return min_width + (max_width - min_width) * (abs(c) - abs_min) / max(abs_max - abs_min, 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for r_i, r_j, c in bonds:
        ax.plot(*np.array([r_i, r_j]).T, color="tab:blue" if c < 0 else "tab:red", lw=width(c),
                solid_capstyle="butt", zorder=1)
    positions = np.array([lat.position(lat.mps2lat_idx(i)) for i in range(lat.N_sites)])
    ax.scatter(*positions.T, s=12, color="k", zorder=2)
    for c in sorted({values[np.argmin(np.abs(values))], values[np.argmax(np.abs(values))]}):  # legend entries
        ax.plot([], [], color="tab:blue" if c < 0 else "tab:red", lw=width(c),
                label=rf"$\langle \mathbf{{S}}_i\cdot\mathbf{{S}}_j\rangle = {c:.4f}$")
    ax.set_title(rf"nearest neighbor $\langle \mathbf{{S}}_i\cdot\mathbf{{S}}_j\rangle$, "
                 rf"range [{values.min():.4f}, {values.max():.4f}]", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    fig.suptitle(_state_label(results_dir))
    if save:
        fig.savefig(results_dir / "nearest_neighbor_correlations.pdf", dpi=200)
    return fig


# the bond types of Fig. 1(a) of Jiang et al. as real space directions, e1 = (1, 0), e2 = (1/2, sqrt(3)/2):
# bond 1 = e2 - e1, bond 2 = e1, bond 3 = e2. On our YC cylinder (wrapping along e2) bonds 1 and 2 are mirror
# partners and bond 3 goes around the cylinder; on our XC cylinder (wrapping along (0, sqrt(3))) bond 2 is along the
# cylinder and bonds 1 and 3 are the mirror partners.
JIANG_BOND_TYPES = {"Bond 1": (-0.5, np.sqrt(3) / 2), "Bond 2": (1.0, 0.0), "Bond 3": (0.5, np.sqrt(3) / 2)}
MATLAB_COLORS = ("#0072BD", "#D95319", "#EDB120")


def _BondType(r_i, r_j):
    """The JIANG_BOND_TYPES name of the nearest neighbor bond r_j - r_i (of either orientation)."""
    d = np.asarray(r_j) - np.asarray(r_i)
    for name, v in JIANG_BOND_TYPES.items():
        if np.isclose(abs(d @ np.asarray(v)), 1.0) and np.isclose(np.linalg.norm(d), 1.0):
            return name
    raise ValueError(f"{d} is not a nearest neighbor bond")


def PlotBondCorrelationsVsLx(results_dir, save=True):
    """
        As the middle panel of Fig. 1 of Jiang et al.: <S_i.S_{i+delta}> on the three types of nearest neighbor
        bonds (JIANG_BOND_TYPES, shown in the inset), averaged around the cylinder, vs. the column l_x of site i:
        bond 1, bond 2 - bond 1 and bond 3. Uses spin_corr_x.csv when it has a longer window along the cylinder
        than the spin_components post-process correlations (older runs), else the latter.
        Saves bond_correlations_vs_lx.pdf in results_dir.
    """
    results_dir = Path(results_dir)
    C, lat = _LoadCorrelationsOnCylinder(results_dir)
    if (results_dir / "spin_corr_x.csv").exists():
        C_long, lat_long = _LoadLongCorrelationsBlock(results_dir)
        if lat_long.N_sites > lat.N_sites:
            C, lat = C_long, lat_long
    Lx, Ly = lat.Ls
    by_column = {(name, x): [] for name in JIANG_BOND_TYPES for x in range(Lx)}
    for x, r_i, r_j, c in _NearestNeighborBonds(C, lat):
        by_column[(_BondType(r_i, r_j), x)].append(c)
    # columns in which every site has its bond of each type (each site has one bond of each type, counted once)
    sites_per_column = Ly * len(lat.unit_cell)
    lxs = np.array([x for x in range(Lx) if all(len(by_column[(name, x)]) == sites_per_column
                                                 for name in JIANG_BOND_TYPES)])
    bond = {name: np.array([np.mean(by_column[(name, x)]) for x in lxs]) for name in JIANG_BOND_TYPES}

    fig, ax = plt.subplots(figsize=(3.6, 3.0), constrained_layout=True)
    for (label, values), color in zip((("Bond 1", bond["Bond 1"]), ("Bond 2 - Bond 1", bond["Bond 2"] - bond["Bond 1"]),
                                       ("Bond 3", bond["Bond 3"])), MATLAB_COLORS):
        ax.plot(lxs, values, "-", color=color, lw=1.5, label=label)
    ax.set_ylim(-0.6, 0.05)
    ax.set_yticks([0, -0.2, -0.4, -0.6])
    ax.set_xlim(lxs.min(), lxs.max())
    ax.set_xlabel(r"$l_x$")
    ax.set_ylabel(r"$\langle \vec{S}_i\cdot\vec{S}_{i+\delta i}\rangle$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="lower left", fontsize="small", fancybox=False, edgecolor="k", framealpha=1)
    ax.text(0.6, 0.45, _state_label(results_dir), transform=ax.transAxes, ha="center", va="center")

    # inset: the bond types on the lattice
    inset = ax.inset_axes([0.72, 0.06, 0.26, 0.3])
    for (name, r), color in zip(JIANG_BOND_TYPES.items(), MATLAB_COLORS):
        r = np.asarray(r)
        inset.plot([0, r[0]], [0, r[1]], color=color, lw=2)
        inset.text(*(0.55 * r), name.split()[1], color=color, fontsize=7, ha="center", va="center",
                   bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    inset.plot(0, 0, "k.", ms=4)
    inset.set_aspect("equal")
    inset.axis("off")
    if save:
        fig.savefig(results_dir / "bond_correlations_vs_lx.pdf", dpi=200)
    return fig, lxs, bond


def GutzwillerJ1J2Energies(psi, results_dir, J2s=JIANG_J2S, Delz=1.0):
    """
        Energy per site of the Gutzwiller MPS psi of results_dir w.r.t. the J1-J2 model (calculateGutzwillerEnergy
        TriangularJ1J2), for each J2; the geometry is read from the directory name.
    """
    name = Path(results_dir).name
    Lx, Ly = (int(L) for L in re.search(r"Lx_(\d+)_Ly_(\d+)", name).groups())
    geometry = _Geometry(results_dir)
    bc_MPS = "finite" if "_finite_" in name else "infinite"
    bc = ("open", "periodic") if bc_MPS == "finite" else ("periodic", "periodic")  # as in GutzwillerDMRGOverlaps
    return np.array([np.real(calculateGutzwillerEnergyTriangularJ1J2(psi, Lx, Ly, bc_MPS, J2, Delz, bc, geometry))
                     for J2 in J2s])


def _Delz(results_dir):
    """Delz of the directory name (DMRG runs of the XXZ model), 1 (isotropic) if absent."""
    delz = re.search(r"_Delz_([0-9.]+?)(?:_|$)", Path(results_dir).name)
    return float(delz.group(1)) if delz else 1.0


def SaveEnergiesVsJ2(results_dir, J2s, energies):
    """Saves ENERGIES_FILE in results_dir: J2, energy per site."""
    np.savetxt(Path(results_dir) / ENERGIES_FILE, np.column_stack([J2s, energies]),
               header=f"J2 energy_per_site; Delz={_Delz(results_dir)}")


def LoadEnergiesVsJ2(results_dir):
    """(J2s, energies per site) of ENERGIES_FILE in results_dir."""
    data = np.loadtxt(Path(results_dir) / ENERGIES_FILE, ndmin=2)
    return data[:, 0], data[:, 1]


def PlotEnergiesVsJ2(results_dir, J2s, energies, save=True):
    """E(J2) per site of the state of results_dir. Saves energy_vs_J2.pdf in results_dir."""
    results_dir = Path(results_dir)
    print(f"{_state_label(results_dir)}: " + ", ".join(f"E(J2={J2:g})={e:.6f}" for J2, e in zip(J2s, energies)))
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.plot(J2s, energies, "o-")
    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$E/N$")
    ax.set_title(_state_label(results_dir))
    if save:
        fig.savefig(results_dir / "energy_vs_J2.pdf", dpi=200)
    return fig


def PlotSchmidtSpectrum(psi, results_dir, bonds=None, xi_max=20.0, save=True):
    """
        Entanglement spectrum xi = -log(lambda^2) by Sz charge sector of the MPS psi of results_dir, on the given
        bonds (default: the cuts between the columns, every Ly * (sites per unit cell) bonds: of the MPS unit cell if
        infinite, the central one if finite). Bond i is left of site i. Saves schmidt_spectrum.pdf in results_dir.
    """
    results_dir = Path(results_dir)
    with open(results_dir / "lattice.pkl", "rb") as f:
        lat = pickle.load(f)
    sites_per_column = lat.Ls[1] * len(lat.unit_cell)
    if bonds is None:
        bonds = (list(range(0, psi.L, sites_per_column)) if not psi.finite else
                 [(psi.L // sites_per_column // 2) * sites_per_column])
    # entanglement_spectrum lists only the nontrivial bonds (all for infinite, 1 .. L-1 for finite)
    nontrivial_bonds = list(range(psi.L + 1)[psi.nontrivial_bonds])
    spectra = psi.entanglement_spectrum(by_charge=True)
    spectra = {bond: spectra[nontrivial_bonds.index(bond)] for bond in bonds}

    fig, axes = plt.subplots(1, len(bonds), figsize=(5 * len(bonds), 5), squeeze=False, constrained_layout=True)
    for ax, bond in zip(axes[0], bonds):
        # the charges of the Gutzwiller MPS are offset by the number of sites to the left, so they are shown
        # relative to their lambda^2 weighted mean
        weights = [(charge[0], np.sum(np.exp(-xis))) for charge, xis in spectra[bond]]
        mean_charge = sum(q * w for q, w in weights) / sum(w for _, w in weights)
        for charge, xis in spectra[bond]:
            xis = xis[xis < xi_max]
            ax.plot(np.full(len(xis), charge[0] - mean_charge), xis, "_", ms=14, mew=1.5, color="tab:blue")
        ax.set_xlabel(r"$2S^z - \langle 2S^z \rangle$ of the left half")
        ax.set_ylabel(r"$\xi = -\log \lambda^2$")
        ax.set_ylim(top=xi_max)
        ax.set_title(f"bond {bond}, S = {psi.entanglement_entropy(bonds=[bond])[0]:.4f}")
    fig.suptitle(_state_label(results_dir))
    if save:
        fig.savefig(results_dir / "schmidt_spectrum.pdf", dpi=200)
    return fig


def AnalyzeCaseAsJiang(psi, results_dir, J2s=JIANG_J2S, n_columns=None):
    """
        The per-case analysis of Jiang et al. of the MPS psi of results_dir, figures and files saved in results_dir:
          - the correlations with a reference site on n_columns columns (CalculateCorrelationRow) and their sign
            Fourier transform (Fig. 5)
          - the nearest neighbor correlations, and by bond type vs. l_x (Fig. 1), if the spin_components post-process
            files are in results_dir
          - the energies vs. J2 w.r.t. the J1-J2 model (ENERGIES_FILE)
          - the Schmidt spectrum
    """
    results_dir = Path(results_dir)
    CalculateCorrelationRow(psi, results_dir, n_columns)
    PlotSignStructureFactor(results_dir)
    if all((results_dir / fname).exists() for fname in CORR_FILES.values()):
        PlotNearestNeighborCorrelations(results_dir)
        PlotBondCorrelationsVsLx(results_dir)
    else:
        print(f"no {list(CORR_FILES.values())} in {results_dir} (spin_components_structure_factor post-process), "
              f"skipping the nearest neighbor correlation plots")
    energies = GutzwillerJ1J2Energies(psi, results_dir, J2s, _Delz(results_dir))
    SaveEnergiesVsJ2(results_dir, J2s, energies)
    PlotEnergiesVsJ2(results_dir, J2s, energies)
    PlotSchmidtSpectrum(psi, results_dir)
    plt.close("all")
