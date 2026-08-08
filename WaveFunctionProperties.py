import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ----------------------------------------------------------------------
# 1) Scalar chirality of a single triangle, <S_i . (S_j x S_k)>
# ----------------------------------------------------------------------
def scalar_chirality_term(psi, i, j, k):
    """Expectation value chi = <psi| S_i . (S_j x S_k) |psi>.

    Written out,

        chi = Sx_i (Sy_j Sz_k - Sz_j Sy_k)
            + Sy_i (Sz_j Sx_k - Sx_j Sz_k)
            + Sz_i (Sx_j Sy_k - Sy_j Sx_k)

    Using Sx = (Sp + Sm)/2 and Sy = -i(Sp - Sm)/2 and expanding, every
    Sx/Sy pair combines into a single Sp/Sm pair, so chi can be written
    purely in terms of Sp, Sm, Sz:

        chi = -(i/2) [ Sm_i Sp_j Sz_k - Sm_i Sz_j Sp_k
                       - Sp_i Sm_j Sz_k + Sz_i Sm_j Sp_k
                       + Sp_i Sz_j Sm_k - Sz_i Sp_j Sm_k ]

    Each term contains exactly one Sp and one Sm (so it conserves total
    Sz), which is why this form works even if the site/MPS uses
    `conserve='Sz'` -- TeNPy does not even define Sx/Sy operators in that
    case, only Sp, Sm, Sz.

    The 6 terms are evaluated with `MPS.expectation_value_term`, which
    TeNPy can evaluate for three arbitrary (not necessarily neighboring)
    sites.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
    i, j, k : int
        MPS site indices of the three corners of the triangle
        (any consistent cyclic order; chi changes sign under an odd
        permutation of i, j, k, as expected for a triple product).

    Returns
    -------
    chi : float
        The (real) scalar spin chirality of the triangle (i, j, k).
    """
    terms = [
        (-0.5j, 'Sm', 'Sp', 'Sz'),
        (+0.5j, 'Sm', 'Sz', 'Sp'),
        (+0.5j, 'Sp', 'Sm', 'Sz'),
        (-0.5j, 'Sz', 'Sm', 'Sp'),
        (-0.5j, 'Sp', 'Sz', 'Sm'),
        (+0.5j, 'Sz', 'Sp', 'Sm'),
    ]
    chi = 0.0
    for coeff, op_i, op_j, op_k in terms:
        chi += coeff * psi.expectation_value_term([(op_i, i), (op_j, j), (op_k, k)])
    # chi is the expectation value of a Hermitian operator -> real
    assert(abs(np.imag(chi)) < 1e-14), "imaginary part of chirality should be 0"
    return np.real(chi)

# ----------------------------------------------------------------------
# 2) Enumerate the elementary 'up' and 'down' triangles of the lattice
# ----------------------------------------------------------------------
def triangular_plaquettes(lattice, u=0):
    """List all elementary triangular plaquettes of a `Triangular` lattice.

    For a `tenpy.models.lattice.Triangular` lattice (one site per unit
    cell, basis vectors a1, a2 at 60 degrees), every unit cell at lattice
    coordinate (x, y) hosts an 'up'-pointing triangle with corners

        (x, y), (x+1, y), (x, y+1)

    and a 'down'-pointing triangle with corners

        (x+1, y), (x, y+1), (x+1, y+1).

    Together these tile the lattice into the two orientations of
    elementary triangles. Plaquettes that would wrap around an *open*
    boundary are skipped; for *periodic* directions, TeNPy's
    `lat2mps_idx` takes care of the wrap-around automatically.

    Parameters
    ----------
    lattice : tenpy.models.lattice.Triangular
    u : int
        Index of the site within the unit cell (0 for `Triangular`,
        which has a single site per unit cell).

    Returns
    -------
    plaquettes : list of dict
        Each entry has keys
            'sites'       : tuple of 3 MPS site indices (i, j, k)
            'corners'     : list of 3 real-space (x, y) corner positions
            'center'      : real-space (x, y) of the triangle's centroid
            'orientation' : 'up' or 'down'
    """
    Lx, Ly = lattice.Ls
    # lattice.bc[d] is True for 'open', False for 'periodic'
    open_x, open_y = bool(lattice.bc[0]), bool(lattice.bc[1])

    def in_bounds(cx, cy):
        if open_x and not (0 <= cx < Lx):
            return False
        if open_y and not (0 <= cy < Ly):
            return False
        return True

    def site_index(cx, cy):
        return int(lattice.lat2mps_idx(np.array([cx, cy, u])))

    def site_pos(cx, cy):
        return lattice.position(np.array([cx, cy, u]))

    plaquettes = []
    for x in range(Lx):
        for y in range(Ly):
            for orientation, corner_list in (
                ('up',   [(x, y), (x + 1, y), (x, y + 1)]),
                ('down', [(x + 1, y), (x + 1, y + 1), (x, y + 1)]),
            ):
                if not all(in_bounds(cx, cy) for cx, cy in corner_list):
                    continue
                sites = tuple(site_index(cx, cy) for cx, cy in corner_list)
                corners = [site_pos(cx, cy) for cx, cy in corner_list]
                center = np.mean(corners, axis=0)
                plaquettes.append({
                    'sites': sites,
                    'corners': corners,
                    'center': center,
                    'orientation': orientation,
                })
    return plaquettes


# ----------------------------------------------------------------------
# 3) Compute chi for every plaquette
# ----------------------------------------------------------------------
def compute_chirality_map(psi, lattice, u=0):
    """Compute the scalar spin chirality for every triangular plaquette.

    Parameters
    ----------
    psi : tenpy.networks.mps.MPS
        The state (e.g. a DMRG ground state) living on `lattice`.
    lattice : tenpy.models.lattice.Triangular
    u : int
        Sublattice index within the unit cell (default 0).

    Returns
    -------
    plaquettes : list of dict
        As returned by `triangular_plaquettes`, with an additional key
        'chirality' containing chi_{ijk} = <S_i . (S_j x S_k)> for the
        three corner sites (i, j, k), taken in the order returned by
        `triangular_plaquettes` (i.e. consistent counter-clockwise
        ordering around the triangle).
    """
    plaquettes = triangular_plaquettes(lattice, u=u)
    for plaq in plaquettes:
        i, j, k = plaq['sites']
        plaq['chirality'] = scalar_chirality_term(psi, i, j, k)
    return plaquettes


# ----------------------------------------------------------------------
# 4) Plot the chirality map
# ----------------------------------------------------------------------
def plot_chirality_map(plaquettes, ax=None, cmap='RdBu_r', vmax=None,
                        show_lattice=True, lattice=None):
    """Plot the scalar spin chirality on every triangular plaquette.

    Each elementary triangle is drawn as a filled patch, colored
    according to its scalar chirality with a diverging colormap centered
    at zero -- this is the type of real-space chirality map used e.g.
    in arXiv:2601.14458 to visualize the (staggered) finite-flux pattern.

    Parameters
    ----------
    plaquettes : list of dict
        Output of `compute_chirality_map`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if not given.
    cmap : str or Colormap
        Diverging colormap (default 'RdBu_r').
    vmax : float, optional
        Symmetric color scale limit; if None, set to
        max(|chi|) over all plaquettes.
    show_lattice : bool
        If True, also scatter the underlying lattice sites.
    lattice : tenpy.models.lattice.Triangular, optional
        Needed only if `show_lattice=True`, to draw the site positions.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    chis = np.array([p['chirality'] for p in plaquettes])
    if vmax is None:
        vmax = np.max(np.abs(chis)) if len(chis) else 1.0
        if vmax == 0:
            vmax = 1.0

    patches = [Polygon(p['corners'], closed=True) for p in plaquettes]
    collection = PatchCollection(patches, cmap=cmap, edgecolor='k', linewidths=0.5)
    collection.set_array(chis)
    collection.set_clim(-vmax, vmax)
    ax.add_collection(collection)

    if show_lattice and lattice is not None:
        Lx, Ly = lattice.Ls
        pts = np.array([lattice.position(np.array([x, y, 0]))
                         for x in range(Lx) for y in range(Ly)])
        ax.scatter(pts[:, 0], pts[:, 1], s=15, color='k', zorder=3)

    all_corners = np.concatenate([p['corners'] for p in plaquettes], axis=0)
    pad = 0.5
    ax.set_xlim(all_corners[:, 0].min() - pad, all_corners[:, 0].max() + pad)
    ax.set_ylim(all_corners[:, 1].min() - pad, all_corners[:, 1].max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    cbar = plt.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\chi = \mathbf{S}_i \cdot (\mathbf{S}_j \times \mathbf{S}_k)$')
    return fig, ax


# ----------------------------------------------------------------------
# Convenience all-in-one wrapper
# ----------------------------------------------------------------------
def plot_scalar_spin_chirality(psi, lattice, u=0, **plot_kwargs):
    """Compute and plot the scalar spin chirality map in one call.

    Returns (plaquettes, ax).
    """
    plaquettes = compute_chirality_map(psi, lattice, u=u)
    fig, ax = plot_chirality_map(plaquettes, lattice=lattice, **plot_kwargs)
    return plaquettes, fig, ax


# ----------------------------------------------------------------------
# 5) Momentum-space spin structure factor
# ----------------------------------------------------------------------
def minimum_image_displacements(lattice, N, wrap_displacements=True):
    """Minimum-image real-space displacement r_i - r_j for every site pair.

    `lattice.position(...)` returns *unwrapped* coordinates: a site at
    lattice-vector index L-1 along a periodic direction is placed L-1
    lattice spacings away from the site at index 0, even though on the
    periodic torus they are actually nearest neighbors (1 lattice spacing
    apart, wrapping the other way). Using the raw, unwrapped positions in
    a Fourier sum would therefore assign the wrong phase to pairs that
    wrap around a periodic boundary.

    This function splits the displacement r_i - r_j into two parts:

    * **Lattice-vector part** ``(n_i - n_j) @ basis``, where (n_i, n_j)
      are the integer unit-cell coordinates returned by
      ``lattice.mps2lat_idx``.  This part is reduced modulo the lattice
      length along each *periodic* direction into (-L/2, L/2] before
      being converted to real space, giving the minimum-image wrap.
      Open directions are left unwrapped.

    * **Sublattice-offset part** ``tau_ui - tau_uj``, where tau_u =
      ``lattice.unit_cell_positions[u]`` is the fractional real-space
      offset of sublattice u within the unit cell.  For a single-site
      unit cell all tau are zero; for multi-site lattices (Kagome,
      Honeycomb, XC, …) the offsets are non-zero fractions of the basis
      vectors.  This part is *never* wrapped: the sublattice offset is
      strictly smaller than one lattice spacing and has nothing to do
      with the periodic boundary conditions.

    The total displacement is therefore::

        rdisp[i, j] = wrap(n_i - n_j) @ basis + (tau_ui - tau_uj)

    which is exactly ``position(i) - position(j)`` evaluated with the
    correct periodic images of the integer unit-cell part.

    Because the minimum image is two-fold degenerate exactly at
    separation L/2 (when L is even), reducing n_i - n_j this way is not
    perfectly antisymmetric under i <-> j at those special pairs (it
    always picks the same branch). To keep the resulting phase matrix,
    and hence the structure factor, manifestly Hermitian/real for *any*
    k, the returned displacement tensor is symmetrized; the antisymmetric
    "tie-breaking" ambiguity then only ever shows up as a slight
    (physically sensible) damping of those particular phase entries,
    rather than as a spurious imaginary part in S(k).

    Returns
    -------
    rdisp : ndarray, shape (N_sites, N_sites, 2)
        rdisp[i, j] is the minimum-image displacement r_i - r_j.
    """
    # N = lattice.N_sites
    lat_idx = np.array([lattice.mps2lat_idx(i) for i in range(N)])
    xy = lat_idx[:, :2].astype(float)  # integer unit-cell coords
    us = lat_idx[:, 2]  # sublattice indices
    tau = lattice.unit_cell_positions[us]  # (N, 2) real-space offsets

    periodic = ~np.asarray(lattice.bc)  # lattice.bc: True = open, False = periodic

    diff_xy = xy[:, None, :] - xy[None, :, :]  # (N, N, 2), n_i - n_j
    if wrap_displacements:
        for d in range(2):
            bc_MPS = lattice.bc_MPS
            infinite = (bc_MPS == "infinite")
            infinite_axis = (infinite and d == 0)
            if periodic[d] and not infinite_axis:
                L = lattice.Ls[d]
                not_half_len = np.where(np.abs(diff_xy[..., d]) != L/2)
                diff_xy[..., d][not_half_len] = (
                        (diff_xy[..., d][not_half_len] + L / 2) % L - L / 2)  # wrap into (-L/2, L/2]

    dtau = tau[:, None, :] - tau[None, :, :]  # (N, N, 2), tau_ui - tau_uj

    # Lattice-vector part (wrapped) converted to real space, plus exact sublattice offset
    return diff_xy @ lattice.basis + dtau  # (N, N, 2)


def _phase_matrix(rdisp, k):
    """
        Hermitian phase matrix exp(i k.(r_i - r_j)).
    """
    return np.exp(1j * (rdisp @ k))


def structure_factor(C, lattice, k, ops=('Sx', 'Sy', 'Sz'), wrap_displacements=True):
    """Static spin structure factor S(k) at a single momentum k.

        S(k) = (1/N) sum_{i,j} e^{i k.(r_i - r_j)} <S_i . S_j>

    where (r_i - r_j) is the *minimum-image* displacement on the
    (possibly periodic) lattice torus -- see `minimum_image_displacements`
    for why this matters: along a periodic direction, the site at the
    far edge is actually only 1 lattice spacing away from the site at
    the near edge (wrapping around), not L-1 spacings away as the raw,
    unwrapped `lattice.position(...)` would suggest. Using unwrapped
    positions would give the wrong Fourier phase to pairs that wrap
    around a periodic boundary.

    The sum runs over *all* ordered pairs (i, j) of sites (i.e. it is
    not restricted to i <= j or to a single reference site), which is
    what makes S(k) manifestly real: writing C_ij = <S_i . S_j>, one has
    C_ji = C_ij^* (since S_i . S_j is Hermitian), so pairing each term
    e^{i k.(r_i-r_j)} C_ij with its partner e^{i k.(r_j-r_i)} C_ji = c.c.
    makes every i != j contribution manifestly real; the i = j (onsite)
    terms are real to begin with. The normalization by 1/N (number of
    sites) is the standard convention so that S(k) stays O(1) when N is
    increased while the local moment is kept fixed (instead of growing
    extensively, as the unnormalized double sum would).

    Parameters
    ----------
    C : numpy.ndarray
        The correlations in x space.
    lattice : tenpy.models.lattice.Lattice
        The (e.g. `Triangular`) lattice psi lives on.
    k : array-like, shape (2,)
        Momentum, in the same units as `lattice.reciprocal_basis`
        (i.e. as a 2-vector in the lab xy-plane, *not* in reduced
        coordinates of the reciprocal lattice vectors).
    ops : tuple of str
        The three spin operators whose dot product is correlated;
        default ``('Sx', 'Sy', 'Sz')`` gives the full SU(2) structure
        factor <S_i . S_j>. Pass e.g. ``('Sz',)`` for the longitudinal
        S^zz(k) structure factor alone.

    Returns
    -------
    Sk : float
        The (real) structure factor at momentum k.
    """
    rdisp = minimum_image_displacements(lattice, C.shape[0], wrap_displacements=wrap_displacements)
    k = np.asarray(k, dtype=float)
    N = lattice.N_sites

    phase = _phase_matrix(rdisp, k)  # shape (N, N), Hermitian
    Sk = np.einsum('ij,ij->', phase, C) / N
    # phase is Hermitian and C is Hermitian, so their entrywise product
    # summed over all i,j is real up to numerical noise; drop the
    # negligible imaginary part explicitly.
    return float(np.real(Sk))


def k_grid(lattice, n1, n2, b1, b2):
    """Regular grid of momenta spanning the lattice's reciprocal cell.

    Builds k = (m1/n1) * b1 + (m2/n2) * b2 for integers m1 in [0, n1)
    and m2 in [0, n2) (shifted to be centered on k=0 if `centered=True`),
    where b1, b2 = `lattice.reciprocal_basis`.

    Parameters
    ----------
    lattice : tenpy.models.lattice.Lattice
    n1, n2 : int
        Number of grid points along each reciprocal basis vector.
        A natural choice is n1, n2 = lattice.Ls, which gives exactly the
        momenta compatible with the (finite, periodic) simulation cell.
    centered : bool
        If True, shift the grid to lie in (roughly) [-0.5, 0.5) x b1, b2
        instead of [0, 1) x b1, b2, i.e. centered around the Gamma point.

    Returns
    -------
    ks : ndarray, shape (n1*n2, 2)
        The grid of momenta, in lab xy-coordinates.
    """
    # b1, b2 = lattice.reciprocal_basis
    #m1 = np.arange(n1)
    #m2 = np.arange(n2)
    m1 = np.linspace(-n1, n1-1, 2*n1)
    m2 = np.linspace(-n2, n2-1, 2*n2)
    #if centered:
    #    m1 = ((m1 + n1 // 2) % n1) - n1 // 2
    #    m2 = ((m2 + n2 // 2) % n2) - n2 // 2
    f1, f2 = np.meshgrid(m1 / n1, m2 / n2, indexing='ij')
    ks = f1.reshape(-1, 1) * b1 + f2.reshape(-1, 1) * b2
    return ks


def _structure_factor_reciprocal_basis(lattice):
    """Reciprocal basis for plotting/sampling the physical spin lattice.

    TeNPy's ``lattice.reciprocal_basis`` is the reciprocal basis of the
    chosen unit cell.  For XC triangular lattices,
    the implementation uses a rectangular two-site unit cell whose
    reciprocal cell is folded relative to the one-site triangular
    Brillouin zone.  The spin structure factor is an extended-zone
    quantity over the physical site positions, so sample it with the
    unfolded triangular primitive reciprocal basis.
    """
    basis = np.asarray(lattice.basis, dtype=float)
    positions = np.asarray(lattice.unit_cell_positions, dtype=float)

    if positions.shape == (2, 2):
        delta = positions[1] - positions[0]
        if np.allclose(2.0 * delta, basis[0] + basis[1]):
            short_axis = int(np.argmin(np.linalg.norm(basis, axis=1)))
            primitive = np.array([basis[short_axis], delta])
            return 2.0 * np.pi * np.linalg.inv(primitive).T

    return np.asarray(lattice.reciprocal_basis, dtype=float)


def compute_structure_factor_grid(C, lattice, n1=None, n2=None,
                                  ops=('Sx', 'Sy', 'Sz'), centered=True, wrap_displacements=True):
    """Evaluate S(k) on a regular grid of momenta.

    Parameters
    ----------
    n1, n2 : int, optional
        Grid resolution; defaults to `lattice.Ls` (the number of unit
        cells along each lattice direction), which is the natural/native
        momentum resolution for a lattice of that size.
    centered : bool
        See `k_grid`.

    Returns
    -------
    ks : ndarray, shape (n1*n2, 2)
        The momenta (lab xy-coordinates).
    Sk : ndarray, shape (n1*n2,)
        S(k) evaluated at each momentum in `ks`.
    """

    # b1, b2 = _structure_factor_reciprocal_basis(lattice)
    b1, b2 = lattice.reciprocal_basis
    if n1 is None or n2 is None:
        n1, n2 = lattice.Ls
    ks = k_grid(lattice, n1, n2, b1, b2)

    # Precompute C and the minimum-image displacements once and reuse
    # across all k (much cheaper than calling `structure_factor`
    # independently for every k, which would recompute the same
    # correlation_function calls n1*n2 times).
    rdisp = minimum_image_displacements(lattice, C.shape[0], wrap_displacements=wrap_displacements)
    N = lattice.N_sites

    Sk = np.empty(len(ks))
    for m, k in enumerate(ks):
        phase = _phase_matrix(rdisp, k)
        Sk[m] = np.real(np.einsum('ij,ij->', phase, C)) / N
    return ks, Sk


# ----------------------------------------------------------------------
# 6) Plot the structure factor as a 2D image
# ----------------------------------------------------------------------
def _tile_kpoints(ks, Sk, lattice, n_tiles):
    """Tile (ks, Sk) periodically over n_tiles BZ images in each direction.

    Returns (ks_tiled, Sk_tiled, central_slice) where central_slice is
    the index slice that selects the original (un-shifted) points from
    the tiled arrays.
    """
    # b1, b2 = lattice.reciprocal_basis
    b1, b2 = _structure_factor_reciprocal_basis(lattice)
    ns = range(-n_tiles, n_tiles + 1)
    shifts = [n1 * b1 + n2 * b2 for n1 in ns for n2 in ns]
    # Find which shift is (0,0) so we can identify the central tile
    zero_idx = next(i for i, s in enumerate(shifts) if np.allclose(s, 0))
    ks_tiled = np.concatenate([ks + s for s in shifts])
    Sk_tiled = np.tile(Sk, len(shifts))
    Nk = len(ks)
    central_slice = slice(zero_idx * Nk, (zero_idx + 1) * Nk)
    return ks_tiled, Sk_tiled, central_slice


def plot_structure_factor(ks, Sk, lat, ax=None, cmap='RdBu',
                          mode='interpolate', n_interp=300,
                          show_kpoints=True, n_tiles=1,
                          edge_color='none'):
    """Plot S(k) as a 2D image in the lab-frame (kx, ky) plane.

    Because the reciprocal lattice vectors b1, b2 of a general lattice
    are not orthogonal (e.g. they are at 120 degrees for the triangular
    lattice), the native k-point grid is *not* a rectangular grid in
    (kx, ky) space -- ``imshow`` or ``pcolormesh`` on a reshaped array
    would place pixels at the wrong positions.

    Two rendering modes are available via the `mode` argument:

    ``'interpolate'`` (default)
        Tiles the data periodically using S(k+G)=S(k), then
        interpolates onto a fine rectangular (kx, ky) pixel grid with
        ``scipy.interpolate.griddata``.  Gives a smooth image; works
        best when the native k-grid is dense enough that interpolation
        does not introduce visible artefacts.

    ``'voronoi'``
        Draws the exact Voronoi cell of each k-point, filled with its
        S(k) value.  No interpolation is performed: every pixel in a
        cell has exactly the value of its k-point.  This is the honest,
        data-faithful view -- for a coarse grid it clearly shows the
        native resolution, and for a fine grid it converges to the same
        image as ``'interpolate'``.

        The Voronoi tessellation is computed on the tiled point set so
        that boundary cells are correctly bounded by the neighbouring BZ
        images, giving proper hexagonal (or other lattice-appropriate)
        cells for all k-points including those near the BZ edge.

    Parameters
    ----------
    ks : ndarray, shape (M, 2)
        k-point coordinates in the lab xy-plane, as returned by
        ``compute_structure_factor_grid``.
    Sk : ndarray, shape (M,)
        Structure factor values at each k-point.
    lat : tenpy.models.lattice.Lattice
        The lattice; its ``reciprocal_basis`` is used to build the
        tiling vectors G = n1*b1 + n2*b2 for the BZ repetition.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if not given.
    cmap : str or Colormap
        Colormap for the S(k) image (default ``'inferno'``).
    mode : {'interpolate', 'voronoi'}
        Rendering mode (default ``'interpolate'``).
    n_interp : int
        Number of pixels along each axis of the rectangular
        interpolation grid; only used when ``mode='interpolate'``
        (default 300).
    show_kpoints : bool
        If True, overlay small white dots at the native k-point
        positions (default True).
    n_tiles : int
        How many BZ images to tile in each direction, i.e. the tiling
        covers offsets n1*b1 + n2*b2 for n1,n2 in [-n_tiles, n_tiles]
        (default 1, meaning the 8 nearest BZ images are added).
        Increase to 2 for very small/sparse grids.
    edge_color : color or 'none'
        Edge color of the Voronoi cells; only used when
        ``mode='voronoi'`` (default ``'none'``, i.e. no edges).
        Pass e.g. ``'white'`` or ``'0.3'`` to draw cell boundaries.

    Returns
    -------
    ax : matplotlib.axes.Axes
    artist : QuadMesh or PatchCollection
        The artist added to the axes (useful for further clim / colorbar
        adjustments).
    """
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ks_tiled, Sk_tiled, central_slice = _tile_kpoints(ks, Sk, lat, n_tiles)

    if mode == 'interpolate':
        from scipy.interpolate import griddata
        kx = np.linspace(ks_tiled[:, 0].min(), ks_tiled[:, 0].max(), n_interp)
        ky = np.linspace(ks_tiled[:, 1].min(), ks_tiled[:, 1].max(), n_interp)
        KX, KY = np.meshgrid(kx, ky)
        SK_img = griddata(ks_tiled, Sk_tiled, (KX, KY), method='linear')
        # pixels outside the convex hull of the tiled data are NaN -> transparent
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)
        artist = ax.pcolormesh(KX, KY, SK_img, cmap=cmap_obj, shading='auto')

    elif mode == 'voronoi':
        from scipy.spatial import Voronoi

        vor = Voronoi(ks_tiled)

        patches = []
        values = []
        for pt_idx in range(central_slice.start, central_slice.stop):
            region_idx = vor.point_region[pt_idx]
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                # Should not happen with tiled data, but skip if it does
                continue
            verts = vor.vertices[region]
            patches.append(MplPolygon(verts, closed=True))
            values.append(Sk_tiled[pt_idx])

        artist = PatchCollection(patches, cmap=cmap,
                                 edgecolor=edge_color, linewidths=0.5)
        artist.set_array(np.array(values))
        ax.add_collection(artist)
        # Let the collection set its own limits
        ax.autoscale_view()

    else:
        raise ValueError(f"mode must be 'interpolate' or 'voronoi', got {mode!r}")

    if show_kpoints:
        ax.scatter(ks_tiled[:, 0], ks_tiled[:, 1],
                   color='white', s=12, linewidths=0,
                   alpha=0.5, zorder=3)

    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    plt.colorbar(artist, ax=ax, label=r'$S(\mathbf{k})$')

    return ax, artist


def getShortestDistanceOnLatticeAxis(ax_coor_site1, ax_coor_site2, ax_bc, ax_L, finite_axis):
    coor_diff = ax_coor_site1 - ax_coor_site2
    coor_dist = abs(coor_diff)
    if ax_bc == "periodic" and finite_axis:
        assert (coor_dist < ax_L)
        dist_orig = coor_dist
        dist_opposite = ax_L - coor_dist
        if dist_orig <= dist_opposite:
            return coor_diff
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


def ComputeMomentumSpaceStructureFactor(corr_x, lat, assert_realness=True,
                                        transform_expectation_value=False, Kx=None, Ky=None,
                                        new_implementation=True):
    if new_implementation:
        return compute_structure_factor_grid(corr_x, lat, wrap_displacements=True)
    print("Warning: using legacy implementation in ComputeMomentumSpaceStructureFactor")
    if transform_expectation_value:
        assert (corr_x.ndim == 1)
    elif lat.bc_MPS != "infinite":
        assert (lat.N_sites == corr_x.shape[0] and lat.N_sites == corr_x.shape[1])

    corr_x_shape = corr_x.shape
    Ls = lat.Ls
    if Kx is None:
        assert (Ky is None), "need to specify momentum along both axes"
        # kx = ky = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        kx = np.linspace(-2 * np.pi, 2 * np.pi, 2 * Ls[0] + 1)
        ky = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        Kx, Ky = np.meshgrid(kx, ky)
    bcs = lat.boundary_conditions
    corr_k = np.zeros(Kx.shape, dtype=complex)
    bc_MPS = lat.bc_MPS
    unit_cell_pos = lat.unit_cell_positions
    basis_vectors = np.asarray(lat.basis, dtype=float)

    for i in range(corr_x_shape[0]):
        coor_i = lat.mps2lat_idx(i)
        if transform_expectation_value:
            coor_center = [(Ls[0] - 1) / 2., (Ls[1] - 1) / 2., len(unit_cell_pos) // 2]
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
        assert (np.max(np.abs(np.imag(corr_k))) < 1e-13)

    return Kx, Ky, corr_k


def testConeStateChirality():
    import numpy as np
    import matplotlib.pyplot as plt
    from tenpy.networks.site import SpinHalfSite
    from tenpy.networks.mps import MPS
    from tenpy.models.lattice import Triangular

    # --- set up a triangular lattice ---------------------------------
    site = SpinHalfSite(conserve=None)
    Lx, Ly = 6, 3
    # finite in x, periodic in y (a typical DMRG "cylinder")
    lattice = Triangular(Lx, Ly, site, bc=['open', 'periodic'], bc_MPS='finite')

    # --- build (or load) a state --------------------------------------
    # Here: a "magnetized cone" state -- three-sublattice 120deg order in
    # the xy-plane, canted out of plane by `theta`. This is the kind of
    # non-coplanar, finite-chirality state studied in arXiv:2601.14458.
    # In practice, replace this block with a DMRG ground state, e.g.
    #
    #   from tenpy.algorithms import dmrg
    #   from tenpy.models.spins import SpinModel
    #   model = SpinModel({'lattice': lattice, 'Jx': 1, 'Jy': 1, 'Jz': 1, ...})
    #   psi = MPS.from_lattice_product_state(lattice, [[...]])
    #   dmrg.run(psi, model, {...})
    theta = np.pi / 4  # canting angle from the z axis

    def spinor(theta, phi):
        return np.array([np.cos(theta / 2),
                         np.sin(theta / 2) * np.exp(1j * phi)], dtype=complex)

    states = []
    for i in range(lattice.N_sites):
        x, y, u = lattice.mps2lat_idx(i)
        sublattice = (x - y) % 3  # 3-sublattice 120deg pattern
        phi = 2 * np.pi * sublattice / 3
        states.append(spinor(theta, phi))

    psi = MPS.from_product_state([site] * lattice.N_sites, states,
                                 bc='finite', dtype=complex)

    # --- compute & plot -------------------------------------------------
    plaquettes, fig, ax = plot_scalar_spin_chirality(psi, lattice)
    ax.set_title(r"Scalar spin chirality $\chi_{ijk}=\mathbf{S}_i\cdot(\mathbf{S}_j\times\mathbf{S}_k)$")
    plt.tight_layout()
    plt.savefig("scalar_spin_chirality.png", dpi=150)
    plt.show()
