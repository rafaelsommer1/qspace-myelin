#!/usr/bin/env python3
"""
qspace-myelin — Publication-Quality Visualization
===================================================
Generates figure-ready outputs from NLD myelin maps.

Outputs:
  *_montage.png      Multi-slice axial montage (NLD overlay on T1)
  *_triplane.png     Three-plane view at a chosen coordinate
  *_mosaic.png       Full-brain mosaic (all slices, small panels)
  *_histogram.png    NLD distribution with tissue compartments

Usage:
  python visualize_nld.py --nld myelin_nld.nii.gz
  python visualize_nld.py --nld myelin_nld.nii.gz --t1 t1_brain.nii.gz
  python visualize_nld.py --nld myelin_nld.nii.gz --t1 t1_brain.nii.gz \\
                          --mask myelin_mask.nii.gz --cmap hot \\
                          --vmin 20 --vmax 95 --alpha 0.75 \\
                          --output figures/sub01
"""

__version__ = "0.1.0"

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel not found. Run: pip install nibabel")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.ndimage import binary_dilation, gaussian_filter

# ── custom colormaps ──────────────────────────────────────────────────────────

def _make_colormaps() -> dict:
    """Build colormaps suited for myelin/NLD visualization."""

    # Paper style: white → blue (Fujiyoshi et al.)
    cmap_paper = LinearSegmentedColormap.from_list(
        "nld_paper", [(1,1,1), (0.05,0.18,0.70)], N=256)

    # Hot myelin: black → red → orange → yellow (high contrast)
    cmap_hot = LinearSegmentedColormap.from_list(
        "nld_hot", [(0,0,0), (0.6,0,0), (1,0.35,0), (1,0.85,0.1)], N=256)

    # Plasma-like: dark purple → magenta → yellow
    cmap_plasma = matplotlib.cm.plasma

    # Thermal: black → blue → cyan → green → yellow → white
    cmap_thermal = LinearSegmentedColormap.from_list(
        "nld_thermal",
        [(0,0,0), (0,0,0.8), (0,0.7,0.9),
         (0.1,0.85,0.3), (0.95,0.9,0.1), (1,1,1)], N=256)

    # Inferno (good for print, colourblind-safe)
    cmap_inferno = matplotlib.cm.inferno

    # Viridis (perceptually uniform, colourblind-safe)
    cmap_viridis = matplotlib.cm.viridis

    return {
        "paper":   cmap_paper,
        "hot":     cmap_hot,
        "thermal": cmap_thermal,
        "plasma":  cmap_plasma,
        "inferno": cmap_inferno,
        "viridis": cmap_viridis,
    }

CMAPS = _make_colormaps()


# ── image helpers ─────────────────────────────────────────────────────────────

def load_volume(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img.affine


def get_slice(vol: np.ndarray, idx: int, axis: int) -> np.ndarray:
    """Extract and orient a 2D slice (radiological convention)."""
    slc = np.take(vol, idx, axis=axis)
    # Rotate to standard view per axis
    if axis == 2:   # axial: L→R, P→A
        return np.rot90(slc)
    elif axis == 1: # coronal: L→R, I→S
        return np.rot90(slc)
    else:           # sagittal: P→A, I→S
        return np.rot90(slc)


def percentile_window(vol: np.ndarray, mask: np.ndarray = None,
                       plow: float = 1.0, phigh: float = 99.0
                       ) -> tuple[float, float]:
    """Robust intensity window from percentiles."""
    data = vol[mask > 0] if mask is not None else vol[vol > 0]
    if len(data) == 0:
        return float(vol.min()), float(vol.max())
    return float(np.percentile(data, plow)), float(np.percentile(data, phigh))


def brain_contour(mask: np.ndarray, axis: int, idx: int,
                  dilate: int = 1) -> np.ndarray:
    """Extract brain boundary as a binary edge map."""
    slc = np.take(mask, idx, axis=axis).astype(bool)
    dilated = binary_dilation(slc, iterations=dilate)
    return (dilated & ~slc).astype(np.uint8)


def find_content_slices(vol: np.ndarray, mask: np.ndarray,
                         axis: int, frac_low: float = 0.10,
                         frac_high: float = 0.90) -> np.ndarray:
    """Return slice indices that contain brain content."""
    sums = mask.sum(axis=tuple(a for a in range(3) if a != axis))
    content = np.where(sums > sums.max() * 0.05)[0]
    lo = content[int(len(content) * frac_low)]
    hi = content[int(len(content) * frac_high)]
    return np.arange(lo, hi + 1)


def pick_slices(valid: np.ndarray, n: int) -> np.ndarray:
    return valid[np.linspace(0, len(valid) - 1, n, dtype=int)]


def _apply_overlay(ax, t1_slc, nld_slc, mask_slc,
                   cmap, norm, alpha, t1_window):
    """Draw T1 background + NLD overlay + mask contour on ax."""
    ax.set_facecolor("black")

    if t1_slc is not None:
        t1_norm = np.clip(
            (t1_slc - t1_window[0]) / (t1_window[1] - t1_window[0] + 1e-8),
            0, 1)
        ax.imshow(t1_norm, cmap="gray", vmin=0, vmax=1,
                  interpolation="bilinear", aspect="equal")

    if nld_slc is not None:
        nld_rgba = cmap(norm(nld_slc))
        if mask_slc is not None:
            nld_rgba[..., 3] = np.where(mask_slc > 0, alpha, 0)
        else:
            nld_rgba[..., 3] = np.where(nld_slc > norm.vmin, alpha, 0)
        ax.imshow(nld_rgba, interpolation="bilinear", aspect="equal")

    ax.axis("off")


# ── figure 1: montage ─────────────────────────────────────────────────────────

def figure_montage(nld: np.ndarray, t1: np.ndarray, mask: np.ndarray,
                    cmap, norm, alpha: float, t1_window: tuple,
                    output_prefix: str, n_cols: int = 6,
                    axis: int = 2, dpi: int = 300):
    """
    Multi-slice axial montage — NLD overlay on T1.
    Clean black background, no axes, shared colorbar.
    """
    valid = find_content_slices(nld, mask, axis)
    n_show = min(n_cols * 4, len(valid))
    n_show = (n_show // n_cols) * n_cols  # round to full rows
    indices = pick_slices(valid, n_show)
    n_rows = n_show // n_cols

    fig_w = n_cols * 2.2
    fig_h = n_rows * 2.2 + 0.6   # extra for colorbar

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        left=0.01, right=0.99,
        top=0.96, bottom=0.08,
        hspace=0.02, wspace=0.02,
    )

    for i, sl in enumerate(indices):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        t1_slc  = get_slice(t1,   sl, axis) if t1   is not None else None
        nld_slc = get_slice(nld,  sl, axis)
        msk_slc = get_slice(mask, sl, axis) if mask is not None else None
        _apply_overlay(ax, t1_slc, nld_slc, msk_slc, cmap, norm, alpha, t1_window)

        # Slice label
        ax.text(0.03, 0.05, f"z={sl}", transform=ax.transAxes,
                color="white", fontsize=6, va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    # Colorbar
    cbar_ax = fig.add_axes([0.20, 0.03, 0.60, 0.025])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                       orientation="horizontal")
    cb.set_label("NLD (a.u.)", color="white", fontsize=9)
    cb.ax.xaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")
    cbar_ax.set_facecolor("black")

    path = f"{output_prefix}_montage.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── figure 2: triplane ────────────────────────────────────────────────────────

def figure_triplane(nld: np.ndarray, t1: np.ndarray, mask: np.ndarray,
                     cmap, norm, alpha: float, t1_window: tuple,
                     output_prefix: str,
                     coord: tuple = None, dpi: int = 300):
    """
    Three-plane view (axial, coronal, sagittal) at a single coordinate.
    Crosshair lines indicate the shared voxel.
    """
    shape = nld.shape

    if coord is None:
        # Default: centre of mass of the NLD map
        from scipy.ndimage import center_of_mass
        cm = center_of_mass(nld * (mask > 0) if mask is not None else nld)
        coord = tuple(int(c) for c in cm)

    x0, y0, z0 = [np.clip(coord[i], 0, shape[i]-1) for i in range(3)]

    fig = plt.figure(figsize=(12, 5), facecolor="black")
    gs = gridspec.GridSpec(1, 4, figure=fig,
                            left=0.01, right=0.98,
                            top=0.93, bottom=0.10,
                            wspace=0.04,
                            width_ratios=[shape[1], shape[0], shape[0], 0.08])

    axes_info = [
        (2, z0, "Axial",    f"z = {z0}"),
        (1, y0, "Coronal",  f"y = {y0}"),
        (0, x0, "Sagittal", f"x = {x0}"),
    ]

    panel_axes = []
    for col, (axis, idx, label, coord_str) in enumerate(axes_info):
        ax = fig.add_subplot(gs[0, col])
        t1_slc  = get_slice(t1,   idx, axis) if t1   is not None else None
        nld_slc = get_slice(nld,  idx, axis)
        msk_slc = get_slice(mask, idx, axis) if mask is not None else None
        _apply_overlay(ax, t1_slc, nld_slc, msk_slc, cmap, norm, alpha, t1_window)

        # Panel label
        ax.set_title(label, color="white", fontsize=11,
                      fontweight="bold", pad=4)
        ax.text(0.97, 0.03, coord_str, transform=ax.transAxes,
                color="#aaaaaa", fontsize=8, ha="right", va="bottom")

        # Crosshairs
        h, w = nld_slc.shape
        cx = {"Axial": y0, "Coronal": x0, "Sagittal": x0}[label]
        cy = {"Axial": x0, "Coronal": z0, "Sagittal": z0}[label]
        # map to rotated coords
        ax.axhline(h - 1 - cy, color="#00ff88", lw=0.6, alpha=0.6, ls="--")
        ax.axvline(cx,          color="#00ff88", lw=0.6, alpha=0.6, ls="--")

        panel_axes.append(ax)

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 3])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("NLD (a.u.)", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cbar_ax.set_facecolor("black")

    path = f"{output_prefix}_triplane.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── figure 3: mosaic ──────────────────────────────────────────────────────────

def figure_mosaic(nld: np.ndarray, mask: np.ndarray,
                   cmap, norm,
                   output_prefix: str,
                   axis: int = 2, dpi: int = 200):
    """
    Full-brain mosaic — all content slices in a compact grid.
    NLD only (no T1), black background. Good for quick QC overview.
    """
    valid = find_content_slices(nld, mask, axis, 0.05, 0.95)
    n = len(valid)
    n_cols = int(np.ceil(np.sqrt(n * 1.6)))
    n_rows = int(np.ceil(n / n_cols))

    fig_w = n_cols * 1.4
    fig_h = n_rows * 1.4 + 0.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            left=0.005, right=0.995,
                            top=0.97, bottom=0.04,
                            hspace=0.01, wspace=0.01)

    for i, sl in enumerate(valid):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        slc = get_slice(nld, sl, axis)
        msk = get_slice(mask, sl, axis) if mask is not None else None

        rgba = cmap(norm(slc))
        if msk is not None:
            rgba[..., 3] = np.where(msk > 0, 1.0, 0.0)
        ax.imshow(rgba, interpolation="bilinear", aspect="equal")
        ax.axis("off")

    # Hide unused panels
    for i in range(n, n_rows * n_cols):
        fig.add_subplot(gs[i // n_cols, i % n_cols]).axis("off")

    # Compact colorbar
    cbar_ax = fig.add_axes([0.25, 0.01, 0.50, 0.018])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label("NLD", color="white", fontsize=7)
    cb.ax.xaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")

    path = f"{output_prefix}_mosaic.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── figure 4: histogram ───────────────────────────────────────────────────────

def figure_histogram(nld: np.ndarray, mask: np.ndarray,
                      norm, cmap,
                      output_prefix: str, dpi: int = 200):
    """
    NLD value distribution within the brain mask.
    Gradient-filled bars coloured by the chosen colormap.
    Marks approximate CSF / GM / WM compartments.
    """
    data = nld[mask > 0].flatten() if mask is not None else nld[nld > 0].flatten()
    data = data[data > 0]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="black")
    ax.set_facecolor("black")

    counts, edges = np.histogram(data, bins=80, range=(norm.vmin, norm.vmax))
    centers = (edges[:-1] + edges[1:]) / 2

    # Colour each bar by the colormap
    bar_colors = cmap(norm(centers))
    ax.bar(centers, counts, width=(edges[1]-edges[0]),
           color=bar_colors, edgecolor="none", alpha=0.92)

    # Compartment annotations (approximate — adjust to your data)
    ymax = counts.max() * 1.15
    for x, label, color in [
        (10,  "CSF",        "#4fc3f7"),
        (35,  "Gray matter","#a5d6a7"),
        (70,  "White matter","#ffcc80"),
    ]:
        ax.axvline(x, color=color, lw=1.0, ls="--", alpha=0.7)
        ax.text(x + 1, ymax * 0.88, label, color=color,
                fontsize=8, va="top",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    ax.set_xlim(norm.vmin, norm.vmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("NLD (a.u.)", color="white", fontsize=11)
    ax.set_ylabel("Voxel count", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    # Stats
    stats = (f"n = {len(data):,}  |  "
             f"median = {np.median(data):.1f}  |  "
             f"IQR = [{np.percentile(data,25):.1f}, {np.percentile(data,75):.1f}]")
    ax.text(0.98, 0.97, stats, transform=ax.transAxes,
            color="#aaaaaa", fontsize=8, ha="right", va="top")

    path = f"{output_prefix}_histogram.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality NLD visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic — NLD only, default colormap
  python visualize_nld.py --nld myelin_nld.nii.gz --mask myelin_mask.nii.gz

  # With T1 underlay, hot colormap, custom window
  python visualize_nld.py --nld myelin_nld.nii.gz --t1 t1_brain.nii.gz \\
                          --mask myelin_mask.nii.gz --cmap hot \\
                          --vmin 15 --vmax 90 --alpha 0.80

  # Paper style (white→blue) full output
  python visualize_nld.py --nld myelin_nld.nii.gz --t1 t1_brain.nii.gz \\
                          --mask myelin_mask.nii.gz --cmap paper \\
                          --output figures/sub01 --dpi 300
        """,
    )

    parser.add_argument("--nld",    required=True, help="NLD map NIfTI")
    parser.add_argument("--t1",     default=None,  help="T1 brain NIfTI (optional underlay)")
    parser.add_argument("--mask",   default=None,  help="Brain mask NIfTI")
    parser.add_argument("--cmap",   default="hot",
                        choices=list(CMAPS.keys()),
                        help=f"Colormap: {', '.join(CMAPS.keys())}  (default: hot)")
    parser.add_argument("--vmin",   type=float, default=None,
                        help="NLD lower display bound (default: P2 of brain voxels)")
    parser.add_argument("--vmax",   type=float, default=None,
                        help="NLD upper display bound (default: P98 of brain voxels)")
    parser.add_argument("--alpha",  type=float, default=0.80,
                        help="NLD overlay opacity 0–1  (default: 0.80)")
    parser.add_argument("--t1vmin", type=float, default=None,
                        help="T1 lower display bound (default: P1)")
    parser.add_argument("--t1vmax", type=float, default=None,
                        help="T1 upper display bound (default: P99)")
    parser.add_argument("--coord",  type=int, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Voxel coordinate for triplane view (default: CoM)")
    parser.add_argument("--axis",   type=int, default=2, choices=[0, 1, 2],
                        help="Primary slice axis for montage/mosaic (default: 2=axial)")
    parser.add_argument("--ncols",  type=int, default=6,
                        help="Columns in montage grid (default: 6)")
    parser.add_argument("--output", default="myelin_vis",
                        help="Output prefix (default: myelin_vis)")
    parser.add_argument("--dpi",    type=int, default=300,
                        help="Output resolution in DPI (default: 300)")
    parser.add_argument("--no-montage",   action="store_true")
    parser.add_argument("--no-triplane",  action="store_true")
    parser.add_argument("--no-mosaic",    action="store_true")
    parser.add_argument("--no-histogram", action="store_true")

    args = parser.parse_args()

    print(f"\n  qspace-myelin visualizer  v{__version__}")
    print(f"  NLD:    {args.nld}")
    print(f"  T1:     {args.t1 or '(none)'}")
    print(f"  Mask:   {args.mask or '(none)'}")
    print(f"  Cmap:   {args.cmap}  |  alpha: {args.alpha}  |  DPI: {args.dpi}\n")

    # Load volumes
    nld, _  = load_volume(args.nld)
    t1      = load_volume(args.t1)[0]   if args.t1   else None
    mask    = load_volume(args.mask)[0].astype(np.uint8) if args.mask else None

    if mask is None:
        mask = (nld > 0).astype(np.uint8)

    # Window/level
    vmin, vmax = percentile_window(nld, mask, 2, 98)
    if args.vmin is not None: vmin = args.vmin
    if args.vmax is not None: vmax = args.vmax
    print(f"  NLD display window: [{vmin:.1f}, {vmax:.1f}]")

    t1_window = (0, 1)
    if t1 is not None:
        t1lo, t1hi = percentile_window(t1, mask, 1, 99)
        if args.t1vmin is not None: t1lo = args.t1vmin
        if args.t1vmax is not None: t1hi = args.t1vmax
        t1_window = (t1lo, t1hi)
        print(f"  T1 display window:  [{t1lo:.0f}, {t1hi:.0f}]")

    cmap = CMAPS[args.cmap]
    norm = Normalize(vmin=vmin, vmax=vmax)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if not args.no_montage:
        print("  Generating montage...")
        figure_montage(nld, t1, mask, cmap, norm, args.alpha, t1_window,
                        args.output, n_cols=args.ncols,
                        axis=args.axis, dpi=args.dpi)

    if not args.no_triplane:
        print("  Generating triplane...")
        figure_triplane(nld, t1, mask, cmap, norm, args.alpha, t1_window,
                         args.output, coord=args.coord, dpi=args.dpi)

    if not args.no_mosaic:
        print("  Generating mosaic...")
        figure_mosaic(nld, mask, cmap, norm,
                       args.output, axis=args.axis, dpi=args.dpi)

    if not args.no_histogram:
        print("  Generating histogram...")
        figure_histogram(nld, mask, norm, cmap, args.output, dpi=args.dpi)

    print(f"\n  Done. Files: {args.output}_{{montage,triplane,mosaic,histogram}}.png\n")


if __name__ == "__main__":
    main()
