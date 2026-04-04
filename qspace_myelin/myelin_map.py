#!/usr/bin/env python3
"""
qDWI Myelin Map — Normalized Leptokurtic Diffusion (NLD) Calculator
====================================================================
Implements the q-space myelin imaging method described in:

    Fujiyoshi K. et al. (2016). Application of q-Space Diffusion MRI for
    the Visualization of White Matter. J. Neurosci., 36(9), 2796–2808.
    https://doi.org/10.1523/JNEUROSCI.1770-15.2016

For each voxel, the water displacement probability density function (PDF)
is estimated via Fourier transform of the signal decay across b-values.
Excess kurtosis of the PDF is averaged across gradient directions and
normalized to produce the NLD index, where higher values indicate greater
diffusion restriction consistent with myelinated white matter.

Usage:
    python myelin_map.py --dwi <dwi.nii.gz> --bval <bvals> --bvec <bvecs>
                         [--mask <mask.nii.gz>] [--norm-mode csf|internal|auto]
                         [--kmax <val>] [--kmin <val>] [--output <prefix>]

Author: Rafael (with AI-assisted implementation)
License: MIT
"""

__version__ = "0.1.0"

import argparse
import logging
import sys

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel not found. Run: pip install nibabel")
    sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter, binary_erosion
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)


def load_bvals_bvecs(bval_file: str, bvec_file: str):
    """Load FSL-style bval and bvec files.

    Parameters
    ----------
    bval_file : str
        Path to FSL .bval file.
    bvec_file : str
        Path to FSL .bvec file (3 × N or N × 3).

    Returns
    -------
    bvals : ndarray, shape (N,)
    bvecs : ndarray, shape (3, N)
    """
    bvals = np.loadtxt(bval_file).flatten()
    bvecs = np.loadtxt(bvec_file)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    return bvals, bvecs


def group_volumes_by_direction(bvals: np.ndarray, bvecs: np.ndarray,
                                b0_threshold: float = 50.0,
                                angle_threshold_deg: float = 5.0):
    """Group DWI volumes by gradient direction.

    Identifies unique gradient orientations using antipodally-equivalent
    angular clustering, then associates each cluster with its b-value
    series (including the shared b=0 reference).

    Parameters
    ----------
    bvals : ndarray, shape (N,)
    bvecs : ndarray, shape (3, N)
    b0_threshold : float
        Volumes with b < b0_threshold are treated as b=0.
    angle_threshold_deg : float
        Angular tolerance for clustering gradient directions.

    Returns
    -------
    unique_bvals : ndarray
        Sorted unique b-values (rounded to nearest 100).
    direction_groups : list of list of (float, list[int])
        For each direction: list of (b-value, [volume_indices]) sorted
        by ascending b-value.
    """
    bvals_r = np.round(bvals / 100) * 100
    unique_bvals = np.sort(np.unique(bvals_r))

    b0_mask = bvals_r < b0_threshold
    b0_indices = np.where(b0_mask)[0]
    dwi_indices = np.where(~b0_mask)[0]

    if len(dwi_indices) == 0:
        raise ValueError("No diffusion-weighted volumes found (b > 0).")

    # Normalize directions; enforce canonical hemisphere (first nonzero > 0)
    dirs = bvecs[:, dwi_indices].T.copy()
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    dirs /= norms
    for d in dirs:
        for component in d:
            if abs(component) > 1e-6:
                if component < 0:
                    d *= -1
                break

    cos_thresh = np.cos(np.radians(angle_threshold_deg))
    assigned = np.zeros(len(dirs), dtype=bool)
    clusters = []

    for i in range(len(dirs)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(dirs)):
            if not assigned[j] and abs(np.dot(dirs[i], dirs[j])) >= cos_thresh:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    log.info("Gradient directions identified: %d", len(clusters))
    log.info("Unique b-values: %s", unique_bvals.astype(int))

    direction_groups = []
    for cluster in clusters:
        group = [(0, b0_indices.tolist())] if len(b0_indices) > 0 else []
        for idx in cluster:
            vol_idx = dwi_indices[idx]
            group.append((bvals_r[vol_idx], [vol_idx]))
        group.sort(key=lambda x: x[0])
        direction_groups.append(group)

    return unique_bvals, direction_groups


def signal_to_pdf(signal: np.ndarray, b_values: np.ndarray,
                  n_interp: int = 256):
    """Estimate the water displacement PDF from the q-space signal decay.

    The normalized signal E(q) = S(q)/S(0) is interpolated onto a uniform
    q-grid, symmetrized (E is an even function), and Fourier-transformed
    to yield the displacement PDF.  Here q ∝ sqrt(b), assuming constant
    diffusion time across shells.

    Parameters
    ----------
    signal : ndarray, shape (n_bvals,)
        Signal intensities sorted by ascending b-value; signal[0] is S(0).
    b_values : ndarray, shape (n_bvals,)
        Corresponding b-values in s/mm².
    n_interp : int
        Number of points for q-space interpolation before FFT.

    Returns
    -------
    displacement : ndarray or None
    pdf : ndarray or None
        Returns (None, None) if S(0) ≤ 0.
    """
    s0 = signal[0]
    if s0 <= 0:
        return None, None

    E = np.clip(signal / s0, 0.0, 1.0)
    q = np.sqrt(b_values)

    q_grid = np.linspace(0, q[-1], n_interp)
    E_grid = np.interp(q_grid, q, E)

    E_sym = np.concatenate([E_grid[::-1], E_grid[1:]])
    pdf = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sym))))

    total = pdf.sum()
    if total > 0:
        pdf /= total

    dq = q_grid[1] - q_grid[0] if len(q_grid) > 1 else 1.0
    displacement = np.fft.fftshift(np.fft.fftfreq(len(pdf), d=dq))

    return displacement, pdf


def excess_kurtosis(pdf: np.ndarray) -> float:
    """Compute excess kurtosis of a PDF.

    K = mean((x - μ)⁴ / σ⁴) - 3

    Parameters
    ----------
    pdf : ndarray
        Probability values (need not be normalized).

    Returns
    -------
    float
        Excess kurtosis; returns 0.0 if the distribution is degenerate.
    """
    if pdf is None or len(pdf) == 0:
        return 0.0
    mu = pdf.mean()
    sigma = pdf.std()
    if sigma < 1e-15:
        return 0.0
    return float(((pdf - mu) ** 4).mean() / sigma ** 4) - 3.0


def compute_nld(kurtosis_map: np.ndarray, k_max: float = None,
                k_min: float = None, mask: np.ndarray = None) -> np.ndarray:
    """Normalize a kurtosis map to the NLD scale (0–100).

    NLD = (K - K_min) / (K_max - K_min) × 100

    where higher NLD corresponds to greater diffusion restriction
    (myelinated white matter).  K_max and K_min default to the P99 and P1
    percentiles of the in-mask distribution when not supplied.

    Parameters
    ----------
    kurtosis_map : ndarray, shape (X, Y, Z)
    k_max : float, optional
    k_min : float, optional
    mask : ndarray of uint8, optional

    Returns
    -------
    nld : ndarray, shape (X, Y, Z), values in [0, 100]
    """
    valid = kurtosis_map[mask > 0] if mask is not None else kurtosis_map[kurtosis_map != 0]

    if len(valid) == 0:
        return np.zeros_like(kurtosis_map)

    if k_max is None:
        k_max = np.percentile(valid, 99)
        log.info("K_max (P99 auto): %.4f", k_max)
    if k_min is None:
        k_min = np.percentile(valid, 1)
        log.info("K_min (P1 auto):  %.4f", k_min)

    if abs(k_max - k_min) < 1e-10:
        log.warning("K_max ≈ K_min; normalization not possible.")
        return np.zeros_like(kurtosis_map)

    nld = np.clip((kurtosis_map - k_min) / (k_max - k_min) * 100.0, 0.0, 100.0)
    if mask is not None:
        nld[mask == 0] = 0.0

    return nld


def estimate_csf_mask(dwi_data: np.ndarray, bvals: np.ndarray,
                       brain_mask: np.ndarray,
                       csf_pct: float = 95.0) -> np.ndarray | None:
    """Automatically identify CSF voxels from the DWI signal ratio.

    CSF exhibits high signal at b=0 and rapid attenuation at high
    b-values due to unrestricted diffusion.  Voxels whose
    S(b=0)/S(b_high) ratio exceeds the `csf_pct` percentile of the
    in-mask distribution are labelled CSF.

    Reducing `csf_pct` (e.g. 80–90) yields a larger, more inclusive
    mask; increasing it toward 99 gives a stricter, smaller mask
    confined to the ventricle cores.  Inspect the QC overlay
    (*_csf_qc.png) to choose an appropriate value for your data.

    Parameters
    ----------
    dwi_data : ndarray, shape (X, Y, Z, N)
    bvals : ndarray, shape (N,)
    brain_mask : ndarray of uint8
    csf_pct : float
        Percentile threshold for the signal-ratio distribution
        (default 95.0 → top 5% of voxels).  Range: 50–99.

    Returns
    -------
    csf_mask : ndarray of uint8 or None
    """
    if not (50.0 <= csf_pct <= 99.0):
        raise ValueError(f"csf_pct must be between 50 and 99 (got {csf_pct}).")

    b0_idx = np.where(bvals < 50)[0]
    hi_idx = np.where(bvals > 5000)[0]

    if len(b0_idx) == 0 or len(hi_idx) == 0:
        log.warning("Could not auto-estimate CSF mask (b=0 or high-b shells missing).")
        return None

    b0_mean = dwi_data[..., b0_idx].mean(axis=3)
    hi_mean = dwi_data[..., hi_idx].mean(axis=3)
    hi_mean = np.where(hi_mean < 1, 1, hi_mean)

    ratio = (b0_mean / hi_mean) * (brain_mask > 0)
    threshold = np.percentile(ratio[brain_mask > 0], csf_pct)
    csf_mask = (ratio >= threshold).astype(np.uint8)

    if HAS_SCIPY:
        csf_mask = binary_erosion(csf_mask, iterations=1).astype(np.uint8)

    log.info("Auto CSF mask (P%.0f threshold): %d voxels", csf_pct, csf_mask.sum())
    return csf_mask


def estimate_wm_reference_mask(kurtosis_map: np.ndarray,
                                brain_mask: np.ndarray) -> np.ndarray | None:
    """Identify a white matter reference region (top 10% kurtosis).

    Parameters
    ----------
    kurtosis_map : ndarray, shape (X, Y, Z)
    brain_mask : ndarray of uint8

    Returns
    -------
    wm_mask : ndarray of uint8 or None
    """
    valid = kurtosis_map[brain_mask > 0]
    if len(valid) == 0:
        return None
    threshold = np.percentile(valid, 90)
    wm_mask = ((kurtosis_map >= threshold) & (brain_mask > 0)).astype(np.uint8)
    log.info("Auto WM reference mask: %d voxels", wm_mask.sum())
    return wm_mask


def compute_nld_csf_referenced(kurtosis_map: np.ndarray,
                                brain_mask: np.ndarray,
                                csf_mask: np.ndarray = None,
                                dwi_data: np.ndarray = None,
                                bvals: np.ndarray = None,
                                csf_pct: float = 95.0):
    """CSF-referenced kurtosis correction.

    Computes K_corrected = K_voxel − K_CSF, anchoring the scale at
    free-water diffusion (kurtosis ≈ 0).  This metric is directly
    comparable across subjects without requiring healthy controls.

    The returned NLD_csf is a [0, 100] rescaling for visualization only.
    Use K_corrected for inter-subject statistical analysis.

    Parameters
    ----------
    kurtosis_map : ndarray, shape (X, Y, Z)
    brain_mask : ndarray of uint8
    csf_mask : ndarray of uint8, optional
    dwi_data : ndarray, shape (X, Y, Z, N), optional
        Required for automatic CSF estimation if csf_mask is None.
    bvals : ndarray, shape (N,), optional
    csf_pct : float
        Percentile threshold passed to :func:`estimate_csf_mask`
        (ignored when csf_mask is supplied explicitly).

    Returns
    -------
    k_corrected : ndarray  (use for statistics)
    nld_csf : ndarray      (use for visualization)
    k_csf : float
    csf_mask : ndarray of uint8
    """
    log.info("Normalization: CSF-referenced (K_corrected = K − K_CSF)")

    if csf_mask is None and dwi_data is not None and bvals is not None:
        csf_mask = estimate_csf_mask(dwi_data, bvals, brain_mask,
                                     csf_pct=csf_pct)

    if csf_mask is not None and csf_mask.sum() > 10:
        csf_k = kurtosis_map[csf_mask > 0]
        k_csf = float(np.median(csf_k))
        log.info("CSF kurtosis — median: %.4f  mean: %.4f  SD: %.4f  n=%d",
                 k_csf, csf_k.mean(), csf_k.std(), len(csf_k))
    else:
        k_csf = float(np.percentile(kurtosis_map[brain_mask > 0], 2))
        log.warning("CSF mask unavailable; using P2 as fallback: %.4f", k_csf)
        csf_mask = None

    k_corrected = np.zeros_like(kurtosis_map)
    k_corrected[brain_mask > 0] = kurtosis_map[brain_mask > 0] - k_csf

    valid_pos = k_corrected[brain_mask > 0]
    valid_pos = valid_pos[valid_pos > 0]
    k_vis_max = float(np.percentile(valid_pos, 99)) if len(valid_pos) > 0 else 1.0

    nld_csf = np.zeros_like(kurtosis_map)
    nld_csf[brain_mask > 0] = np.clip(
        k_corrected[brain_mask > 0] / k_vis_max * 100.0, 0.0, 100.0
    )

    log.info("K_corrected range: [%.4f, %.4f]",
             k_corrected[brain_mask > 0].min(),
             k_corrected[brain_mask > 0].max())
    log.info("NLD_csf scale: 0 = CSF level, 100 = P99 tissue (K=%.4f)", k_vis_max)

    return k_corrected, nld_csf, k_csf, csf_mask


def compute_nld_internal_reference(kurtosis_map: np.ndarray,
                                    brain_mask: np.ndarray,
                                    csf_mask: np.ndarray = None,
                                    wm_ref_mask: np.ndarray = None,
                                    dwi_data: np.ndarray = None,
                                    bvals: np.ndarray = None,
                                    csf_pct: float = 95.0):
    """Internal reference normalization using CSF (K_min) and dense WM (K_max).

    Anchors the NLD scale to subject-specific tissue references, avoiding
    dependence on population-derived normalization constants.

    Parameters
    ----------
    kurtosis_map : ndarray, shape (X, Y, Z)
    brain_mask : ndarray of uint8
    csf_mask : ndarray of uint8, optional
    wm_ref_mask : ndarray of uint8, optional
    dwi_data : ndarray, shape (X, Y, Z, N), optional
    bvals : ndarray, shape (N,), optional
    csf_pct : float
        Percentile threshold passed to :func:`estimate_csf_mask`
        (ignored when csf_mask is supplied explicitly).

    Returns
    -------
    nld_map : ndarray
    k_min : float
    k_max : float
    csf_mask : ndarray of uint8
    wm_ref_mask : ndarray of uint8
    """
    log.info("Normalization: internal reference (CSF + WM)")

    if csf_mask is None and dwi_data is not None and bvals is not None:
        csf_mask = estimate_csf_mask(dwi_data, bvals, brain_mask,
                                     csf_pct=csf_pct)

    if csf_mask is not None and csf_mask.sum() > 10:
        csf_k = kurtosis_map[csf_mask > 0]
        k_min = float(np.median(csf_k))
        log.info("K_min from CSF: %.4f (n=%d)", k_min, len(csf_k))
    else:
        k_min = float(np.percentile(kurtosis_map[brain_mask > 0], 1))
        log.warning("CSF mask unavailable; using P1 as K_min: %.4f", k_min)
        csf_mask = None

    if wm_ref_mask is None:
        wm_ref_mask = estimate_wm_reference_mask(kurtosis_map, brain_mask)

    if wm_ref_mask is not None and wm_ref_mask.sum() > 10:
        wm_k = kurtosis_map[wm_ref_mask > 0]
        k_max = float(np.median(wm_k))
        log.info("K_max from WM ref: %.4f (n=%d)", k_max, len(wm_k))
    else:
        k_max = float(np.percentile(kurtosis_map[brain_mask > 0], 99))
        log.warning("WM mask unavailable; using P99 as K_max: %.4f", k_max)
        wm_ref_mask = None

    nld_map = compute_nld(kurtosis_map, k_max=k_max, k_min=k_min, mask=brain_mask)
    log.info("NLD range: [%.1f, %.1f]",
             nld_map[brain_mask > 0].min(), nld_map[brain_mask > 0].max())

    return nld_map, k_min, k_max, csf_mask, wm_ref_mask


def compute_myelin_map(dwi_data: np.ndarray, bvals: np.ndarray,
                        bvecs: np.ndarray, mask: np.ndarray = None,
                        k_max: float = None, k_min: float = None,
                        smooth_sigma: float = 0.5,
                        n_interp: int = 128):
    """Compute the NLD myelin map using vectorized q-space processing.

    For each gradient direction, the DWI signal across b-values is
    Fourier-transformed to estimate the displacement PDF, from which
    excess kurtosis is derived.  The mean kurtosis across directions is
    then normalized to the NLD scale.

    Parameters
    ----------
    dwi_data : ndarray, shape (X, Y, Z, N)
    bvals : ndarray, shape (N,)
    bvecs : ndarray, shape (3, N)
    mask : ndarray of uint8, optional
        Brain mask.  Auto-generated from the b=0 image if None.
    k_max : float, optional
    k_min : float, optional
    smooth_sigma : float
        Sigma for Gaussian smoothing of the kurtosis map (0 = disabled).
    n_interp : int
        Number of q-space interpolation points per direction.

    Returns
    -------
    kurtosis_map : ndarray, shape (X, Y, Z)
    nld_map : ndarray, shape (X, Y, Z)
    mask : ndarray of uint8, shape (X, Y, Z)
    """
    nx, ny, nz = dwi_data.shape[:3]
    log.info("Volume: %d × %d × %d, %d volumes", nx, ny, nz, dwi_data.shape[3])

    _, direction_groups = group_volumes_by_direction(bvals, bvecs)

    if mask is None:
        b0_idx = np.where(bvals < 50)[0]
        b0_mean = dwi_data[..., b0_idx].mean(axis=3) if len(b0_idx) > 0 \
                  else dwi_data[..., 0]
        thr = np.percentile(b0_mean[b0_mean > 0], 10)
        mask = (b0_mean > thr).astype(np.uint8)

    mi = np.where(mask > 0)
    n_voxels = len(mi[0])
    log.info("Voxels in mask: %d", n_voxels)

    vox = dwi_data[mi[0], mi[1], mi[2], :]  # (n_voxels, N)
    all_kurtosis = np.zeros((n_voxels, len(direction_groups)))

    for d_idx, group in enumerate(direction_groups):
        log.info("Processing direction %d/%d...", d_idx + 1, len(direction_groups))

        bvals_d = np.array([bv for bv, _ in group])
        signals = np.column_stack([
            vox[:, vidx].mean(axis=1) for _, vidx in group
        ])  # (n_voxels, n_bvals)

        s0 = signals[:, 0].copy()
        s0[s0 <= 0] = 1e-10
        E = np.clip(signals / s0[:, np.newaxis], 0.0, 1.0)

        q = np.sqrt(bvals_d)
        q_grid = np.linspace(0, q[-1], n_interp)

        E_grid = np.zeros((n_voxels, n_interp))
        for v in range(n_voxels):
            E_grid[v] = np.interp(q_grid, q, E[v])

        E_sym = np.concatenate([E_grid[:, ::-1], E_grid[:, 1:]], axis=1)
        pdf = np.abs(np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(E_sym, axes=1), axis=1), axes=1
        ))

        pdf_sum = pdf.sum(axis=1, keepdims=True)
        pdf_sum[pdf_sum <= 0] = 1.0
        pdf /= pdf_sum

        mu = pdf.mean(axis=1, keepdims=True)
        sigma = pdf.std(axis=1, keepdims=True)
        sigma[sigma < 1e-15] = 1e-15
        all_kurtosis[:, d_idx] = ((pdf - mu) ** 4 / sigma ** 4).mean(axis=1) - 3.0

    mean_kurtosis = all_kurtosis.mean(axis=1)

    kurtosis_map = np.zeros((nx, ny, nz))
    kurtosis_map[mi[0], mi[1], mi[2]] = mean_kurtosis

    if smooth_sigma > 0 and HAS_SCIPY:
        smoothed = gaussian_filter(kurtosis_map, sigma=smooth_sigma)
        kurtosis_map[mask > 0] = smoothed[mask > 0]

    log.info("Kurtosis range: [%.4f, %.4f]",
             kurtosis_map[mask > 0].min(), kurtosis_map[mask > 0].max())

    nld_map = compute_nld(kurtosis_map, k_max=k_max, k_min=k_min, mask=mask)

    return kurtosis_map, nld_map, mask


def compute_myelin_map_voxelwise(dwi_data: np.ndarray, bvals: np.ndarray,
                                  bvecs: np.ndarray, mask: np.ndarray = None,
                                  k_max: float = None, k_min: float = None):
    """Voxel-by-voxel NLD computation (reference implementation).

    Slower than :func:`compute_myelin_map` but useful for debugging or
    when memory is constrained.  Calls :func:`signal_to_pdf` and
    :func:`excess_kurtosis` directly.
    """
    nx, ny, nz = dwi_data.shape[:3]
    _, direction_groups = group_volumes_by_direction(bvals, bvecs)

    if mask is None:
        b0_idx = np.where(bvals < 50)[0]
        b0_mean = dwi_data[..., b0_idx].mean(axis=3) if len(b0_idx) > 0 \
                  else dwi_data[..., 0]
        thr = np.percentile(b0_mean[b0_mean > 0], 10)
        mask = (b0_mean > thr).astype(np.uint8)

    kurtosis_map = np.zeros((nx, ny, nz))
    total = mask.sum()
    done = 0

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if mask[x, y, z] == 0:
                    continue

                k_values = []
                for group in direction_groups:
                    bvals_d = np.array([bv for bv, _ in group])
                    sig = np.array([
                        dwi_data[x, y, z, vidx].mean() for _, vidx in group
                    ])
                    _, pdf = signal_to_pdf(sig, bvals_d)
                    if pdf is not None:
                        k_values.append(excess_kurtosis(pdf))

                if k_values:
                    kurtosis_map[x, y, z] = np.mean(k_values)

                done += 1
                if done % 10000 == 0:
                    log.info("Progress: %d/%d (%.1f%%)", done, total,
                             100 * done / total)

    nld_map = compute_nld(kurtosis_map, k_max=k_max, k_min=k_min, mask=mask)
    return kurtosis_map, nld_map, mask


def save_nifti(data: np.ndarray, affine: np.ndarray, header,
               path: str, dtype=np.float32):
    """Save a numpy array as a NIfTI file."""
    nib.save(nib.Nifti1Image(data.astype(dtype), affine, header), path)
    log.info("Saved: %s", path)


def visualize_nld(nld_volume: np.ndarray, output_prefix: str,
                  slice_axis: int = 2):
    """Save PNG visualizations of the NLD map.

    Generates a color-coded (white-to-blue) and a grayscale montage,
    consistent with the display style of Fujiyoshi et al. (2016).

    Parameters
    ----------
    nld_volume : ndarray, shape (X, Y, Z)
    output_prefix : str
    slice_axis : int
        0 = sagittal, 1 = coronal, 2 = axial.
    """
    if not HAS_MPL:
        log.warning("matplotlib not available; skipping PNG output.")
        return

    cmap_nld = LinearSegmentedColormap.from_list(
        "nld", [(1, 1, 1), (0, 0, 0.8)], N=256
    )

    n_slices = nld_volume.shape[slice_axis]
    n_show = min(16, n_slices)
    indices = np.linspace(int(n_slices * 0.15), int(n_slices * 0.85),
                          n_show, dtype=int)
    n_cols, n_rows = 4, int(np.ceil(n_show / 4))

    def _get_slice(vol, sl, ax):
        return np.rot90(np.take(vol, sl, axis=ax))

    for suffix, cmap in [("_nld.png", cmap_nld), ("_nld_gray.png", "gray")]:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for i, sl in enumerate(indices):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(_get_slice(nld_volume, sl, slice_axis),
                      cmap=cmap, vmin=0, vmax=100, interpolation="bilinear")
            ax.set_title(f"slice {sl}", fontsize=9)
            ax.axis("off")
        for i in range(n_show, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].axis("off")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
        fig.colorbar(sm, cax=cbar_ax).set_label("NLD (a.u.)", fontsize=11)
        fig.suptitle("NLD Myelin Map", fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        path = output_prefix + suffix
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="black" if "gray" not in suffix else "white")
        plt.close()
        log.info("Saved: %s", path)


def visualize_csf_overlay(nld_volume: np.ndarray, csf_mask: np.ndarray,
                           output_prefix: str, slice_axis: int = 2):
    """Save a QC PNG with the CSF mask overlaid on the NLD map.

    Selects slices that contain CSF voxels, displays the NLD map in
    grayscale, and renders the CSF mask as a semi-transparent red
    contour overlay.  Useful for verifying automatic CSF detection
    before accepting CSF-referenced normalization results.

    Parameters
    ----------
    nld_volume : ndarray, shape (X, Y, Z)
    csf_mask : ndarray of uint8, shape (X, Y, Z)
    output_prefix : str
    slice_axis : int
        0 = sagittal, 1 = coronal, 2 = axial.
    """
    if not HAS_MPL:
        log.warning("matplotlib not available; skipping CSF overlay PNG.")
        return

    def _get_slice(vol, sl, ax):
        return np.rot90(np.take(vol, sl, axis=ax))

    # Identify slices that contain at least one CSF voxel
    csf_sums = csf_mask.sum(axis=tuple(a for a in range(3) if a != slice_axis))
    csf_slices = np.where(csf_sums > 0)[0]

    if len(csf_slices) == 0:
        log.warning("CSF mask is empty; skipping overlay visualization.")
        return

    # Pick up to 16 evenly spaced slices from the CSF-containing range
    n_show = min(16, len(csf_slices))
    indices = csf_slices[
        np.linspace(0, len(csf_slices) - 1, n_show, dtype=int)
    ]

    n_cols = 4
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             facecolor="black")

    csf_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "csf_red", [(1, 0, 0, 0), (1, 0.15, 0.15, 0.75)], N=2
    ) if hasattr(plt.cm, "colors") else None

    for i, sl in enumerate(indices):
        ax = axes[i // n_cols, i % n_cols]
        nld_sl  = _get_slice(nld_volume, sl, slice_axis)
        csf_sl  = _get_slice(csf_mask,   sl, slice_axis).astype(float)

        ax.imshow(nld_sl, cmap="gray", vmin=0, vmax=100,
                  interpolation="bilinear")

        # Overlay: mask as solid red with transparency
        csf_rgba = np.zeros((*csf_sl.shape, 4))
        csf_rgba[csf_sl > 0] = [1.0, 0.15, 0.15, 0.65]
        ax.imshow(csf_rgba, interpolation="nearest")

        n_csf = int(csf_sl.sum())
        ax.set_title(f"sl {sl}  ({n_csf} vx)", fontsize=8, color="white")
        ax.axis("off")

    for i in range(n_show, n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_facecolor("black")
        ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend = fig.legend(
        handles=[Patch(color=(1, 0.15, 0.15, 0.65), label="CSF mask")],
        loc="lower center", ncol=1, fontsize=10,
        facecolor="#1a1a1a", edgecolor="none", labelcolor="white",
    )
    fig.suptitle("CSF mask QC — overlay on NLD map", fontsize=13,
                 color="white", y=1.01)
    plt.tight_layout()

    path = f"{output_prefix}_csf_qc.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    log.info("Saved: %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="NLD Myelin Map — Fujiyoshi et al. (2016)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # auto normalization
  python myelin_map.py --dwi dwi.nii.gz --bval data.bval --bvec data.bvec

  # CSF-referenced (recommended for MS studies)
  python myelin_map.py --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \\
                       --mask brain.nii.gz --norm-mode csf

  # internal reference (CSF + WM anchors)
  python myelin_map.py --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \\
                       --mask brain.nii.gz --norm-mode internal

  # fixed Kmax/Kmin from controls
  python myelin_map.py --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \\
                       --mask brain.nii.gz --kmax 15.0 --kmin -2.0
        """,
    )
    parser.add_argument("--dwi",      required=True, help="4D DWI NIfTI file")
    parser.add_argument("--bval",     required=True, help="FSL .bval file")
    parser.add_argument("--bvec",     required=True, help="FSL .bvec file")
    parser.add_argument("--mask",     default=None,  help="Brain mask NIfTI (optional)")
    parser.add_argument("--kmax",     type=float, default=None)
    parser.add_argument("--kmin",     type=float, default=None)
    parser.add_argument("--norm-mode",
                        choices=["auto", "manual", "internal", "csf"],
                        default="auto", dest="norm_mode",
                        help=("auto: P1/P99 from data  |  "
                              "csf: K − K_CSF (recommended for MS)  |  "
                              "internal: CSF + WM anchors  |  "
                              "manual: use --kmax/--kmin"))
    parser.add_argument("--csf-mask",    default=None, dest="csf_mask",
                        help="CSF mask NIfTI (auto-estimated if omitted)")
    parser.add_argument("--csf-pct",     type=float, default=95.0, dest="csf_pct",
                        help=("Percentile threshold for automatic CSF detection "
                              "(default: 95 → top 5%% of voxels by signal ratio). "
                              "Lower values (e.g. 80-90) produce a larger, more "
                              "inclusive mask; higher values (e.g. 98-99) restrict "
                              "the mask to ventricle cores. Inspect *_csf_qc.png "
                              "to evaluate the result."))
    parser.add_argument("--wm-ref-mask", default=None, dest="wm_ref_mask",
                        help="WM reference mask NIfTI (auto-estimated if omitted)")
    parser.add_argument("--output",   default="myelin", help="Output prefix")
    parser.add_argument("--method",   choices=["fast", "slow"], default="fast")
    parser.add_argument("--smooth",   type=float, default=0.5,
                        help="Gaussian smoothing sigma in voxels (0 = off)")
    parser.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2],
                        help="Slice axis for visualization (0=sag, 1=cor, 2=ax)")
    parser.add_argument("--version",  action="version", version=__version__)

    args = parser.parse_args()

    print(f"\n  qDWI Myelin Map  v{__version__}")
    print("  Fujiyoshi et al., J. Neurosci., 2016\n")

    log.info("[1/4] Loading data")
    dwi_img = nib.load(args.dwi)
    dwi_data = dwi_img.get_fdata(dtype=np.float32).astype(np.float64)
    affine, header = dwi_img.affine, dwi_img.header
    log.info("Shape: %s", dwi_data.shape)

    bvals, bvecs = load_bvals_bvecs(args.bval, args.bvec)
    log.info("b-values: %s", np.unique(np.round(bvals / 100) * 100).astype(int))
    log.info("Volumes: %d", len(bvals))

    mask = None
    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(np.uint8)

    norm_mode = "manual" if (args.kmax is not None and args.kmin is not None) \
                else args.norm_mode

    log.info("[2/4] Computing kurtosis and NLD")
    compute_fn = compute_myelin_map if args.method == "fast" \
                 else compute_myelin_map_voxelwise

    kurtosis_map, nld_map, mask = compute_fn(
        dwi_data, bvals, bvecs, mask=mask,
        k_max=args.kmax, k_min=args.kmin,
        **({"smooth_sigma": args.smooth} if args.method == "fast" else {})
    )

    k_corrected_map = None
    csf_mask_out = wm_ref_mask_out = None

    csf_mask_in = nib.load(args.csf_mask).get_fdata().astype(np.uint8) \
                  if args.csf_mask else None

    if norm_mode == "csf":
        log.info("[2b/4] CSF-referenced normalization")
        k_corrected_map, nld_map, _, csf_mask_out = compute_nld_csf_referenced(
            kurtosis_map, mask, csf_mask=csf_mask_in,
            dwi_data=dwi_data, bvals=bvals, csf_pct=args.csf_pct
        )
    elif norm_mode == "internal":
        log.info("[2b/4] Internal reference normalization")
        wm_mask_in = nib.load(args.wm_ref_mask).get_fdata().astype(np.uint8) \
                     if args.wm_ref_mask else None
        nld_map, _, _, csf_mask_out, wm_ref_mask_out = \
            compute_nld_internal_reference(
                kurtosis_map, mask, csf_mask=csf_mask_in,
                wm_ref_mask=wm_mask_in, dwi_data=dwi_data, bvals=bvals,
                csf_pct=args.csf_pct
            )

    log.info("[3/4] Saving outputs")
    out = args.output
    save_nifti(nld_map,       affine, header, f"{out}_nld.nii.gz")
    save_nifti(kurtosis_map,  affine, header, f"{out}_kurtosis.nii.gz")
    save_nifti(mask,          affine, header, f"{out}_mask.nii.gz", dtype=np.uint8)

    if k_corrected_map is not None:
        save_nifti(k_corrected_map, affine, header, f"{out}_kcorr.nii.gz")
        log.info("  → Use *_kcorr.nii.gz for inter-subject statistics")
    if csf_mask_out is not None:
        save_nifti(csf_mask_out,    affine, header, f"{out}_csf_mask.nii.gz",
                   dtype=np.uint8)
    if wm_ref_mask_out is not None:
        save_nifti(wm_ref_mask_out, affine, header, f"{out}_wm_ref_mask.nii.gz",
                   dtype=np.uint8)

    log.info("[4/4] Generating visualizations")
    visualize_nld(nld_map, out, slice_axis=args.slice_axis)
    if csf_mask_out is not None:
        visualize_csf_overlay(nld_map, csf_mask_out, out,
                              slice_axis=args.slice_axis)

    print(f"\n  Done. Outputs written to: {out}_*\n")


if __name__ == "__main__":
    main()
