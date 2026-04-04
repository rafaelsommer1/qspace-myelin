# qspace-myelin

**NLD myelin mapping from q-space diffusion MRI**

[![CI](https://github.com/YOUR_USERNAME/qspace-myelin/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/qspace-myelin/actions)
[![PyPI](https://img.shields.io/pypi/v/qspace-myelin)](https://pypi.org/project/qspace-myelin/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Implements the **Normalized Leptokurtic Diffusion (NLD)** myelin mapping method described in:

> Fujiyoshi K. et al. (2016). *Application of q-Space Diffusion MRI for the Visualization of White Matter.* **Journal of Neuroscience**, 36(9), 2796–2808. https://doi.org/10.1523/JNEUROSCI.1770-15.2016

For each voxel, the water displacement probability density function (PDF) is estimated by Fourier transform of the multi-shell DWI signal decay. Excess kurtosis of the PDF, averaged across gradient directions, is normalized to produce the NLD index — a quantitative marker sensitive to myelin content.

---

## Installation

**Core + CLI:**
```bash
pip install qspace-myelin
```

**With graphical interface:**
```bash
pip install "qspace-myelin[gui]"
```

**From source:**
```bash
git clone https://github.com/YOUR_USERNAME/qspace-myelin.git
cd qspace-myelin
pip install -e ".[gui]"
```

### Standalone binary (no Python required)

Download the pre-built binary for your platform from the [Releases](https://github.com/YOUR_USERNAME/qspace-myelin/releases) page:

| Platform | File |
|---|---|
| Linux x86-64 | `myelin-map-gui-linux-x86_64` |
| macOS (Intel) | `myelin-map-gui-macos-x86_64` |
| macOS (Apple Silicon) | `myelin-map-gui-macos-arm64` |
| Windows | `myelin-map-gui-windows.exe` |

---

## Usage

### Graphical interface

```bash
myelin-map-gui
```

### Command line

```bash
# Auto normalization
myelin-map --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \
           --mask brain.nii.gz --output sub01/myelin

# CSF-referenced normalization (recommended for MS studies)
myelin-map --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \
           --mask brain.nii.gz --norm-mode csf --output sub01/myelin

# Adjust CSF threshold and inspect QC overlay
myelin-map --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \
           --mask brain.nii.gz --norm-mode csf --csf-pct 88 \
           --output sub01/myelin
```

### Full preprocessing pipeline

```bash
# With pre-stripped T1 (skips BET)
bash pipeline_myelin_map.sh \
  --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \
  --t1brain t1_brain.nii.gz \
  --norm_mode csf --csf_pct 90 \
  --output_dir results/sub01/
```

---

## Data requirements

| Parameter | Value |
|---|---|
| Acquisition | Multi-shell DWI |
| b-values | multiple shells, validated with 9 steps: 0–10,000 s/mm² |
| Format | NIfTI + FSL-style bval/bvec |

---

## Output files

| File | Description |
|---|---|
| `*_nld.nii.gz` | NLD myelin map (0–100, visualization) |
| `*_kurtosis.nii.gz` | Raw excess kurtosis map |
| `*_kcorr.nii.gz` | CSF-corrected kurtosis (**use for statistics**) |
| `*_csf_mask.nii.gz` | Auto-detected CSF mask |
| `*_nld.png` | Color-coded montage (white → blue) |
| `*_csf_qc.png` | CSF mask QC overlay |

---

## Normalization modes

| Mode | Description | Use case |
|---|---|---|
| `auto` | P1/P99 percentiles from data | Quick exploration |
| `csf` | K_corrected = K − K_CSF | **MS studies, no controls needed** |
| `internal` | CSF (Kmin) + dense WM (Kmax) | Single-subject normalization |
| `manual` | Fixed Kmax/Kmin from controls | Multi-site studies |

For inter-subject comparisons, use `--norm-mode csf` and the `*_kcorr.nii.gz` output.

---

## External dependencies (pipeline only)

The full pipeline script requires:
- [FSL](https://fsl.fmrib.ox.ac.uk/) ≥ 6.0 (eddy, BET, FAST)
- [ANTs](https://github.com/ANTsX/ANTs) ≥ 2.3 (recommended; FSL fnirt used as fallback)
- [MRtrix3](https://www.mrtrix.org/) (optional; for denoising and Gibbs correction)

---

## Citing

If you use qspace-myelin in your research, please cite both the software and the original method:

**Software:**
```
qspace-myelin: NLD myelin mapping from q-space diffusion MRI.
GitHub. https://github.com/rafaelsommer1/qspace-myelin
```

**Method:**
```
Fujiyoshi K. et al. (2016). Application of q-Space Diffusion MRI for the
Visualization of White Matter. Journal of Neuroscience, 36(9), 2796–2808.
https://doi.org/10.1523/JNEUROSCI.1770-15.2016
```

> **Note on AI-assisted development:** The implementation was developed by the authors, using
> AI-assisted code generation (Claude, Anthropic) to structure and optimize code.
> on the methodology in Fujiyoshi et al. (2016). All scientific decisions,
> parameter choices, and validation were performed by the authors.

---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
