#!/bin/bash
# =============================================================================
# qDWI Myelin Map Pipeline
# =============================================================================
# Full preprocessing + NLD myelin map computation.
# Reference: Fujiyoshi et al., J. Neurosci., 2016, 36(9):2796-2808.
#
# Steps:
#   0.  Dependency check and directory setup
#   1.  T1 preprocessing (reorient, BET if needed, FAST)
#   2.  DWI denoising (MRtrix3 dwidenoise, optional)
#   3.  Gibbs unringing (MRtrix3 mrdegibbs, optional)
#   4.  Eddy current + motion correction (FSL eddy)
#   5.  DWI brain extraction (BET on b0)
#   6.  b0 → T1 registration (ANTs SyN; FSL flirt+fnirt fallback)
#   7.  NLD myelin map (myelin_map.py, native DWI space)
#   8.  Myelin map → T1 space
#   9.  Inverse transform (T1 → DWI space, for overlay)
#   10. QC report
#
# Usage:
#   bash pipeline_myelin_map.sh \
#     --dwi  <dwi.nii.gz>   --bval <data.bval>  --bvec <data.bvec> \
#     --t1   <t1.nii.gz>    [OR  --t1brain <t1_brain.nii.gz>] \
#     --pe_dir <j|j-|i|i->  --output_dir <dir/>
#
# Requirements:
#   - FSL >= 6.0
#   - ANTs >= 2.3  (recommended; FSL fnirt used as fallback)
#   - Python 3 with nibabel, numpy, scipy, matplotlib
#   - MRtrix3 (optional: denoising and Gibbs correction)
# =============================================================================

set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  [${1}] ${2}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}
log_info() { echo -e "${YELLOW}  → ${1}${NC}"; }
log_ok()   { echo -e "${GREEN}  ✓ ${1}${NC}"; }
log_warn() { echo -e "${YELLOW}  ⚠ ${1}${NC}"; }
log_err()  { echo -e "${RED}  ✗ ERROR: ${1}${NC}"; }

# ── defaults ─────────────────────────────────────────────────────────────────
DWI=""; BVAL=""; BVEC=""
T1=""           # full T1 with skull (optional if --t1brain given)
T1_BRAIN=""     # pre-stripped T1 brain (optional; skips BET if provided)
PE_DIR="j-"     # phase-encoding direction: j- (AP) j (PA) i- (RL) i (LR)
READOUT_TIME=0.05
OUTPUT_DIR="output_myelin"
NTHREADS=4
BET_F_T1=0.35
BET_F_DWI=0.3
SMOOTH_SIGMA=0.5
KMAX=""; KMIN=""
NORM_MODE="auto"    # auto | csf | internal | manual
CSF_PCT=95          # percentile threshold for auto CSF detection
CSF_MASK=""         # optional manual CSF mask
SKIP_DENOISE=false
SKIP_EDDY=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── usage ────────────────────────────────────────────────────────────────────
usage() {
    cat << EOF

Usage: bash pipeline_myelin_map.sh [options]

Required:
  --dwi  <file>         4D DWI NIfTI (.nii.gz)
  --bval <file>         FSL .bval file
  --bvec <file>         FSL .bvec file

T1 input (at least one required):
  --t1       <file>     Full T1 with skull (.nii.gz)
  --t1brain  <file>     Pre-stripped T1 brain (.nii.gz)
                        If provided, BET is skipped.
                        If --t1 is also given, it is used for BBR registration.
                        If only --t1brain is given, linear registration is used.

Acquisition:
  --pe_dir   <dir>      Phase-encoding direction: j, j-, i, i-  (default: j-)
  --readout  <float>    Total readout time in seconds            (default: 0.05)

Processing:
  --output_dir <dir>    Output directory                         (default: output_myelin)
  --nthreads   <int>    Number of threads                        (default: 4)
  --bet_f_t1   <float>  BET fractional intensity for T1          (default: 0.35)
  --bet_f_dwi  <float>  BET fractional intensity for b0          (default: 0.3)
  --smooth     <float>  Kurtosis map smoothing sigma in voxels   (default: 0.5)

NLD normalization:
  --norm_mode  <mode>   auto | csf | internal | manual           (default: auto)
                          auto     : P1/P99 percentiles from data
                          csf      : K_corrected = K − K_CSF  [recommended for MS]
                          internal : CSF (Kmin) + dense WM (Kmax) anchors
                          manual   : fixed --kmax / --kmin values
  --csf_pct    <float>  Percentile for auto CSF detection (50–99) (default: 95)
                        Lower values → larger, more inclusive mask.
                        Inspect *_csf_qc.png to evaluate.
  --csf_mask   <file>   Manual CSF mask NIfTI (overrides auto detection)
  --kmax       <float>  Kmax for manual normalization
  --kmin       <float>  Kmin for manual normalization

Flags:
  --skip_denoise        Skip dwidenoise step
  --skip_eddy           Skip eddy correction step
  -h, --help            Show this help

Examples:
  # Minimal run (auto normalization)
  bash pipeline_myelin_map.sh \\
    --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \\
    --t1 t1.nii.gz --output_dir results/

  # Pre-stripped T1, CSF-referenced normalization
  bash pipeline_myelin_map.sh \\
    --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \\
    --t1brain t1_brain.nii.gz --norm_mode csf --output_dir results/

  # Adjust CSF threshold and inspect QC overlay
  bash pipeline_myelin_map.sh \\
    --dwi dwi.nii.gz --bval data.bval --bvec data.bvec \\
    --t1brain t1_brain.nii.gz --norm_mode csf --csf_pct 88 --output_dir results/

EOF
    exit 1
}

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dwi)          DWI="$2";          shift 2;;
        --bval)         BVAL="$2";         shift 2;;
        --bvec)         BVEC="$2";         shift 2;;
        --t1)           T1="$2";           shift 2;;
        --t1brain)      T1_BRAIN="$2";     shift 2;;
        --pe_dir)       PE_DIR="$2";       shift 2;;
        --readout)      READOUT_TIME="$2"; shift 2;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2;;
        --nthreads)     NTHREADS="$2";     shift 2;;
        --bet_f_t1)     BET_F_T1="$2";     shift 2;;
        --bet_f_dwi)    BET_F_DWI="$2";    shift 2;;
        --smooth)       SMOOTH_SIGMA="$2"; shift 2;;
        --norm_mode)    NORM_MODE="$2";    shift 2;;
        --csf_pct)      CSF_PCT="$2";      shift 2;;
        --csf_mask)     CSF_MASK="$2";     shift 2;;
        --kmax)         KMAX="$2";         shift 2;;
        --kmin)         KMIN="$2";         shift 2;;
        --skip_denoise) SKIP_DENOISE=true; shift;;
        --skip_eddy)    SKIP_EDDY=true;    shift;;
        -h|--help)      usage;;
        *) log_err "Unknown option: $1"; usage;;
    esac
done

# ── input validation ──────────────────────────────────────────────────────────
if [[ -z "$DWI" || -z "$BVAL" || -z "$BVEC" ]]; then
    log_err "--dwi, --bval and --bvec are required."
    usage
fi

if [[ -z "$T1" && -z "$T1_BRAIN" ]]; then
    log_err "At least one of --t1 or --t1brain is required."
    usage
fi

for f in "$DWI" "$BVAL" "$BVEC" \
         ${T1:+"$T1"} ${T1_BRAIN:+"$T1_BRAIN"} ${CSF_MASK:+"$CSF_MASK"}; do
    [[ -f "$f" ]] || { log_err "File not found: $f"; exit 1; }
done

if [[ -n "$KMAX" && -n "$KMIN" ]]; then
    NORM_MODE="manual"
fi

valid_norm="auto csf internal manual"
if [[ ! " $valid_norm " =~ " $NORM_MODE " ]]; then
    log_err "Invalid --norm_mode '$NORM_MODE'. Choose: $valid_norm"
    exit 1
fi

# ── step 0: setup ─────────────────────────────────────────────────────────────
log_step "0" "Setup and dependency check"

mkdir -p "${OUTPUT_DIR}"/{preproc,t1,registration,myelin_map,qc}
PREPROC="${OUTPUT_DIR}/preproc"
T1_DIR="${OUTPUT_DIR}/t1"
REG_DIR="${OUTPUT_DIR}/registration"
MYELIN_DIR="${OUTPUT_DIR}/myelin_map"
QC_DIR="${OUTPUT_DIR}/qc"

HAS_FSL=false; HAS_ANTS=false; HAS_MRTRIX=false

if command -v fsl &>/dev/null || [[ -n "${FSLDIR:-}" ]]; then
    HAS_FSL=true
    log_ok "FSL: ${FSLDIR:-$(dirname "$(which bet)")}"
else
    log_err "FSL not found. Install FSL >= 6.0."
    exit 1
fi

if command -v antsRegistration &>/dev/null; then
    HAS_ANTS=true
    log_ok "ANTs: $(which antsRegistration)"
else
    log_warn "ANTs not found. Falling back to FSL flirt + fnirt."
fi

if command -v dwidenoise &>/dev/null; then
    HAS_MRTRIX=true
    log_ok "MRtrix3: $(which dwidenoise)"
else
    log_warn "MRtrix3 not found. Denoising will be skipped."
    SKIP_DENOISE=true
fi

command -v python3 &>/dev/null || { log_err "python3 not found."; exit 1; }
log_ok "Python: $(python3 --version)"

cp "$DWI" "${PREPROC}/dwi_orig.nii.gz"
cp "$BVAL" "${PREPROC}/dwi.bval"
cp "$BVEC" "${PREPROC}/dwi.bvec"

NVOLS=$(fslnvols "${PREPROC}/dwi_orig.nii.gz")
log_info "DWI: ${NVOLS} volumes"
log_info "b-values: $(tr ' ' '\n' < "${PREPROC}/dwi.bval" | sort -n | uniq | tr '\n' ' ')"
log_info "Normalization mode: ${NORM_MODE}"
[[ "$NORM_MODE" == "csf" || "$NORM_MODE" == "internal" ]] && \
    log_info "CSF percentile threshold: ${CSF_PCT}"

# Save run parameters
cat << EOF > "${OUTPUT_DIR}/pipeline_params.txt"
qDWI Myelin Map Pipeline — run parameters
==========================================
Date:          $(date)
DWI:           $(readlink -f "$DWI")
BVAL:          $(readlink -f "$BVAL")
BVEC:          $(readlink -f "$BVEC")
T1:            ${T1:-(not provided)}
T1 brain:      ${T1_BRAIN:-(auto via BET)}
PE direction:  ${PE_DIR}
Readout time:  ${READOUT_TIME} s
BET f (T1):    ${BET_F_T1}
BET f (DWI):   ${BET_F_DWI}
Smooth sigma:  ${SMOOTH_SIGMA}
Norm mode:     ${NORM_MODE}
CSF pct:       ${CSF_PCT}
CSF mask:      ${CSF_MASK:-(auto)}
Kmax:          ${KMAX:-auto}
Kmin:          ${KMIN:-auto}
Threads:       ${NTHREADS}
EOF
log_info "Parameters saved to ${OUTPUT_DIR}/pipeline_params.txt"

# ── step 1: T1 preprocessing ──────────────────────────────────────────────────
log_step "1" "T1 preprocessing"

if [[ -n "$T1_BRAIN" ]]; then
    # ── case A: pre-stripped T1 provided ──────────────────────────────────────
    log_info "Using pre-stripped T1: ${T1_BRAIN}"
    cp "$T1_BRAIN" "${T1_DIR}/t1_brain.nii.gz"

    # Generate a binary brain mask from the brain image itself
    fslmaths "${T1_DIR}/t1_brain.nii.gz" -bin "${T1_DIR}/t1_brain_mask.nii.gz"
    log_ok "Brain mask derived from T1 brain image."

    if [[ -n "$T1" ]]; then
        # Full T1 also provided → reorient and keep for BBR
        log_info "Full T1 also provided; reorienting for BBR registration."
        fslreorient2std "$T1" "${T1_DIR}/t1_reoriented.nii.gz"
        T1_FULL_AVAILABLE=true
    else
        log_warn "--t1 not provided. BBR registration unavailable; linear registration will be used."
        T1_FULL_AVAILABLE=false
        # Placeholder so downstream scripts have something to reference
        cp "${T1_DIR}/t1_brain.nii.gz" "${T1_DIR}/t1_reoriented.nii.gz"
    fi
else
    # ── case B: full T1 — run BET ────────────────────────────────────────────
    log_info "Running brain extraction (BET f=${BET_F_T1})..."
    fslreorient2std "$T1" "${T1_DIR}/t1_reoriented.nii.gz"
    bet "${T1_DIR}/t1_reoriented.nii.gz" "${T1_DIR}/t1_brain" \
        -m -f "${BET_F_T1}" -R
    log_ok "T1 brain:  ${T1_DIR}/t1_brain.nii.gz"
    log_ok "Brain mask: ${T1_DIR}/t1_brain_mask.nii.gz"
    T1_FULL_AVAILABLE=true
fi

# Tissue segmentation (used for BBR and as anatomical reference)
log_info "Tissue segmentation (FAST)..."
fast -t 1 -n 3 -o "${T1_DIR}/t1_fast" "${T1_DIR}/t1_brain.nii.gz"
log_ok "Segmentation done."

# ── step 2: DWI denoising ─────────────────────────────────────────────────────
log_step "2" "DWI denoising"

DWI_CURRENT="${PREPROC}/dwi_orig.nii.gz"

if [[ "$SKIP_DENOISE" == true ]]; then
    log_warn "Denoising skipped."
    cp "${DWI_CURRENT}" "${PREPROC}/dwi_denoised.nii.gz"
else
    log_info "dwidenoise (Marchenko-Pastur PCA)..."
    dwidenoise "${DWI_CURRENT}" "${PREPROC}/dwi_denoised.nii.gz" \
        -noise "${PREPROC}/noise_map.nii.gz" \
        -nthreads "${NTHREADS}"
    fslmaths "${DWI_CURRENT}" -sub "${PREPROC}/dwi_denoised.nii.gz" \
        "${QC_DIR}/denoise_residual.nii.gz"
    log_ok "Noise map: ${PREPROC}/noise_map.nii.gz"
fi

DWI_CURRENT="${PREPROC}/dwi_denoised.nii.gz"

# ── step 3: Gibbs unringing ───────────────────────────────────────────────────
log_step "3" "Gibbs unringing"

if [[ "$HAS_MRTRIX" == true ]] && command -v mrdegibbs &>/dev/null; then
    log_info "mrdegibbs..."
    mrdegibbs "${DWI_CURRENT}" "${PREPROC}/dwi_degibbs.nii.gz" \
        -nthreads "${NTHREADS}"
    DWI_CURRENT="${PREPROC}/dwi_degibbs.nii.gz"
    log_ok "Gibbs unringing done."
else
    log_warn "mrdegibbs not available; skipping."
fi

# ── step 4: eddy correction ───────────────────────────────────────────────────
log_step "4" "Eddy current and motion correction"

if [[ "$SKIP_EDDY" == true ]]; then
    log_warn "Eddy skipped."
    cp "${DWI_CURRENT}" "${PREPROC}/dwi_eddy.nii.gz"
    cp "${PREPROC}/dwi.bvec" "${PREPROC}/dwi_eddy.eddy_rotated_bvecs"
else
    log_info "Extracting b0 for eddy mask..."
    fslroi "${DWI_CURRENT}" "${PREPROC}/b0_pre_eddy" 0 1
    bet "${PREPROC}/b0_pre_eddy" "${PREPROC}/b0_brain_pre_eddy" \
        -m -f "${BET_F_DWI}"

    log_info "Creating acqparams.txt (PE_DIR=${PE_DIR})..."
    case "${PE_DIR}" in
        j-)  echo "0 -1 0 ${READOUT_TIME}" > "${PREPROC}/acqparams.txt";;
        j)   echo "0  1 0 ${READOUT_TIME}" > "${PREPROC}/acqparams.txt";;
        i-)  echo "-1 0 0 ${READOUT_TIME}" > "${PREPROC}/acqparams.txt";;
        i)   echo "1  0 0 ${READOUT_TIME}" > "${PREPROC}/acqparams.txt";;
        *)   log_err "Invalid PE_DIR: ${PE_DIR}"; exit 1;;
    esac

    log_info "Creating index.txt..."
    NVOLS_EDDY=$(fslnvols "${DWI_CURRENT}")
    printf '%s ' $(seq 1 "${NVOLS_EDDY}" | xargs -I{} echo 1) \
        > "${PREPROC}/index.txt"

    EDDY_CMD="eddy"
    command -v eddy_openmp &>/dev/null && EDDY_CMD="eddy_openmp"
    command -v eddy_cuda   &>/dev/null && EDDY_CMD="eddy_cuda"
    log_info "Using: ${EDDY_CMD}"

    "${EDDY_CMD}" \
        --imain="${DWI_CURRENT}" \
        --mask="${PREPROC}/b0_brain_pre_eddy_mask.nii.gz" \
        --acqp="${PREPROC}/acqparams.txt" \
        --index="${PREPROC}/index.txt" \
        --bvals="${PREPROC}/dwi.bval" \
        --bvecs="${PREPROC}/dwi.bvec" \
        --out="${PREPROC}/dwi_eddy" \
        --data_is_shelled \
        --verbose

    log_ok "Eddy done: ${PREPROC}/dwi_eddy.nii.gz"

    [[ -f "${PREPROC}/dwi_eddy.eddy_movement_rms" ]] && \
        cp "${PREPROC}/dwi_eddy.eddy_movement_rms" "${QC_DIR}/"
fi

DWI_CURRENT="${PREPROC}/dwi_eddy.nii.gz"
BVEC_CURRENT="${PREPROC}/dwi_eddy.eddy_rotated_bvecs"
[[ -f "${BVEC_CURRENT}" ]] || {
    BVEC_CURRENT="${PREPROC}/dwi.bvec"
    log_warn "Rotated bvecs not found; using original bvecs."
}

# ── step 5: DWI brain extraction ──────────────────────────────────────────────
log_step "5" "DWI brain extraction (post-eddy)"

log_info "Extracting b0 (post-eddy)..."
fslroi "${DWI_CURRENT}" "${PREPROC}/b0_eddy" 0 1

log_info "BET on b0 (f=${BET_F_DWI})..."
bet "${PREPROC}/b0_eddy" "${PREPROC}/b0_brain" -m -f "${BET_F_DWI}"

fslmaths "${DWI_CURRENT}" -mas "${PREPROC}/b0_brain_mask.nii.gz" \
    "${PREPROC}/dwi_eddy_brain.nii.gz"
log_ok "Masked DWI: ${PREPROC}/dwi_eddy_brain.nii.gz"

slicer "${PREPROC}/b0_brain.nii.gz" \
    -a "${QC_DIR}/b0_brain_qc.png" 2>/dev/null || true

# ── step 6: b0 → T1 registration ─────────────────────────────────────────────
log_step "6" "b0 → T1 registration (EPI distortion correction)"

REG_METHOD=""

if [[ "$HAS_ANTS" == true ]]; then
    log_info "ANTs SyN registration..."
    antsRegistrationSyN.sh \
        -d 3 \
        -f "${T1_DIR}/t1_brain.nii.gz" \
        -m "${PREPROC}/b0_brain.nii.gz" \
        -o "${REG_DIR}/epi2t1_" \
        -t s \
        -n "${NTHREADS}"

    if [[ -f "${REG_DIR}/epi2t1_1Warp.nii.gz" ]]; then
        REG_METHOD="ants_syn"
        log_ok "Warp:   ${REG_DIR}/epi2t1_1Warp.nii.gz"
        log_ok "Affine: ${REG_DIR}/epi2t1_0GenericAffine.mat"
        log_ok "b0 in T1 space: ${REG_DIR}/epi2t1_Warped.nii.gz"
    else
        log_err "ANTs SyN failed. Falling back to FSL."
        HAS_ANTS=false
    fi
fi

if [[ "$HAS_ANTS" == false ]]; then
    log_info "FSL registration fallback..."

    if [[ "$T1_FULL_AVAILABLE" == true && -f "${T1_DIR}/t1_fast_seg.nii.gz" ]]; then
        log_info "BBR (boundary-based registration)..."
        epi_reg \
            --epi="${PREPROC}/b0_brain.nii.gz" \
            --t1="${T1_DIR}/t1_reoriented.nii.gz" \
            --t1brain="${T1_DIR}/t1_brain.nii.gz" \
            --out="${REG_DIR}/epi2t1_bbr"
        cp "${REG_DIR}/epi2t1_bbr.mat" "${REG_DIR}/epi2t1_linear.mat"
    else
        [[ "$T1_FULL_AVAILABLE" == false ]] && \
            log_warn "Full T1 not available; BBR skipped. Using 6-DOF flirt."
        flirt \
            -in "${PREPROC}/b0_brain.nii.gz" \
            -ref "${T1_DIR}/t1_brain.nii.gz" \
            -omat "${REG_DIR}/epi2t1_linear.mat" \
            -out "${REG_DIR}/epi2t1_linear.nii.gz" \
            -dof 6 -cost mutualinfo
    fi

    log_info "fnirt (non-linear distortion correction)..."
    fnirt \
        --in="${PREPROC}/b0_brain.nii.gz" \
        --ref="${T1_DIR}/t1_brain.nii.gz" \
        --aff="${REG_DIR}/epi2t1_linear.mat" \
        --cout="${REG_DIR}/epi2t1_warpcoef.nii.gz" \
        --iout="${REG_DIR}/epi2t1_fnirt.nii.gz" \
        --subsamp=4,2,1,1

    REG_METHOD="fsl_fnirt"
    log_ok "FSL registration done."
fi

[[ "$REG_METHOD" == "ants_syn" ]] && \
    slicer "${T1_DIR}/t1_brain.nii.gz" "${REG_DIR}/epi2t1_Warped.nii.gz" \
        -a "${QC_DIR}/registration_overlay.png" 2>/dev/null || true

# ── step 7: NLD myelin map ────────────────────────────────────────────────────
log_step "7" "NLD myelin map computation"

log_info "Computing in native DWI space (preserves original resolution)."

MYELIN_CMD="python3 ${SCRIPT_DIR}/myelin_map.py \
    --dwi    ${PREPROC}/dwi_eddy_brain.nii.gz \
    --bval   ${PREPROC}/dwi.bval \
    --bvec   ${BVEC_CURRENT} \
    --mask   ${PREPROC}/b0_brain_mask.nii.gz \
    --output ${MYELIN_DIR}/myelin \
    --smooth ${SMOOTH_SIGMA} \
    --norm-mode ${NORM_MODE} \
    --csf-pct ${CSF_PCT} \
    --method fast"

[[ -n "$KMAX"     ]] && MYELIN_CMD="${MYELIN_CMD} --kmax ${KMAX}"
[[ -n "$KMIN"     ]] && MYELIN_CMD="${MYELIN_CMD} --kmin ${KMIN}"
[[ -n "$CSF_MASK" ]] && MYELIN_CMD="${MYELIN_CMD} --csf-mask ${CSF_MASK}"

log_info "Command: $(echo ${MYELIN_CMD} | tr -s ' ')"
eval "${MYELIN_CMD}"

log_ok "Myelin map (native): ${MYELIN_DIR}/myelin_nld.nii.gz"

# Copy CSF QC image to the main QC directory for easy access
[[ -f "${MYELIN_DIR}/myelin_csf_qc.png" ]] && \
    cp "${MYELIN_DIR}/myelin_csf_qc.png" "${QC_DIR}/csf_mask_qc.png" && \
    log_ok "CSF QC overlay copied to ${QC_DIR}/csf_mask_qc.png"

# ── step 8: myelin map → T1 space ─────────────────────────────────────────────
log_step "8" "Myelin map → T1 space"

_apply_transform() {
    local input="$1" output="$2" interp="${3:-Linear}"
    if [[ "$REG_METHOD" == "ants_syn" ]]; then
        antsApplyTransforms -d 3 \
            -i "$input" -r "${T1_DIR}/t1_brain.nii.gz" \
            -t "${REG_DIR}/epi2t1_1Warp.nii.gz" \
            -t "${REG_DIR}/epi2t1_0GenericAffine.mat" \
            -o "$output" -n "$interp"
    else
        applywarp \
            --in="$input" --ref="${T1_DIR}/t1_brain.nii.gz" \
            --warp="${REG_DIR}/epi2t1_warpcoef.nii.gz" \
            --out="$output" --interp=trilinear
    fi
}

_apply_transform "${MYELIN_DIR}/myelin_nld.nii.gz" \
                 "${MYELIN_DIR}/myelin_nld_in_t1.nii.gz"
log_ok "NLD in T1 space: ${MYELIN_DIR}/myelin_nld_in_t1.nii.gz"

_apply_transform "${MYELIN_DIR}/myelin_kurtosis.nii.gz" \
                 "${MYELIN_DIR}/myelin_kurtosis_in_t1.nii.gz"

_apply_transform "${PREPROC}/b0_brain_mask.nii.gz" \
                 "${MYELIN_DIR}/dwi_mask_in_t1.nii.gz" \
                 "NearestNeighbor"

# K_corrected (CSF-referenced) if present
if [[ -f "${MYELIN_DIR}/myelin_kcorr.nii.gz" ]]; then
    _apply_transform "${MYELIN_DIR}/myelin_kcorr.nii.gz" \
                     "${MYELIN_DIR}/myelin_kcorr_in_t1.nii.gz"
    log_ok "K_corrected in T1 space: ${MYELIN_DIR}/myelin_kcorr_in_t1.nii.gz"
fi

# ── step 9: inverse transform (T1 → DWI) ─────────────────────────────────────
log_step "9" "Inverse transform: T1 → DWI space"

if [[ "$REG_METHOD" == "ants_syn" ]]; then
    antsApplyTransforms -d 3 \
        -i "${T1_DIR}/t1_brain.nii.gz" \
        -r "${PREPROC}/b0_brain.nii.gz" \
        -t "[${REG_DIR}/epi2t1_0GenericAffine.mat,1]" \
        -t "${REG_DIR}/epi2t1_1InverseWarp.nii.gz" \
        -o "${REG_DIR}/t1_in_dwi.nii.gz" -n Linear
else
    invwarp \
        --ref="${PREPROC}/b0_brain.nii.gz" \
        --warp="${REG_DIR}/epi2t1_warpcoef.nii.gz" \
        --out="${REG_DIR}/t12epi_warpcoef.nii.gz"
    applywarp \
        --in="${T1_DIR}/t1_brain.nii.gz" \
        --ref="${PREPROC}/b0_brain.nii.gz" \
        --warp="${REG_DIR}/t12epi_warpcoef.nii.gz" \
        --out="${REG_DIR}/t1_in_dwi.nii.gz" --interp=trilinear
fi
log_ok "T1 in DWI space: ${REG_DIR}/t1_in_dwi.nii.gz"

# ── step 10: QC and report ────────────────────────────────────────────────────
log_step "10" "QC and final report"

if command -v fsleyes &>/dev/null; then
    fsleyes render \
        --outfile "${QC_DIR}/myelin_on_t1.png" \
        --size 1200 400 \
        "${T1_DIR}/t1_brain.nii.gz" \
        "${MYELIN_DIR}/myelin_nld_in_t1.nii.gz" \
            --cmap hot --alpha 60 --displayRange 20 100 \
        2>/dev/null || log_warn "fsleyes render failed."
else
    slicer "${MYELIN_DIR}/myelin_nld_in_t1.nii.gz" \
        -a "${QC_DIR}/myelin_t1space_qc.png" 2>/dev/null || true
    slicer "${MYELIN_DIR}/myelin_nld.nii.gz" \
        -a "${QC_DIR}/myelin_native_qc.png"  2>/dev/null || true
fi

cat << EOF > "${OUTPUT_DIR}/REPORT.txt"
╔══════════════════════════════════════════════════════════════╗
║            qDWI Myelin Map Pipeline — Report                ║
╠══════════════════════════════════════════════════════════════╣
║ Date:      $(date)
║ Reference: Fujiyoshi et al., J. Neurosci., 2016, 36(9):2796
╠══════════════════════════════════════════════════════════════╣
║ INPUT
║   DWI:         $(readlink -f "$DWI")
║   Volumes:     ${NVOLS}
║   T1:          ${T1:-(not provided)}
║   T1 brain:    ${T1_BRAIN:-(generated via BET)}
║   PE direction: ${PE_DIR}
╠══════════════════════════════════════════════════════════════╣
║ PREPROCESSING
║   Denoising:    $( [[ "$SKIP_DENOISE" == true ]] && echo "skipped" || echo "dwidenoise (MP-PCA)")
║   Eddy:         $( [[ "$SKIP_EDDY"    == true ]] && echo "skipped" || echo "${EDDY_CMD:-eddy}")
║   T1 BET:       $( [[ -n "$T1_BRAIN" ]] && echo "skipped (pre-stripped T1 provided)" || echo "bet f=${BET_F_T1}")
║   Registration: ${REG_METHOD}
╠══════════════════════════════════════════════════════════════╣
║ NLD NORMALIZATION
║   Mode:         ${NORM_MODE}
║   CSF pct:      ${CSF_PCT}
║   CSF mask:     ${CSF_MASK:-(auto-estimated)}
║   Kmax:         ${KMAX:-auto}
║   Kmin:         ${KMIN:-auto}
╠══════════════════════════════════════════════════════════════╣
║ OUTPUT FILES
║
║ Native DWI space:
║   NLD map:            ${MYELIN_DIR}/myelin_nld.nii.gz
║   Kurtosis map:       ${MYELIN_DIR}/myelin_kurtosis.nii.gz
║   Brain mask:         ${MYELIN_DIR}/myelin_mask.nii.gz
$( [[ -f "${MYELIN_DIR}/myelin_kcorr.nii.gz" ]] && \
   echo "║   K_corrected:        ${MYELIN_DIR}/myelin_kcorr.nii.gz  ← USE FOR STATISTICS" )
$( [[ -f "${MYELIN_DIR}/myelin_csf_mask.nii.gz" ]] && \
   echo "║   CSF mask:           ${MYELIN_DIR}/myelin_csf_mask.nii.gz" )
║
║ T1 space:
║   NLD map:            ${MYELIN_DIR}/myelin_nld_in_t1.nii.gz
║   Kurtosis map:       ${MYELIN_DIR}/myelin_kurtosis_in_t1.nii.gz
$( [[ -f "${MYELIN_DIR}/myelin_kcorr_in_t1.nii.gz" ]] && \
   echo "║   K_corrected:        ${MYELIN_DIR}/myelin_kcorr_in_t1.nii.gz" )
║
║ Cross-references:
║   T1 in DWI space:    ${REG_DIR}/t1_in_dwi.nii.gz
║
║ QC:
║   b0 brain:           ${QC_DIR}/b0_brain_qc.png
║   Registration:       ${QC_DIR}/registration_overlay.png
$( [[ -f "${QC_DIR}/csf_mask_qc.png" ]] && \
   echo "║   CSF mask overlay:   ${QC_DIR}/csf_mask_qc.png  ← inspect before accepting normalization" )
║   NLD (color):        ${MYELIN_DIR}/myelin_nld.png
║   NLD (gray):         ${MYELIN_DIR}/myelin_nld_gray.png
╚══════════════════════════════════════════════════════════════╝
EOF

log_ok "Report: ${OUTPUT_DIR}/REPORT.txt"

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   Pipeline complete!                        ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  NLD (native): ${MYELIN_DIR}/myelin_nld.nii.gz${NC}"
echo -e "${GREEN}║  NLD (T1):     ${MYELIN_DIR}/myelin_nld_in_t1.nii.gz${NC}"
if [[ -f "${MYELIN_DIR}/myelin_kcorr.nii.gz" ]]; then
echo -e "${GREEN}║  K_corr:       ${MYELIN_DIR}/myelin_kcorr.nii.gz  ← statistics${NC}"
fi
if [[ -f "${QC_DIR}/csf_mask_qc.png" ]]; then
echo -e "${GREEN}║  CSF QC:       ${QC_DIR}/csf_mask_qc.png  ← review before use${NC}"
fi
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Visualize:                                                 ║${NC}"
echo -e "${GREEN}║    fsleyes ${T1_DIR}/t1_brain.nii.gz \\                     ║${NC}"
echo -e "${GREEN}║      ${MYELIN_DIR}/myelin_nld_in_t1.nii.gz \\               ║${NC}"
echo -e "${GREEN}║      --cmap hot --alpha 50 --displayRange 20 100            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
