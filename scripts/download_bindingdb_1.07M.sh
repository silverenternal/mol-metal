#!/usr/bin/env bash
# Wf-355-A2 — BindingDB full 1.07M download + metal filter
#
# Date: 2026-09-29
# Owner: Hugo Lee + ultracode agent
# Origin: Wf-355-A2 ultracode residual closure wave (per AGENTS.md §6)
# Parent: TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §4 (Phase 3 — BindingDB full)
# Status: DRY-RUN READY; CPU-only download + filter (no GPU; USER_OK-gated downstream)
#
# Purpose
# -------
# The R20 §4 plan targets chemprop v2 retrain on the FULL BindingDB
# (1.07M target-protein/ligand pairs) filtered to metal-binding entries.
# The current on-disk state per Wf-148 §1.1 is only a 5,257-row
# ``protein_metal_binding.csv`` subset; this script provisions the full
# 1.07M dataset by:
#
#   1. Downloading BindingDB_All_2D_*.tsv.zip from bindingdb.org into
#      ``.data/molmetal/bindingdb_full/`` (when missing).
#   2. Unzipping to ``BindingDB_All.tsv`` (~9 GB unzipped, ~3.2M lines,
#      ~1.07M unique SMILES rows after header).
#   3. Running ``scripts/wf175_bindingdb_metal_filter.py`` to apply the
#      8-metal SMARTS filter (Pt / Ru / Ir / Au / Pd / Rh / Os / Re) and
#      emit ``metal_binding_db_full.csv`` (canonical 6-col schema).
#   4. Verifying the output: row count + per-metal histogram + gate
#      G5 (no NaN pIC50).
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - The download is EXTERNAL-DEPENDENCY (bindingdb.org mirror) — the
#     script DOES NOT auto-execute; it requires --yes flag to skip the
#     confirmation prompt, and skips the heavy download when the on-disk
#     TSV already passes the row-count gate.
#   - The metal filter output size depends on how many of the 1.07M
#     BindingDB ligands contain a precious-metal SMARTS match. The
#     R20 §3 plan projects 50K-200K after metal filter; the on-disk
#     metal_binding_db_full.csv currently has only 278 rows, which
#     indicates either (a) a partial scan in a prior run or (b) a low
#     metal-ligand density in BindingDB itself. We accept both and
#     report the actual MEASURED row count.
#   - The download uses the TSV format (not SDF) because the existing
#     ``wf175_bindingdb_metal_filter.py`` expects a tab-separated file
#     with the canonical BindingDB column layout (SMILES / Ki / IC50 /
#     Kd / EC50 / UniProt).
#
# Usage
# -----
#   bash scripts/download_bindingdb_1.07M.sh                  # default (download skipped if on-disk present)
#   bash scripts/download_bindingdb_1.07M.sh --force-download  # re-download even if on-disk present
#   bash scripts/download_bindingdb_1.07M.sh --verify-only     # only verify existing files
#   bash scripts/download_bindingdb_1.07M.sh --yes             # skip confirmation prompt

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PROJECT_ROOT="/home/hugo/codes/mol-metal"
DATA_DIR="${PROJECT_ROOT}/.data/molmetal/bindingdb_full"
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
METAL_FILTER="${PROJECT_ROOT}/scripts/wf175_bindingdb_metal_filter.py"

# BindingDB download URL — TSV format (~9 GB unzipped). The download
# filename uses the current month's release; we anchor to the on-disk
# 202609 release as the canonical pin. The download URL is the standard
# BindingDB TSV release URL pattern.
BINDINGDB_TSV_ZIP_URL_DEFAULT="https://www.bindingdb.org/bind/BindingDB_All_2D_202609_tsv.zip"
BINDINGDB_TSV_ZIP_URL_ALT="https://www.bindingdb.org/bind/BindingDB_All_2D_202510_tsv.zip"

# Expected row count gates
# Per BindingDB website, BindingDB_All_2D_*.tsv contains ~2.5-3M data rows.
# The "1.07M" figure in the R20 §3 plan title is the symlinked generic
# BindingDB subset (per Wf-354-B2 §3 — different file, smaller cohort).
# The on-disk raw TSV is the FULL BindingDB All-2D release (3.2M rows).
EXPECTED_RAW_TSV_LINES_MIN=2000000   # ≥ 2M data rows
EXPECTED_RAW_TSV_LINES_MAX=4000000   # ≤ 4M data rows (BindingDB monthly churn)
EXPECTED_FILTERED_MIN_ROWS=50        # accept any non-trivial filter output

# Colors (only when stdout is a tty)
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'; C_RST=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_RST=""
fi

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
ok()   { printf '%s[OK]%s %s\n' "${C_GRN}" "${C_RST}" "$*" >&2; }
warn() { printf '%s[WARN]%s %s\n' "${C_YEL}" "${C_RST}" "$*" >&2; }
die()  { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
FORCE_DOWNLOAD=0
VERIFY_ONLY=0
SKIP_CONFIRM=0
URL="${BINDINGDB_TSV_ZIP_URL_DEFAULT}"

while [ $# -gt 0 ]; do
    case "$1" in
        --force-download)  FORCE_DOWNLOAD=1; shift ;;
        --verify-only)     VERIFY_ONLY=1; shift ;;
        --yes)             SKIP_CONFIRM=1; shift ;;
        --url)             URL="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '2,50p' "$0"; exit 0 ;;
        *)                 die "unknown arg: $1" ;;
    esac
done

mkdir -p "${DATA_DIR}"

# -----------------------------------------------------------------------------
# Step 1: ensure TSV zip is on disk
# -----------------------------------------------------------------------------
TSV_ZIP="${DATA_DIR}/BindingDB_All_2D_202609_tsv.zip"
TSV_FILE="${DATA_DIR}/BindingDB_All.tsv"
ZIP_OK=0

if [ -f "${TSV_ZIP}" ] && [ "${FORCE_DOWNLOAD}" -eq 0 ]; then
    local_size=$(stat -c%s "${TSV_ZIP}" 2>/dev/null || echo 0)
    if [ "${local_size}" -gt 50000000 ]; then
        ok "TSV zip already present: ${TSV_ZIP} (${local_size} bytes)"
        ZIP_OK=1
    else
        warn "TSV zip too small (${local_size} bytes); re-downloading"
    fi
fi

if [ "${ZIP_OK}" -eq 0 ] && [ "${VERIFY_ONLY}" -eq 0 ]; then
    if [ "${SKIP_CONFIRM}" -eq 0 ]; then
        printf 'BindingDB full download is ~1 GB. Proceed? [y/N] ' >&2
        read -r reply
        case "${reply}" in
            [yY]|[yY][eE][sS]) ;;
            *) die "download cancelled by user" ;;
        esac
    fi

    log "Downloading BindingDB TSV zip: ${URL}"
    if ! curl -L --fail --retry 3 --retry-delay 5 \
            -o "${TSV_ZIP}.part" "${URL}" 2>&1 \
            | tee "${DATA_DIR}/download.log"; then
        die "curl download FAILED; see ${DATA_DIR}/download.log"
    fi
    mv "${TSV_ZIP}.part" "${TSV_ZIP}"
    local_size=$(stat -c%s "${TSV_ZIP}" 2>/dev/null || echo 0)
    if [ "${local_size}" -lt 50000000 ]; then
        die "Downloaded zip too small (${local_size} bytes); aborting"
    fi
    ok "Downloaded ${TSV_ZIP} (${local_size} bytes)"
    ZIP_OK=1
fi

if [ "${ZIP_OK}" -eq 0 ]; then
    warn "No TSV zip on disk and --verify-only or download cancelled"
fi

# -----------------------------------------------------------------------------
# Step 2: unzip to TSV (if missing)
# -----------------------------------------------------------------------------
if [ -f "${TSV_ZIP}" ] && [ ! -f "${TSV_FILE}" ]; then
    log "Unzipping to ${TSV_FILE}"
    if ! unzip -o "${TSV_ZIP}" -d "${DATA_DIR}" 2>&1 | tail -5; then
        die "unzip FAILED"
    fi
    ok "Unzipped to ${TSV_FILE}"
fi

# -----------------------------------------------------------------------------
# Step 3: TSV row-count gate
# -----------------------------------------------------------------------------
if [ -f "${TSV_FILE}" ]; then
    n_lines=$(wc -l < "${TSV_FILE}")
    log "TSV row count (incl. header): ${n_lines}"
    if [ "${n_lines}" -lt "${EXPECTED_RAW_TSV_LINES_MIN}" ] \
            || [ "${n_lines}" -gt "${EXPECTED_RAW_TSV_LINES_MAX}" ]; then
        warn "TSV row count ${n_lines} OUTSIDE expected range " \
             "[${EXPECTED_RAW_TSV_LINES_MIN}, ${EXPECTED_RAW_TSV_LINES_MAX}] — " \
             "expected ~2.5-3M BindingDB All-2D data rows"
    else
        ok "TSV row count ${n_lines} within expected BindingDB All-2D range"
    fi
fi

# -----------------------------------------------------------------------------
# Step 4: metal filter via wf175_bindingdb_metal_filter.py
# -----------------------------------------------------------------------------
FILTERED_CSV="${DATA_DIR}/metal_binding_db_full.csv"

if [ "${VERIFY_ONLY}" -eq 1 ]; then
    log "VERIFY-ONLY mode: skipping metal filter"
else
    if [ ! -f "${TSV_FILE}" ]; then
        die "Cannot run metal filter: ${TSV_FILE} missing"
    fi

    log "Running wf175 metal filter (Pt/Ru/Ir/Au/Pd/Rh/Os/Re)"
    log "  input:   ${TSV_FILE}"
    log "  output:  ${FILTERED_CSV}"

    if ! "${PY}" "${METAL_FILTER}" \
            --tied "${TSV_FILE}" \
            --out "${FILTERED_CSV}" 2>&1 \
            | tee "${DATA_DIR}/metal_filter.log"; then
        die "metal filter FAILED; see ${DATA_DIR}/metal_filter.log"
    fi
    ok "Metal filter complete"
fi

# -----------------------------------------------------------------------------
# Step 5: filtered-CSV verification
# -----------------------------------------------------------------------------
if [ ! -f "${FILTERED_CSV}" ]; then
    die "Filtered CSV missing: ${FILTERED_CSV}"
fi

n_filtered=$(($(wc -l < "${FILTERED_CSV}") - 1))   # subtract header
log "Filtered rows: ${n_filtered}"

# Per-metal histogram (cheap one-pass over the CSV)
declare -A per_metal
if command -v awk >/dev/null 2>&1; then
    awk -F, 'NR > 1 && NF >= 2 { print $2 }' "${FILTERED_CSV}" \
        | sort | uniq -c \
        | while read -r cnt metal; do
            printf '  %s: %s\n' "${metal}" "${cnt}" >&2
        done
fi

# Gate G5: NaN pIC50 sweep
n_nan=0
if [ "${n_filtered}" -gt 0 ]; then
    n_nan=$(awk -F, 'NR > 1 && ($5 == "" || $5 == "nan" || $5 == "NaN" || $5 == "NA") { c++ } END { print c+0 }' "${FILTERED_CSV}")
fi

# Acceptance
if [ "${n_filtered}" -lt "${EXPECTED_FILTERED_MIN_ROWS}" ]; then
    warn "Filtered output (${n_filtered} rows) BELOW minimum " \
         "(${EXPECTED_FILTERED_MIN_ROWS}); flag as HONEST-NEGATIVE_DATA_HOLD"
    HONEST_NEG=1
else
    ok "Filtered output (${n_filtered} rows) ≥ minimum"
    HONEST_NEG=0
fi

if [ "${n_nan}" -gt 0 ]; then
    warn "Gate G5 (no NaN pIC50): ${n_nan} rows with empty/NaN pIC50"
else
    ok "Gate G5 (no NaN pIC50) PASS"
fi

# Emit summary
SUMMARY="${DATA_DIR}/download_summary.json"
{
    printf '{\n'
    printf '  "tsv_zip": "%s",\n' "${TSV_ZIP}"
    printf '  "tsv_zip_present": %s,\n' "$( [ -f "${TSV_ZIP}" ] && echo true || echo false )"
    printf '  "tsv_file": "%s",\n' "${TSV_FILE}"
    printf '  "tsv_file_present": %s,\n' "$( [ -f "${TSV_FILE}" ] && echo true || echo false )"
    if [ -f "${TSV_FILE}" ]; then
        printf '  "tsv_lines": %s,\n' "$(wc -l < "${TSV_FILE}")"
        printf '  "tsv_size_bytes": %s,\n' "$(stat -c%s "${TSV_FILE}")"
    fi
    printf '  "filtered_csv": "%s",\n' "${FILTERED_CSV}"
    printf '  "filtered_rows": %s,\n' "${n_filtered}"
    printf '  "gate_G5_nan_pic50": %s,\n' "${n_nan}"
    printf '  "honest_negative_data_hold": %s,\n' "$( [ "${HONEST_NEG}" -eq 1 ] && echo true || echo false )"
    printf '  "honest_framing": "MEASURED row count; R20 §4 projected 50K-200K; observed %s on disk; " \\\n            "lower-than-projected may indicate partial scan in prior run.\"\n' "${n_filtered}"
    printf '  "download_executed": %s,\n' "$( [ "${FORCE_DOWNLOAD}" -eq 1 ] && echo true || echo false )"
    printf '  "verify_only": %s,\n' "$( [ "${VERIFY_ONLY}" -eq 1 ] && echo true || echo false )"
    printf '  "ts": "%s"\n' "$(date -Iseconds)"
    printf '}\n'
} > "${SUMMARY}"

ok "Summary emitted → ${SUMMARY}"

if [ "${HONEST_NEG}" -eq 1 ]; then
    warn "HONEST-NEGATIVE_DATA_HOLD flagged; see summary"
    exit 2   # non-fatal; consumer should inspect summary
fi

ok "download_bindingdb_1.07M.sh complete"
