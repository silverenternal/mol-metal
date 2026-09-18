#!/bin/bash
# Fake vina binary for WF-1 verification.
# Real vina needs receptor + ligand + config; we have none. The script's
# try/except around docking will catch subprocess errors and still write
# n_decoded to the report. We just need a binary that exists and exits 0.
exit 0
