# CLAUDE.md

Project context, HPC workflow, conventions, and known pitfalls live in
**AGENTS.md** — read that first. Result-file locations are in **RESULTS.md**.

Quick rules:
- Vision pipeline never reads `S12.nc` or radiometric °C values.
- All heavy work runs as parallel SLURM jobs on Spartan (`gpu-h100` for GPU,
  default CPU partition otherwise). Submit jobs concurrently when independent.
- Review rendering: rectangular bounding boxes (never filled dots), no
  trajectory trails, "detect"/"tracking" status label above each box.
- Run `python3 vision_autolabel.py --self-test` after touching the labeler.
