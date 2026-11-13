# Demos

Self-contained, unrelated demos kept for reference. They are **not** part of
the ML pipeline and should not be coupled to it.

| Directory / file | Topic |
|---|---|
| `260205-isatty/` | `isatty` C/Python behavior |
| `260322-jax/` | JAX installation and device execution |
| `260410-endia/` | Endianness (x86 / MIPS) |
| `260428-MCP-SKILL/` | MCP server/client and Claude skills demos |
| `260615-docker/` | Docker + X11 (`xeyes`) |
| `260917-algo_PID/` | PID control tuning |
| `avx2-demos/` | AVX2 intrinsics, dispatch, Eigen, CUDA |
| `demo03-thynodu.py` | YOLO training/prediction against the thyroid dataset |
| `demo4SAM2InUltralytics.py` | SAM2 in Ultralytics |
| `inspectingSettings.py` | Inspect Ultralytics settings |
| `printTrueValueItemCoordinates.py` | NumPy `where` demo |
| `demo-awk_var.sh`, `demo_getRelativePath.py` | Bash/Python path demos |
| `streamlit-01-app.py` | Minimal Streamlit app |
| `thyroid/` | Thyroid DICOM demos (`printAllDicomTags.py`, `determineImageFileFormatInDicom.py`, `opencvTrackbarpos.py`) |
| `bash_script_-n.sh`, `claudeCode-bootstrap.sh` | Shell/Claude Code bootstrap demos |

Build artifacts under `avx2-demos/build/` are git-ignored and were removed from
the working tree during the merge.
