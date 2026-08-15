# Submission checklist

Mapped to Track 5 “submission requirements”. Official filenames at repo root stay English.

| Official item | Our file | Note |
|---------------|----------|------|
| `skill.md` (required) | `skill.md` | One document; `skills/` is appendix |
| Project source | `spectral_conv/`, `fno_ns/` | SUPA kernels + FNO |
| Deps / build / run | `README.md`; `spectral_conv/build.sh`; `scripts/validate.sh` | Source SDK, then build |
| Accuracy scripts and results | `spectral_conv/test_accuracy.py`; `results/run_logs/` | rel ≤ `1e-4` |
| Perf scripts and report | `spectral_conv/test_perf.py`; `results/run_logs/` | 64 / 128 / 256; formal idle frozen |
| Run logs or screenshots | `results/run_logs/`; `demo/media/` | includes `brsmi` snapshot |
| Agent development log (≥5 entries, ≥3 scene types) | `development_log.md`; `AGENT_OFFICIAL.md`; `agent_logs/` | audit page first |
| Test-result write-up | `results.md` | same official data |
| Demo materials (recommended) | `demo/` | official-data flow fields |

Root filenames: `README.md`, `skill.md`, `results.md`, `development_log.md`.
