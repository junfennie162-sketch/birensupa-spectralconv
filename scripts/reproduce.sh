#!/usr/bin/env bash
# Back-compat: same as validate.sh (build + accuracy + unofficial pruned probe).
# Does not run test_perf.py.
exec "$(cd "$(dirname "$0")" && pwd)/validate.sh"
