#!/usr/bin/env bash
# Load Biren / SUPA / torch_br.
set -euo pipefail

SDK="${SUPA_BASE:-/usr/local/birensupa/sdk/1.11.0.0.rc2}"
ENV_SCRIPT="${SDK}/scripts/brsw_set_env.sh"

if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "ERROR: missing ${ENV_SCRIPT}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_SCRIPT}"
export SUPA_BASE="${SDK}"

echo "SUPA_BASE=${SUPA_BASE}"
echo "env ready. next: bash scripts/validate.sh (serial GPU only)"
