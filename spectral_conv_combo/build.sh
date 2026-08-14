#!/usr/bin/env bash
set -euo pipefail

SUPA_BASE=${SUPA_BASE:-/usr/local/birensupa/sdk/1.11.0.0.rc2}
EXT_SUFFIX=$(python3-config --extension-suffix)

python3 - <<'PY' > /tmp/spectral_conv_torch_flags.sh
from torch.utils.cpp_extension import include_paths, library_paths
print('TORCH_INCS="' + ' '.join('-I'+p for p in include_paths()) + ' -I/usr/include/python3.10"')
print('TORCH_LIBDIRS="' + ' '.join('-L'+p for p in library_paths()) + '"')
PY
# shellcheck disable=SC1091
source /tmp/spectral_conv_torch_flags.sh

g++ -O2 -std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 ${TORCH_INCS} \
  -I/usr/local/lib/python3.10/dist-packages/torch_br/include \
  -I${SUPA_BASE}/supa/include \
  -I${SUPA_BASE}/sufft/include \
  -c spectral_conv_ext.cpp -o spectral_conv_ext_cpp.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  spectral_conv_ext.su -c -o spectral_conv_ext_su.o

${SUPA_BASE}/brcc/bin/brcc --supa-link -shared -fPIC \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  spectral_conv_ext_cpp.o spectral_conv_ext_su.o \
  ${TORCH_LIBDIRS} \
  -L/usr/local/lib/python3.10/dist-packages/torch_br/lib \
  -L${SUPA_BASE}/supa/lib \
  -L${SUPA_BASE}/sufft/lib \
  -Wl,-rpath,${SUPA_BASE}/sufft/lib \
  -lc10 -ltorch -ltorch_cpu -ltorch_python -ltorch_br -lsupa-runtime -lsufft \
  -o spectral_conv_ext${EXT_SUFFIX}

echo "built spectral_conv_ext${EXT_SUFFIX} (with suFFT)"
