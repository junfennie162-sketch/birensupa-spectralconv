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

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fft.su -c -o pruned_fft_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_geo.su -c -o pruned_geo_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_irfft_w64.su -c -o pruned_irfft_w64_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_ifft_h_fact.su -c -o pruned_ifft_h_fact_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_rfft_w_fact.su -c -o pruned_rfft_w_fact_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_rfft_w_fact256_n16.su -c -o pruned_rfft_w_fact256_n16_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fft_h_fact.su -c -o pruned_fft_h_fact_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fft_h_fact256.su -c -o pruned_fft_h_fact256_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_ifft_h_fact256.su -c -o pruned_ifft_h_fact256_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fwd_fact64.su -c -o pruned_fwd_fact64_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_irfft_w256_pair.su -c -o pruned_irfft_w256_pair_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_irfft_w128_vec4.su -c -o pruned_irfft_w128_vec4_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused256.su -c -o pruned_inv_fused256_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused256_n4.su -c -o pruned_inv_fused256_n4_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused128.su -c -o pruned_inv_fused128_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused128_n4.su -c -o pruned_inv_fused128_n4_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused128_n8.su -c -o pruned_inv_fused128_n8_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused64.su -c -o pruned_inv_fused64_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_inv_fused64_n8.su -c -o pruned_inv_fused64_n8_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fwd_fused64.su -c -o pruned_fwd_fused64_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fwd_fused128.su -c -o pruned_fwd_fused128_su.o

${SUPA_BASE}/brcc/bin/brcc -fPIC -O2 \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  -I${SUPA_BASE}/supa/include \
  pruned_fwd_fused256.su -c -o pruned_fwd_fused256_su.o

${SUPA_BASE}/brcc/bin/brcc --supa-link -shared -fPIC \
  --supa-gpu-arch=br100 \
  --supa-path=${SUPA_BASE}/supa \
  spectral_conv_ext_cpp.o spectral_conv_ext_su.o pruned_fft_su.o pruned_geo_su.o pruned_irfft_w64_su.o pruned_ifft_h_fact_su.o pruned_rfft_w_fact_su.o pruned_rfft_w_fact256_n16_su.o pruned_fft_h_fact_su.o pruned_fft_h_fact256_su.o pruned_ifft_h_fact256_su.o pruned_fwd_fact64_su.o pruned_irfft_w256_pair_su.o pruned_irfft_w128_vec4_su.o pruned_inv_fused256_su.o pruned_inv_fused256_n4_su.o pruned_inv_fused128_su.o pruned_inv_fused128_n4_su.o pruned_inv_fused128_n8_su.o pruned_inv_fused64_su.o pruned_inv_fused64_n8_su.o pruned_fwd_fused64_su.o pruned_fwd_fused128_su.o pruned_fwd_fused256_su.o \
  ${TORCH_LIBDIRS} \
  -L/usr/local/lib/python3.10/dist-packages/torch_br/lib \
  -L${SUPA_BASE}/supa/lib \
  -L${SUPA_BASE}/sufft/lib \
  -Wl,-rpath,${SUPA_BASE}/sufft/lib \
  -lc10 -ltorch -ltorch_cpu -ltorch_python -ltorch_br -lsupa-runtime -lsufft \
  -o spectral_conv_ext${EXT_SUFFIX}

echo "built spectral_conv_ext${EXT_SUFFIX} (with suFFT)"
