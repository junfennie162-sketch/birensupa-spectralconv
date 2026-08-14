[INFO] BIREN software environment variables have been set successfully!
[94mYou can use command 'brsw' to show current enabled Biren software.[0m
SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
env ready. next: build spectral_conv / run tests (serial GPU only)
[test] spectral_conv accuracy  ← 文件: spectral_conv/test_accuracy.py
built spectral_conv_ext.cpython-310-x86_64-linux-gnu.so (with suFFT)
{'torch': '2.9.0+cu128', 'supa_empty_ok': 'supa:0', 'cwd': '/workspace/ai4s-f/submission/spectral_conv'}
{'case': 'tiny_8x8', 'shape': 'B2_Cin2_Cout3_8x8', 'modes': '2x2', 'max_abs': 2.9802322387695312e-08, 'max_rel': 4.67537856728397e-08, 'threshold': 0.0001, 'ok': True}
{'case': 'small_32x32', 'shape': 'B2_Cin4_Cout4_32x32', 'modes': '8x8', 'max_abs': 5.960464477539063e-08, 'max_rel': 1.1374232639616424e-07, 'threshold': 0.0001, 'ok': True}
{'case': 'target_64x64', 'shape': 'B2_Cin4_Cout4_64x64', 'modes': '12x12', 'max_abs': 5.960464477539063e-08, 'max_rel': 1.3066267136808956e-07, 'threshold': 0.0001, 'ok': True}
{'summary': '/workspace/ai4s-f/submission/results/summary.json', 'run_log': '/workspace/ai4s-f/submission/results/run_logs/spectral_accuracy_2026-07-25.md', 'worst_rel': 1.3066267136808956e-07, 'ok': True}
{'task': 'spectral_conv_accuracy', 'ok': True}
[test] spectral_conv suFFT   ← 文件: spectral_conv/test_sufft_accuracy.py
built spectral_conv_ext.cpython-310-x86_64-linux-gnu.so (with suFFT)
{'task': 'spectral_conv_sufft_accuracy', 'device': 'supa'}
{'case': 'tiny_8x8', 'path': 'sufft_r2c_supa_mul_sufft_c2r', 'shape': 'B2_Cin2_Cout3_8x8', 'modes': '2x2', 'max_abs': 1.4901161193847656e-08, 'max_rel': 8.007249089634666e-08, 'threshold': 0.0001, 'ok': True}
{'case': 'small_32x32', 'path': 'sufft_r2c_supa_mul_sufft_c2r', 'shape': 'B2_Cin4_Cout4_32x32', 'modes': '8x8', 'max_abs': 8.195638656616211e-08, 'max_rel': 1.8928233107327004e-07, 'threshold': 0.0001, 'ok': True}
{'case': 'target_64x64', 'path': 'sufft_r2c_supa_mul_sufft_c2r', 'shape': 'B2_Cin4_Cout4_64x64', 'modes': '12x12', 'max_abs': 8.195638656616211e-08, 'max_rel': 2.1701555345060285e-07, 'threshold': 0.0001, 'ok': True}
{'worst_rel': 2.1701555345060285e-07, 'run_log': '/workspace/ai4s-f/submission/results/run_logs/spectral_sufft_accuracy_2026-07-25.md', 'ok': True}
[test] spectral_conv perf      ← 文件: spectral_conv/test_perf.py
{'torch': '2.9.0+cu128', 'device': 'supa', 'task': 'spectral_conv_perf'}

--- 性能测试（自研 SUPA Extension）---
分辨率 64x64: 前向 48.800ms, 显存 119.4MB
分辨率 128x128: 前向 49.814ms, 显存 127.4MB
分辨率 256x256: 前向 69.851ms, 显存 582.7MB
{'summary': '/workspace/ai4s-f/submission/results/summary.json', 'run_log': '/workspace/ai4s-f/submission/results/run_logs/spectral_perf_2026-07-25.md'}
{'task': 'spectral_conv_perf', 'ok': True}
[test] spectral_conv suFFT perf ← 文件: spectral_conv/test_sufft_perf.py
{'task': 'spectral_conv_sufft_perf', 'iters': 100}
[cpu_fft_supa_mul] 64x64: 49.210ms, 9.5MB
[sufft_fft_supa_mul] 64x64: 20.383ms, 41.7MB
[cpu_fft_supa_mul] 128x128: 50.089ms, 22.6MB
[sufft_fft_supa_mul] 128x128: 28.572ms, 150.0MB
[cpu_fft_supa_mul] 256x256: 227.071ms, 46.9MB
[sufft_fft_supa_mul] 256x256: 67.784ms, 558.7MB
{'run_log': '/workspace/ai4s-f/submission/results/run_logs/spectral_sufft_perf_2026-07-25.md', 'ok': True}
[test] spectral_conv backward ← 文件: spectral_conv/test_backward.py
{'task': 'spectral_mul_backward', 'threshold': 0.0001}
{'shape': 'B2_Cin2_Cout3_4x4', 'fwd_rel': 4.5491461264646205e-08, 'grad_x_rel': 5.4103335145327947e-08, 'grad_w_rel': 6.252869866330002e-08, 'ok': True}
{'shape': 'B2_Cin4_Cout4_8x8', 'fwd_rel': 4.1477115075849724e-08, 'grad_x_rel': 4.481754345420086e-08, 'grad_w_rel': 5.082164378222842e-08, 'ok': True}
{'shape': 'B2_Cin4_Cout4_12x12', 'fwd_rel': 4.445655577001162e-08, 'grad_x_rel': 5.31553325799905e-08, 'grad_w_rel': 5.5756842698428954e-08, 'ok': True}
{'worst_grad_rel': 6.252869866330002e-08, 'ok': True}
{'summary': '/workspace/ai4s-f/submission/results/summary.json', 'run_log': '/workspace/ai4s-f/submission/results/run_logs/spectral_backward_2026-07-25.md', 'ok': True}
[test] spectral_conv 3d      ← 文件: spectral_conv/test_3d_accuracy.py
{'task': 'spectral_conv3d_accuracy', 'threshold': 0.0001}
{'case': 'tiny_8', 'shape': 'B2_Cin2_Cout3_8x8x8', 'modes': '2x2x2', 'max_abs': 2.9802322387695312e-08, 'max_rel': 8.784265048689122e-08, 'threshold': 0.0001, 'ok': True}
{'case': 'small_16', 'shape': 'B2_Cin4_Cout4_16x16x16', 'modes': '4x4x4', 'max_abs': 8.940696716308594e-08, 'max_rel': 1.0731816502129732e-07, 'threshold': 0.0001, 'ok': True}
{'worst_rel': 1.0731816502129732e-07, 'ok': True}
{'summary': '/workspace/ai4s-f/submission/results/summary.json', 'run_log': '/workspace/ai4s-f/submission/results/run_logs/spectral_3d_accuracy_2026-07-25.md', 'ok': True}
[test] fno_ns forward+viz      ← 文件: fno_ns/test_forward.py + visualize.py
{'task': 'fno_ns_checkpoint_evaluation', 'mode': 'evaluate_checkpoint', 'data_source': 'generated_ns_like_v2', 'data_shape': [1024, 30, 64, 64], 'parameters': 2106145, 'history_epochs': 110}
{'rel_l2_torch': 0.00951623497530818, 'rel_l2_supa': 0.00951623567380011, 'torch_supa_l2_delta': 6.984919309616089e-10, 'ok': True}
{'figure': '/workspace/ai4s-f/submission/results/figures/fno_ns_pred_vs_gt_2026-07-25.png', 'demo_copy': '/workspace/ai4s-f/submission/demo/media/fno_ns_pred_vs_gt_2026-07-25.png', 'shared_vmin': -1.3666377067565918, 'shared_vmax': 1.3666377067565918, 'sample_relative_l2': 0.006609991192817688, 'data': 'generated_ns_like_v2'}
[test] fno_ns chain consistency ← 文件: fno_ns/test_chain_cpu_supa_consistency.py
{
  "status": "pass",
  "threshold": 0.0001,
  "random_model": {
    "relative_error": 6.580167246283963e-05,
    "threshold": 0.0001,
    "finite": true,
    "ok": true,
    "input_shape": [
      4,
      10,
      64,
      64
    ]
  },
  "checkpoint_model": {
    "relative_error": 4.595319114741869e-05,
    "threshold": 0.0001,
    "finite": true,
    "ok": true,
    "input_shape": [
      4,
      10,
      64,
      64
    ]
  },
  "fallback": "SUPA-resident fused input round-trips through CPU before suFFT",
  "measured_at": "2026-07-25T12:19:01Z"
}
[test] fno_ns batch16 perf     ← 文件: fno_ns/benchmark_fno_batch16.py
{
  "status": "measured",
  "measured_at": "2026-07-25T12:19:17Z",
  "device": "Biren106B / supa",
  "data": "generated_ns_like_v2",
  "data_disclosure": "self-generated NS-like v2; not public NS64",
  "config": {
    "batch_size": 16,
    "height": 64,
    "width": 64,
    "warmup": 10,
    "iters": 50,
    "seed": 20260722,
    "dtype": "float32"
  },
  "chain_consistency": {
    "relative_error": 4.60252704215236e-05,
    "threshold": 0.0001,
    "finite": true,
    "ok": true,
    "input_shape": [
      16,
      10,
      64,
      64
    ]
  },
  "pure_forward": {
    "grid_points_per_second": 1173260.7463697502,
    "samples_per_second": 286.4406119066773,
    "milliseconds_per_sample": 3.491125065484084,
    "forward_milliseconds_per_batch": 55.85800104774535,
    "peak_memory_MB": 177.58447265625
  },
  "with_dataloader": {
    "grid_points_per_second": 1140295.991205603,
    "samples_per_second": 278.3925759779304,
    "milliseconds_per_sample": 3.5920498112682253,
    "forward_milliseconds_per_batch": 57.472796980291605,
    "peak_memory_MB": 179.58447265625
  }
}
[test] OK mode=all
results: /workspace/ai4s-f/submission/results/summary.json  |  /workspace/ai4s-f/submission/results.md
