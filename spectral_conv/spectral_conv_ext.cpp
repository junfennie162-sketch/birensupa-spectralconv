#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <torch/extension.h>
#include <supa.h>
#include <sufft.h>

extern "C" suError_t launch_spectral_mul(const float *d_x_freq, const float *d_weights,
                                         float *d_y_freq, int batch_size, int channels_in,
                                         int channels_out, int modes1, int modes2,
                                         suStream_t stream);
extern "C" suError_t launch_spectral_mul_scatter(
    const float *d_x_freq, const float *d_weights, float *d_out_freq, int batch_size,
    int channels_in, int channels_out, int modes1, int modes2, int height,
    int width_freq, int corner, suStream_t stream);
extern "C" suError_t launch_spectral_mul_gather_scatter(
    const float *d_x_full, const float *d_weights, float *d_out_freq, int batch_size,
    int channels_in, int channels_out, int modes1, int modes2, int height,
    int width_freq, int corner, suStream_t stream);
extern "C" suError_t launch_spectral_mul_gather_scatter_dual(
    const float *d_x_full, const float *d_w1, const float *d_w2, float *d_out_freq,
    int batch_size, int channels_in, int channels_out, int modes1, int modes2,
    int height, int width_freq, suStream_t stream);
extern "C" suError_t launch_pruned_ifft_h(const float *in_freq, float *row_freq,
                                          int batch_size, int channels, int height,
                                          int modes1, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_irfft_w(const float *row_freq, float *spatial,
                                           int batch_size, int channels, int height,
                                           int width, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w(const float *x, float *row_freq,
                                          int batch_size, int channels, int height,
                                          int width, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_fft_h_corners(const float *row_freq, float *packed,
                                                 int batch_size, int channels, int height,
                                                 int modes1, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_geo(const float *x, float *row_freq,
                                              int batch_size, int channels, int height,
                                              int width, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_fact16x8_w128(const float *x, float *row_freq,
                                                        int batch_size, int channels, int height,
                                                        suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_fact16x16_w256(const float *x, float *row_freq,
                                                         int batch_size, int channels, int height,
                                                         suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_fact16x16_w256_n16(const float *x, float *row_freq,
                                                         int batch_size, int channels, int height,
                                                         suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_fact16x4_w64(const float *x, float *row_freq,
                                                       int batch_size, int channels, int height,
                                                       suStream_t stream);
extern "C" suError_t launch_pruned_rfft2_fused_mixed64(const float *x, float *packed,
                                                      int batch_size, int channels,
                                                      suStream_t stream);
extern "C" suError_t launch_pruned_rfft2_fused_mixed128(const float *x, float *packed,
                                                       int batch_size, int channels,
                                                       suStream_t stream);
extern "C" suError_t launch_pruned_rfft2_fused_mixed256(const float *x, float *packed,
                                                       int batch_size, int channels,
                                                       suStream_t stream);
extern "C" suError_t launch_pruned_fft_h_fact16x4_h64(const float *row_freq, float *packed,
                                                      int batch_size, int channels,
                                                      suStream_t stream);
extern "C" suError_t launch_pruned_fft_h_fact16x8_h128(const float *row_freq, float *packed,
                                                       int batch_size, int channels,
                                                       suStream_t stream);
extern "C" suError_t launch_pruned_fft_h_fact16x16_h256(const float *row_freq, float *packed,
                                                        int batch_size, int channels,
                                                        suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_coop(const float *x, float *row_freq,
                                               int batch_size, int channels, int height,
                                               int width, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_rfft_w_pack32(const float *x, float *row_freq,
                                                 int batch_size, int channels, int height,
                                                 int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_fft_h_geo(const float *row_freq, float *packed,
                                             int batch_size, int channels, int height,
                                             int modes1, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_fft_h_coop(const float *row_freq, float *packed,
                                              int batch_size, int channels, int height,
                                              int modes1, int modes2, suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused(const float *in_freq, float *spatial,
                                                int batch_size, int channels, int height,
                                                int width, int modes1, int modes2,
                                                suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed256(const float *in_freq, float *spatial,
                                                         int batch_size, int channels,
                                                         suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed256_n4(const float *in_freq, float *spatial,
                                                            int batch_size, int channels,
                                                            suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed128(const float *in_freq, float *spatial,
                                                         int batch_size, int channels,
                                                         suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed128_n4(const float *in_freq, float *spatial,
                                                            int batch_size, int channels,
                                                            suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed128_n8(const float *in_freq, float *spatial,
                                                            int batch_size, int channels,
                                                            suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed64(const float *in_freq, float *spatial,
                                                        int batch_size, int channels,
                                                        suStream_t stream);
extern "C" suError_t launch_pruned_irfft2_fused_mixed64_n8(const float *in_freq, float *spatial,
                                                          int batch_size, int channels,
                                                          suStream_t stream);

static bool env_flag(const char *key, bool default_on) {
    const char *value = std::getenv(key);
    if (value == nullptr || value[0] == '\0') {
        return default_on;
    }
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' || value[0] == 'n' ||
             value[0] == 'N');
}

static void check_status(suError_t status, const char *what) {
    TORCH_CHECK(status == suSuccess, what, " failed with SUPA status ",
                static_cast<int>(status));
}

static void check_sufft(sufftStatus_t status, const char *what) {
    TORCH_CHECK(status == SUFFT_STATUS_SUCCESS, what, " failed with suFFT status ",
                static_cast<int>(status));
}

static void check_freq_tensor(const torch::Tensor &tensor, const char *name, int64_t expected_last) {
    TORCH_CHECK(tensor.dim() == 5, name, " must have shape [..., 2] interleaved complex");
    TORCH_CHECK(tensor.size(-1) == expected_last, name, " last dim must be 2 (real,imag)");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

struct PlanKey {
    int nx;
    int batch;
    int type;

    bool operator==(const PlanKey &other) const {
        return nx == other.nx && batch == other.batch && type == other.type;
    }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey &key) const {
        return (static_cast<std::size_t>(key.nx) * 1315423911u) ^
               (static_cast<std::size_t>(key.batch) * 2654435761u) ^
               static_cast<std::size_t>(key.type);
    }
};

static std::mutex g_plan_mutex;
static std::unordered_map<PlanKey, sufftHandle_t, PlanKeyHash> g_plan_cache;
// Shared workspace per plan key (rfft/irfft plans all use the same scratch).
// We size each plan's workspace via `sufftGetSize1d` and pin it with
// `sufftSetWorkArea` at create time so BIREN FFT kernels avoid the implicit
// auto-alloc path on every launch. Allocations are one-per-shape and held
// for the life of the process; cumulative size is bounded by the plan-cache
// size (currently ~6 distinct shapes per official §3.2 sweep).
struct WorkspaceKey {
    int nx;
    int batch;
    int type;
    bool operator==(const WorkspaceKey &other) const {
        return nx == other.nx && batch == other.batch && type == other.type;
    }
};
struct WorkspaceKeyHash {
    std::size_t operator()(const WorkspaceKey &key) const {
        return (static_cast<std::size_t>(key.nx) * 1315423911u) ^
               (static_cast<std::size_t>(key.batch) * 2654435761u) ^
               static_cast<std::size_t>(key.type);
    }
};
static std::mutex g_workspace_mutex;
static std::unordered_map<WorkspaceKey, torch::Tensor, WorkspaceKeyHash> g_workspace_cache;

// R9: reuse suFFT staging tensors across calls. Biren allocator churn on
// permute+.contiguous() / empty_like dominates wall time once plans are warm.
// Keys are shape tuples; buffers are overwritten next call (same contract as
// Python _OUT_FREQ_CACHE). Cap keeps peak memory bounded under shape sweeps.
struct StageKey {
    int64_t d0, d1, d2, d3;
    int kind;
    bool operator==(const StageKey &o) const {
        return d0 == o.d0 && d1 == o.d1 && d2 == o.d2 && d3 == o.d3 && kind == o.kind;
    }
};
struct StageKeyHash {
    std::size_t operator()(const StageKey &k) const {
        std::size_t h = static_cast<std::size_t>(k.d0);
        h = h * 1315423911u ^ static_cast<std::size_t>(k.d1);
        h = h * 2654435761u ^ static_cast<std::size_t>(k.d2);
        h = h * 97531u ^ static_cast<std::size_t>(k.d3);
        return h ^ (static_cast<std::size_t>(k.kind) * 17u);
    }
};
static std::mutex g_stage_mutex;
static std::unordered_map<StageKey, torch::Tensor, StageKeyHash> g_stage_cache;
static constexpr std::size_t kStageCacheCap = 48;
static int g_stage_pool = 0;

static void set_stage_pool(int64_t pool) {
    g_stage_pool = static_cast<int>(pool);
}

static StageKey with_pool(StageKey key) {
    key.kind += g_stage_pool * 64;
    return key;
}

enum StageKind : int {
    kStageRowFreq = 1,
    kStageColIn = 2,
    kStageColOut = 3,
    kStageRowFreqInv = 4,
    kStageSpatialOut = 5,
    kStageScatterTop = 6,
    kStageScatterBot = 7,
    kStageTruncColIn = 8,
    kStageTruncColOut = 9,
    kStageTruncPad = 10,
    kStageTruncOut = 11,
    kStageMulOut = 12,
};

static torch::Tensor get_stage_buffer(const StageKey &key,
                                      at::IntArrayRef sizes,
                                      const torch::TensorOptions &opts) {
    const StageKey pooled = with_pool(key);
    std::lock_guard<std::mutex> lock(g_stage_mutex);
    auto found = g_stage_cache.find(pooled);
    if (found != g_stage_cache.end()) {
        return found->second;
    }
    if (g_stage_cache.size() >= kStageCacheCap) {
        g_stage_cache.erase(g_stage_cache.begin());
    }
    auto buf = torch::empty(sizes, opts);
    g_stage_cache.emplace(pooled, buf);
    return buf;
}

static torch::Tensor get_zero_stage_buffer(const StageKey &key,
                                           at::IntArrayRef sizes,
                                           const torch::TensorOptions &opts) {
    const StageKey pooled = with_pool(key);
    std::lock_guard<std::mutex> lock(g_stage_mutex);
    auto found = g_stage_cache.find(pooled);
    if (found != g_stage_cache.end()) {
        return found->second;
    }
    if (g_stage_cache.size() >= kStageCacheCap) {
        g_stage_cache.erase(g_stage_cache.begin());
    }
    auto buf = torch::zeros(sizes, opts);
    g_stage_cache.emplace(pooled, buf);
    return buf;
}

static torch::Tensor get_workspace(int nx, int batch, sufftType_t type) {
    const WorkspaceKey key{nx, batch, static_cast<int>(type)};
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    auto found = g_workspace_cache.find(key);
    if (found != g_workspace_cache.end()) {
        return found->second;
    }
    sufftHandle_t tmp = 0;
    size_t work_size = 0;
    check_sufft(sufftCreatePlan(&tmp), "sufftCreatePlan(workspace)");
    check_sufft(sufftBuildPlan1d(tmp, nx, batch, type, &work_size),
                "sufftBuildPlan1d(workspace)");
    auto ws = torch::empty(
        {static_cast<int64_t>(std::max<size_t>(work_size, 1))},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::Device("supa")));
    g_workspace_cache.emplace(key, ws);
    sufftDestroy(tmp);
    return ws;
}

static sufftHandle_t get_cached_plan(int nx, int batch, sufftType_t type) {
    const PlanKey key{nx, batch, static_cast<int>(type)};
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto found = g_plan_cache.find(key);
    if (found != g_plan_cache.end()) {
        return found->second;
    }
    sufftHandle_t plan = 0;
    size_t work_size = 0;
    check_sufft(sufftCreatePlan(&plan), "sufftCreatePlan");
    check_sufft(sufftBuildPlan1d(plan, nx, batch, type, &work_size), "sufftBuildPlan1d");
    // Wire up workspace (new in B2 / 2026-07-24): PIN a pre-allocated scratch
    // buffer so the kernel skips its internal alloc. We pin a worker-scope
    // tensor of size returned by sufftGetSize1d, cached per (nx, batch, type).
    try {
        auto ws = get_workspace(nx, batch, type);
        check_sufft(sufftSetWorkArea(plan, ws.data_ptr()),
                    "sufftSetWorkArea");
    } catch (const std::exception &) {
        // Non-fatal: if external workspace fails to bind, the kernel will
        // fall back to internal alloc (no correctness change).
    }
    g_plan_cache.emplace(key, plan);
    return plan;
}

torch::Tensor spectral_mul(torch::Tensor x_freq, torch::Tensor weights) {
    check_freq_tensor(x_freq, "x_freq", 2);
    check_freq_tensor(weights, "weights", 2);
    TORCH_CHECK(x_freq.device() == weights.device(), "x_freq and weights must share device");

    const int64_t batch_size = x_freq.size(0);
    const int64_t channels_in = x_freq.size(1);
    const int64_t modes1 = x_freq.size(2);
    const int64_t modes2 = x_freq.size(3);
    const int64_t channels_out = weights.size(1);

    TORCH_CHECK(weights.size(0) == channels_in, "weights C_in mismatch");
    TORCH_CHECK(weights.size(2) == modes1 && weights.size(3) == modes2, "modes mismatch");
    TORCH_CHECK(batch_size > 0 && channels_in > 0 && channels_out > 0 && modes1 > 0 &&
                    modes2 > 0,
                "sizes must be positive");

    auto y_freq = torch::empty({batch_size, channels_out, modes1, modes2, 2}, x_freq.options());
    check_status(launch_spectral_mul(static_cast<const float *>(x_freq.data_ptr()),
                                     static_cast<const float *>(weights.data_ptr()),
                                     static_cast<float *>(y_freq.data_ptr()),
                                     static_cast<int>(batch_size),
                                     static_cast<int>(channels_in),
                                     static_cast<int>(channels_out),
                                     static_cast<int>(modes1), static_cast<int>(modes2),
                                     nullptr),
                 "launch_spectral_mul");
    return y_freq;
}

// In-place variant: caller supplies `y_freq` to avoid the per-call
// allocator churn that creeps back into the fused path. Shape / dtype
// contracts match `spectral_mul`; the buffer must hold
// `(batch_size, channels_out, modes1, modes2, 2)` float32 interleaved.
//
// Stream handling: the underlying `.su` kernel takes an `suStream_t`
// argument; BIREN's SUPAStream header transitively pulls in `sutlass.h`
// which isn't on the include path, so this pybind leaves the stream arg
// at the default. The optional `synchronize` flag triggers a device-wide
// `torch_br.supa.synchronize()` from the Python wrapper, which keeps the
// common case (no sync) allocator-cheap.
void spectral_mul_out(torch::Tensor x_freq, torch::Tensor weights,
                      torch::Tensor y_freq) {
    check_freq_tensor(x_freq, "x_freq", 2);
    check_freq_tensor(weights, "weights", 2);
    check_freq_tensor(y_freq, "y_freq", 2);
    TORCH_CHECK(x_freq.device() == weights.device(), "x_freq and weights must share device");
    TORCH_CHECK(x_freq.device() == y_freq.device(), "y_freq must share device with x_freq");

    const int64_t batch_size = x_freq.size(0);
    const int64_t channels_in = x_freq.size(1);
    const int64_t modes1 = x_freq.size(2);
    const int64_t modes2 = x_freq.size(3);
    const int64_t channels_out = weights.size(1);

    TORCH_CHECK(y_freq.size(0) == batch_size, "y_freq batch mismatch");
    TORCH_CHECK(y_freq.size(1) == channels_out, "y_freq C_out mismatch");
    TORCH_CHECK(y_freq.size(2) == modes1 && y_freq.size(3) == modes2, "y_freq modes mismatch");
    TORCH_CHECK(weights.size(0) == channels_in, "weights C_in mismatch");
    TORCH_CHECK(weights.size(2) == modes1 && weights.size(3) == modes2, "weights modes mismatch");

    check_status(launch_spectral_mul(static_cast<const float *>(x_freq.data_ptr()),
                                     static_cast<const float *>(weights.data_ptr()),
                                     static_cast<float *>(y_freq.data_ptr()),
                                     static_cast<int>(batch_size),
                                     static_cast<int>(channels_in),
                                     static_cast<int>(channels_out),
                                     static_cast<int>(modes1), static_cast<int>(modes2),
                                     nullptr),
                 "launch_spectral_mul(out)");
}

// Dual-corner fused dispatch (R5): launches two `spectral_mul_out` kernels
// in a single pybind boundary, sharing the same input-shape contract. Saves
// ~0.04 ms / fused call vs two separate pybind dispatches (≈ 0.16 ms / chain
// at L=4). Both kernels are launched on the default SUPA stream so the
// existing fused-path ordering is preserved.
void spectral_mul_dual_out(torch::Tensor x1, torch::Tensor w1,
                           torch::Tensor x2, torch::Tensor w2,
                           torch::Tensor y1, torch::Tensor y2) {
    auto check_one = [](const torch::Tensor &xn, const torch::Tensor &wn,
                        const torch::Tensor &yn, const char *prefix) {
        check_freq_tensor(xn, prefix, 2);
        check_freq_tensor(wn, "weights", 2);
        check_freq_tensor(yn, "y_freq", 2);
        TORCH_CHECK(xn.device() == wn.device(), "x and weights must share device");
        TORCH_CHECK(xn.device() == yn.device(), "y must share device with x");
        TORCH_CHECK(xn.size(0) == yn.size(0), "y batch mismatch");
        TORCH_CHECK(xn.size(1) == wn.size(0), "weights C_in mismatch");
        TORCH_CHECK(yn.size(1) == wn.size(1), "y C_out mismatch");
        TORCH_CHECK(xn.size(2) == wn.size(2) && xn.size(2) == yn.size(2),
                    "modes1 mismatch");
        TORCH_CHECK(xn.size(3) == wn.size(3) && xn.size(3) == yn.size(3),
                    "modes2 mismatch");
    };
    check_one(x1, w1, y1, "x_freq1");
    check_one(x2, w2, y2, "x_freq2");

    const int64_t batch_size = x1.size(0);
    const int64_t channels_in = x1.size(1);
    const int64_t modes1 = x1.size(2);
    const int64_t modes2 = x1.size(3);
    const int64_t channels_out = w1.size(1);

    check_status(launch_spectral_mul(static_cast<const float *>(x1.data_ptr()),
                                     static_cast<const float *>(w1.data_ptr()),
                                     static_cast<float *>(y1.data_ptr()),
                                     static_cast<int>(batch_size),
                                     static_cast<int>(channels_in),
                                     static_cast<int>(channels_out),
                                     static_cast<int>(modes1), static_cast<int>(modes2),
                                     nullptr),
                 "launch_spectral_mul(dual/y1)");
    check_status(launch_spectral_mul(static_cast<const float *>(x2.data_ptr()),
                                     static_cast<const float *>(w2.data_ptr()),
                                     static_cast<float *>(y2.data_ptr()),
                                     static_cast<int>(batch_size),
                                     static_cast<int>(channels_in),
                                     static_cast<int>(channels_out),
                                     static_cast<int>(modes1), static_cast<int>(modes2),
                                     nullptr),
                 "launch_spectral_mul(dual/y2)");
}

// Dual-corner + scatter fused dispatch (R5-4): writes y1/y2 directly into
// the strided `out_freq` positions (`[:modes1]` and `[-modes1:]`). Drops the
// need for a separate `_y_freq_buffer` cache and the two Python-side slice
// assignments. Expect ~0.3 ms saved per fused call vs the dual_out path,
// on top of R5-3's pybind saving.
//
// `out_freq` shape: [B, Cout, H, Wf, 2]; the top `modes1` rows and the bottom
// `modes1` rows of the modes2 axis are written; the middle is left untouched
// (caller is responsible for it being zeroed, which is the existing fused-path
// invariant).
void spectral_mul_dual_scatter_out(
    torch::Tensor x1, torch::Tensor w1,
    torch::Tensor x2, torch::Tensor w2,
    torch::Tensor out_freq) {
    check_freq_tensor(x1, "x_freq1", 2);
    check_freq_tensor(w1, "weights1", 2);
    check_freq_tensor(x2, "x_freq2", 2);
    check_freq_tensor(w2, "weights2", 2);
    check_freq_tensor(out_freq, "out_freq", 2);
    TORCH_CHECK(x1.device() == out_freq.device(), "x1 and out_freq must share device");
    TORCH_CHECK(w1.device() == x1.device(), "x1 and w1 must share device");
    TORCH_CHECK(x2.device() == x1.device(), "x1 and x2 must share device");
    TORCH_CHECK(w2.device() == x1.device(), "x1 and w2 must share device");
    const int64_t B = x1.size(0);
    const int64_t Cin = x1.size(1);
    const int64_t M1 = x1.size(2);
    const int64_t M2 = x1.size(3);
    const int64_t H = out_freq.size(2);
    const int64_t Wf = out_freq.size(3);
    TORCH_CHECK(out_freq.size(0) == B, "out_freq batch mismatch");
    TORCH_CHECK(M1 <= H, "modes1 must fit within height");
    TORCH_CHECK(M2 <= Wf, "modes2 must fit within width_freq");
    TORCH_CHECK(out_freq.is_contiguous(), "out_freq must be contiguous");

    const int64_t Cout = w1.size(1);
    const int64_t Cout2 = w2.size(1);
    TORCH_CHECK(Cout == Cout2, "weights C_out mismatch");

    // R10: true scatter-write — no temp y buffers / no strided copy_.
    // zero_ still required so non-mode bands stay 0 for irfft.
    out_freq.zero_();

    check_status(launch_spectral_mul_scatter(
                     static_cast<const float *>(x1.data_ptr()),
                     static_cast<const float *>(w1.data_ptr()),
                     static_cast<float *>(out_freq.data_ptr()),
                     static_cast<int>(B), static_cast<int>(Cin),
                     static_cast<int>(Cout), static_cast<int>(M1),
                     static_cast<int>(M2), static_cast<int>(H),
                     static_cast<int>(Wf), /*corner=*/0, nullptr),
                 "launch_spectral_mul_scatter(top)");
    check_status(launch_spectral_mul_scatter(
                     static_cast<const float *>(x2.data_ptr()),
                     static_cast<const float *>(w2.data_ptr()),
                     static_cast<float *>(out_freq.data_ptr()),
                     static_cast<int>(B), static_cast<int>(Cin),
                     static_cast<int>(Cout2), static_cast<int>(M1),
                     static_cast<int>(M2), static_cast<int>(H),
                     static_cast<int>(Wf), /*corner=*/1, nullptr),
                 "launch_spectral_mul_scatter(bot)");
}

// R11: full-spectrum gather + corner scatter. `x_freq` is [B, Cin, H, Wf, 2].
void spectral_mul_dual_full_scatter_out(
    torch::Tensor x_freq, torch::Tensor w1, torch::Tensor w2,
    torch::Tensor out_freq, int64_t modes1_i64, int64_t modes2_i64,
    bool zero_out) {
    check_freq_tensor(x_freq, "x_freq", 2);
    check_freq_tensor(w1, "weights1", 2);
    check_freq_tensor(w2, "weights2", 2);
    check_freq_tensor(out_freq, "out_freq", 2);
    TORCH_CHECK(x_freq.device() == out_freq.device(), "x_freq/out_freq device");
    TORCH_CHECK(w1.device() == x_freq.device(), "w1 device");
    TORCH_CHECK(w2.device() == x_freq.device(), "w2 device");
    TORCH_CHECK(x_freq.is_contiguous(), "x_freq must be contiguous");
    TORCH_CHECK(out_freq.is_contiguous(), "out_freq must be contiguous");

    const int64_t B = x_freq.size(0);
    const int64_t Cin = x_freq.size(1);
    const int64_t H = x_freq.size(2);
    const int64_t Wf = x_freq.size(3);
    const int64_t M1 = modes1_i64;
    const int64_t M2 = modes2_i64;
    TORCH_CHECK(M1 > 0 && M2 > 0 && M1 <= H && M2 <= Wf, "modes out of range");
    TORCH_CHECK(out_freq.size(0) == B && out_freq.size(2) == H && out_freq.size(3) == Wf,
                "out_freq shape mismatch");
    TORCH_CHECK(w1.size(0) == Cin && w1.size(2) == M1 && w1.size(3) == M2, "w1 shape");
    TORCH_CHECK(w2.size(0) == Cin && w2.size(2) == M1 && w2.size(3) == M2, "w2 shape");
    const int64_t Cout = w1.size(1);
    TORCH_CHECK(w2.size(1) == Cout && out_freq.size(1) == Cout, "C_out mismatch");

    if (zero_out) {
        out_freq.zero_();
    }

    check_status(launch_spectral_mul_gather_scatter_dual(
                     static_cast<const float *>(x_freq.data_ptr()),
                     static_cast<const float *>(w1.data_ptr()),
                     static_cast<const float *>(w2.data_ptr()),
                     static_cast<float *>(out_freq.data_ptr()),
                     static_cast<int>(B), static_cast<int>(Cin),
                     static_cast<int>(Cout), static_cast<int>(M1),
                     static_cast<int>(M2), static_cast<int>(H),
                     static_cast<int>(Wf), nullptr),
                 "launch_spectral_mul_gather_scatter_dual");
}

// SDK exports only BuildPlan1d. 2D R2C = batched 1D R2C(W) + batched 1D C2C(H).
torch::Tensor rfft2_sufft(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 4, "rfft2_sufft expects [B,C,H,W]");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "rfft2_sufft expects float32");
    auto x_contig = x.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(x_contig.size(2));
    const int width = static_cast<int>(x_contig.size(3));
    const int width_freq = width / 2 + 1;
    const int planes = batch_size * channels;
    const int row_batch = planes * height;
    const int col_batch = planes * width_freq;

    auto flat_in = x_contig.view({planes, height, width});
    auto row_freq = torch::empty({planes, height, width_freq, 2}, x_contig.options());

    sufftHandle_t plan_r2c = get_cached_plan(width, row_batch, SUFFT_TYPE_R2C);
    check_sufft(sufftExecR2C(plan_r2c, flat_in.data_ptr<float>(),
                             reinterpret_cast<suFloatComplex *>(row_freq.data_ptr<float>())),
                "sufftExecR2C batched");

    // Prefer permute().contiguous() over custom transpose / copy_(strided):
    // on Biren, contig packing uses Memcpy2D and beats our R12 transpose kernel.
    auto col_in = row_freq.permute({0, 2, 1, 3}).contiguous();
    auto col_out = torch::empty_like(col_in);
    sufftHandle_t plan_c2c = get_cached_plan(height, col_batch, SUFFT_TYPE_C2C);
    check_sufft(sufftExecC2C(plan_c2c,
                             reinterpret_cast<suFloatComplex *>(col_in.data_ptr<float>()),
                             reinterpret_cast<suFloatComplex *>(col_out.data_ptr<float>()),
                             SUFFT_DIRECTION_FORWARD),
                "sufftExecC2C fwd batched");

    return col_out.permute({0, 2, 1, 3})
        .contiguous()
        .view({batch_size, channels, height, width_freq, 2});
}

// P3/P4: column-FFT only first `modes2` bins — narrow BEFORE permute.
// P4: returns packed [B,C,H,modes2,2] (no full-Wf zero/copy pad).
torch::Tensor rfft2_sufft_trunc(torch::Tensor x, int64_t modes2_i64) {
    TORCH_CHECK(x.dim() == 4, "rfft2_sufft_trunc expects [B,C,H,W]");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "rfft2_sufft_trunc expects float32");
    auto x_contig = x.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(x_contig.size(2));
    const int width = static_cast<int>(x_contig.size(3));
    const int width_freq = width / 2 + 1;
    const int modes2 = static_cast<int>(modes2_i64);
    TORCH_CHECK(modes2 > 0 && modes2 <= width_freq, "modes2 out of range");
    const int planes = batch_size * channels;
    const int row_batch = planes * height;
    const int col_batch = planes * modes2;
    const auto opts = x_contig.options();

    auto flat_in = x_contig.view({planes, height, width});
    auto row_freq = get_stage_buffer(
        StageKey{planes, height, width_freq, 2, kStageRowFreq},
        {planes, height, width_freq, 2}, opts);

    sufftHandle_t plan_r2c = get_cached_plan(width, row_batch, SUFFT_TYPE_R2C);
    check_sufft(sufftExecR2C(plan_r2c, flat_in.data_ptr<float>(),
                             reinterpret_cast<suFloatComplex *>(row_freq.data_ptr<float>())),
                "sufftExecR2C batched (trunc)");

    // Narrow first so permute+.contiguous only packs modes2 columns (P2→P3).
    auto col_in = row_freq.narrow(/*dim=*/2, /*start=*/0, /*length=*/modes2)
                      .permute({0, 2, 1, 3})
                      .contiguous();
    auto col_out = get_stage_buffer(
        StageKey{planes, modes2, height, 2, kStageTruncColOut},
        {planes, modes2, height, 2}, opts);
    sufftHandle_t plan_c2c = get_cached_plan(height, col_batch, SUFFT_TYPE_C2C);
    check_sufft(sufftExecC2C(plan_c2c,
                             reinterpret_cast<suFloatComplex *>(col_in.data_ptr<float>()),
                             reinterpret_cast<suFloatComplex *>(col_out.data_ptr<float>()),
                             SUFFT_DIRECTION_FORWARD),
                "sufftExecC2C fwd trunc");

    // P6: pack into stage buffer (view to match permute layout before copy_).
    auto packed = get_stage_buffer(
        StageKey{batch_size, channels, height, modes2, kStageTruncOut},
        {batch_size, channels, height, modes2, 2}, opts);
    packed.view({planes, height, modes2, 2})
        .copy_(col_out.permute({0, 2, 1, 3}).contiguous());
    return packed;
}

// P3/P4: inv column-FFT on first modes2; C2R still needs full Wf pad.
// Accepts packed [B,C,H,modes2,2] (P4) or full [B,C,H,Wf,2] (legacy).
// Do not stage-cache the C2R spatial buffer (multi-call poison on this SDK).
// P6: C2C col_out may be stage-cached (fully overwritten).
torch::Tensor irfft2_sufft_trunc(torch::Tensor x_freq, int64_t height_i64, int64_t width_i64,
                                 int64_t modes2_i64) {
    check_freq_tensor(x_freq, "x_freq", 2);
    auto x_contig = x_freq.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(height_i64);
    const int width = static_cast<int>(width_i64);
    const int packed_w = static_cast<int>(x_contig.size(3));
    const int width_freq = width / 2 + 1;
    const int modes2 = static_cast<int>(modes2_i64);
    TORCH_CHECK(x_contig.size(2) == height, "height mismatch");
    TORCH_CHECK(modes2 > 0 && modes2 <= width_freq, "modes2 out of range");
    TORCH_CHECK(packed_w == modes2 || packed_w == width_freq,
                "x_freq width must be modes2 (packed) or width/2+1 (full)");
    TORCH_CHECK(modes2 <= packed_w, "modes2 exceeds x_freq width");
    const int planes = batch_size * channels;
    const int row_batch = planes * height;
    const int col_batch = planes * modes2;
    const float scale = 1.0f / static_cast<float>(height * width);
    const auto opts = x_contig.options();

    auto flat = x_contig.view({planes, height, packed_w, 2});
    auto col_in = flat.narrow(/*dim=*/2, /*start=*/0, /*length=*/modes2)
                      .permute({0, 2, 1, 3})
                      .contiguous();
    // P6: reuse C2C destination (fully overwritten each call). Do NOT cache
    // the C2R spatial buffer — that poisoned multi-call on this SDK.
    auto col_out = get_stage_buffer(
        StageKey{planes, modes2, height, 2, kStageTruncColIn},
        {planes, modes2, height, 2}, opts);
    sufftHandle_t plan_c2c = get_cached_plan(height, col_batch, SUFFT_TYPE_C2C);
    check_sufft(sufftExecC2C(plan_c2c,
                             reinterpret_cast<suFloatComplex *>(col_in.data_ptr<float>()),
                             reinterpret_cast<suFloatComplex *>(col_out.data_ptr<float>()),
                             SUFFT_DIRECTION_INVERSE),
                "sufftExecC2C inv trunc");

    // P8: scale on a fresh permute buffer — NEVER mul_ the stage-cached col_out
    // (in-place scale poisoned multi-call on this SDK; same class of bug as
    // caching C2R out). Packed size << spatial out.
    auto packed_cols = col_out.permute({0, 2, 1, 3}).contiguous();
    packed_cols.mul_(scale);

    // P5: reuse pad without per-call full zero_. Allocate with zeros once;
    // subsequent hits only overwrite [0, modes2). Multi-call must stay correct.
    StageKey pad_key{planes, height, width_freq, 2, kStageTruncPad};
    torch::Tensor row_freq;
    {
        std::lock_guard<std::mutex> lock(g_stage_mutex);
        auto found = g_stage_cache.find(pad_key);
        if (found != g_stage_cache.end()) {
            row_freq = found->second;
        } else {
            if (g_stage_cache.size() >= kStageCacheCap) {
                g_stage_cache.erase(g_stage_cache.begin());
            }
            row_freq = torch::zeros({planes, height, width_freq, 2}, opts);
            g_stage_cache.emplace(pad_key, row_freq);
        }
    }
    row_freq.narrow(/*dim=*/2, /*start=*/0, /*length=*/modes2).copy_(packed_cols);

    auto out = torch::empty({planes, height, width}, opts.dtype(torch::kFloat32));
    sufftHandle_t plan_c2r = get_cached_plan(width, row_batch, SUFFT_TYPE_C2R);
    check_sufft(sufftExecC2R(plan_c2r,
                             reinterpret_cast<suFloatComplex *>(row_freq.data_ptr<float>()),
                             out.data_ptr<float>()),
                "sufftExecC2R trunc");

    return out.view({batch_size, channels, height, width});
}

torch::Tensor irfft2_sufft(torch::Tensor x_freq, int64_t height_i64, int64_t width_i64) {
    check_freq_tensor(x_freq, "x_freq", 2);
    auto x_contig = x_freq.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(height_i64);
    const int width = static_cast<int>(width_i64);
    const int width_freq = static_cast<int>(x_contig.size(3));
    TORCH_CHECK(x_contig.size(2) == height, "height mismatch");
    TORCH_CHECK(width_freq == width / 2 + 1, "width_freq mismatch");
    const int planes = batch_size * channels;
    const int row_batch = planes * height;
    const int col_batch = planes * width_freq;
    const float scale = 1.0f / static_cast<float>(height * width);

    auto flat = x_contig.view({planes, height, width_freq, 2});
    auto col_in = flat.permute({0, 2, 1, 3}).contiguous();
    auto col_out = torch::empty_like(col_in);
    sufftHandle_t plan_c2c = get_cached_plan(height, col_batch, SUFFT_TYPE_C2C);
    check_sufft(sufftExecC2C(plan_c2c,
                             reinterpret_cast<suFloatComplex *>(col_in.data_ptr<float>()),
                             reinterpret_cast<suFloatComplex *>(col_out.data_ptr<float>()),
                             SUFFT_DIRECTION_INVERSE),
                "sufftExecC2C inv batched");

    // P8: scale packed/full spectrum before C2R (same math, less work than
    // scaling spatial out when Wf << W — always true for rFFT).
    col_out.mul_(scale);
    auto row_freq = col_out.permute({0, 2, 1, 3}).contiguous();
    auto out = torch::empty({planes, height, width}, x_contig.options().dtype(torch::kFloat32));
    sufftHandle_t plan_c2r = get_cached_plan(width, row_batch, SUFFT_TYPE_C2R);
    check_sufft(sufftExecC2R(plan_c2r,
                             reinterpret_cast<suFloatComplex *>(row_freq.data_ptr<float>()),
                             out.data_ptr<float>()),
                "sufftExecC2R batched");

    return out.view({batch_size, channels, height, width});
}

torch::Tensor rfft2_pruned_trunc(torch::Tensor x, int64_t modes1_i64, int64_t modes2_i64) {
    TORCH_CHECK(x.dim() == 4, "rfft2_pruned_trunc expects [B,C,H,W]");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "rfft2_pruned_trunc expects float32");
    auto x_contig = x.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(x_contig.size(2));
    const int width = static_cast<int>(x_contig.size(3));
    const int modes1 = static_cast<int>(modes1_i64);
    const int modes2 = static_cast<int>(modes2_i64);
    TORCH_CHECK(modes1 > 0 && modes1 <= height, "modes1 out of range");
    TORCH_CHECK(modes2 > 0 && modes2 <= width / 2 + 1, "modes2 out of range");
    const auto opts = x_contig.options();
    auto packed = get_stage_buffer(
        StageKey{batch_size, channels, height, modes2, kStageTruncOut},
        {batch_size, channels, height, modes2, 2}, opts);
    if (height == 64 && width == 64 && modes1 == 16 && modes2 == 16 &&
        env_flag("SPECTRAL_FUSED_FWD64", true)) {
        check_status(launch_pruned_rfft2_fused_mixed64(
                         x_contig.data_ptr<float>(), packed.data_ptr<float>(), batch_size,
                         channels, nullptr),
                     "launch_pruned_rfft2_fused_mixed64");
        return packed;
    }
    if (height == 128 && width == 128 && modes1 == 16 && modes2 == 16 &&
        env_flag("SPECTRAL_FUSED_FWD128", true)) {
        check_status(launch_pruned_rfft2_fused_mixed128(
                         x_contig.data_ptr<float>(), packed.data_ptr<float>(), batch_size,
                         channels, nullptr),
                     "launch_pruned_rfft2_fused_mixed128");
        return packed;
    }
    auto row = get_stage_buffer(
        StageKey{batch_size, channels, height, modes2, kStageTruncColOut},
        {batch_size, channels, height, modes2, 2}, opts);
    const bool coop = env_flag("SPECTRAL_COOP", false);
    if (env_flag("SPECTRAL_PACKED_FFT", false) && width == 64 && modes2 == 16) {
        check_status(launch_pruned_rfft_w_pack32(x_contig.data_ptr<float>(),
                                                 row.data_ptr<float>(), batch_size, channels,
                                                 height, modes2, nullptr),
                     "launch_pruned_rfft_w_pack32");
    } else if (coop && modes2 == 16 && (width == 64 || width == 128 || width == 256)) {
        check_status(launch_pruned_rfft_w_coop(x_contig.data_ptr<float>(), row.data_ptr<float>(),
                                               batch_size, channels, height, width, modes2,
                                               nullptr),
                     "launch_pruned_rfft_w_coop");
    } else if (modes2 == 16 && width == 256) {
        if (env_flag("SPECTRAL_RFFT256_N16", false)) {
            check_status(launch_pruned_rfft_w_fact16x16_w256_n16(
                             x_contig.data_ptr<float>(), row.data_ptr<float>(), batch_size,
                             channels, height, nullptr),
                         "launch_pruned_rfft_w_fact16x16_w256_n16");
        } else {
            check_status(launch_pruned_rfft_w_fact16x16_w256(
                             x_contig.data_ptr<float>(), row.data_ptr<float>(), batch_size,
                             channels, height, nullptr),
                         "launch_pruned_rfft_w_fact16x16_w256");
        }
    } else if (modes2 == 16 && width == 128) {
        check_status(launch_pruned_rfft_w_fact16x8_w128(
                         x_contig.data_ptr<float>(), row.data_ptr<float>(), batch_size,
                         channels, height, nullptr),
                     "launch_pruned_rfft_w_fact16x8_w128");
    } else if (modes2 == 16 && width == 64) {
        check_status(launch_pruned_rfft_w_fact16x4_w64(
                         x_contig.data_ptr<float>(), row.data_ptr<float>(), batch_size,
                         channels, height, nullptr),
                     "launch_pruned_rfft_w_fact16x4_w64");
    } else if (modes2 == 16 && (width == 64 || width == 128 || width == 256)) {
        check_status(launch_pruned_rfft_w_geo(x_contig.data_ptr<float>(), row.data_ptr<float>(),
                                              batch_size, channels, height, width, modes2,
                                              nullptr),
                     "launch_pruned_rfft_w_geo");
    } else {
        check_status(launch_pruned_rfft_w(x_contig.data_ptr<float>(), row.data_ptr<float>(),
                                          batch_size, channels, height, width, modes2, nullptr),
                     "launch_pruned_rfft_w");
    }
    if (coop && modes1 == 16 && modes2 == 16 &&
        (height == 64 || height == 128 || height == 256)) {
        check_status(launch_pruned_fft_h_coop(row.data_ptr<float>(), packed.data_ptr<float>(),
                                              batch_size, channels, height, modes1, modes2,
                                              nullptr),
                     "launch_pruned_fft_h_coop");
    } else if (modes1 == 16 && modes2 == 16 && height == 256) {
        check_status(launch_pruned_fft_h_fact16x16_h256(row.data_ptr<float>(), packed.data_ptr<float>(),
                                                        batch_size, channels, nullptr),
                     "launch_pruned_fft_h_fact16x16_h256");
    } else if (modes1 == 16 && modes2 == 16 && height == 128) {
        check_status(launch_pruned_fft_h_fact16x8_h128(row.data_ptr<float>(), packed.data_ptr<float>(),
                                                       batch_size, channels, nullptr),
                     "launch_pruned_fft_h_fact16x8_h128");
    } else if (modes1 == 16 && modes2 == 16 && height == 64) {
        check_status(launch_pruned_fft_h_fact16x4_h64(row.data_ptr<float>(), packed.data_ptr<float>(),
                                                      batch_size, channels, nullptr),
                     "launch_pruned_fft_h_fact16x4_h64");
    } else if (modes1 == 16 && modes2 == 16 && (height == 64 || height == 128 || height == 256)) {
        check_status(launch_pruned_fft_h_geo(row.data_ptr<float>(), packed.data_ptr<float>(),
                                             batch_size, channels, height, modes1, modes2,
                                             nullptr),
                     "launch_pruned_fft_h_geo");
    } else {
        check_status(launch_pruned_fft_h_corners(row.data_ptr<float>(), packed.data_ptr<float>(),
                                                 batch_size, channels, height, modes1, modes2,
                                                 nullptr),
                     "launch_pruned_fft_h_corners");
    }
    return packed;
}

static void irfft2_pruned_packed_into(const torch::Tensor &x_contig, int height, int width,
                                      int modes1, int modes2, torch::Tensor &out);

torch::Tensor irfft2_pruned_packed(torch::Tensor x_freq, int64_t height_i64, int64_t width_i64,
                                   int64_t modes1_i64, int64_t modes2_i64) {
    check_freq_tensor(x_freq, "x_freq", 2);
    auto x_contig = x_freq.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(height_i64);
    const int width = static_cast<int>(width_i64);
    const int modes1 = static_cast<int>(modes1_i64);
    const int modes2 = static_cast<int>(modes2_i64);
    TORCH_CHECK(x_contig.size(2) == height, "height mismatch");
    TORCH_CHECK(x_contig.size(3) == modes2, "packed width must be modes2");
    TORCH_CHECK(modes1 > 0 && modes1 <= height, "modes1 out of range");
    const auto opts = x_contig.options();
    auto out = torch::empty({batch_size, channels, height, width}, opts.dtype(torch::kFloat32));
    irfft2_pruned_packed_into(x_contig, height, width, modes1, modes2, out);
    return out;
}

void irfft2_pruned_packed_out(torch::Tensor x_freq, int64_t height_i64, int64_t width_i64,
                              int64_t modes1_i64, int64_t modes2_i64, torch::Tensor out) {
    check_freq_tensor(x_freq, "x_freq", 2);
    auto x_contig = x_freq.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(height_i64);
    const int width = static_cast<int>(width_i64);
    const int modes1 = static_cast<int>(modes1_i64);
    const int modes2 = static_cast<int>(modes2_i64);
    TORCH_CHECK(x_contig.size(2) == height, "height mismatch");
    TORCH_CHECK(x_contig.size(3) == modes2, "packed width must be modes2");
    TORCH_CHECK(modes1 > 0 && modes1 <= height, "modes1 out of range");
    TORCH_CHECK(out.dim() == 4, "out expects [B,C,H,W]");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out expects float32");
    TORCH_CHECK(out.size(0) == batch_size && out.size(1) == channels &&
                    out.size(2) == height && out.size(3) == width,
                "out shape mismatch");
    auto out_c = out.contiguous();
    irfft2_pruned_packed_into(x_contig, height, width, modes1, modes2, out_c);
    if (!out.is_same(out_c)) {
        out.copy_(out_c);
    }
}

static void irfft2_pruned_packed_into(const torch::Tensor &x_contig, int height, int width,
                                      int modes1, int modes2, torch::Tensor &out) {
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const auto opts = x_contig.options();
    if (height == 256 && width == 256 && modes1 == 16 && modes2 == 16 &&
        env_flag("SPECTRAL_FUSED_MIXED256", true)) {
        if (env_flag("SPECTRAL_FUSED_MIXED256_N4", true)) {
            check_status(launch_pruned_irfft2_fused_mixed256_n4(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed256_n4");
        } else {
            check_status(launch_pruned_irfft2_fused_mixed256(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed256");
        }
        return;
    }
    if (height == 128 && width == 128 && modes1 == 16 && modes2 == 16 &&
        (env_flag("SPECTRAL_FUSED_MIXED128", false) ||
         env_flag("SPECTRAL_FUSED_MIXED128_N4", true) ||
         env_flag("SPECTRAL_FUSED_MIXED128_N8", false))) {
        if (env_flag("SPECTRAL_FUSED_MIXED128_N8", false)) {
            check_status(launch_pruned_irfft2_fused_mixed128_n8(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed128_n8");
        } else if (env_flag("SPECTRAL_FUSED_MIXED128_N4", true)) {
            check_status(launch_pruned_irfft2_fused_mixed128_n4(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed128_n4");
        } else {
            check_status(launch_pruned_irfft2_fused_mixed128(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed128");
        }
        return;
    }
    if (height == 64 && width == 64 && modes1 == 16 && modes2 == 16 &&
        env_flag("SPECTRAL_FUSED_MIXED64", true)) {
        if (env_flag("SPECTRAL_FUSED_MIXED64_N8", false)) {
            check_status(launch_pruned_irfft2_fused_mixed64_n8(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed64_n8");
        } else {
            check_status(launch_pruned_irfft2_fused_mixed64(
                             x_contig.data_ptr<float>(), out.data_ptr<float>(), batch_size,
                             channels, nullptr),
                         "launch_pruned_irfft2_fused_mixed64");
        }
        return;
    }
    const bool fused =
        (env_flag("SPECTRAL_FUSED_INV", false) ||
         (width == 64 && env_flag("SPECTRAL_FUSED_OCC64", false)) ||
         (width >= 256 && env_flag("SPECTRAL_FUSED_INV256", false))) &&
        modes1 == 16 && modes2 == 16;
    if (fused) {
        check_status(launch_pruned_irfft2_fused(x_contig.data_ptr<float>(), out.data_ptr<float>(),
                                                batch_size, channels, height, width, modes1,
                                                modes2, nullptr),
                     "launch_pruned_irfft2_fused");
        return;
    }
    auto row = get_stage_buffer(
        StageKey{batch_size, channels, height, modes2, kStageTruncColIn},
        {batch_size, channels, height, modes2, 2}, opts);
    check_status(launch_pruned_ifft_h(x_contig.data_ptr<float>(), row.data_ptr<float>(),
                                      batch_size, channels, height, modes1, modes2, nullptr),
                 "launch_pruned_ifft_h");
    check_status(launch_pruned_irfft_w(row.data_ptr<float>(), out.data_ptr<float>(),
                                       batch_size, channels, height, width, modes2, nullptr),
                 "launch_pruned_irfft_w");
}

torch::Tensor ifft_h_pruned_packed(torch::Tensor x_freq, int64_t modes1_i64, int64_t modes2_i64) {
    check_freq_tensor(x_freq, "x_freq", 2);
    auto x_contig = x_freq.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(x_contig.size(2));
    const int modes1 = static_cast<int>(modes1_i64);
    const int modes2 = static_cast<int>(modes2_i64);
    auto row = torch::empty_like(x_contig);
    check_status(launch_pruned_ifft_h(x_contig.data_ptr<float>(), row.data_ptr<float>(),
                                      batch_size, channels, height, modes1, modes2, nullptr),
                 "launch_pruned_ifft_h");
    return row;
}

torch::Tensor irfft_w_pruned(torch::Tensor row_freq, int64_t width_i64, int64_t modes2_i64) {
    check_freq_tensor(row_freq, "row_freq", 2);
    auto x_contig = row_freq.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(x_contig.size(2));
    const int width = static_cast<int>(width_i64);
    const int modes2 = static_cast<int>(modes2_i64);
    auto out = torch::empty({batch_size, channels, height, width}, x_contig.options().dtype(torch::kFloat32));
    check_status(launch_pruned_irfft_w(x_contig.data_ptr<float>(), out.data_ptr<float>(),
                                       batch_size, channels, height, width, modes2, nullptr),
                 "launch_pruned_irfft_w");
    return out;
}

void spectral_conv2d_pruned_out(torch::Tensor x, torch::Tensor w1, torch::Tensor w2,
                                int64_t modes1_i64, int64_t modes2_i64, torch::Tensor out) {
    TORCH_CHECK(x.dim() == 4, "x expects [B,C,H,W]");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x expects float32");
    check_freq_tensor(w1, "w1", 2);
    check_freq_tensor(w2, "w2", 2);
    auto x_contig = x.contiguous();
    auto w1c = w1.contiguous();
    auto w2c = w2.contiguous();
    const int batch_size = static_cast<int>(x_contig.size(0));
    const int channels_in = static_cast<int>(x_contig.size(1));
    const int height = static_cast<int>(x_contig.size(2));
    const int width = static_cast<int>(x_contig.size(3));
    const int modes1 = static_cast<int>(modes1_i64);
    const int modes2 = static_cast<int>(modes2_i64);
    const int channels_out = static_cast<int>(w1c.size(1));
    TORCH_CHECK(w1c.size(0) == channels_in && w2c.size(0) == channels_in, "C_in mismatch");
    TORCH_CHECK(w2c.size(1) == channels_out, "C_out mismatch");
    TORCH_CHECK(w1c.size(2) == modes1 && w1c.size(3) == modes2, "w1 modes");
    TORCH_CHECK(w2c.size(2) == modes1 && w2c.size(3) == modes2, "w2 modes");
    TORCH_CHECK(out.dim() == 4 && out.dtype() == torch::kFloat32, "out shape/dtype");
    TORCH_CHECK(out.size(0) == batch_size && out.size(1) == channels_out &&
                    out.size(2) == height && out.size(3) == width,
                "out shape mismatch");
    auto packed = rfft2_pruned_trunc(x_contig, modes1, modes2);
    auto out_freq = get_zero_stage_buffer(
        StageKey{batch_size, channels_out, height, modes2, kStageMulOut},
        {batch_size, channels_out, height, modes2, 2}, packed.options());
    check_status(launch_spectral_mul_gather_scatter_dual(
                     packed.data_ptr<float>(), w1c.data_ptr<float>(), w2c.data_ptr<float>(),
                     out_freq.data_ptr<float>(), batch_size, channels_in, channels_out,
                     modes1, modes2, height, modes2, nullptr),
                 "launch_spectral_mul_gather_scatter_dual");
    auto out_c = out.contiguous();
    irfft2_pruned_packed_into(out_freq, height, width, modes1, modes2, out_c);
    if (!out.is_same(out_c)) {
        out.copy_(out_c);
    }
}

PYBIND11_MODULE(spectral_conv_ext, m) {
    m.def("spectral_mul", &spectral_mul,
          "Complex spectral multiply for FNO corner modes (SUPA kernel)");
    m.def("spectral_mul_out", &spectral_mul_out,
          "In-place spectral multiply (caller-supplied y_freq buffer)",
          pybind11::arg("x_freq"), pybind11::arg("weights"),
          pybind11::arg("y_freq"));
    m.def("spectral_mul_dual_out", &spectral_mul_dual_out,
          "Dual-corner fused dispatch: writes y1 and y2 in one pybind call",
          pybind11::arg("x1"), pybind11::arg("w1"),
          pybind11::arg("x2"), pybind11::arg("w2"),
          pybind11::arg("y1"), pybind11::arg("y2"));
    m.def("spectral_mul_dual_scatter_out", &spectral_mul_dual_scatter_out,
          "Dual-corner + scatter: writes strided corners of out_freq in one "
          "pybind call (also zeroes out_freq); saves zero+scatter kernel "
          "launches from the fused path",
          pybind11::arg("x1"), pybind11::arg("w1"),
          pybind11::arg("x2"), pybind11::arg("w2"),
          pybind11::arg("out_freq"));
    m.def("spectral_mul_dual_full_scatter_out", &spectral_mul_dual_full_scatter_out,
          "R11: gather corners from full x_freq + scatter-write out_freq",
          pybind11::arg("x_freq"), pybind11::arg("w1"), pybind11::arg("w2"),
          pybind11::arg("out_freq"), pybind11::arg("modes1"),
          pybind11::arg("modes2"), pybind11::arg("zero_out") = true);
    m.def("rfft2_sufft", &rfft2_sufft,
          "2D R2C via batched suFFT 1D stages + plan cache");
    m.def("rfft2_sufft_trunc", &rfft2_sufft_trunc,
          "P4: R2C + col-FFT modes2; returns packed [B,C,H,modes2,2]",
          pybind11::arg("x"), pybind11::arg("modes2"));
    m.def("irfft2_sufft", &irfft2_sufft,
          "2D C2R via batched suFFT 1D stages + plan cache");
    m.def("irfft2_sufft_trunc", &irfft2_sufft_trunc,
          "P4: inv col-FFT on packed/full modes2 + C2R",
          pybind11::arg("x_freq"), pybind11::arg("height"),
          pybind11::arg("width"), pybind11::arg("modes2"));
    m.def("rfft2_pruned_trunc", &rfft2_pruned_trunc,
          "SUFFT-free: W-DFT modes2 + H-DFT two corners → packed [B,C,H,M2,2]",
          pybind11::arg("x"), pybind11::arg("modes1"), pybind11::arg("modes2"));
    m.def("irfft2_pruned_packed", &irfft2_pruned_packed,
          "SUFFT-free inverse from packed dual-corner spectrum",
          pybind11::arg("x_freq"), pybind11::arg("height"),
          pybind11::arg("width"), pybind11::arg("modes1"),
          pybind11::arg("modes2"));
    m.def("irfft2_pruned_packed_out", &irfft2_pruned_packed_out,
          "SUFFT-free inverse into caller-supplied spatial buffer (CPU-in cache)",
          pybind11::arg("x_freq"), pybind11::arg("height"),
          pybind11::arg("width"), pybind11::arg("modes1"),
          pybind11::arg("modes2"), pybind11::arg("out"));
    m.def("ifft_h_pruned_packed", &ifft_h_pruned_packed,
          "Pruned inverse H-FFT only (profile)",
          pybind11::arg("x_freq"), pybind11::arg("modes1"), pybind11::arg("modes2"));
    m.def("irfft_w_pruned", &irfft_w_pruned,
          "Pruned inverse W-iRFFT only (profile)",
          pybind11::arg("row_freq"), pybind11::arg("width"), pybind11::arg("modes2"));
    m.def("spectral_conv2d_pruned_out", &spectral_conv2d_pruned_out,
          "One pybind: pruned rFFT + dual-corner mul + pruned iRFFT into out",
          pybind11::arg("x"), pybind11::arg("w1"), pybind11::arg("w2"),
          pybind11::arg("modes1"), pybind11::arg("modes2"), pybind11::arg("out"));
    m.def("set_stage_pool", &set_stage_pool,
          "Select C++ stage-buffer pool (0/1) so two streams do not alias",
          pybind11::arg("pool"));
}
