#include <cstdint>
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

// In-place variant: caller provides `out` of shape (B, Cout, M1, M2, 2).
// Saves the per-call `cudaMallocAsync`-equivalent for hot-loop reuse.
void spectral_mul_out(torch::Tensor x_freq, torch::Tensor weights,
                      torch::Tensor out) {
    check_freq_tensor(x_freq, "x_freq", 2);
    check_freq_tensor(weights, "weights", 2);
    TORCH_CHECK(x_freq.device() == weights.device(),
                "x_freq and weights must share device");
    TORCH_CHECK(out.device() == x_freq.device(),
                "out must share device with x_freq");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int64_t batch_size = x_freq.size(0);
    const int64_t channels_in = x_freq.size(1);
    const int64_t modes1 = x_freq.size(2);
    const int64_t modes2 = x_freq.size(3);
    const int64_t channels_out = weights.size(1);

    TORCH_CHECK(out.size(0) == batch_size, "out B mismatch");
    TORCH_CHECK(out.size(1) == channels_out, "out Cout mismatch");
    TORCH_CHECK(out.size(2) == modes1 && out.size(3) == modes2, "out modes mismatch");
    TORCH_CHECK(out.size(4) == 2, "out last dim must be 2 (complex)");

    check_status(launch_spectral_mul(static_cast<const float *>(x_freq.data_ptr()),
                                     static_cast<const float *>(weights.data_ptr()),
                                     static_cast<float *>(out.data_ptr()),
                                     static_cast<int>(batch_size),
                                     static_cast<int>(channels_in),
                                     static_cast<int>(channels_out),
                                     static_cast<int>(modes1), static_cast<int>(modes2),
                                     nullptr),
                 "launch_spectral_mul(out)");
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

    auto row_freq = col_out.permute({0, 2, 1, 3}).contiguous();
    auto out = torch::empty({planes, height, width}, x_contig.options().dtype(torch::kFloat32));
    sufftHandle_t plan_c2r = get_cached_plan(width, row_batch, SUFFT_TYPE_C2R);
    check_sufft(sufftExecC2R(plan_c2r,
                             reinterpret_cast<suFloatComplex *>(row_freq.data_ptr<float>()),
                             out.data_ptr<float>()),
                "sufftExecC2R batched");

    out.mul_(scale);
    return out.view({batch_size, channels, height, width});
}

PYBIND11_MODULE(spectral_conv_ext, m) {
    m.def("spectral_mul", &spectral_mul,
          "Complex spectral multiply for FNO corner modes (SUPA kernel)");
    m.def("spectral_mul_out", &spectral_mul_out,
          "In-place spectral_mul writing into a pre-allocated out tensor");
    m.def("rfft2_sufft", &rfft2_sufft,
          "2D R2C via batched suFFT 1D stages + plan cache");
    m.def("irfft2_sufft", &irfft2_sufft,
          "2D C2R via batched suFFT 1D stages + plan cache");
}
