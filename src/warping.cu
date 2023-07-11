#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "warp.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__device__ inline T sqr(T x) {
    return x * x;
}

namespace kernels {
template <typename scalar_t, const int N_CHANNELS>
__global__ void applyWarpKernel(torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> img,
                                torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> homog,
                                torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> warped) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    auto x_coord = homog[n][0][0] * ix + homog[n][0][1] * iy + homog[n][0][2];
    auto y_coord = homog[n][1][0] * ix + homog[n][1][1] * iy + homog[n][1][2];
    auto z_coord = homog[n][2][0] * ix + homog[n][2][1] * iy + homog[n][2][2];
    x_coord /= z_coord;
    y_coord /= z_coord;
    int64_t x_coord_floor = floor(x_coord);
    int64_t y_coord_floor = floor(y_coord);
    auto res_x = x_coord - x_coord_floor;
    auto res_y = y_coord - y_coord_floor;
    if (iy < warped.size(HEIGHT_DIM) and ix < warped.size(WIDTH_DIM)) {
        if (x_coord_floor >= 0 and x_coord_floor < img.size(WIDTH_DIM) - 1 and y_coord_floor >= 0 and
            y_coord_floor < img.size(HEIGHT_DIM) - 1) {
            for (int c = 0; c < N_CHANNELS; ++c) {
                auto top_left = img[n][c][y_coord_floor][x_coord_floor];
                auto top_right = img[n][c][y_coord_floor][x_coord_floor + 1];
                auto bottom_left = img[n][c][y_coord_floor + 1][x_coord_floor];
                auto bottom_right = img[n][c][y_coord_floor + 1][x_coord_floor + 1];
                auto top_interp = res_x * top_right + (1.f - res_x) * top_left;
                auto bottom_interp = res_x * bottom_right + (1.f - res_x) * bottom_left;
                warped[n][c][iy][ix] = res_y * bottom_interp + (1.f - res_y) * top_interp;
            }
        } else {
            for (int c = 0; c < N_CHANNELS; ++c) {
                warped[n][c][iy][ix] = 0.f;
            }
        }
    }
}

} // namespace kernels

void launchWarpKernel(const torch::Tensor& image, const torch::Tensor& homography, torch::Tensor& warped) {
    // get CUDA stream
    const auto deviceIndex = image.device().index();
    auto stream = c10::cuda::getCurrentCUDAStream(deviceIndex);

    // setup thread grid
    // @TODO:kind of default
    static const dim3 threads(32, 32, 1);
    const dim3 blocks((warped.size(WIDTH_DIM) + threads.x - 1) / threads.x,
                      (warped.size(HEIGHT_DIM) + threads.y - 1) / threads.y,
                      warped.size(BATCH_DIM));

    AT_DISPATCH_FLOATING_TYPES(image.type(), "warping", [&] {
        auto imagePtr = image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
        auto outPtr = warped.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
        auto homographyPtr = homography.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        kernels::applyWarpKernel<scalar_t, 3><<<blocks, threads, 0, stream.stream()>>>(imagePtr, homographyPtr, outPtr);
    });

    // check error
    C10_CUDA_CHECK(cudaGetLastError());
}