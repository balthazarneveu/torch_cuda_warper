#include "warp.hpp"

#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <vector>

// -------- CUDA forward declarations
void launchWarpKernel(const torch::Tensor& image, const torch::Tensor& homography, torch::Tensor& warped);

// -------- C++ interface

/**
 * @brief Applies an homograpphy to a batch of 2D images
 *
 * @param image     batch of images to process, a NCHW tensor, C is the nimber of channels / colors, N is the batch dimension
 * @param homography the homography 3x3 to apply, a N,3,3 tensor - Convention: indirect order
 */
torch::Tensor warpImage(torch::Tensor& image,
                        const torch::Tensor& homography,
                        const std::optional<int> width,
                        const std::optional<int> height) {
    // ensure equal batch size between image and homography
    AT_ASSERTM(image.size(BATCH_DIM) == homography.size(BATCH_DIM),
               "Expecting the image and the homography to have the same batch size N.");
    // check image shape and layout
    AT_ASSERTM(image.dim() == NUM_DIMS, "Expecting a 4-dimensional image tensor");
    AT_ASSERTM(homography.dim() == 3, "Expecting a 3-dimensional homography tensor");
    AT_ASSERTM(homography.size(1) == 3, "Expecting a N,3,3 tensor");
    AT_ASSERTM(homography.size(2) == 3, "Expecting a N,3,3 tensor");

    torch::Tensor out_image;
    if (width && height) {
        out_image = torch::zeros({image.size(BATCH_DIM), image.size(CHANNELS_DIM), height.value(), width.value()},
                                image.options());
    } else {
        out_image = torch::zeros_like(image);
    }
    if (image.is_cuda()) {
        AT_ASSERTM(image.is_cuda(), "Expecting the image tensor to be stored in GPU memory");
        AT_ASSERTM(homography.is_cuda(), "Expecting the homography tensor to be stored in GPU memory");
        launchWarpKernel(image, homography, out_image);
    } else {
        AT_ASSERTM(!homography.is_cuda(), "Expecting the homography tensor to be stored in CPU memory");
        // CPU ACCESSORS
        // https://pytorch.org/cppdocs/notes/tensor_basics.html#cpu-accessors
        // @TODO: parallel for implementation
        auto out = out_image.accessor<float, NUM_DIMS>();
        auto img = image.accessor<float, NUM_DIMS>();
        auto homog = homography.accessor<float, 3>();
        for (int n = 0; n < out_image.size(BATCH_DIM); n++) {
            for (int y = 0; y < out_image.size(HEIGHT_DIM); y++) {
                for (int x = 0; x < out_image.size(WIDTH_DIM); x++) {
                    auto x_coord = homog[n][0][0] * x + homog[n][0][1] * y + homog[n][0][2];
                    auto y_coord = homog[n][1][0] * x + homog[n][1][1] * y + homog[n][1][2];
                    auto z_coord = homog[n][2][0] * x + homog[n][2][1] * y + homog[n][2][2];
                    x_coord /= z_coord;
                    y_coord /= z_coord;

                    // BILINEAR INTERPOLATION IMPLEMENTATION
                    int64_t x_coord_floor = floor(x_coord);
                    int64_t y_coord_floor = floor(y_coord);
                    auto res_x = x_coord - x_coord_floor;
                    auto res_y = y_coord - y_coord_floor;
                    if (x_coord_floor >= 0 and x_coord_floor < image.size(WIDTH_DIM) - 1 and y_coord_floor >= 0 and
                        y_coord_floor < image.size(HEIGHT_DIM) - 1) {
                        for (int c = 0; c < img.size(1); c++) {
                            auto top_left = img[n][c][y_coord_floor][x_coord_floor];
                            auto top_right = img[n][c][y_coord_floor][x_coord_floor + 1];
                            auto bottom_left = img[n][c][y_coord_floor + 1][x_coord_floor];
                            auto bottom_right = img[n][c][y_coord_floor + 1][x_coord_floor + 1];
                            auto top_interp = res_x * top_right + (1.f - res_x) * top_left;
                            auto bottom_interp = res_x * bottom_right + (1.f - res_x) * bottom_left;
                            out[n][c][y][x] = res_y * bottom_interp + (1.f - res_y) * top_interp;
                        }
                    } else {
                        for (int c = 0; c < img.size(CHANNELS_DIM); c++) {
                            out[n][c][y][x] = 0.f;
                        }
                    }
                }
            }
        }
    }
    return out_image;
};

// -------- pybind11 module definition

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("warp_image",
               &warpImage,
               py::arg("image"),
               py::arg("homography"),
               py::arg("width") = py::none(),
               py::arg("height") = py::none(), "Applies a warp to an image based on a homography");
}