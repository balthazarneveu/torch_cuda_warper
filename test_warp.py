import pytest
from torch.utils.benchmark import Timer
import logging
import torch
from warper_utils import transform_metric_to_normalized_torch
try:
    import torch_warper
except:
    torch_warper = None
    logging.warning("Need to build the torch C++/Cuda kernel first")
current_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="module", autouse=True)
def device_fixture():
    current_device = torch.device("cuda", 0)
    torch.cuda.set_device(current_device)


@torch.jit.script
def torch_homography(img: torch.Tensor, homography: torch.Tensor, align_corners: bool = True):
    """Apply homography transformation using torch operators

    Parameters
    ----------
    x : input tensor in NCHW
    homography : tensor N33 with the homography transformation
    """
    with torch.no_grad():
        grid = torch.nn.functional.affine_grid(homography[:, :2, :], img.shape, align_corners=align_corners)
        yt = torch.nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=align_corners)
    return yt, grid


@pytest.mark.parametrize("spatial_size", [64])
@pytest.mark.parametrize("rot_angle", [0, 30, 45, 60, 90])
def test_warp_forward(spatial_size, rot_angle):
    n, ch = 1, 3  # batch and channels
    img = torch.eye(spatial_size, dtype=torch.float32).repeat(n, ch, 1, 1).to(current_device)  # NCHW
    phi = torch.tensor([rot_angle * 3.1415 / 180])
    s = torch.sin(phi)
    c = torch.cos(phi)
    homography = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]).repeat(n, 1, 1).to(current_device)
    yp = torch_warper.warp_image(img, homography)
    yp_cpu = torch_warper.warp_image(img.cpu(), homography.cpu())
    assert torch.allclose(yp.cpu(), yp_cpu, rtol=1.e-4)
    homography_pytorch = transform_metric_to_normalized_torch(homography.double(),
                                                              img.shape[-2:],
                                                              img.shape[-2:],
                                                              is_affinity=False).to(current_device).float()
    yt, _grid = torch_homography(img, homography_pytorch, align_corners=True)
    assert torch.allclose(yp, yt, rtol=1., atol=1.e-6)


def time_forward(x, homography, torch_ref=False):
    stmt = "torch_homography(x, homography)" if torch_ref else "torch_warper.warp_image(x, homography)"
    t = Timer(stmt=stmt,
              globals={
                  "x": x,
                  "homography": homography,
                  "torch_homography": torch_homography,
                  "torch_warper": torch_warper
              })
    res = t.blocked_autorange(min_run_time=1.)
    timec = res.median
    return timec


def warp_perf(spatial_size):
    n, ch = 16, 32  # batch and channels
    data_size = 4
    nbytes_read_write = data_size * (3 * 3 * n + 2 * ch * n
                                     )  # read homography matrix + read and write (2*) one float per channel per batch
    nbytes_read_write_torch = data_size * (2 * 3 * n + 2 * n + 2 * n * ch
                                           )  # read affine + write grid + read and write warp
    img = torch.eye(spatial_size, dtype=torch.float32).repeat(n, ch, 1, 1).to(current_device)
    phi = torch.rand(n) * 3.141592
    s, c = torch.sin(phi), torch.cos(phi)
    r1 = torch.stack([c, -s, torch.zeros(n)], 1)
    r2 = torch.stack([s, c, torch.zeros(n)], 1)
    r3 = torch.stack([torch.zeros(n), torch.zeros(n), torch.ones(n)], 1)
    homography = torch.stack([r1, r2, r3], 1).to(current_device)
    timec = time_forward(img, homography)
    print(
        f"Warp forward custom cuda kernel {timec:.4f} s, bandwidth {spatial_size**2*nbytes_read_write*1e-9/timec:.4f}GB/s"
    )
    homography = transform_metric_to_normalized_torch(homography.double(), img.shape[-2:], img.shape[-2:],
                                                      is_affinity=False).to(current_device).float()
    timec_ref = time_forward(img, homography, torch_ref=True)
    print(
        f"Warp forward torch {timec_ref:.4f} s, bandwidth {spatial_size**2*nbytes_read_write_torch*1e-9/timec_ref:.4f}GB/s"
    )
    print(f"Speed up {timec_ref/timec:.1f}x")


if __name__ == "__main__":
    warp_perf(512)