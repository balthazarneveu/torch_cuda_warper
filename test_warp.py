import pytest

import torch  # noqa: F401  # isort:skip
import torch_warper  #isort:skip

pytestmark = pytest.mark.gputest
import time

current_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from torch.utils.benchmark import Timer


@pytest.fixture(scope="module", autouse=True)
def device_fixture():
    current_device = torch.device("cuda", 0)
    torch.cuda.set_device(current_device)


@torch.jit.script
def torch_homography(x: torch.Tensor, homography: torch.Tensor, align_corners: bool = True):
    """Apply homography transformation using torch operators

    Parameters
    ----------
    x : input tensor in NCHW
    homography : tensor N33 with the homography transformation
    """
    with torch.no_grad():
        grid = torch.nn.functional.affine_grid(homography[:, :2, :], x.shape, align_corners=align_corners)
        yt = torch.nn.functional.grid_sample(x, grid, mode="bilinear", align_corners=align_corners)
    return yt, grid


@pytest.mark.parametrize("spatial_size", [8, 9])
@pytest.mark.parametrize("rot_angle", [0, 30, 45, 60, 90])
def test_warp_forward(spatial_size, rot_angle):
    n, ch = 1, 1  # batch and channels
    # NHWC tensor format
    x = torch.eye(spatial_size, dtype=torch.float32).repeat(n, ch, 1, 1).permute(0, 2, 3, 1).to(current_device)
    phi = torch.tensor([rot_angle * 3.1415 / 180])
    s = torch.sin(phi)
    c = torch.cos(phi)
    homography = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]).repeat(n, 1, 1).to(current_device)

    yp = torch_warper.warp_image(x, homography)
    yt, grid = torch_homography(x.permute(0, 3, 1, 2), homography, align_corners=True)
    yt = yt.permute(0, 2, 3, 1)
    assert torch.allclose(yp, yt)


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
    # NHWC tensor format for the inputs
    x = torch.eye(spatial_size, dtype=torch.float32).repeat(n, ch, 1, 1).permute(0, 2, 3, 1).to(current_device)
    phi = torch.rand(n) * 3.141592
    s, c = torch.sin(phi), torch.cos(phi)
    r1 = torch.stack([c, -s, torch.zeros(n)], 1)
    r2 = torch.stack([s, c, torch.zeros(n)], 1)
    r3 = torch.stack([torch.zeros(n), torch.zeros(n), torch.ones(n)], 1)
    homography = torch.stack([r1, r2, r3], 1).to(current_device)
    timec = time_forward(x, homography)
    print(
        f"Warp forward custom cuda kernel {timec:.4f} s, bandwidth {spatial_size**2*nbytes_read_write*1e-9/timec:.4f}GB/s"
    )
    timec_ref = time_forward(x.permute(0, 3, 1, 2).clone(), homography, torch_ref=True)
    print(
        f"Warp forward torch {timec_ref:.4f} s, bandwidth {spatial_size**2*nbytes_read_write_torch*1e-9/timec_ref:.4f}GB/s"
    )
    print(f"Speed up {timec_ref/timec:.1f}x")


if __name__ == "__main__":
    warp_perf(512)
