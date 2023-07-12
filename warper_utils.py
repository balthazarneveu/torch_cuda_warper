import torch


def normalized_to_metric(h: int, w: int, dtype=torch.float64, device='cpu') -> torch.Tensor:
    """Referential change 3x3 matrix
    - from "pytorch" centered [0, 0] with `[-1,1] x [-1, 1]` normalized coordinates
    - to "classic" top left corner referential classic image coordinates `[0, h-1] x [0, w-1]`

    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """
    scale = torch.tensor([[(w - 1) / 2., 0., 0.], [0., (h - 1) /
                         2., 0.], [0., 0., 1.]], device=device, dtype=dtype)
    center_offset = torch.tensor(
        [[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]], device=device, dtype=dtype)
    return torch.matmul(scale, center_offset).to(dtype=dtype)


def metric_to_normalized(h: int, w: int, dtype=torch.float64, device='cpu'):
    """Referential change 3x3 matrix
    - from "classic" top left corner referential classic image coordinates `[0, h-1] x [0, w-1]`
    - to "pytorch" centered [0, 0] with `[-1,1] x [-1, 1]` normalized coordinates
    """
    return torch.linalg.inv(normalized_to_metric(h, w, dtype=dtype, device=device))


def transform_metric_to_normalized_torch(transforms_metric: torch.Tensor,
                                         input_size: tuple,
                                         output_size: tuple,
                                         is_affinity: bool = False) -> torch.Tensor:
    """Transform a tensor of homographies / affinities
    - from metric (classic image top left corner definition)
    - to normalized referential (pytorch definition)
    """
    if transforms_metric.shape[-2] == 2:  # 2x3 matrix to 3x3 matrix
        third_row = torch.tensor(
            [[[0., 0., 1.]]], dtype=transforms_metric.dtype, device=transforms_metric.device)
        third_row = third_row.repeat(transforms_metric.shape[-3], 1, 1)
        transforms_metric = torch.cat([transforms_metric, third_row], dim=-2)
    m2n = metric_to_normalized(
        input_size[0], input_size[1], device=transforms_metric.device)  # 3x3 matrix
    n2m = normalized_to_metric(
        output_size[0], output_size[1], device=transforms_metric.device)  # 3x3 matrix
    normed_transform = torch.matmul(
        m2n, torch.matmul(transforms_metric, n2m))  # 3x3 matrix
    if is_affinity:
        normed_transform = normed_transform[:2, :]  # return 2x3 matrix
    return normed_transform
