# ------------------------------------------------------------------------------------------------
# DCNv4 Function Utilities
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
"""Shared utility functions for DCNv4 operations."""

import math


def factors(n: int) -> list[int]:
    """Compute all factors of N.

    Args:
        n: The number to factorize.

    Returns:
        List of all factors of n in ascending order.

    """
    result = []
    for i in range(1, n + 1):
        if n % i == 0:
            result.append(i)
    return result


def compute_offset_mask_channels(
    group: int,
    kernel_size: int,
    remove_center: int = 0,
) -> int:
    """Compute the padded offset_mask channel dimension.

    The offset_mask contains offsets (2 per point) + weights (1 per point) for each group.
    The total is padded to be divisible by 8 for tensor core requirements.

    Args:
        group: Number of groups for spatial aggregation.
        kernel_size: Spatial size of the sampling grid.
        remove_center: Whether to exclude the center point (0 or 1).

    Returns:
        Padded channel dimension divisible by 8.

    """
    k_points = kernel_size * kernel_size - remove_center
    total = group * k_points * 3
    return int(math.ceil(total / 8) * 8)


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2.

    Power-of-2 dimensions are more efficient for CUDA kernels due to
    better memory alignment and vectorized load/store operations.

    Args:
        n: The number to check.

    Returns:
        True if n is a power of 2, False otherwise.

    Raises:
        ValueError: If n is not a non-negative integer.

    """
    if (not isinstance(n, int)) or (n < 0):
        msg = f"invalid input for is_power_of_2: {n} (type: {type(n)})"
        raise ValueError(msg)
    return (n & (n - 1) == 0) and n != 0


__all__ = ["compute_offset_mask_channels", "factors", "is_power_of_2"]
