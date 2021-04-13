"""Crops image around segmentations."""
import itertools

from typing import Any, Optional, Sequence, Tuple

import pytest
import torch


def resize_range(lo: int,
                 hi: int,
                 to: int,
                 lb: Optional[int] = None,
                 ub: Optional[int] = None) -> Tuple[int, int]:
    """Expand or contract range (lo, hi) to given size.

    Centers on original range if possible.

    Respects lower bound and upperbound constraints, i.e. will never extend
    lo value beyond lb nor hi value above up.

    For example, resize(3, 5, 4) will return the range [2, 6) and
    resize(1, 3, 5, lb=0) will return [0, 5).

    Args:
        lo (int): Low value of range, inclusive.
        hi (int): High value of range, exclusive.
        to (int): Size for resized range.
        lb (Optional[int], optional): Lower bound for lo. By default, no bound
            is enforced.
        ub (Optional[int], optional): Upper bonud for hi. By default, no bound
            is enforced.

    Returns:
        Tuple[int, int]: The resized range.

    Raises:
        ValueError: If [lo, hi) is not valid range or if ub prevents growth
            to desired size.

    """
    if lo > hi:
        raise ValueError(f'bad range: [{lo}, {hi})')
    if lb is not None and ub is not None:
        if lb >= ub:
            raise ValueError(f'lb {lb} cannot be >= ub {ub}')
        if ub - lb < to:
            raise ValueError(f'impossible resize, cannot resize {ub - lb} '
                             f'size range to {to}')

    size = hi - lo
    if size == to:
        # Easy, nothing to change.
        return lo, hi
    elif size < to:
        # Tricky. We must grow the range, watching the boundaries if set.
        growth = to - size
        hi_delta = growth // 2
        lo_delta = hi_delta - growth

        # If lower delta crosses the bound, move the excess to upper end.
        if lb is not None and lo + lo_delta < lb:
            hi_delta += lb - lo - lo_delta
            assert ub is None or hi + hi_delta <= ub, 'impossible resize?'
            return lb, hi + hi_delta

        # Ditto, but for upper delta.
        if ub is not None and hi + hi_delta > ub:
            lo_delta -= hi + hi_delta - ub
            assert lb is None or lo + lo_delta >= lb, 'impossible resize?'
            return lo + lo_delta, ub

        return lo + lo_delta, hi + hi_delta
    else:
        # Easy! We can just shrink the range.
        shrinkage = size - to
        lo_delta = shrinkage // 2
        hi_delta = lo_delta - shrinkage
        return lo + lo_delta, hi + hi_delta


def find_largest_segment(
    mask: torch.Tensor,
    neighbor_offsets: Sequence[int] = (-1, 0, 1),
) -> Sequence[Tuple[int, int]]:
    """Return the largest segment in the mask.

    For example, the largest segment in the mask 1101 is 1100.
    This implementation, given a tensor of N elements, uses O(N) space
    and time.

    Args:
        mask (torch.Tensor): Binary mask describing image segments.
            Should be two-dimensional.
        neighbor_offsets (Sequence[int], optional): Offsets to consider
            as pixel neighbors when computing contiguity of segments.

    Returns:
        Sequence[Tuple[int, int]]: A mask of the same size as the input mask,
            but with all but the largest segment zeroed out.

    Raises:
        ValueError: If mask is not two-dimensional.

    """
    if len(mask.shape) != 2:
        raise ValueError(f'mask must be 2D, got {len(mask.shape)}D')
    height, width = mask.shape

    # Determine all possible starting points, i.e. all unmasked values.
    starts = mask.nonzero().tolist()

    # Find all connected components through repeated DFS.
    components = []
    visited = set()
    while starts:
        start = tuple(starts.pop())
        if start in visited:
            continue

        # Perform DFS.
        component = []
        frontier = [start]
        while frontier:
            current = i, j = frontier.pop()
            if current in visited or not mask[i, j]:
                continue
            visited.add(current)
            component.append(current)

            # Add neighbors to frontier.
            offsets = itertools.product(neighbor_offsets, neighbor_offsets)
            for i_offset, j_offset in offsets:
                i_new = i + i_offset
                j_new = j + j_offset
                if i_new < 0 or i_new >= height or j_new < 0 or j_new >= width:
                    continue
                frontier.append((i_new, j_new))

        components.append(component)

    # Split mask by connected components.
    components = sorted(components, key=len)
    return tuple(components[-1]) if components else ()


def crop_by_mask(
        image: torch.Tensor,
        mask: torch.Tensor,
        size: Optional[int] = None,
        segment: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop the image to be the smallest size containing the mask.

    Args:
        image (torch.Tensor): The image to crop.
            Should have shape (3, height, width).
        mask (Union[torch.Tensor, Sequence[Tuple[int, int]]]): The mask to
            crop around. If a tensor, should have shape (height, width).
            Otherwise should be a list of points comprising the mask.
        size (int): Size for the cropped image. Defaults to smallest size
            containing the mask.
        segment (Optional[Sequence[Tuple[int, int]]]], optional): The specific
            points to crop around. Otherwise will be inferred from the mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cropped image and mask.

    """
    if len(image.shape) != 3:
        raise ValueError(f'image must be 3D, got {len(image.shape)}D')
    if len(mask.shape) != 2:
        raise ValueError(f'mask must be 2D, got {len(mask.shape)}D')
    _, height, width = image.shape

    if segment is None:
        segment = mask.nonzero().tolist()
    assert segment is not None, 'null segment?'

    # Something of a heuristic; if the mask is empty, just take something like
    # "the whole image," ignoring black pixels for convenience.
    if not segment:
        segment = image.sum(dim=0).nonzero().tolist()

    # Determine mask bounding box.
    low_i, low_j = height, width
    hi_i, hi_j = -1, -1
    for i, j in segment:
        if i < low_i:
            low_i = i
        if j < low_j:
            low_j = j
        if i > hi_i:
            hi_i = i
        if j > hi_j:
            hi_j = j
    assert low_i <= hi_i, 'bad bounding box height'
    assert low_j <= hi_j, 'bad bounding box width'

    # Upper bound should be exclusive for correctness.
    hi_i += 1
    hi_j += 1

    # Determine the crop.
    if size is not None:
        low_i, hi_i = resize_range(low_i, hi_i, size, lb=0, ub=height)
        low_j, hi_j = resize_range(low_j, hi_j, size, lb=0, ub=width)

    subimage = image[:, low_i:hi_i, low_j:hi_j]
    submask = mask[low_i:hi_i, low_j:hi_j]
    return subimage, submask


def crop_by_largest_segment(
        image: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find largest contiguous segment in the mask, and crop image to it.

    Segment size is determined by the number of contiguous 1s.

    Args:
        image (torch.Tensor): The image to crop.
        mask (torch.Tensor): The mask to split into segments.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cropped image and mask.

    """
    assert 'segment' not in kwargs, 'set segment in crop_by_largest_segment'
    largest_segment = find_largest_segment(mask)
    return crop_by_mask(image, mask, segment=largest_segment, **kwargs)


###############
# Unit tests. #
###############


@pytest.mark.parametrize('lo,hi,to,kwargs,expected', (
    (3, 5, 4, dict(), (2, 6)),
    (3, 7, 2, dict(), (4, 6)),
    (1, 3, 5, dict(lb=0), (0, 5)),
    (1, 3, 5, dict(ub=4), (-1, 4)),
    (1, 3, 3, dict(lb=2, ub=5), (2, 5)),
    (1, 3, 3, dict(lb=-1, ub=2), (-1, 2)),
))
def test_resize_range(lo, hi, to, kwargs, expected):
    """Test resize_range works for valid use cases."""
    actual = resize_range(lo, hi, to, **kwargs)
    assert actual == expected


@pytest.mark.parametrize('lo,hi,to,kwargs,error_pattern', (
    (2, 1, 1, dict(), '.*bad range.*'),
    (1, 2, 1, dict(lb=2, ub=1), '.*cannot be >=.*'),
    (1, 2, 1, dict(lb=2, ub=2), '.*cannot be >=.*'),
    (1, 2, 4, dict(lb=1, ub=2), '.*impossible resize*'),
))
def test_resize_range_bad_input(lo, hi, to, kwargs, error_pattern):
    """Test resize_range explodes on invalid inputs."""
    with pytest.raises(ValueError, match=error_pattern):
        resize_range(lo, hi, to, **kwargs)


@pytest.mark.parametrize('mask,expecteds', (
    (torch.zeros(3, 3, dtype=torch.long), ()),
    (
        torch.ones(3, 3, dtype=torch.long),
        torch.ones(3, 3, dtype=torch.long).nonzero().tolist(),
    ),
    (
        torch.tensor([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]),
        torch.tensor([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]).nonzero().tolist(),
    ),
))
def test_find_largest_segment(mask, expecteds):
    """Test split_by_segment on basic use cases."""
    actuals = {tuple(point) for point in find_largest_segment(mask)}
    assert len(actuals) == len(expecteds)

    expecteds = {tuple(point) for point in expecteds}
    assert actuals == expecteds


@pytest.fixture
def image():
    """Return a fake image for testing."""
    return torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])


@pytest.mark.parametrize('mask,size,expected_image,expected_mask', (
    (
        torch.tensor([
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]),
        2,
        torch.tensor([
            [1, 2],
            [4, 5],
        ]),
        torch.tensor([
            [1, 1],
            [0, 0],
        ]),
    ),
    (
        torch.tensor([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ]),
        2,
        torch.tensor([
            [1, 2],
            [4, 5],
        ]),
        torch.tensor([
            [1, 1],
            [1, 1],
        ]),
    ),
    (
        torch.tensor([
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
        ]),
        2,
        torch.tensor([
            [2, 3],
            [5, 6],
        ]),
        torch.tensor([
            [1, 1],
            [1, 1],
        ]),
    ),
))
def test_crop_by_mask(image, mask, size, expected_image, expected_mask):
    """Test crop_by_mask crops around image correctly."""
    image = image.unsqueeze(0).repeat(3, 1, 1)
    expected_image = expected_image.unsqueeze(0).repeat(3, 1, 1)
    actual_image, actual_mask = crop_by_mask(image, mask, size=size)
    assert actual_image.equal(expected_image)
    assert actual_mask.equal(expected_mask)
