import pytest
import numpy as np
from engine.base_classes import Location, AbsLoc, RelLoc


@pytest.fixture(params=[
    [1, 2], [[2, 3]], (3, 3), [(3, 4)], [(3, 3), (4, 4)],
    np.random.randint(0, 10, (5, 2))])
def coord2d(request):
    return request.param


@pytest.fixture
def loc2d(coord2d):
    return Location(coord2d)


@pytest.fixture(params=[1, 2, 3, 4, 5])
def ndim(request):
    return request.param


@pytest.fixture(params=[0, 1] + np.random.randint(2, 10, 3).tolist())
def ncoord(request):
    return request.param


@pytest.fixture
def coord_nd(ncoord, ndim, hypercube_grid_shape_nd):
    return np.random.randint(0, hypercube_grid_shape_nd[0], (ncoord, ndim))


@pytest.fixture
def loc_nd(coord_nd):
    return Location(coord_nd)


@pytest.fixture
def hypercube_grid_shape_nd(ndim):
    return (10, ) * ndim


@pytest.fixture(params=[[[[3, 3]]], np.random.randint(0, 1, (3, 2, 2))])
def invalid_coord(request):
    return request.param


@pytest.fixture
def loc10d():
    return Location(np.random.randint(0, 3, 10))


def test_Location_class_type(loc2d, coord2d):
    assert issubclass(Location, np.ndarray)


def test_Location_init(loc2d, coord2d):
    assert np.array_equal(np.array(coord2d, ndmin=2, dtype='int64'), loc2d)


def test_Location_init2(loc_nd, coord_nd, ndim, ncoord):
    assert np.array_equal(coord_nd, Location(coord_nd))
    assert Location(coord_nd).shape == (ncoord, ndim)


def test_Location_init_invalid(invalid_coord):
    with pytest.raises(AssertionError):
        Location(invalid_coord)


def test_Location_dtype():
    assert Location((3.0, 3.0)).dtype == np.int64


def test_Location_intersect1(loc10d, loc_nd, hypercube_grid_shape_nd):
    with pytest.raises(UserWarning):
        Location.intersect(loc10d, loc_nd, hypercube_grid_shape_nd)
    with pytest.raises(UserWarning):
        Location.intersect(loc_nd, loc10d, hypercube_grid_shape_nd)


def test_Location_intersect2(loc_nd, hypercube_grid_shape_nd):
    assert np.array_equal(
        loc_nd, Location.intersect(loc_nd, loc_nd, hypercube_grid_shape_nd))


def test_Location_intersect_mask(loc_nd, hypercube_grid_shape_nd):
    assert np.array_equal(
        np.ones(loc_nd.shape[0], dtype=np.bool),
        Location.intersect_mask(loc_nd, loc_nd, hypercube_grid_shape_nd))


@pytest.mark.parametrize("a,b,expected", [
    (Location([[3,3], [3,4]]), Location([3,3]), np.array([True, False])),
    (Location([3,3]), Location([[3,3], [3,4]]), np.array([True])),
    (Location([3,6]), Location([[3,3], [3,4]]), np.array([False])),
    (Location([[3,6], [6,3]]), Location([[3,3]]), np.array([False, False])),
])
def test_Location_intersect_mask2(a, b, expected):
    assert np.array_equal(Location.intersect_mask(a, b, (10, 10)), expected)
