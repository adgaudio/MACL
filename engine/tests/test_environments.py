import pytest
import numpy as np
from engine.api.cartesian import CartesianEnvironment, AbsLoc, RelLoc


@pytest.fixture(
    params=list(reversed([(5,), (4, 5), (5, 4, 3), (3, 4, 5, 3)])),
    ids=list(reversed(['5', '4x5', '5x4x3', '3x4x5x2'])))
def grid_shape_M(request):
    return request.param


@pytest.fixture
def ndim_M(grid_shape_M):
    return len(grid_shape_M)


@pytest.fixture
def cart_M(request, grid_shape_M):
    return CartesianEnvironment(grid_shape_M)


@pytest.fixture
def cart_big():
    return CartesianEnvironment((8, 9, 7, 6, 10))


@pytest.fixture
def cart_small():
    return CartesianEnvironment((3, 3))


@pytest.fixture
def origin_small():
    return AbsLoc((0,0))


@pytest.fixture
def relloc_small():
    return AbsLoc((1, -1))


@pytest.fixture
def cart_big_twos(cart_big):
    return 1 + AbsLoc(np.ones((1, cart_big.grid_ndim), dtype='int'))


@pytest.fixture
def loc_with_3coord_M(grid_shape_M, ndim_M):
    return AbsLoc(np.array([
        np.array(grid_shape_M) - 1,  # max point on each dimension
        (np.array(grid_shape_M) - 1) // 2,  # middle point in each dimension
        np.repeat(0, ndim_M),  # min point on each dimension
    ], dtype='int'), ndim_M)


@pytest.fixture
def rel_with_3coord_M(ndim_M):
    rel = RelLoc(np.array([
        np.zeros(ndim_M),
        np.zeros(ndim_M),
        np.zeros(ndim_M),
    ], dtype='int'), ndim_M)
    rel[0][-1] = -1
    rel[1][-1] = 0
    rel[2][0] = 1
    return rel


@pytest.fixture
def loc_at_origin_M(ndim_M):
    return AbsLoc(np.zeros((1, ndim_M), dtype='int'), ndim_M)


@pytest.fixture
def loc_at_ones_M(ndim_M):
    return AbsLoc(np.ones((1, ndim_M), dtype='int'), ndim_M)


@pytest.fixture(params=['origin', 'end', 'middle', '3coord'],
                ids=['loc0', 'locEnd', 'locMiddle', 'loc3coord'])
def absloc(ndim_M, grid_shape_M, request, loc_with_3coord_M):
    if request.param == 'middle':
        loc = AbsLoc((np.array(grid_shape_M) - 1) // 2, ndim_M)
        assert loc.shape == (1, ndim_M)
        return loc
    elif request.param == 'end':
        loc = AbsLoc(np.array(grid_shape_M) - 1, ndim_M)
        assert loc.shape == (1, ndim_M)
        return loc
    elif request.param == 'origin':
        loc = AbsLoc(np.repeat(0, ndim_M), ndim_M)
        assert loc.shape == (1, ndim_M)
        return loc
    elif request.param == '3coord':
        loc = loc_with_3coord_M
        assert loc.shape == (3, ndim_M)
        return loc
    raise UserWarning("misconfiguration of test fixture")


@pytest.fixture
def relloc_M(request, ndim_M):
    '''offset by 1 along last dimension'''
    a = np.zeros((1, ndim_M), dtype='int')
    a[0][-1] = 1
    return RelLoc(a, ndim_M)


@pytest.fixture
def invalid_loc(grid_shape_M, absloc):
    return absloc + grid_shape_M


@pytest.fixture
def agent_id(request):
    return 101010


@pytest.fixture
def agent_ids():
    return [0, 1, 2, 3]


@pytest.fixture(params=[0, 1, 2], ids=['0hops', '1hops', '2hops'])
def nhops(request):
    return request.param


def test_grid_shape(cart_M, grid_shape_M):
    assert hasattr(cart_M, 'grid_shape')
    assert isinstance(cart_M.grid_shape, tuple)
    assert cart_M.grid_shape == grid_shape_M


def test_grid_ndim(cart_M, ndim_M):
    assert hasattr(cart_M, 'grid_ndim')
    assert isinstance(cart_M.grid_ndim, int)
    assert cart_M.grid_ndim == ndim_M


def test_neighbors1(cart_M, absloc, nhops, ndim_M):
    '''does env.neighbors(...) return expected shape?'''
    ncoord = absloc.shape[0]

    expected_shape = (ncoord, (2 * nhops + 1)**ndim_M, ndim_M)
    nei = cart_M.neighbors(absloc, nhops)
    assert expected_shape == nei.shape


def test_neighbors2(cart_M, absloc, nhops, grid_shape_M):
    '''are all neighbors in bounds?
    better would be to check all neighbors within nhops'''
    nei = cart_M.neighbors(absloc, nhops)
    assert (nei < grid_shape_M).all()


def test_neighbors3(cart_M, loc_at_ones_M):
    '''does it return no duplicates and stay within bounds?'''
    nei = cart_M.neighbors(loc_at_ones_M, 1)
    assert np.abs(nei).max() == 2
    assert np.abs(nei).min() == 0
    # no dups check:
    assert len(set(tuple(x) for x in nei.reshape(-1, nei.shape[-1]))) \
        == \
        nei.shape[0] * nei.shape[1]


def test_neighbors4(cart_big, cart_big_twos, nhops):
    '''another in bounds test'''
    nei = cart_big.neighbors(cart_big_twos, nhops=nhops)
    assert np.abs(nei).max() == nhops + 2
    assert np.abs(nei).min() == 2 - nhops


def test_relative_neighbors(cart_small):
    arr1 = cart_small.relative_neighbors(1)

    assert isinstance(arr1, RelLoc)
    assert np.array_equal(arr1, RelLoc([
        [-1, -1], [0, -1], [1, -1],
        [-1,  0], [0,  0], [1,  0],
        [-1,  1], [0,  1], [1,  1]]))

    arr2 = cart_small.relative_neighbors(2)
    assert np.array_equal(arr2, RelLoc([
        [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2],
        [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
        [-2,  0], [-1,  0], [0,  0], [1,  0], [2,  0],
        [-2,  1], [-1,  1], [0,  1], [1,  1], [2,  1],
        [-2,  2], [-1,  2], [0,  2], [1,  2], [2,  2]]))


def test_add_agent1(cart_M, agent_id, grid_shape_M):
    '''agent added to random location'''
    assert cart_M._agents == dict()
    cart_M.add_agent(agent_id)
    assert len(cart_M._agents) == 1
    assert (cart_M._agents[agent_id] < grid_shape_M).all()


def test_add_agent2(cart_M, absloc, agent_id):
    '''agent added to given location'''
    cart_M.add_agent(agent_id, absloc)
    assert (cart_M._agents[agent_id] == absloc).all()


def test_add_agent3(cart_M, agent_id):
    '''agent can only be added once'''
    cart_M.add_agent(agent_id)
    with pytest.raises(UserWarning):
        cart_M.add_agent(agent_id)
    assert len(cart_M._agents) == 1


def test_agent_location(cart_M, absloc, agent_id):
    cart_M.add_agent(agent_id, absloc)
    assert (cart_M.agent_location(agent_id) == absloc).all()


def test_agents_at_location(cart_M, absloc, agent_ids):
    '''Should return correct location for all agents given a location'''
    for agent_id in agent_ids:
        cart_M.add_agent(agent_id, absloc)

    for agent_id, agent_loc in cart_M.agents_at_location(absloc):
        expected_agent_loc = cart_M.agent_location(agent_id)
        assert np.array_equal(agent_loc, expected_agent_loc)


def test_agents_at_location2(cart_M, absloc, invalid_loc, agent_ids):
    '''Should return no locations and no agents agents'''
    assert list(cart_M.agents_at_location(invalid_loc)) == []


def test_agents_at_location3(cart_M, loc_with_3coord_M, agent_id):
    '''should return partial location of agent'''
    cart_M.add_agent(agent_id, loc_with_3coord_M[:2])

    for a_id, agent_loc in cart_M.agents_at_location(loc_with_3coord_M[1:]):
        assert a_id == agent_id
        assert (agent_loc == loc_with_3coord_M[1]).all()


def test_agents_at_location4(cart_M, absloc, agent_ids):
    '''should return locations of all agents'''
    [cart_M.add_agent(x) for x in agent_ids]
    found_all_agent_ids = 0
    for agent_id, agent_loc in cart_M.agents_at_location():
        found_all_agent_ids += 1
        assert agent_id in agent_ids
        assert (agent_loc == cart_M.agent_location(agent_id)).all()
    assert found_all_agent_ids == len(agent_ids)


def test_visit_statistics(cart_small, agent_ids, relloc_small, origin_small):
    assert cart_small.visit_statistics() == {
        'uniq_coords_visited': 0,
        'total_visit_counts': 0,
        'num_agents_sharing_coords': 0,
        'visit_counts_variance': 0.0,
        'visit_counts_mean': 0.0,
        }

    a1, a2 = agent_ids[:2]
    cart_small.add_agent(a1, origin_small)
    cart_small.add_agent(a2, origin_small)
    cart_small.process_request(a1, move={a1: relloc_small})
    cart_small.process_request(a1, move={a1: relloc_small})
    cart_small.process_request(a1, move={a1: relloc_small})
    cart_small.process_request(a2, move={a2: relloc_small})
    assert cart_small.visit_statistics() == {
        'uniq_coords_visited': 3,
        'total_visit_counts': 4,
        'num_agents_sharing_coords': 0,
        'visit_counts_variance': 0.46913580246913583,
        'visit_counts_mean': 0.44444444444444442,
        }


def test_visit_counts(cart_M, agent_id, grid_shape_M):
    '''returns all zeros even after agent added to a location'''
    cart_M.add_agent(agent_id)
    visit_counts = cart_M.visit_counts()

    is_initialized_with_no_visits = (visit_counts == 0).all()
    assert is_initialized_with_no_visits
    has_correct_number_counts = (np.prod(grid_shape_M), )
    assert visit_counts.shape == has_correct_number_counts


def test_visit_counts2(
        cart_M, agent_id, loc_at_origin_M, relloc_M, grid_shape_M):
    '''reflects agent moves'''
    cart_M.add_agent(agent_id, loc_at_origin_M)

    # go forwards
    cart_M._process_agent_move(agent_id, relloc_M)

    expected_zeros = np.zeros(loc_at_origin_M.shape[0])
    expected_ones = np.ones(loc_at_origin_M.shape[0])
    expected_twos = np.ones(loc_at_origin_M.shape[0]) * 2
    assert (cart_M.visit_counts(loc_at_origin_M) == expected_zeros).all()
    assert (cart_M.visit_counts(loc_at_origin_M + relloc_M) ==
            expected_ones).all()

    # go backwards and forwards
    cart_M._process_agent_move(agent_id, -relloc_M)
    cart_M._process_agent_move(agent_id, relloc_M)
    assert (cart_M.visit_counts(loc_at_origin_M) == expected_ones).all()
    assert (cart_M.visit_counts(loc_at_origin_M + relloc_M) ==
            expected_twos).all()

    # check all visit counts
    expected_visit_counts = np.zeros((np.prod(grid_shape_M), ))
    expected_visit_counts[0] = 1
    expected_visit_counts[1] = 2
    assert (expected_visit_counts == cart_M.visit_counts()).all()


def test_visit_counts3(cart_M, agent_id, loc_with_3coord_M, rel_with_3coord_M):
    '''do visit_counts update properly for distributed agents?'''
    cart_M.add_agent(agent_id, loc_with_3coord_M)
    cart_M._process_agent_move(agent_id, rel_with_3coord_M)
    assert (cart_M.visit_counts(loc_with_3coord_M + rel_with_3coord_M) ==
            np.array([1, 0, 1])).all()


def test_visit_counts4(cart_M, agent_id, loc_at_origin_M, loc_at_ones_M):
    '''when a distributed agent moves such that two of its coordinates are the
    same, ensure we count that visit as 1 rather than 2 visits
    '''
    # absolute loc at [[0, ...], [1, ...]]
    loc_2coords = AbsLoc(np.concatenate(
        [loc_at_origin_M, loc_at_ones_M]),
        ndim=loc_at_origin_M.shape[1])
    # relative loc at [[1, ...], [0, ...]]
    relloc = RelLoc(np.zeros(loc_2coords.shape, dtype='int'))
    relloc[0] = 1
    relloc[-1] = 0

    cart_M.add_agent(agent_id, loc_2coords)
    cart_M._process_agent_move(agent_id, relloc)
    assert (cart_M.visit_counts(loc_2coords + relloc) ==
            np.array([1, 1])).all()
    # sanity check that the location of each coordinate is the same
    newloc = cart_M.agent_location(agent_id)
    assert (newloc == newloc[[0]].repeat(2, axis=0)).all()


def test_agent_locations_nearby(cart_M, agent_ids, loc_at_origin_M, ndim_M):
    '''Should return correct location for all agents given a location'''
    nhops = 1
    # add all agents except last one to origin
    for agent_id in agent_ids[:2]:
        cart_M.add_agent(agent_id, loc_at_origin_M)
    # find agents near to the first agent
    nearbylocs = list(cart_M.agent_locations_nearby(agent_ids[0], nhops))
    assert [x[0] for x in nearbylocs] == agent_ids[:2]
    for agent_id, mask in nearbylocs:
        assert mask.dtype == np.bool
        assert mask.shape == (
            loc_at_origin_M.shape[0], (2 * nhops + 1) ** ndim_M)
        assert mask.sum() == 1

        expected_mask = np.zeros(
            (loc_at_origin_M.shape[0], (2*nhops+1)**ndim_M), dtype=np.bool)
        expected_mask[0][(2*nhops+1)**ndim_M // 2] = True  # origin location
        assert np.array_equal(mask, expected_mask)


def test_visit_counts_nearby(cart_M, agent_ids, nhops, loc_at_ones_M,
                             grid_shape_M, ndim_M):
    for agent_id in agent_ids:
        cart_M.add_agent(agent_id, loc_at_ones_M)

    vcounts = cart_M.visit_counts_nearby(agent_id, nhops)
    expected_shape = (loc_at_ones_M.shape[0],
                      cart_M.relative_neighbors(nhops).shape[0])
    assert np.array_equal(vcounts, np.zeros(expected_shape))

    cart_M._process_agent_move(agent_id, RelLoc(loc_at_ones_M))
    vcounts = cart_M.visit_counts_nearby(agent_id, nhops)
    expected_vcounts = np.zeros(expected_shape)
    expected_vcounts[0][expected_shape[1]//2] = 1
    assert np.array_equal(expected_vcounts, vcounts)


def test_process_request(
        cart_small, agent_ids, origin_small, relloc_small):
    '''requests should process a move and a message'''
    for agent_id in agent_ids:
        cart_small.add_agent(agent_id, origin_small)
    a1 = agent_ids[0]
    a2 = agent_ids[1]
    cart_small.process_request(
        a1,
        {a1: relloc_small, a2: -relloc_small},
        {a1: 'hello self', a2: 'banshee'})
    # moves properly
    assert np.array_equal(
        cart_small.agent_location(a1),
        (origin_small + relloc_small) % cart_small.grid_shape)
    assert np.array_equal(cart_small.agent_location(a2), origin_small)


def test_agent_inbox1(cart_small, agent_id):
    '''should return empty dict when agent never received msg'''
    cart_small.add_agent(agent_id)
    assert cart_small.agent_inbox(agent_id) == cart_small._new_empty_inbox()


def test_agent_inbox2(cart_small, agent_ids, relloc_small):
    '''should flush messages'''
    a1 = agent_ids[0]
    a2 = agent_ids[1]
    for agent_id in agent_ids:
        cart_small.add_agent(agent_id)

    cart_small.process_request(
        a1,
        move={a1: relloc_small, a2: relloc_small},
        msg={a1: 'hello self'})
    cart_small.process_request(
        a2,
        move={a1: -relloc_small},
        msg={a1: 'lost message'})
    cart_small.process_request(a2, msg={a1: 'hello from a2'})

    inbox1 = cart_small.agent_inbox(a1, flush=False)
    inbox2 = cart_small.agent_inbox(a1, flush=True)
    expected_inbox = {'msg': {a1: "hello self", a2: 'hello from a2'},
                      'move': {a2: -relloc_small}}
    for inbox in [inbox1, inbox2]:
        assert inbox.keys() == expected_inbox.keys()
        assert inbox['msg'] == expected_inbox['msg']
        assert inbox['move'].keys() == expected_inbox['move'].keys()
        assert np.array_equal(inbox['move'][a2],  expected_inbox['move'][a2])
    assert cart_small._new_empty_inbox() == cart_small.agent_inbox(a1)
