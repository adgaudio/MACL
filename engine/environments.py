from collections import defaultdict
from typing import Tuple, Dict, Any, Sequence, Generator, DefaultDict, Iterator

import numpy as np
from engine.base_classes import \
    AgentId, AbsLoc, RelLoc, BaseEnvironment, BaseAgent


_Inbox = Dict[str, Any]  # type alias


class CartesianEnvironment(BaseEnvironment):
    '''
    A traversable n-dimensional space of pre-defined shape
    that defines the ways agents can interact within this space

    Agents use the environment to send each other messages,
    check the state of the grid, and move to new locations.
    '''

    def __init__(self, grid_shape: Sequence[int]) -> None:
        '''
        Initialize an environment.

        - `grid_shape` defines the maximum size of each dimension of the space.
          A shape of (4,5,6) has 3 dimensions will require us to manage state
          for 4*5*6 distinct coordinates.
        '''
        self.grid_shape = tuple(grid_shape)  # type: Tuple[int, ...]
        self.grid_ndim = len(self.grid_shape)  # type: int
        # mutable grid that defines all possible locations.
        # initialized as never unvisited (value=0)
        self._grid_state = np.zeros(self.grid_shape, dtype='int')
        self._agents = {}  # type: Dict[AgentId, AbsLoc]
        self._agent_inbox = defaultdict(self._new_empty_inbox
                                        )  # type: DefaultDict[AgentId, _Inbox]

    def __repr__(self):
        return '%s:%s %s agents' % (
            super().__repr__(), self.grid_shape, len(self._agents))

    def _new_empty_inbox(self) -> _Inbox:
        return {'msg': {}, 'move': {}}

    def visit_statistics(self) -> Dict[str, int]:
        '''Return a small report summarizing the grid state'''
        if self._agents:
            all_coords = np.concatenate(list(self._agents.values()))
            shared_coords = np.unique(
                all_coords.view([('', all_coords.dtype)] * all_coords.shape[1]),
                return_counts=True)[1]
            num_shared_coords = shared_coords[shared_coords > 1].sum()
        else:
            num_shared_coords = 0
        return {
            'uniq_coords_visited': (self._grid_state > 0).sum(),
            'total_visit_counts': (self._grid_state.sum()),
            'num_agents_sharing_coords': num_shared_coords,
            'visit_counts_variance': self._grid_state.var(),
            'visit_counts_mean': self._grid_state.mean(),
        }

    def random_coordinate(self, n_coord: int) -> AbsLoc:
        return AbsLoc(
            np.apply_along_axis(
                np.random.randint, 0, np.array([self.grid_shape]),
                size=n_coord))

    def _relative_neighbors(self, nhops: int):
        return np.array(np.meshgrid(*(
            np.arange(-1 * nhops, nhops + 1) for _ in range(self.grid_ndim))))\
            .reshape(self.grid_ndim, -1).T

    def relative_neighbors(self, nhops: int):
        """Convenience function to shows relative distance
        of each neighboring coordinate from an origin point.

        This function is useful to lookup the way this environment orders
        nearby coordinates relative to some reference point.

        The number of neighbors returned is (2*nhops+1)**grid_ndim
        """
        return self._relative_neighbors(nhops).view(RelLoc)


    def neighbors(self, loc: AbsLoc, nhops: int) -> np.ndarray:
        '''
        Return the list of all coordinates neighboring each coordinate in the
        given location, up to n hops away.

        `loc` a location with one or more coordinates
        `nhops` how many units away in each dimension we consider

        Return array of absolute locations.  The return value has shape:
            (loc n_coords, num_neighbors_per_coord, grid_ndim)
            where:
                n_coords = number of coordinates in given `loc`
                num_neighbors_per_coord = (2*nhops+1)**grid_ndim
                grid_ndim = number of dimensions.

        - The returned set of coordinates represents a location that is the
        cartesian product of 2*nhops+1 positions for each dimension.
        - The total number coordinates returned are:
            n_coord_returned = (2*nhops+1)**ndim * loc.shape[0]
        - Will return duplicate coordinates if the input `loc` has coordinates
        that share neighbors.
        - A coordinate is a neighbor of itself.
        '''
        broadcasted_loc = np.expand_dims(loc, axis=1)
        relative_offsets = self._relative_neighbors(nhops=nhops)
        return (relative_offsets + broadcasted_loc) % self.grid_shape

    def visit_counts(self, loc: AbsLoc = None) -> np.ndarray:
        '''
        Show how many times each coordinate in given location has been visited
        If `loc` is None, return the count for all coordinates, where
            the first returned count value corresponds to coordinate at origin
            (0, ...), and the last one corresponds to the coordinate at the end
            of each dimension in the space (`grid_shape` - 1).
        '''
        if loc is None:
            return self._grid_state.ravel()
        return self._grid_state[list(loc.T)]

    def visit_counts_nearby(self, agent: AgentId, nhops: int) -> np.ndarray:
        '''
        Show many times each valid coordinate neighboring the agent has been
        visited.

        Agents may exist in many coordinates at once, and the nearby coordinates
        are relative to each coordinate in the current location of `agent`.
        There are (2*nhops+1)**ndim visit_counts returned for each successive
        coordinate of `agent`.

        Returns the visit count as an array of shape:
            (n_coordinates, 2*nhops+1)
        '''
        nei = self.neighbors(self._agents[agent], nhops=nhops)
        return self.visit_counts(AbsLoc(nei.reshape(-1, self.grid_ndim)))\
            .reshape(nei.shape[:-1])

    def _is_valid_location(self, node: AbsLoc) -> bool:
        '''
        Can an agent move to this location?
        '''
        in_bounds = ((node >= 0) & (node < self.grid_shape))  # type: np.array[int]
        in_bounds = in_bounds.all(axis=1)
        return in_bounds.view(np.ndarray)
        #  in_bounds = in_bounds * 2 - 1  # set True to 1, False to -1
        #  return in_bounds * node

    def _process_agent_move(self, agent: AgentId, node: RelLoc) -> None:
        '''
        manages updates to the grid state when an agent changes location
        '''
        # Calculate new position agent wishes to move to
        newabsloc = (self._agents[agent] + node) % self.grid_shape
        # This new position may be invalid.  Do not move to invalid positions.
        valid_locs = self._is_valid_location(newabsloc)
        newabsloc[~valid_locs] = \
            self._agents[agent][~valid_locs]
        # Move agent to the new position
        self._agents[agent] = newabsloc
        # Mark that we have visited another Location
        # fyi: in case where newabsloc has duplicate coordinates, this method
        # increases the visit_count at that coordinate by only 1.
        # and do not count visit if processing a relative move of 0 in all
        # directions.
        #  self._grid_state[tuple(newabsloc.T)] += 1
        no_visits = (node.view(np.ndarray) == 0).all(1)
        self._grid_state[tuple(newabsloc[~no_visits].T)] += 1

    def _process_agent_inbox(
            self, from_agent: AgentId,
            move: Dict[AgentId, RelLoc],
            msg: Dict[AgentId, Sequence[int]]) -> None:
        '''updates an agent's mailbox when it receives messages'''
        for to_agent, relloc in move.items():
            self._agent_inbox[to_agent]['move'][from_agent] = relloc
        for to_agent, msgv in msg.items():
            self._agent_inbox[to_agent]['msg'][from_agent] = msgv

    def process_request(
            self, agent: AgentId,
            move: Dict[AgentId, RelLoc]={},
            msg: Dict[AgentId, Sequence[int]]={}) -> None:
        '''
        Agents will call this to try to perform certain actions in the
        space.  This method does not return any feedback, so an agent can
        discover whether a request has been successful only by observing the
        environment with other methods.

        `agent` identifies the agent making this request
        `move` a dict specifying where agent would like to move itself and
            other agents relative to each agent's respective current position
          - key: agent B that we would like to ask to do something
          - value: what would we like to ask the other agent to do?
          Note: An agent should pass its own ID to move itself.
        `msg`: a dict allowing the agent to send a message to other agents.
        For instance, this field could define whether or not agent approved of
        other agent's previous request.
          - key: agent B, whom we would like to send a message
          - value: arbitrary numeric data.

        If an agent sends a message to another agent, and the other agent
        has not yet processed the message, the older message is overwritten.
        '''
        if agent in move:
            self._process_agent_move(agent, move.pop(agent))
        self._process_agent_inbox(agent, move, msg)

    def add_agent(self, agent: AgentId, pos: AbsLoc=None,
                  n_coord: int=1) -> None:
        '''Assign agent an absolute location in the grid.

        - `pos` - Defines the absolute location of the agent in the environment
          If undefined, a random location is chosen
        - `n_coord` - Defines how many coordinates in agent's location
          Not necessary if pos is defined
        '''
        if isinstance(agent, BaseAgent):
            raise UserWarning(
                'Must receive an AgentId.  Received %s' % type(agent))
        if pos is None:
            pos = self.random_coordinate(n_coord)
        if agent in self._agents:
            raise UserWarning("Already added this agent to the environment")
        self._agents[agent] = pos

    def agent_location(self, agent: AgentId) -> AbsLoc:
        '''Return the location of this agent in the environment'''
        return self._agents[agent]

    def agents_at_location(self, loc: AbsLoc = None) -> \
            Iterator[Tuple[AgentId, AbsLoc]]:
        '''
        Return the agents and their locations, or as many coordinates of the
        nearby agent that are shared in common with the given location.
        Locations may have many coordinates, and therefore agents may exist in
        many coordinates at once.

        If `loc` is not defined, return all agents and their locations
        '''
        if loc is None:
            return (x for x in self._agents.items())
        gen = ((a, AbsLoc.intersect(pos, loc, self.grid_shape))
               for a, pos in self._agents.items())
        return ((agent, pos) for agent, pos in gen if pos.size)

    def agent_locations_nearby(self, agent: AgentId, nhops: int) -> \
            Generator[Tuple[AgentId, np.ndarray], None, None]:
        '''
        Return the relative location of each nearby agents that is visible to
        `agent`.  Agents may exist in many coordinates at once, and the
        coordinates of nearby agents are identified as relative to one or more
        reference coordinates, as defined by current location of `agent`.

        fyi: An agent can also see itself in terms of relative distances
        between its coordinates.

        Returns generator of tuples in form:
            (nearby AgentId,
             bitmask specifying which neighbors contain that agent)

            The returned bitmask has the same shape
            as self.neighors(..., nhops).  That is:
                ( "n_coords in agent loc" , (2*nhops+1)**grid_ndim )
        '''
        agent_loc = self._agents[agent]
        nei = self.neighbors(agent_loc, nhops=nhops)
        # convert neighbors to an absolute location
        nei_abs = nei.reshape(-1, nei.shape[-1])
        for a2, a2loc in self._agents.items():
            # find neighbors that contain this agent
            nei_mask = AbsLoc.intersect_mask(nei_abs, a2loc, self.grid_shape)
            yield (a2, nei_mask.reshape(nei.shape[:-1]))

    def agent_inbox(self, agent: AgentId, flush: bool=True) -> _Inbox:
        '''
        Agents have an inbox that allows them to receive messages from other
        agents.  By default, an agent's inbox is emptied when it is read.
        Agents can use the inbox to receive knowledge or feedback from other
        agents.
        '''
        dct = self._agent_inbox[agent]
        if flush:
            self._agent_inbox[agent] = self._new_empty_inbox()
        return dct
