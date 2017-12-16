import abc
from typing import Sequence, Union, NewType, TypeVar, Generic
import numpy as np


class Location(np.ndarray, metaclass=abc.ABCMeta):
    '''
    Represents a location as a collection of spatial integer coordinates
    where each coordinate references a specific position in n-dimensional space.
    A Location is a numpy array in form [coord1, coord2, ...]
    '''

    def __new__(
            cls, loc: Union[np.ndarray, Sequence[int], Sequence[Sequence[int]]],
            ndim: int=None) -> 'Location':
        '''
        - `loc` - one or more integer coordinates, each representing a point in
            space.
        - `ndim` - an optional number specifying number of dimensions a
        coordinate represents.  Only useful to sanity check your input.
          In 2D space: (x,y) or [(x,y), ...]
          In 3D space, (x,y,z) or [(x,y,z), ...]
        '''
        if isinstance(loc, np.ndarray):
            rv = loc  # type: np.ndarray
            if len(rv.shape) == 1:
                rv = rv.reshape(1, -1)
        else:
            rv = np.array(loc, dtype='int64', ndmin=2)
        if ndim is not None:
            assert rv.shape[-1] == ndim, \
                "%s(...) received coordinates with %s dimensions,"\
                "but expected %s dimensions." % (
                    cls.__name__, rv.shape[-1], ndim)
        return rv.view(cls)

    def __array_finalize__(self, loc):
        assert len(loc.shape) == 2, \
            "Location should have 2 dimensions.  Found %s with shape %s" % (
                len(loc.shape), loc.shape)

    @staticmethod
    def intersect(loc1: 'Location', loc2: 'Location', grid_shape: Sequence[int]
                  ) -> 'Location':
        '''
        Given two locations, find the points in common.

        http://stackoverflow.com/questions/9269681/intersection-of-2d-numpy-ndarrays
        '''
        mask = Location.intersect_mask(loc1, loc2, grid_shape)
        return loc1[mask]

    @staticmethod
    def intersect_mask(loc1: 'Location', loc2: 'Location',
                       grid_shape: Sequence[int]) -> np.ndarray:
        '''
        Return bitmask specifying which coordinates of loc1 are also in loc2
        '''
        if loc1.shape[1] != loc2.shape[1]:
            raise UserWarning(
                "Cannot intersect on locations containing"
                "coordinates of different num dimensions."
                "\nloc1=%s\nloc2=%s" % (loc1, loc2))
        return np.in1d(
            np.ravel_multi_index(loc1.T, grid_shape),
            np.ravel_multi_index(loc2.T, grid_shape)
        )


class AbsLoc(Location):
    '''An Absolute Location in n-dimensional space

    To refer to an absolute location in a space, the space should define the
    reference position.  For instance, a 2D Grid Space might refer to the "top
    left" corner of the space  with GridNode((0,0))
    '''
    pass


class RelLoc(Location):
    '''A Relative Location in n-dimensional space

    All locations are relative to some reference point.  This reference point
    is typically an agent location.  A relative location in a 2D
    space such as GridNode((0, 1)) means "to the right one space."

    We might not know exactly where in GridSpace this we are.
    '''
    pass


class BaseEnvironment(object, metaclass=abc.ABCMeta):
    '''
    A virtual world that agents can interact with.

    All environments should subclass from this one.
    '''
    def __repr__(self) -> str:
        return '%s' % (self.__class__.__name__)

AgentId = NewType('AgentId', int)
_BaseEnvironment = TypeVar('_BaseEnvironment', bound=BaseEnvironment)


class BaseAgent(Generic[_BaseEnvironment], metaclass=abc.ABCMeta):
    '''
    An agent interacts with the environment to perform actions or receive
    sensory input.  It is identifiable by an integer id.
    '''

    def __repr__(self) -> str:
        return '%s:%s' % (self.__class__.__name__, id(self))

    @property
    def ID(self) -> AgentId:
        return AgentId(id(self))

    @abc.abstractmethod
    def next_action(self, env: _BaseEnvironment) -> None:
        '''
        This method is the primary way an agent interacts with the environment.
        '''
        NotImplemented
