from typing import Sequence, Union
from engine.base_classes import BaseAgent, RelLoc
from engine.environments import CartesianEnvironment
import numpy as np
import keras


class RandomAgent(BaseAgent):
    def __init__(self, n_coord: int, walk: bool=False) -> None:
        self.n_coord = n_coord
        self.walk = walk

    def next_action(self, env: CartesianEnvironment) -> None:
        if self.walk:
            move = RelLoc(np.random.randint(
                -1, 2, (self.n_coord, env.grid_ndim)))
        else:
            move = env.random_coordinate(self.n_coord)
        env.process_request(self.ID, move={self.ID: move})


class DeterministicAgent(BaseAgent):
    '''
    A simple agent that moves in regular repeating sequence.
    It does nothing else

    Some examples:
    - A boring agent that moves in a square.  It could be distributed if
    env.add_agent specifies a starting location with many starting coordinates,
    and this would work because each relative location in the move sequence
    defined below will get broadcasted to all coordinates.  In this case, each
    distributed part of the agent does the same thing.
    >>> a = DeterministicAgent(RelLoc([(1,0), (0,1), (-1, 0), (0, -1)], 2))
    >>> env.add_agent(a, AbsLoc([(0,0), (10,0)]))  # agent with 2 coordinates

    - A distributed agent that has 500 coordinates in 2D space.
      Each coordinate has its own random sequence.
      >>>  DeterministicAgent(np.random.randint(-1, 2, (100, 500, 2)))
    >>> env.add_agent(a, n_coord=500)
    '''

    def __init__(self, move_sequence: Sequence[RelLoc]) -> None:
        #  self._pos = start_position
        self._seq = move_sequence
        self._seqidx = 0

    def next_action(self, env: CartesianEnvironment) -> None:
        '''
        Very simple agent that moves in a repeating pattern.
        '''
        move = RelLoc(self._seq[self._seqidx])
        self._seqidx = (1 + self._seqidx) % len(self._seq)
        env.process_request(self.ID, move={self.ID: move})


class SeekAnyLeastVisitedAgent(BaseAgent):
    '''Simple greedy agent that simply goes to the first least visited
    neighbor coordinate it sees within 1 hop of its current location.
    '''
    def __init__(self, mode='argmin'):
        '''
        - `mode` either 'argmin' or 'sample'
          - "argmin" says to choose the next move as one with min number of
            visit counts. breaks ties by choosing first value
          - "argmin_sample_ties" same as argmin, but break ties randomly
        '''
        self.mode = mode

    def next_action(self, env: CartesianEnvironment) -> None:

        vc = env.visit_counts_nearby(self.ID, 1)

        if self.mode == 'argmin_sample_ties':
            mask = vc == vc.min(axis=1).reshape(-1, 1)
            numties = mask.sum(axis=1)

            idxs = np.empty_like(numties)
            # find argmin where there are no ties.
            idxs[numties == 1] = vc[numties == 1].argmin(axis=1)
            # where there are ties, sample a possible minimum uniformly
            probs = (1 / numties[numties > 1])\
                .repeat(vc.shape[1]).reshape(-1, vc.shape[1])
            probs[~mask[numties > 1]] = 0
            idxs[numties > 1] = \
                (probs.cumsum(axis=1) > np.random.sample((probs.shape[0], 1)))\
                .argmax(axis=1)
        elif self.mode == 'argmin':
            idxs = vc.argmin(axis=1)
        else:
            raise NotImplemented('unrecognized mode: %s' % self.mode)
        move = env.relative_neighbors(1)[idxs]
        env.process_request(
            self.ID, move={self.ID: move})


class KerasAgent(BaseAgent):
    '''
    This agent uses a neural network to choose its next move
    Pass in a pre-trained model, predict next move for each coordinate, move
    Assume:
        - model input is receives visit_counts_nearby(nhops=1)
        - model output is 9 values whose indexes correspond with those for
        relative_neighbors(nhops=1)
    '''
    def __init__(self, keras_model_or_fp: Union[str, keras.models.Model],
                 mode: str='sample') -> None:
        '''
        - `keras_model_or_fp` - filepath to an h5 file containing a keras model,
            or an instance of a model itself.
        - `mode` - 'argmax' or 'sample'. determines how to choose an output.
          - "argmax" says to choose the next move as one with max probability.
            breaks ties by choosing first value
          - "sample" says to choose the next move by treating predictions as
            a probability distribution and sampling from that distribution
        '''
        if isinstance(keras_model_or_fp, keras.models.Model):
            self.mdl = keras_model_or_fp  # type: keras.models.Model
        else:
            self.mdl = keras.models.load_model(keras_model_or_fp)
        self.mode = mode
        self.last_mdl_output = None  # type: np.ndarray
        self.last_mdl_input = None  # type: np.ndarray

        # assume model has a fixed input size that matches the number of nearby
        # visit_counts nhops away.
        self._nhops_in = None  # type: int

    def _next_action_model_input(self, env: CartesianEnvironment) -> np.ndarray:
        if self._nhops_in is None:
            self._nhops_in = int(
                (np.sqrt(self.mdl.input_shape[1]) - 1) / env.grid_ndim)
        X = env.visit_counts_nearby(self.ID, self._nhops_in)
        return X

    def next_action(self, env: CartesianEnvironment) -> None:
        # generate probabilities for each possible next move
        X = self._next_action_model_input(env)
        target = self.mdl.predict(X)
        self.last_mdl_input = X
        self.last_mdl_output = target
        if np.isnan(target).any():
            raise Exception("Nulls in predicted output")
        if self.mode == 'sample':
            # treat output indices as probability distribution and sample it.
            # sample next move from the probability distribution
            cs = target.cumsum(axis=1)
            idxs = (cs > (np.random.sample((target.shape[0], 1)) * cs[:, -1:])
                    ).argmax(axis=1)
        elif self.mode == 'argmax':
            idxs = target.argmax(axis=1)
        else:
            raise NotImplemented('unrecognized mode: %s' % self.mode)
        # process next move
        move = env.relative_neighbors(1)[idxs]
        env.process_request(self.ID, move={self.ID: move})
