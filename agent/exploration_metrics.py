import numpy as np


class StateVisitationTracker:
    """Tracks the fraction of a discretized state space visited so far.

    A continuous state space has infinitely many states, so "percentage of
    possible states visited" only makes sense once the space is discretized
    into a finite grid. Pick a small subset of dimensions and bin counts, or
    total_possible_states explodes and coverage stays at ~0% no matter how
    much exploration actually happens.
    """

    def __init__(self, low, high, bins=10, dims=None):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)

        self.dims = list(range(len(low))) if dims is None else list(dims)
        self.low = low[self.dims]
        self.high = high[self.dims]

        bins = np.asarray(bins)
        self.bins = np.full(len(self.dims), bins, dtype=np.int64) if bins.ndim == 0 else bins.astype(np.int64)

        self.total_possible_states = int(np.prod(self.bins))
        self.visited = set()

    def _discretize(self, state):
        state = np.asarray(state, dtype=np.float64)[self.dims]
        clipped = np.clip(state, self.low, self.high)
        span = np.where(self.high > self.low, self.high - self.low, 1.0)
        normalized = (clipped - self.low) / span
        bin_idx = np.minimum((normalized * self.bins).astype(np.int64), self.bins - 1)
        return tuple(bin_idx.tolist())

    def update(self, state):
        self.visited.add(self._discretize(state))

    @property
    def unique_states_visited(self):
        return len(self.visited)

    @property
    def coverage(self):
        return self.unique_states_visited / self.total_possible_states
