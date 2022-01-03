import numpy as np

from numpy.linalg import norm
from typing import List
from utils import Vec
from mac import MACArray

class Levelset(MACArray):

    INFINITY = 1e10

    def __new__(cls, data):
        # Data is a numpy-convertible collection of data, construct
        # it and get a view of our own subclass
        obj = np.asarray(data).view(cls)

        obj.off = Vec(0.5, 0.5)

        # Return the newly create object
        return obj

    def __array_finalize__(self, obj):
        # This function can be called in three distinct ways:
        # 1) We are in the middle of a MACField.__new__ call cascade
        # 2) A view of a MACField was requested via arr.view(MACField)
        # 3) A slice of a MACField was requested via MACarr[:slice]
        # In the first option, everything is already set, obj is None
        # and nothing needs to be done. In the remaining scenarios we
        # need to actually set the off field, as we did not pass through
        # the appropriate call cascade
        if obj is None: return

        self.off = getattr(obj, 'off', (0, 0))

    def update(self, P : List[Vec]) -> None:
        # Reset everything to infinity and prepare the index array
        self[...] = self.INFINITY
        tidx = np.ones_like(self, dtype = np.int) * -1

        # * Initialize the array over the input geometry
        # NOTE: We swap the rows and columns to adapt to C order
        #       in the gidx array
        pnts = np.array(P)[:, [1, 0]] 
        gidx = np.floor(pnts).astype(np.int)
        dist = norm(gidx - pnts, axis = 1)

        self[gidx[:, 0], gidx[:, 1]] = dist
        tidx[gidx[:, 0], gidx[:, 1]] = np.arange(len(pnts))

        # * Propagate the closest point distance estimate
        Idx = np.indices(self.shape).transpose(1, 2, 0)
        
        def _sweep(s, t1, t2, idx):
            mask = t1 >= 0
            dist = norm(idx[mask] - pnts[t1[mask]], axis = 1)

            dmask = dist < s[mask]

            s [mask] = np.where(dmask, dist, s[mask])
            t2[mask] = np.where(dmask, t1[mask], t2[mask])

        for args in zip(self[1:], tidx, tidx[1:], Idx[1:]):
            _sweep(*args)

        for args in zip(self[-2::-1], tidx[::-1], tidx[-2::-1], Idx[-2::-1]):
            _sweep(*args)

        for args in zip(self[:, 1:].T, tidx.T, tidx[:, 1:].T, Idx[:, 1:].transpose(1, 0, 2)):
            _sweep(*args)

        for args in zip(self[:, -2::-1].T, tidx[:, ::-1].T, tidx[:, -2::-1].T, Idx[:, -2::-1].transpose(1, 0, 2)):
            _sweep(*args)


        # for i in range(self.shape[0] - 1):
        #     mask = tidx[i] >= 0

        #     dist = norm(Idx[i+1, mask] - pnts[tidx[i, mask]], axis = 1)
            
        #     dmask = dist < self[i+1, mask]

        #     self[i+1, mask] = np.where(dmask, dist, self[i+1, mask]) 
        #     tidx[i+1, mask] = np.where(dmask, tidx[i, mask], tidx[i+1, mask])

        # for j in range(self.shape[1]):
            # mask = t1 >= 0
            # dist = norm(idx[mask] - pnts[t1[mask]], axis = 1)

            # dmask = dist < s[mask]

            # s[mask] = np.where(dmask, dist, s[mask])
            # t2[mask] = np.where(dmask, t1[mask], t2[mask])
            

        # for i in range(self.shape[0] - 1, 0, -1):
            # mask = tidx[i] >= 0

            # dist = norm(Idx[i-1, mask] - pnts[tidx[i, mask]], axis = 1)
            
            # dmask = dist < self[i-1, mask]

            # self[i-1, mask] = np.where(dmask, dist, self[i-1, mask]) 
            # tidx[i-1, mask] = np.where(dmask, tidx[i, mask], tidx[i-1, mask])

        # for j in range(self.shape[1] - 1):
        #     mask = tidx[:, j] >= 0

        #     dist = norm(Idx[mask, j+1] - pnts[tidx[mask, j]], axis = 1)

        #     dmask = dist < self[mask, j+1]

        #     self[mask, j+1] = np.where(dmask, dist, self[mask, j+1])
        #     tidx[mask, j+1] = np.where(dmask, tidx[mask, j], tidx[mask, j+1]) 

        # for j in range(self.shape[1] - 1, 0, -1):
        #     mask = tidx[:, j] >= 0

        #     dist = norm(Idx[mask, j-1] - pnts[tidx[mask, j]], axis = 1)

        #     dmask = dist < self[mask, j-1]

        #     self[mask, j-1] = np.where(dmask, dist, self[mask, j-1])
        #     tidx[mask, j-1] = np.where(dmask, tidx[mask, j], tidx[mask, j-1]) 

    
    # def _fnidx2d(self, idx : np.ndarray, K : int = 1) -> np.ndarray:
    #     def bound(idx): return np.clip(idx, 0, np.array(self.shape) - 1)

    #     line = range(-K, K + 1)
    #     nidx = [bound(idx + np.array((i, j))) for i in line 
    #             for j in line if abs(i) + abs(j) <= K]

    #     return np.unique(nidx, axis = 0)

    # def _fnidx2(self, idx : np.ndarray) -> np.ndarray:
    #     shape = np.array(self.shape).reshape(2, 1, 1)

    #     nidx = np.array([
    #         np.clip(idx + np.array([-1, 0]).reshape(2, 1, 1), 0, shape), # North
    #         np.clip(idx + np.array([+1, 0]).reshape(2, 1, 1), 0, shape), # South
    #         np.clip(idx + np.array([0, -1]).reshape(2, 1, 1), 0, shape), # West
    #         np.clip(idx + np.array([0, +1]).reshape(2, 1, 1), 0, shape)  # East
    #     ])

    #     return nidx