import numpy as np

from numpy.linalg import norm
from typing import List
from utils import Vec
from mac import MACArray

from math import sqrt

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

    def update(self, pnts : np.ndarray, r : float, rep : int = 2) -> None:
        # Reset everything to infinity and prepare the index array
        self[...] = self.INFINITY
        tidx = np.ones_like(self, dtype = np.int) * -1

        # * Initialize the array over the input geometry
        # NOTE: We swap the rows and columns to adapt to C order
        #       in the gidx array
        # pnts = np.array(P)[:, [1, 0]] 
        pnts = pnts[:, [1, 0]] - self.off
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

        for _ in range(rep):
            for args in zip(self[1:], tidx, tidx[1:], Idx[1:]):
                _sweep(*args)

            for args in zip(self[-2::-1], tidx[::-1], tidx[-2::-1], Idx[-2::-1]):
                _sweep(*args)

            for args in zip(self[:, 1:].T, tidx.T, tidx[:, 1:].T, Idx[:, 1:].transpose(1, 0, 2)):
                _sweep(*args)

            for args in zip(self[:, -2::-1].T, tidx[:, ::-1].T, tidx[:, -2::-1].T, Idx[:, -2::-1].transpose(1, 0, 2)):
                _sweep(*args)

        # * Subtract the particle radius from the levelset
        self -= r

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

    def lift(self, lvst : np.ndarray, lv : float = 1.) -> None:
        '''
            This function sets the values identified as the negative values
            of by the provided Levelset to a lifted-value lv. This is used
            to impose that a Fluid-Levelset takes positive values where a
            Solid-Levelset is negative, thus ensuring a coherent description
            of the state of the system.
        '''
        self[lvst < 0] = np.maximum(self[lvst < 0], lv)

    def redistance(self, dx : float = 1.) -> None:
        '''
            This function exploits the Eikonal equation to refine the initial
            levelset particle-based construction. It is only needed where
            the levelset is negative.
        '''
        mask = self < 0

        Idx = np.indices(self.shape).transpose(1, 2, 0)[mask]

        # We should be guaranteed that we do not hit any out-of-bounds
        # index in neighbors as we are only inspecting the fluid interior
        nIdx = np.stack([Idx + np.array([0, -1]), Idx + np.array([-1, 0])])

        def _update(idx, nidx):
            # Sort the neighbors based on their values on the levelset
            phis = np.sort(self[tuple(nidx.T)])
            
            d = phis[0] + dx
            if d > phis[1]:
                d = .5 * (phis[0] + phis[1] + sqrt(2 * dx**2 - (phis[1] - phis[0])**2))

            self[tuple(idx)] = min(self[tuple(idx)], d)

        [_update(idx, nidx) for idx, nidx in zip(Idx, nIdx)]

    def fill_holes(self, rep : int = 1) -> None:
        idx = np.indices(self.shape)

        left, right = np.array([[[0]], [[-1]]]), np.array([[[0]], [[1]]])
        top, bottom = np.array([[[1]], [[0]]]), np.array([[[-1]], [[0]]])

        for _ in range(rep):
            avg = 0.25 * (self[tuple(np.maximum(idx + left, 0))] + 
                        self[tuple(np.minimum(idx + right, self.shape[1] -1))] +
                        self[tuple(np.maximum(idx + bottom, 0))] +
                        self[tuple(np.minimum(idx + top, self.shape[0] - 1))])
            
            self[...] = np.minimum(self, avg)

    def closest(self, Psc : np.ndarray, eps : float = 1e-2, max_iter : int = 1000) -> np.ndarray:
        '''
            Find the set of closest point on this Levelset surfact from
            a set of arbitrary points.
        '''
        Ps = Psc.copy().reshape(-1, 2)
        Ny, Nx = np.gradient(self)

        nablaX = MACArray(Nx, off = self.off)
        nablaY = MACArray(Ny, off = self.off)

        Fs = self(Ps).reshape(-1, 1)
        
        Ds = np.vstack([nablaX(Ps), nablaY(Ps)]).T

        Qs = np.empty(Ps.shape)
        
        alphas = np.ones((len(Fs), 1))
        active = np.ones(len(Ps), dtype = np.bool)

        # No not move point that are already inside
        active[Fs.squeeze() < 0] = False

        t = 0
        while active.any() and t < max_iter:
            # Get only active points
            Qs[active] = Ps[active] - alphas[active] * Fs[active] * Ds[active]

            mask = np.abs(self(Qs)) < np.abs(Fs).squeeze()
            mask = np.logical_and(mask, active)

            Ps[mask] = Qs[mask]
            Fs[mask] = self(Qs[mask]).reshape(-1, 1)
            Ds[mask, 0] = nablaX(Qs[mask])
            Ds[mask, 1] = nablaY(Qs[mask])

            # Mark the point that have reached convergence
            active = np.logical_and(active, np.abs(Fs).squeeze() > eps)

            alphas[~mask] = 0.7 * alphas[~mask]

            t += 1

        if active.any():
            raise ValueError('Convergence failed for some point in closest.')

        return Ps

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