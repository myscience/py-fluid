import numpy as np
from numpy.lib.index_tricks import nd_grid
import utils

from scipy.interpolate import griddata

from math import floor, ceil
from typing import Tuple, Any, List

from itertools import starmap
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

from utils import Vec

class MACArray(np.ndarray):
    '''
        Basic implementation of a Marker-And-Cell (MAC) staggered grid.

        Examples:
            p[i, j, k] = p_i,j,k
            u[i, j, k] = u_i-1/2,j,k
            v[i, j, k] = v_i,j-1/2,k
            w[i, j, k] = w_i,j,k-1/2
    '''

    def __new__(cls, data, off : Vec = None):
        # Data is a numpy-convertible collection of data, construct
        # it and get a view of our own subclass
        obj = np.asarray(data).view(cls)

        # Add the offset parameter to out augmented array
        off = Vec() if off is None else off
        if abs(off.x) > 1 or abs(off.y) > 1:
            msg = f'MACArray only support offset in range (-1, 1). Got {off}'
            raise ValueError(msg)
        obj.off = off

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

    @classmethod
    def zeros(cls, shape : Tuple [int, int], off : Tuple[int, int] = (0, 0)) -> np.ndarray:
        # Create the underlying zeros data
        data = np.zeros(shape)

        # Return the correct construction of the MAC field
        return cls.__new__(cls, data, off)

    @classmethod
    def const(cls, shape : Tuple[int, int], value : float, off : Tuple[int, int] = (0, 0)) ->np.ndarray:
        # Create the underlying const-value data
        data = np.ones(shape) * value

        # Return the correct construction of the MAC field
        return cls.__new__(cls, data, off)

    def __call__(self, P : np.ndarray) -> np.ndarray:
        '''
            Fast interpolation of values represented by this array
            on continous positions identified by the Vec array P. 
        '''

        key = P - self.off
        key = np.clip(key, [0] * len(self.shape), self.shape)[:, [1, 0]]

        # Create the interpolation object 
        x, y = np.arange(self.shape[0]), np.arange(self.shape[1])

        f = RectBivariateSpline(x, y, self)
        return f(*key.T, grid = False).squeeze()

        # x, y = np.arange(self.shape[1]), np.arange(self.shape[0])
        # f = interp2d(x, y, self, kind = interp)

        # Run the interpolation for every point provided
        # return np.array(list(starmap(f, key))).squeeze()

    def at(self, *args: Any, interp = 'linear') -> List[float]:
        '''
            Calling the at function is used to inspect the array in a
            continuous location, using interpolation to return non-grid
            values and taking into account the grid field offset. 
            NOTE: This is a SLOW function call because it tries to parse
                  possibly different arguments. Fast simulations should
                  use the more narrowly-scoped __call__ method of the class.
        '''
        # Here we dispatch the different ways this function can be called:
        # 1) 2(3)-arguments, interpreted as x, y, (z) coordinates of a single point
        # 2) A single tuple of correct length, interpreted as the coordiantes of a point
        # 3) A tuple|list of tuples|lists, interpreted as coordinates of multiple points
        if len(args) > 0 and len(args) == len(self.shape):
            pos = Vec(*args)
            key = [(pos - self.off)]

        elif isinstance(args[0],    (tuple, list)) and\
             isinstance(args[0][0], (int, float)):
            msg = 'Shape mismatch in grid interpolation. Wrong dimensions in tuple.'
            assert len(args[0]) == len(self.shape), msg

            pos = Vec(*args[0])
            key = [(pos - self.off)]

        elif isinstance(args[0], (tuple, list, np.ndarray)) and\
             isinstance(args[0][0], (tuple, list, np.ndarray)):
            msg = 'Shape mismatch in grid interpolation. Wrong dimensions in tuple list.'
            assert np.all([len(arg) == len(self.shape) for arg in args[0]]), msg

            pos = [Vec(*arg) for arg in args[0]]
            key = [(p - self.off) for p in pos]

        else:
            raise ValueError(f'Invalid type for field interpolation. Got {args}')

        key = np.clip(key, [0] * len(self.shape), self.shape)[:, [1, 0]]

        # Create the interpolation object 
        x, y = np.arange(self.shape[0]), np.arange(self.shape[1])

        f = RectBivariateSpline(x, y, self)
        return f(*key.T, grid = False).squeeze()

    # Taken directly from: 
    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    def fastfit(self, values, vtx, wts, fill_value = np.nan) -> None:
        ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
        ret[np.any(wts < 0, axis = 1)] = fill_value
        
        self[...] = ret.reshape(self.shape)

    def gridfit(self, P : np.ndarray, values : List[Vec]) -> None:
        # We make use of the scipy griddata interpolation to translate
        # the values of the point back into the grid
        points = P - self.off

        x, y = np.arange(self.shape[1]), np.arange(self.shape[0])
        X, Y = np.meshgrid(x, y)

        self[...] = griddata(points, values, (X, Y), method = 'linear')

    def fit(self, P, key : str, comp : str = None, nidx2 : np.ndarray = None) -> None:
        # Prepare the empty new MACArray and the weight grid
        newself = self.zeros(self.shape, off = self.off)
        weights = np.zeros(self.shape)

        # * Get the collection of particles location and subtract the grid offset
        Pos = np.clip(P.locate(), 0, np.array(self.shape) - 1)
        Val = [getattr(p.pld[key], comp) if comp else p.pld[key] for p in P]
        
        Grd = np.floor(Pos).astype(np.int)

        # Pre-compute the subarrays of nearest neighbors indices for faster lookup
        if nidx2 is None:
            idx = np.indices(self.shape)
            nidx2 = self.fnidx2d(self.shape, idx, K = 2)

        # We iterate through the particles to update the grid where needed
        # def update(p, gidx):
        for p, v, gidx in zip(Pos, Val, Grd):
            # We update only relevant grid positions, close enough to considered particle
            # nidx = self._fnidx2d(gidx, K = 2)
            # ? DO WE NEED TO FIND THE UNIQUE VALUES?
            nidx = np.unique(nidx2[...,gidx[0], gidx[1]], axis = 0)

            pos = p - self.off

            k = self._h2(pos.x - nidx[..., 1]) * self._h2(pos.y - nidx[..., 0])

            weights[nidx[:, 0], nidx[:, 1]] += k
            newself[nidx[:, 0], nidx[:, 1]] += k * v
        # [update(p, gidx) for p, gidx in zip(P, Grd)]

        self[...] = newself.copy() / (weights + 1e-10)

        # * Extrapolate to unknown values. Known values are wherever
        # * the weights came up as non-zero.
        self.extrapolate(weights == 0)

    def extrapolate(self, marker : np.ndarray, max_iter = 50000) -> None:
        MAX_V = 100000

        # No need to extrapolate for an all-known grid
        if not np.any(marker): return

        # Prepare an integer helper array
        marker = marker.astype(np.int) * MAX_V

        # Get the grid indices and corresponding neighbors
        Idx = np.indices(self.shape).transpose(1, 2, 0)
        nIdx = self._fnidx2d(Idx).transpose(1, 2, 0, 3)

        # * This is the wavefront list, which is initialized with
        # * unknwon indices for which a known neighbour exists
        # W = [tuple(wfidx) for wfidx in Idx.reshape(-1, 2) 
        #     if marker[tuple(wfidx)] > 0 and np.any(marker[tuple(self._fnidx2d(wfidx).T)] == 0)]
        
        W = [tuple(wfidx) for wfidx, nidx in zip(Idx.reshape(-1, 2), nIdx.reshape(-1, 5, 2)) 
            if marker[tuple(wfidx)] > 0 and np.any(marker[tuple(nidx.T)] == 0)]
        marker[tuple(np.array(W).T)] = 1

        # * Propagate the wavefront information to unknown values
        t = 0
        while t < len(W):
            idx = W[t]
            # nidx = self._fnidx2d(idx)
            nidx = nIdx[tuple(idx)]

            # Set the value to the average of better known values
            mask = marker[tuple(nidx.T)] < marker[idx]
            self[idx] = np.mean(self[tuple(nidx[mask].T)]) 
            
            # Propagate the wavefront further
            mask = marker[tuple(nidx.T)] == MAX_V
            marker[tuple(nidx[mask].T)] = marker[idx] + 1
            W += [tuple(idx) for idx in nidx[mask]]
            t += 1

            assert t < max_iter, 'Runaway while loop'

    def _h2(self, R : np.ndarray) -> np.ndarray:
        out = np.zeros_like(R)
        
        mask1 = np.logical_and(R >= -1.5, R < -0.5)
        mask2 = np.logical_and(R >= -0.5, R < +0.5)
        mask3 = np.logical_and(R >= +0.5, R < +1.5)

        out[mask1] = 0.5 * (R[mask1] + 1.5)**2
        out[mask2] = 0.75 - R[mask2]**2
        out[mask3] = 0.5 * (1.5 - R[mask3])**2

        return out

    # def _h2(self, r : float) -> float:
    #     if r >= -1.5 and r < -.5: return 0.5 * (r + 1.5)**2
    #     elif r >= -.5 and r < .5: return 0.75 - r**2
    #     elif r >= .5 and r < 1.5: return 0.5 * (1.5 - r)**2
    #     else: return 0.

    @classmethod
    def fnidx2d(cls, shape : Tuple[int, int], idx : np.ndarray, K : int = 1) -> np.ndarray:
        def bound(idx): return np.clip(idx, 0, (np.array(shape) - 1).reshape(2, 1, 1))

        line = range(-K, K + 1)
        nidx = [bound(idx + np.array([[[i]], [[j]]])) for i in line 
                for j in line if abs(i) + abs(j) <= K]

        return np.unique(nidx, axis = 0)

    @classmethod
    def fnidx1d(cls, shape : Tuple[int, int], idx : np.ndarray, K : int = 1) -> np.ndarray:
        def bound(idx): return np.clip(idx, 0, (np.array(shape) - 1).reshape(2, 1, 1))

        h, w = shape
        line = range(-K, K + 1)
        nidx = [bound(idx + i + j * w) for i in line for j in line if abs(i) + abs(j) <= K]

        return np.unique(nidx, axis = 0)


    def _fnidx2d(self, idx : np.ndarray, K : int = 1) -> np.ndarray:
        # def bound(idx): return np.clip(idx, 0, np.array(self.shape) - 1)

        # line = range(-K, K + 1)
        # nidx = [bound(idx + np.array((i, j))) for i in line 
        #         for j in line if abs(i) + abs(j) <= K]

        # return np.unique(nidx, axis = 0)
        def bound(idx): return np.clip(idx, 0, np.array(self.shape) - 1)

        line = range(-K, K + 1)
        nidx = [bound(idx + np.array((i, j))) for i in line 
                for j in line if abs(i) + abs(j) <= K]

        return np.unique(nidx, axis = 0)
