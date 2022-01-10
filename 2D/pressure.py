import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import factorized, cg
from operator import mul
from functools import reduce
from typing import Tuple

from levelset import Levelset
from math import sqrt

class Solver:
    '''
        This class implements a Pressure Solver for a 2D incompressible
        fluid system. Internally it stores the system matrix A efficiently
        as a sparse matrix, and when called it solves the linear system that
        expects the rhs to be the field divergence.
    '''

    def __init__(self, shape : Tuple[int, int]) -> None:
        self.shape = shape
        self.size  = reduce(mul, shape)

        # Construct the A fluid system matrix and its preconditioner M
        self.A = sp.lil_matrix((self.size, self.size))
        self.M = None

    def __call__(self, rhs : np.ndarray) -> np.ndarray:
        '''
            This call solves the linear system associate to the current status
            of the A matrix (which should have been prepared prior to the solver
            call for correct solution).
        '''

        # solve = factorized(self.A.tocsc())
        # return solve(rhs)

        p0 = np.zeros(rhs.size)
        p, info = cg(self.A.tocsc(), rhs.ravel(), x0 = p0, M = self.M, tol = 1e-5)

        return p.reshape(*self.shape), info

    def prepare(self, fluid, solid, scale : float) -> None:
        '''
            This function updates the internal sparse matrix A that represents
            the fluid system to adapt it to the current situation, thus preparing
            the object for the subsequent project call.
        '''

        fmask = fluid < 0
        smask = solid > 0

        # The diagonal entries are equal to the number of non-solid neighbors
        fshape = self.shape[0], self.shape[1]
        idxs = np.indices(fshape)

        left, right = np.array([[[0]], [[-1]]]), np.array([[[0]], [[1]]])
        top, bottom = np.array([[[1]], [[0]]]), np.array([[[-1]], [[0]]])

        # Compute the number of non-solid neighbors to a given idx
        nsnb = smask[tuple(np.maximum(idxs + left, 0))].astype(np.int) +\
               smask[tuple(np.minimum(idxs + right, self.shape[1] - 1))].astype(np.int) +\
               smask[tuple(np.minimum(idxs + top, self.shape[0] - 1))].astype(np.int) +\
               smask[tuple(np.maximum(idxs + bottom, 0))].astype(np.int)

        # Set to zero the idxs that are not fluid voxels
        self.Adiag = (fmask * nsnb).ravel() * scale

        # self.Adiag = [nonsolid(idx) * scale for idx in np.ndindex(self.shape)]

        # The AplusX entries are equal to the -scale constant if the positive
        # x neighbor is a fluid voxel, otherwise they are kept to zero
        rshape = self.shape[0], self.shape[1] - 1
        tshape = self.shape[0] - 1, self.shape[1]
        self.AplusX = [-scale if (fmask[idx] and fmask[idx[0], idx[1] + 1]) else 0 
                    for idx in np.ndindex(rshape)]
        self.AplusY = [-scale if (fmask[idx] and fmask[idx[0] + 1, idx[1]]) else 0
                    for idx in np.ndindex(tshape)]

        w = self.shape[0]
        ridx = np.ravel_multi_index(np.indices(rshape), self.shape).ravel()
        tidx = np.ravel_multi_index(np.indices(tshape), self.shape).ravel()

        self.A.setdiag(self.Adiag)

        self.A[ridx, ridx + 1] = self.AplusX
        self.A[ridx + 1, ridx] = self.AplusX

        self.A[tidx, tidx + w] = self.AplusY
        self.A[tidx + w, tidx] = self.AplusY        
        
    def precondition(self, fluid : Levelset, tau : float = 0.97, sigma : float = 0.25) -> None:
        '''
            This function compute the preconditioner to the A system matrix
            useful for speeding up the computation of the iterative Conjugate
            Gradient algorithm.
        '''
        # Prepare the preconditioner
        _, w = self.shape
        self.M = sp.lil_matrix((self.size, self.size))
    
        # Only update fluid region
        mask = fluid < 0
        ridx = np.ravel_multi_index(np.indices(self.shape)[:, mask], self.shape)

        for idx in ridx:
            e = self.Adiag[idx] - (self.AplusX[idx - 1] * self.M[idx - 1])**2\
                                - (self.AplusY[idx - w] * self.M[idx - w])**2\
                    - tau * (self.AplusX[idx - 1] * self.AplusY[idx - 1] * self.M[idx - 1]**2 +
                             self.AplusY[idx - w] * self.AplusX[idx - w] * self.M[idx - w]**2)

            e = self.Adiag[idx] if e < sigma * self.Adiag[idx] else e

            self.M[idx] = 1 / sqrt(e)

