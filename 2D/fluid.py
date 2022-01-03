import numpy as np
from numpy.core.fromnumeric import shape

from PIL import Image

from utils import Vec
from mac import MACArray
from levelset import Levelset
from particles import Swarm
from pressure import Solver

from typing import List, Tuple

class Fluid:
    '''
        This is the overarching class that represents a full-fledged fluid.
    '''

    UNKNWN = 1000000
    KNWN = 0

    def __init__(self, shape : Tuple[int, int], 
            fls : Levelset, 
            sls : Levelset, 
            ppc : int = 4,  # Particles per Cell
            rho : float = 1.,
            dx  : float = 1.,
            pl  : dict = None
            ) -> None:
        w, h = shape
        self.shape = (h, w)

        self.fls = fls
        self.sls = sls

        self.rho = rho
        self.dx  = dx

        # Prepare the velocity field components
        self.U = MACArray.zeros((h, w + 1), off = Vec(0., 0.5))
        self.V = MACArray.zeros((h + 1, w), off = Vec(0.5, 0.))

        # Initialize the pressure solver
        self.solver = Solver(self.shape)

        # Initialize the particle swarm inside the fluid region
        rng = np.random.default_rng()

        idx = np.indices(self.shape)[:, fls < 0]
        pos = [Vec(x + rng.random(), y + rng.random()) for y, x in idx.T for _ in range(ppc)]

        self.S = Swarm((0, w, 0, h), len(pos), pos = pos, vel = Vec(0., 0.), payload = pl)

    def step(self, dt : float) -> None:
        self.dt = dt

        # * Update the fluid levelset based on new particle location
        # self.fls.update(self.S.locate())
        # self.fls -= 0.9

        # * Trasfer particles value to the grid
        self.U.gridfit(self.S, 'vel', 'x')
        self.V.gridfit(self.S, 'vel', 'y')
        # self.U.fit(self.S, 'vel', 'x')
        # self.V.fit(self.S, 'vel', 'y')

        # Extrapolate data to unknown positions
        self.U.extrapolate(np.isnan(self.U))
        self.V.extrapolate(np.isnan(self.V))

        # * Set the velocities to zero on solid boundaries
        self.enforce_boundary(self.sls)

        # * Compute the divergence of the velocity field
        div = self.div([self.U, self.V], self.fls, 1. / self.dx) 

        # * Compute the pressure term
        # Prepare the new system A matrix
        self.solver.prepare(self.fls, self.sls, self.dt / (self.rho * self.dx**2))

        # Solve the pressure system using Conjugate Gradient
        # ? NOTE: We can improve here by providing the correct preconditioner!
        p, info = self.solver(div)

        if info > 0:
            self.p = p
            import matplotlib.pyplot as plt

            plt.imshow(p)
            plt.show()

        assert info == 0, info

        # * Project the velocity field so to make them divergence-free
        newU, newV = self.project(p, self.fls, self.sls)

        # * Trasfer value back to particles using PIC|FLIP update
        self.S.picflip(self.U, newU, 'vel', 'x')
        self.S.picflip(self.V, newV, 'vel', 'y')
        # self.S.fit(newU, 'vel', 'x')
        # self.S.fit(newV, 'vel', 'y')

        self.U[...] = newU.copy()
        self.V[...] = newV.copy()

        # * Advect the particles around
        self.S.advect(dt, self.dx, self.U, self.V)

    def screenshot(self, dpi = 200):
        '''
            This function is used to create a screenshot of the current fluid
            condition.
        '''

        imgR = MACArray.zeros(self.shape, off = Vec(0.5, 0.5))
        imgG = MACArray.zeros(self.shape, off = Vec(0.5, 0.5))
        imgB = MACArray.zeros(self.shape, off = Vec(0.5, 0.5))

        imgR.gridfit(self.S, 'ink', 'r')
        imgG.gridfit(self.S, 'ink', 'g')
        imgB.gridfit(self.S, 'ink', 'b')
        imgs = [imgR, imgG, imgB]

        return Image.fromarray(np.stack(imgs, axis = -1).astype(np.uint8))
    

    def sprinkle(self, newS : Swarm):
        self.S.merge(newS)

    def project(self, p : np.ndarray, fls, sls) -> None:
        scale = self.dt / (self.rho * self.dx)

        idx = np.indices(self.shape)

        # * Update only U voxels for which voxel (i, j) is FLUID or (i - 1, j) is fluid
        # NOTE: Remember that to go left we need to move along the column dimension, which is
        #       the second one according to the numpy storage convention 
        left, bottom = np.array([[[0]], [[-1]]]), np.array([[[-1]], [[0]]])

        ufidx = np.logical_or(fls[tuple(idx)] < 0, fls[tuple(idx + left  )] < 0)
        vfidx = np.logical_or(fls[tuple(idx)] < 0, fls[tuple(idx + bottom)] < 0)

        usidx = np.logical_or(sls[tuple(idx)] < 0, sls[tuple(idx + left  )] < 0)
        vsidx = np.logical_or(sls[tuple(idx)] < 0, sls[tuple(idx + bottom)] < 0)

        # Prepare flag arrays for known where to extrapolate U and V
        mU, mV = np.ones_like(self.U) * self.UNKNWN, np.ones_like(self.V) * self.UNKNWN

        # Erase the velocity fields with a known value to mark an UNKWN state
        mUmask = np.logical_or(ufidx, usidx)
        mVmask = np.logical_or(vfidx, vsidx)
        mU[:, :-1][mUmask] = self.KNWN
        mV[:-1, :][mVmask] = self.KNWN

        # Create novel U and V field for PIC|FLIP update
        newU = self.U.copy()
        newV = self.V.copy()

        # Compute and apply the pressure gradient where appropriate
        newU[:, :-1][ufidx] -= scale * (p[ufidx] - p[tuple((idx + left  )[:, ufidx])])
        newU[:, :-1][usidx] = 0.

        newV[:-1, :][vfidx] -= scale * (p[vfidx] - p[tuple((idx + bottom)[:, vfidx])])
        newV[:-1, :][vsidx] = 0.

        # Extrapolate the U, V fields where they are unknown
        newU.extrapolate(mU == self.UNKNWN)
        newV.extrapolate(mV == self.UNKNWN)

        return newU, newV

    def div(self, fields : List[MACArray], fluid : Levelset, scale : float) -> MACArray:
        # Prepare a centered MACArray containing the resulting
        # divergence field
        shape = fluid.shape
        divf = MACArray.zeros(shape, off = Vec(.5, .5))

        # Unpack the field components
        U, V = fields

        mask = fluid < 0

        # Get the indices of fluid voxels
        idx = np.indices(shape)[:, mask]

        divf[mask] = -scale * (U[tuple(idx + np.array([[0], [1]]))] - U[tuple(idx)] +
                               V[tuple(idx + np.array([[1], [0]]))] - V[tuple(idx)])

        return divf

    def enforce_boundary(self, sls):
        idx = np.indices(self.shape)
        left, bottom = np.array([[[0]], [[-1]]]), np.array([[[-1]], [[0]]])

        usidx = np.logical_or(sls[tuple(idx)] < 0, sls[tuple(idx + left  )] < 0)
        vsidx = np.logical_or(sls[tuple(idx)] < 0, sls[tuple(idx + bottom)] < 0)
        self.U[:, :-1][usidx] = 0.
        self.V[:-1, :][vsidx] = 0.
        self.U[:, -1] = 0
        self.V[-1, :] = 0
