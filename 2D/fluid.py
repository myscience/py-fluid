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

        self.S = Swarm((1, w - 1, 1, h - 1), len(pos), pos = pos, vel = Vec(0., 0.), payload = pl)

        # External force fields
        self.fX = 0.
        self.fY = 0.

    def step(self, dt : float) -> None:
        self.dt = dt

        # * Update the fluid levelset based on new particle location
        self.fls.update(self.S.locate(), 1.)
        self.fls.redistance()
        self.fls.fill_holes(1)
        self.fls.lift(self.sls)

        # * Trasfer particles value to the grid
        # We gridify swarm location
        # NOTE: Second call to gridify is cheaper than the first
        #       because it re-uses the triangulation
        uvtx, uwts = self.S.gridify(self.U.shape, self.fls, off = self.U.off)
        vvtx, vwts = self.S.gridify(self.V.shape, self.fls, off = self.V.off)

        U = self.S.collect('vel', 'x')
        V = self.S.collect('vel', 'y')

        self.U.fastfit(U, uvtx, uwts)
        self.V.fastfit(V, vvtx, vwts)

        # * Add external forces
        self.U += dt * self.fX
        self.V += dt * self.fY

        # * Extrapolate data to unknown positions
        self.U.extrapolate(np.isnan(self.U))
        self.V.extrapolate(np.isnan(self.V))

        # * Set the velocities to zero on solid boundaries
        self.enforce_boundary(self.sls)

        # * Compute the divergence of the velocity field
        div = self.div([self.U, self.V], self.fls, self.sls, 1. / self.dx) 

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

        self.U[...] = newU.copy()
        self.V[...] = newV.copy()

        # * Advect the particles around
        self.S.advect(dt, self.dx, self.U, self.V)

        # * Backtrack those particles that ended up inside solid
        self.backtrace()

    def screenshot(self, mask = None, bkg = Vec(0., 0., 0.), mask_value = Vec(0., 0., 0.)):
        '''
            This function is used to create a screenshot of the current fluid
            condition.
        '''

        imgR = MACArray.const(self.shape, bkg.r, off = Vec(0.5, 0.5))
        imgG = MACArray.const(self.shape, bkg.g, off = Vec(0.5, 0.5))
        imgB = MACArray.const(self.shape, bkg.b, off = Vec(0.5, 0.5))

        R = self.S.collect('ink', 'r')
        G = self.S.collect('ink', 'g')
        B = self.S.collect('ink', 'b')

        # NOTE: We can speedup the interpolation for the [R, G, B]
        #       channels by exploiting the fact that we are interpolating
        #       on grids of the same shape, just the values are different.
        #       we precumpute the gridification of the Swarm, and pass the
        #       different color values accordingly.
        vtx, wts = self.S.gridify(self.shape, self.fls, off = Vec(0.5, 0.5))

        imgR.fastfit(R, vtx, wts)
        imgG.fastfit(G, vtx, wts)
        imgB.fastfit(B, vtx, wts)

        # Remove NaNs by filling with background
        imgR[np.isnan(imgR)] = bkg.r
        imgG[np.isnan(imgG)] = bkg.g
        imgB[np.isnan(imgB)] = bkg.b

        # imgR.fit(self.S, 'ink', 'r')
        # imgG.fit(self.S, 'ink', 'g')
        # imgB.fit(self.S, 'ink', 'b')

        # We can optionally mask the image to compensate for the convex-hull
        # fit that might paint unwnated regions
        if mask is not None:
            imgR[mask] = mask_value.r
            imgG[mask] = mask_value.g
            imgB[mask] = mask_value.b

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

    def div(self, fields : List[MACArray], fluid : Levelset, solid : Levelset, scale : float) -> MACArray:
        # Prepare a centered MACArray containing the resulting
        # divergence field
        shape = fluid.shape
        divf = MACArray.zeros(shape, off = Vec(.5, .5))

        # Unpack the field components
        U, V = fields

        fmask = fluid < 0
        smask = solid < 0

        # Get the indices of fluid voxels
        idx = np.indices(shape)[:, fmask]

        right = np.array([[0], [1]])
        top   = np.array([[1], [0]])

        left   = np.array([[0], [-1]])
        bottom = np.array([[-1], [0]]) 

        divf[fmask] = -scale * (U[tuple(idx + right)] - U[tuple(idx)] +
                                V[tuple(idx + top)]   - V[tuple(idx)])

        # * Correct divergence to take into account solid boundaries
        idx_l = idx[:, smask[tuple(idx + left)]]
        idx_r = idx[:, smask[tuple(idx + right)]]
        idx_t = idx[:, smask[tuple(idx + top)]]
        idx_b = idx[:, smask[tuple(idx + bottom)]]

        divf[tuple(idx_l)] -= scale * (U[tuple(idx_l)] - 0.)
        divf[tuple(idx_r)] += scale * (U[tuple(idx_r + right)] - 0.)

        divf[tuple(idx_b)] -= scale * (V[tuple(idx_b)] - 0.)
        divf[tuple(idx_t)] += scale * (V[tuple(idx_t + top)] - 0.)
        
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

    def backtrace(self) -> None:
        P = self.S.get(self.sls)
        
        if len(P) > 0:
            nP = self.fls.closest(np.array([p.pos for p in P]))

            # Re-fit the new velocity based on updated position
            pU, pV = self.U(nP), self.V(nP)

            [p.update('pos', px, 'x') for p, px in zip(P, nP[:, 0])]
            [p.update('pos', py, 'y') for p, py in zip(P, nP[:, 1])]

            try:
                [p.update('vel', vx, 'x') for p, vx in zip(P, pU)]
                [p.update('vel', vy, 'y') for p, vy in zip(P, pV)]

            # We intercept the exception cause when a single particles
            # is out of bounds, in which case pU and pV are scalar
            # number which do not support iteration
            except TypeError as e:
                [p.update('vel', vx, 'x') for p, vx in zip(P, [pU])]
                [p.update('vel', vy, 'y') for p, vy in zip(P, [pV])]
