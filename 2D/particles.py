import numpy as np
from numpy.lib.utils import deprecate
import utils
import mac
from utils import Vec, Particle
from levelset import Levelset

import scipy.spatial.qhull as qhull

from typing import List, Tuple, Union
from itertools import repeat

class Swarm:
    '''
        A simple class for representing a particle swarm. This is used
        as the particle-component of the PIC|FLIP method for fluid
        simulation: particles are sprinkled into the fluid domain and
        advected aroud. They carry a payload of quantities (ink) that
        is fitted to the domain grid and vice versa they can fit their
        payload from a given field.
    '''

    def __init__(self, 
            domain, 
            num : int,
            pos : Union[List[Vec], np.ndarray, None] = None,
            vel : Union[Vec, List[Vec], np.ndarray, None] = None,
            payload = None
        ) -> None:
        
        # * Input parsing
        self.domain = domain

        if pos is None: 
            # We sprinkle the particles randomly in domain            
            rng = np.random.default_rng()
            X, Y = rng.uniform(*domain[:2], num), rng.uniform(*domain[2:], num)

            L = [Vec(x, y) for x, y in zip(X, Y)]
        elif isinstance(pos, (np.ndarray, list)):
            msg = f'Unmatched numerosity and initial position set. Got N: {num} and V: {len(pos)}'
            assert len(pos) == num, msg
            L = pos
        else: raise ValueError(f'Unknown position type. Got {type(pos)}')

        if vel is None: V = repeat(None)
        elif isinstance(vel, utils.Vec): V = (vel.copy() for _ in range(num))
        elif isinstance(vel, (tuple, list, np.ndarray)): 
            msg = f'Unmatched numerosity and initial velocity set. Got N: {num} and V: {len(vel)}'
            assert len(vel) == num, msg
            V = vel
        else: raise ValueError(f'Unknown velocity type. Got {type(vel)}')

        if payload is None: P = repeat({})
        elif isinstance(payload, dict): P = (payload.copy() for _ in range(num))
        elif isinstance(payload, (tuple, list, np.ndarray)): 
            msg = f'Unmatched numerosity and initial payload. Got N: {num} and V: {len(payload)}'
            assert len(payload) == num, msg            
            P = payload
        else: raise ValueError(f'Unknown payload type. Got {type(payload)}') 

        self.swarm = [Particle({'pos' : l, 'vel' : v, **p}) for l, v, p in zip(L, V, P)]

        # Point triangulation stored for faster lookup
        self.tri = None

    def __len__(self) -> int:
        return len(self.swarm)

    def __getitem__(self, index : int) -> Particle:
        return self.swarm[index]

    def merge(self, S):
        msg = 'Cannot merge swarms of different domains.'
        assert self.domain == S.domain, msg

        self.swarm += S.swarm 

    def fit(self, field : mac.MACArray, key : str, comp : str = None) -> None:
        # Locate the particles in the grid and get interpolated values
        # based on the underlying field
        values = field(self.locate())

        # Set the provived key entry of the particles payload
        [p.update(key, float(v), comp) for p, v in zip(self.swarm, values)]

    def picflip(self, fold : mac.MACArray, fnew : mac.MACArray, key : str, comp : str = None, alpha = 0.03) -> None:
        '''
            This function implements the PIC|FLIP update
        '''
        # Get the difference between the new and old field 
        fdiff = fnew - fold

        # Get the set of particles positions & old payload values
        P = self.locate()
        V = np.array([getattr(p.pld[key], comp) if comp else p.pld[key] for p in self.swarm])

        # Use the PIC|FLIP update as new value for the key payload component
        values = alpha * fnew(P) + (1 - alpha) * (V + fdiff(P))

        [p.update(key, float(v), comp) for p, v in zip(self.swarm, values)]

    def advect(self, dt : float, dx : float, U : mac.MACArray, V : mac.MACArray) -> None:
        '''
            This function implements the RK3 time integration step, moving
            the particle swarm in the provided U-V velocity field
        '''

        # NOTE: Velocity is expressed in physical units (meter/second)
        #       so when dealing with its interpolated values (the Ks)
        #       we mutiply them by dt/dx to obtain the absolute number
        #       that actual refers to the grid-units, as the positions
        #       (the Ps) are actually expressed in grid-units.
        P1 = self.locate()
        K1 = np.stack([U(P1), V(P1)], axis = -1)

        P2 = P1 + .5 * dt / dx * K1
        K2 = np.stack([U(P2), V(P2)], axis = -1)

        P3 = P1 + .75 * dt / dx * K2
        K3 = np.stack([U(P3), V(P3)], axis = -1)

        Pf = P1 + dt / dx * (2 * K1 + 3 * K2 + 4 * K3) / 9.

        # Update particle position with grid-based units
        [p.update('pos', np.clip(px, *self.domain[:2]), 'x') for p, px in zip(self.swarm, Pf[:, 0])]
        [p.update('pos', np.clip(py, *self.domain[2:]), 'y') for p, py in zip(self.swarm, Pf[:, 1])]

        # Invalidate previous triangulation
        self.tri = None

    def get(self, mask : Levelset) -> List[Particle]:
        return [P for P, v in zip(self.swarm, mask([P.pos for P in self.swarm]) < 0) if v]

    def locate(self, mask : Levelset = None) -> List[Vec]:
        '''
            This function locates the particles on a grid.
            NOTE: Because grid uses a row-major format, the y position is return
                  BEFORE the x position, so that when used to index the grid the
                  correct location is used. So return format is [y, x].
        '''
        if mask is None: return np.array([P.pos for P in self.swarm]) 
        else: return np.array([P.pos for P, v in zip(self.swarm, mask([P.pos for P in self.swarm]) < 0) if v])
   
    def collect(self, key : str, comp : str = None) -> List:
        return [getattr(P.pld[key], comp) if comp else P.pld[key] for P in self.swarm]

    def triangulate(self, fls : Levelset, pos = None) -> qhull.Delaunay:
        pos = self.locate() if pos is None else pos

        self.tri = qhull.Delaunay(pos)

        # * From triangulation we remove those triangles whose barycentric
        # * coordinate lies outside the fluid domain
        bary = np.mean(pos[self.tri.simplices], axis = 1)
        mask = fls(bary) > 0

        # ! FIXME: Altering the simplices array breaks the subsequent
        # !        call to find_simplex
        self.tri.deprecated = {s_idx for s_idx in np.arange(len(mask))[mask]}
        # self.tri.neighbors = self.tri.neighbors[mask]
        # self.tri.equations = self.tri.equations[mask]

        return self.tri

    # Taken from:
    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    def gridify(self, grid : Tuple[int, int], fls : Levelset, off = None) -> Tuple[np.ndarray, np.ndarray]:
        d = len(grid)

        x, y = np.arange(grid[1]), np.arange(grid[0])
        off = Vec([0] * d) if off is None else off

        # Create the array storing the grid point location, optionally
        # offset by a costant vector
        uvw = np.stack(np.meshgrid(x, y), axis = -1).reshape(-1, d) + off

        # Get the point triangulation if not pre-computed
        tri = self.triangulate(fls) if self.tri is None else self.tri

        simplex = tri.find_simplex(uvw)

        # Filter out simplex indices if they are deprecated
        simplex = [s if s not in tri.deprecated else -1 for s in simplex]

        vertices = np.take(tri.simplices, simplex, axis = 0)
        temp = np.take(tri.transform, simplex, axis = 0)
        delta = uvw - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))