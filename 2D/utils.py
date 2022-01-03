import numpy as np
from numpy.core.numeric import zeros_like

from typing import Tuple

class Vec(np.ndarray):
    '''
        Basic extension of numpy array to represent a two- or three-component vector.

        NOTE: To better integrate this object with the default numpy row-major storage
              the array stores the x-location (namely the column-coordinate on a grid)
              at index [1] and not at [0] as one might reasonably expect. Vice versa,
              the y-coordinate is stored at idx [0] and not at [1].
    '''

    def __new__(cls, *data):
        msg = f'Creating Vec with more than three components. Got {data} with len: {len(data)}'
        assert len(data) < 4, msg

        # Data is a numpy-convertible collection of data, construct
        # it and get a view of our own subclass
        obj = np.asarray(data).view(cls)

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

    @classmethod
    def rand(cls, size : int, range : Tuple[float, float] = (0, 1)) -> np.ndarray:
        # Create the underlying zeros data
        data = np.random.uniform(*range, size = size)

        # Return the correct construction of the MAC field
        return cls.__new__(cls, *data)

    def setx(self, x): self[0] = float(x)
    def sety(self, y): self[1] = float(y)        
    def setz(self, z): self[2] = float(z)

    def __repr__(self) -> str:
        if len(self) == 2: msg = f'[x : {self[0]}, y : {self[1]}]'
        elif len(self) == 3: msg = f'[x : {self[0]}, y : {self[1]}, z : {self[2]}]'
        else: raise ValueError('Cannot represent Vector with more than three components.')

        return msg

    x = property(lambda self: float(self[0]), setx)
    y = property(lambda self: float(self[1]), sety)
    z = property(lambda self: float(self[2]), setz)

    r = property(lambda self: float(self[0]), setx)
    g = property(lambda self: float(self[1]), sety)
    b = property(lambda self: float(self[2]), setz)

class Particle:
    '''
        Basic class for representing a fluid particle capable of carrying an
        arbitrary payload encoded as a dictionary of signature {'name' : property}.
    '''

    def __init__(self, payload : dict = None) -> None:
        self.pld = {} if payload is None else payload

    def __repr__(self) -> str:
        return ''.join([f'{k} : {v} ' for k, v in self.pld.items()])

    def update(self, k, v, c = None, throw = True):
        if throw: assert k in self.pld, f'Unknown payload quantity {k} in particle update.'
        
        if c: 
            if throw: getattr(self.pld[k], c)
            setattr(self.pld[k], c, v)
        else: self.pld[k] = v

    def setpos(self, pos): self.pld['pos'] = pos
    def setvel(self, vel): self.pld['vel'] = vel

    pos = property(lambda self: self.pld['pos'], setpos)
    vel = property(lambda self: self.pld['vel'], setvel)
