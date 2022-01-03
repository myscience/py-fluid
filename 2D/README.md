# Simple Fluid Simulation in Python

![Van_Gogh](res/van_gogh.gif)

A minimal python implementation of a `PIC|FLIP` particle-grid fluid solver. Heavily based on Robert Bridson's book [*Fluid Simulation for Computer Graphics*](http://wiki.cgt3d.cn/mediawiki/images/4/43/Fluid_Simulation_for_Computer_Graphics_Second_Edition.pdf).

## How to run this code

This code is based on the following libraries:

```python
numpy.py      # Data manipulation
scipy.sparse  # Solving the pressure system via sparse matrices
PIL.Image     # Visualization of the fluid
```

To run this code simply run the `Jupyter notebook` `Fluid Simulation.ipynb`.

**WARNING:** As it stands right now this code is highly non-optimized and takes a comically long time to complete even a modest 2D simulation. Further optimizations are planned as future work.