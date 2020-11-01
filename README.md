# EulerSolver
An implementation of Euler Forwards / Euler Backwards numberical solvers for ODEs and DAEs in Python

## Code Attributes

### Getting the results
There are several nice-to-haves that have been incorporated into this package. In addition to having a forward and backward solver,
the solvers are set up to consistently return the time series and y values with consistent axes. For example, the following code:

```python
y, t = forward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p)

plt.plot(t, y[0,:])
```

will always return the 0th element of the solution vs time, regardless of the number of equations.

### Dynamic Time Steps

The code takes dynamic time steps in order to approximate a desired level of accuracy, dictated by the `abs_tol` argument. This can save
computational time, especially for non-stiff equations where larger steps are allowed.

### Time Interpolation

The code takes time steps dynamically (by default - pass `ddynamic_steps=False` to disable this) in order to achieve a target accuracy.
If you would like to return the result at only specific times (for instance, every 10 seconds), passing a numpy array of pre-discretized times
using `out_time=<array-of-pre-discretized-time>` will interpolate these answers for you using a cubic 1d interpolation, per-equation.

### Solving DAEs

Euler backward, being an implicit solver, can solve some subset of DAE systems. An example can be found in `eulersolver/tests.py`, but is replicated here
with more context:

```python
import numpy as np
def dae_function(y: np.array, p: np.array = None):
    """A small Nickel Electrode battery model"""
    F = 96487  # Faraday's constant (C/mol)
    R = 8.314  # Universal gas constant (J/mol K)
    T = 298.15  # Temperature (K)
    phi1 = 0.420  # Equilibrium potential (V)
    phi2 = 0.303  # Equilibrium potential (V)
    W = 92.7  # Mass of active material (g)
    V = 1E-5  # Volume (m^3)
    i01 = 1E-4  # Exchange current density (A/cm^2)
    i02 = 1E-10  # Exchange current density (A/cm^2)
    iapp = 1E-5  # Applied current density (A/cm^2)
    rho = 3.4  # Density (g/cm^3)

    dy = np.zeros(y.shape)
    dy[0] = (W * i01 / (rho * V * F)) * (
                2 * (1 - y[0]) * np.exp((0.5 * F / (R * T)) * (y[1] - phi1)) - 2 * y[0] * np.exp(
            (-0.5 * F / (R * T)) * (y[1] - phi1)))
    dy[1] = i01 * (2 * (1 - y[0]) * np.exp((0.5 * F / (R * T)) * (y[1] - phi1)) - 2 * y[0] * np.exp(
        (-0.5 * F / (R * T)) * (y[1] - phi1))) + i02 * (
                        np.exp((F / (R * T)) * (y[1] - phi2)) - np.exp((-F / (R * T)) * (y[1] - phi2))) - iapp
    return dy
```

Above is the DAE system. The first equation is an ODE, and the second equation is just algebraic. In order to pass this information to the solver,
we need to supply the integer index at which the ODEs end. This means that, in order to solve DAEs using this solver, the ODEs must be at the
beginning of the vector, and the AEs must be at the end.

```python
import numpy as np
from euler_solver import EulerSolver

y_0 = np.array([0.05, 0.35])
backward_solver = EulerSolver(style='backward', abs_tol=1e-6)
y, t = backward_solver.solve(dae_function, y_0, 60, number_odes=1)
```

Due to the stiffness of the equations, it may be necessary to tighten the `abs_tol` to a relatively small number to reduce the size of the first timestep, which allows the equations
to initialize properly.

## Usage

Examples can be founds in `/eulersolver/tests.py`, but some will be replicated here.

### A -> B -> C rate equation example

In this example, there are two chemical reactions: A -> B, and B -> C. These have rates which are parameterized in
array `p`. The equations are as follows:

<img src="https://latex.codecogs.com/gif.latex?\\&space;y_1(0)&space;=&space;1;&space;\\&space;y_2(0)&space;=&space;0;&space;\\&space;y_3(0)&space;=&space;0;&space;\\&space;\frac{dy_1}{dt}&space;=&space;-&space;k_1y_1&space;\\&space;\frac{dy_2}{dt}&space;=&space;k_1y_1&space;-&space;k_2y_2&space;\\&space;\frac{dy_3}{dt}&space;=&space;k_3y_3" title="\\ y_1(0) = 1; \\ y_2(0) = 0; \\ y_3(0) = 0; \\ \frac{dy_1}{dt} = - k_1y_1 \\ \frac{dy_2}{dt} = k_1y_1 - k_2y_2 \\ \frac{dy_3}{dt} = k_3y_3" />

This equation is modeled by the following function:

```python
def complex_function_with_parameters(y: np.array, p: np.array = None) -> np.array:
    """An equation representing A -> B -> C reaction with rate constants p."""
    dy = np.zeros(len(y))
    dy[0] = -p[0] * y[0]  # rate consumed of A
    dy[1] = p[0] * y[0] - p[1] * y[1]  # rate of A generated minus rate of B consumed
    dy[2] = p[1] * y[1]
    return dy

```

To solve this equation, the following code can be used:

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from euler_solver import EulerSolver


y_0 = np.array([1.0, 0.0, 0.0])
p = np.array([0.1, 0.2, 0.5])

forward_solver = EulerSolver(style='forward', abs_tol=1e-5)
backward_solver = EulerSolver(style='backward', abs_tol=1e-4)

y1, t1 = forward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p)
y2, t2 = backward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p, out_times=t1)

plt.plot(t1, y1[0,:], label = 'A - forward')
plt.plot(t1, y1[1,:], label = 'B - forward')
plt.plot(t1, y1[2,:], label = 'C - forward')

plt.plot(t2, y2[0,:], label = 'A - backward')
plt.plot(t2, y2[1,:], label = 'B - backward')
plt.plot(t2, y2[2,:], label = 'C - backward')
plt.legend()
```

## Run the Tests

The tests can be run using:

```bash
pipenv install
pipenv run nosetests eulersolver/tests.py
```
