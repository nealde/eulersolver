import numpy as np


class ForwardSolver:
    def __init__(self):
        """Forward Euler is represented using the following mathematical expression:
        y_(n+1) = y_n + h*f(t_n, y_n)

        This means, effectively, that the point at the next timestep is dictated by the derivative
        at the current time step. Because of this, it can be solved explicitly in a simple for-loop
        with no jacobian or numerical solver. This makes it extremely fast, although it is subject to
        compounding errors.
        """
        self.order = 2  # this is a first-order solver, but the dynamic step size works better with this equal to 2

    @staticmethod
    def single_step(func: callable, y_current: np.array, step: float, parameters: np.array = None, number_odes: int = None) -> np.array:
        if parameters is not None:
            return y_current + step * func(y_current, parameters)
        return y_current + step * func(y_current)





