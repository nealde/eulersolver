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
        """
        Takes a single step according to Euler Forwards.
        Parameters
        ----------
        func: callable
            The function that represents the system of equations.
        y_current: np.array
            The current state of the equations
        step: float
            The step size to take
        parameters: np.array
            An array of parameters to pass to the function, if applicable.
        number_odes: int
            The number of ODEs in the system, if the system is a set of DAEs. It is not used by this function, but
                needs to conform to the interface set by Euler Backward

        Returns
        -------
        np.array containing the derivative at the next time step, per Euler Forward.
        """
        if parameters is not None:
            return y_current + step * func(y_current, parameters)
        return y_current + step * func(y_current)





