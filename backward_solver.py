import numpy as np


class BackwardSolver:
    def __init__(self, nr_tolerance: float = 1e-7):
        """
        Backward Euler is represented using the following mathematical expression:
        y_(n+1) = y_n + h*f(t_n+1, y_n+1)

        This means that the state at the next timestep is dictated by the derivative at the next
        time step. This significantly increases the accuracy and stability of this solver,
        but at the cost of higher complexity and more computational load.


        Parameters
        ----------
        nr_tolerance: float
            The tolerance for newton-rhapson to declare convergence.
        """
        self.order = 2
        self.NR_tol = nr_tolerance
        self._function = None
        self._di = None
        self._n = None
        self._dfdy = None
        self._parameters = None
        self._step = None
        self._mass_matrix = None
        self._b = None

    def _func_wrap(self, y_current: np.array) -> np.array:
        """
        A simple function wrapper that inserts the parameters or not, depending upon whether or not they have
        been passed. This is attempting to simulate scipy's optimize's `args` argument.
        Parameters
        ----------
        y_current: np.array
            The current state of the equations

        Returns
        -------
        The derivative of the equations at the current state.
        """
        if self._parameters is None:
            return self._function(y_current)
        return self._function(y_current, self._parameters)

    def single_step(self, func: callable, y_current: np.array, step: float, parameters: np.array = None, number_odes: int = None) -> np.array:
        """
        Takes a single step according to Euler Backwards.
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
            The number of ODEs in the system, if the system is a set of DAEs.

        Returns
        -------
        np.array containing the derivative at the next time step, per Euler Backward.
        """
        self._function = func
        self._n = len(y_current)
        self._step = step
        self._parameters = parameters
        if self._dfdy is None or self._dfdy.shape[0] != self._n or self._number_odes != number_odes:
            self._initialize_in_place_arrays(number_odes)

        self._b = np.zeros(self._n)
        return self.newton_rhapson(y_current)

    def _initialize_in_place_arrays(self, number_odes: int):
        """
        Due to the mass matrix solution style, we need to know how many of the equations are ODEs. This builds
            parameters that are used to make those calculations in-place, increasing performance.
        Parameters
        ----------
        number_odes: int
            The number of ordinary differential equations in the system, given that all of the ODEs are at the beginning
            of the system of equations.

        Returns
        -------
        None
        """
        self._number_odes = self._n
        if number_odes:
            self._number_odes = number_odes
        self._mass_matrix = np.zeros((self._n, self._n))
        for i in range(self._number_odes):
            self._mass_matrix[i, i] = 1.0
        self._di = np.zeros(self._n)
        self._dfdy = np.zeros((self._n, self._n))

    def _calculate_numerical_jacobian(self, y_current: np.array) -> np.array:
        """
        Given a function, calculate the numerical jacobian by permuting each input variable a small amount.
        Parameters
        ----------
        y_current: np.array
            The current state of the system of equations

        Returns
        -------
        (n x n) np.array containing the jacobian
        """
        for i in range(self._n):
            self._di[i] = 1.0
            delta = max(1e-12, 1e-5 * y_current[i])
            # perturb one input by a small amount
            dy_delta = self._func_wrap(y_current + delta * self._di)
            dy = self._func_wrap(y_current)
            self._dfdy[:, i] = (dy_delta - dy) / delta
            self._di[i] = 0.0

    def newton_rhapson(self, y_current: np.array):
        """
        The mass-matrix form of the Newton Rhapson method, which allows for the solving of DAEs
        Parameters
        ----------
        y_current: np.array
            The current state of the system of equations

        Returns
        -------
        np.array containing the derivative of the system of equations at the next timestep.
        """
        y_old = np.copy(y_current)
        y_new = np.copy(y_current)

        for i in range(10):
            self._calculate_numerical_jacobian(y_current)
            Jac = self._dfdy
            Jac[:self._number_odes, :] *= self._step
            Jac -= self._mass_matrix

            dy = self._func_wrap(y_new)

            self._b[:self._number_odes] = self._step * dy[:self._number_odes] + y_old[:self._number_odes] - y_new[:self._number_odes]
            self._b[self._number_odes:] = dy[self._number_odes:]

            dy_new = np.linalg.solve(Jac, self._b)
            y_new -= dy_new
            err = np.linalg.norm(dy_new) / np.linalg.norm(y_new)

            if err < self.NR_tol:
                break
        return y_new



