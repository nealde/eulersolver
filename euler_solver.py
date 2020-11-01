import numpy as np

from typing import Tuple
from scipy.interpolate import interp1d

from .backward_solver import BackwardSolver
from .forward_solver import ForwardSolver


class EulerSolver:
    def __init__(self, style: str = 'backward', dynamic_steps: bool = True, abs_tol: float = 1e-3):
        """
        A wrapper function for the Euler Forwards and Backwards numberical ODE / DAE solvers.
        Parameters
        ----------
        style: string
            either 'forward' or 'backward', this selects the solving method to use.
        dynamic_steps: bool
            If True, allow the solvers to take dynamic steps in order to increase the performance and allow
                a desired error tolerrance to control the time stepping.
        abs_tol: float
            The desired tolerance for the solution.
        """
        self._solvers = {'forward': ForwardSolver, 'backward': BackwardSolver}
        assert style in self._solvers, f'only {self._solvers.keys()} solving styles are accepted, received {style}'

        self.style = style
        self.absolute_tolerance = abs_tol
        self.relative_tolerance = 10 * abs_tol
        self.solver = self._solvers[self.style]()

        self._take_dynamic_steps = dynamic_steps
        self._default_step_size = max(self.absolute_tolerance, 1e-4)

    def _dynamic_step(self, func: callable, y_current: np.array, step: float, parameters: np.array = None, number_odes: int = None) -> Tuple[np.array, float]:
        """
        Take a dynamic step, which will suggest a modification of the next step size.
        Parameters
        ----------
        func: callable
            The function which represents the system of equations
        y_current: np.array
            The current state of the system of equations
        step: float
            The step size to take
        parameters: np.array
            Optional, a set of parameters to pass to the user-defined function
        number_odes: int
            The index at which the ODEs switch to AEs, if the system is a set of DAEs.

        Returns
        -------
        [solution: np.array, next_step_size: float]
            The solution contains the new values at the next time step.
            The next_step_size represents the suggested next step size to maintain the same level of error.
        """
        half_step = step / 2
        next_step_size = step
        solution = self.solver.single_step(func, y_current, step, parameters, number_odes=number_odes)
        if self._take_dynamic_steps:
            half_step_solution = self.solver.single_step(func, y_current, half_step, parameters, number_odes=number_odes)
            two_step_solution = self.solver.single_step(func, half_step_solution, half_step, parameters, number_odes=number_odes)
            next_step_size = step * self._adjust_step_size(two_step_solution, solution)
            solution = two_step_solution  # take the better answer, since you calculated it anyway

        return solution, next_step_size

    def _adjust_step_size(self, two_step_solution: np.array, one_step_solution: np.array) -> float:
        """
        In order to control the error at each time step, 3 total steps must be taken: one of the original time step size,
        and two more each of half that step size. By comparing the errors at the solution time, information about
        how big the next time step must be can be gleaned.
        Parameters
        ----------
        two_step_solution: np.array
            The values taken with two half-steps, which will be more accurate than the one-step solution.
        one_step_solution: np.array
            The values taken with a single step, which will be less accurate.

        Returns
        -------
        growth_factor: float
            The factor by which to change the step size to maintain the same level of error.
        """
        residual = (2 ** self.solver.order * two_step_solution - one_step_solution) / (2 ** self.solver.order - 1)
        error = abs(residual - two_step_solution) / (self.absolute_tolerance + self.relative_tolerance * abs(residual))
        largest_error = max(self.absolute_tolerance, np.linalg.norm(error))
        growth_factor = 0.9 * (1 / largest_error) ** (1 / self.solver.order)
        return growth_factor

    def solve(self, func: callable, y_initial: np.array, final_time: float, step: float = None, parameters: np.array = None, out_times: np.array = None, number_odes: int = None) -> np.array:
        """
        Solve func, any callable python function, which acts as a system of equations for the solver to act upon.
        This does work with DAEs if the backwards style is selected.
        Example:
        def a_to_b_to_c_reaction(y: np.array, p: np.array = None) -> np.array:
            dy = np.zeros(len(y))
            dy[0] = -p[0] * y[0]  # rate consumed of A
            dy[1] = p[0] * y[0] - p[1] * y[1]  # rate of A generated minus rate of B consumed
            dy[2] = p[1] * y[1]  # rate of C generated
            return dy

        with initial conditions:
        y_0 = np.array([1.0, 0.0, 0.0])
        rate_constants = np.array([0.1, 0.2, 0.5])

        forward_solver = EulerSolver(style='forward', abs_tol=1e-6)
        forward_solver.solve(a_to_b_to_c_reaction, y_0,

        Parameters
        ----------
        func: callable
            The function representing the system of equations to solve.
        y_initial: np.array
            The initial conditions for the Initial Value Problem
        final_time: float
            The final time to solve to.
        step: float
            The initial step size, or every step size, if error correction is not enabled.
        parameters: np.array
            An array containing a set of parameters to pass to func
        out_times: np.array
            If desired, an array of pre-determined times can be supplied, and the values at these times
                will be interpolated using cubic splines. Each of the equations will be interpolated separately.
        number_odes: int
            If the system is a DAE system, the index at which the equations switch from ODEs to AEs.

        Returns
        -------
        y, t: np.array, np.array
            The y array contains the equations along the 0th axis and the solution at each time along the 1st axis.
            The t array contains the time values.
            The data can be accessed like this:
            plt.plot(t, y[0,:]), where this is plotting the 0th element of y vs time.
        """
        _time = 0
        y = y_initial
        n = len(y_initial)
        y_values = [y]
        time_values = [_time]
        if step is None:
            step = self._default_step_size
        while final_time > _time + step:
            _time += step
            y, step = self._dynamic_step(func, y, step, parameters, number_odes=number_odes)
            y_values.append(y)
            time_values.append(_time)

        # perform final step
        step = final_time - _time
        y, _ = self._dynamic_step(func, y, step, parameters, number_odes=number_odes)
        _time += step
        y_values.append(y)
        time_values.append(_time)
        y_values = np.transpose(y_values)

        if out_times:  # interpolate at out_times
            y_values = np.array([interp1d(time_values, y_values[row_of_data, :], kind='cubic', bounds_error=False)(out_times) for row_of_data in range(y_values.shape[0])])
            time_values = out_times

        return y_values, time_values

