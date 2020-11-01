import numpy as np

from typing import Tuple
from scipy.interpolate import interp1d

from .backward_solver import BackwardSolver
from .forward_solver import ForwardSolver


class EulerSolver:
    def __init__(self, style: str = 'backward', dynamic_steps: bool = True, abs_tol: float = 1e-3):
        self._solvers = {'forward': ForwardSolver, 'backward': BackwardSolver}
        assert style in self._solvers, f'only {self._solvers.keys()} solving styles are accepted, received {style}'

        self.style = style
        self.absolute_tolerance = abs_tol
        self.relative_tolerance = 10 * abs_tol
        self.solver = self._solvers[self.style]()

        self._take_dynamic_steps = dynamic_steps
        self._default_step_size = max(self.absolute_tolerance, 1e-4)

    def _dynamic_step(self, func: callable, y_current: np.array, step: float, parameters: np.array = None, number_odes: int = None) -> Tuple[np.array, float]:
        half_step = step / 2
        next_step_size = step
        solution = self.solver.single_step(func, y_current, step, parameters, number_odes=number_odes)
        if self._take_dynamic_steps:
            half_step_solution = self.solver.single_step(func, y_current, half_step, parameters, number_odes=number_odes)
            two_step_solution = self.solver.single_step(func, half_step_solution, half_step, parameters, number_odes=number_odes)
            next_step_size = step * self._adjust_step_size(two_step_solution, solution)
            solution = two_step_solution  # take the better answer, since you calculated it anyway

        return solution, next_step_size

    def _adjust_step_size(self, two_step_solution, one_step_solution):
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
        func
        y_initial
        final_time
        step
        parameters
        out_times
        number_odes

        Returns
        -------

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
        # y_values = np.array(y_values).reshape(n, -1, order='C')
        # y_values = np.rot90(np.array(y_values), k=3)[:, ::-1]
        # y_values = np.rot90(np.array(y_values))  # rotate in order to have time and Y share slicing direction

        if out_times:  # interpolate at out_times
            y_values = np.array([interp1d(time_values, y_values[row_of_data, :], kind='cubic', bounds_error=False)(out_times) for row_of_data in range(y_values.shape[0])])
            time_values = out_times

        return y_values, time_values

