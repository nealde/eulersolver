import numpy as np

from .euler_solver import EulerSolver


def simple_function(y: np.array) -> np.array:
    """A simple quadratic function. Returns the derivative given current state and parameters."""
    return y


def exact_simple_solution(time: np.array) -> np.array:
    return np.exp(time).reshape(1,-1)


def complex_function_with_parameters(y: np.array, p: np.array = None) -> np.array:
    """An equation representing A -> B -> C reaction with rate constants p."""
    dy = np.zeros(len(y))
    dy[0] = -p[0] * y[0]  # rate consumed of A
    dy[1] = p[0] * y[0] - p[1] * y[1]  # rate of A generated minus rate of B consumed
    dy[2] = p[1] * y[1]
    return dy


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


def test_initialization():
    forward_solver = EulerSolver(style='forward')
    backward_solver = EulerSolver(style='backward')
    try:
        fail_solver = EulerSolver(style='anything_else')
    except AssertionError:
        pass


def test_forward_single_step():
    y_0 = np.array([1])
    step = 0.5  # seconds
    forward_solver = EulerSolver(style='forward')
    y_1 = forward_solver.solver.single_step(simple_function, y_0, step)
    np.testing.assert_almost_equal(y_1, np.array([1.5]))


def test_backward_single_step():
    y_0 = np.array([1.0])
    step = 0.5  # seconds
    forward_solver = EulerSolver(style='backward')
    y_1 = forward_solver.solver.single_step(simple_function, y_0, step)
    np.testing.assert_almost_equal(y_1, np.array([2.0]))


def test_forward_dynamic_single_step():
    y_0 = np.array([1.0])
    step = 0.5  # seconds
    forward_solver = EulerSolver(style='forward')
    y_1 = forward_solver._dynamic_step(simple_function, y_0, step)
    np.testing.assert_almost_equal(y_1[0], np.array([1.5625]))


def test_backward_dynamic_single_step():
    y_0 = np.array([1.0])
    step = 0.5  # seconds
    forward_solver = EulerSolver(style='backward')
    y_1 = forward_solver._dynamic_step(simple_function, y_0, step)
    np.testing.assert_almost_equal(y_1[0], np.array([1.7777778]))


def test_forward_simple_solution():
    y_0 = np.array([1.0])
    forward_solver = EulerSolver(style='forward', abs_tol=1e-8)
    y, t = forward_solver.solve(simple_function, y_0, 4.0)
    exact_solution = exact_simple_solution(t)
    np.testing.assert_almost_equal(y, exact_solution, decimal=1)


def test_backward_simple_solution():
    y_0 = np.array([1.0])
    backward_solver = EulerSolver(style='backward', abs_tol=1e-8)
    y, t = backward_solver.solve(simple_function, y_0, 4.0)
    exact_solution = exact_simple_solution(t)
    np.testing.assert_almost_equal(y, exact_solution, decimal=1)


def test_forward_a_b_c_reaction_with_interpolation():
    y_0 = np.array([1.0, 0.0, 0.0])
    p = np.array([0.1, 0.2, 0.5])
    forward_solver = EulerSolver(style='forward', abs_tol=1e-6)
    backward_solver = EulerSolver(style='backward', abs_tol=1e-5)
    y1, t1 = forward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p)
    y2, t2 = backward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p, out_times=t1)
    np.testing.assert_almost_equal(y1, y2, decimal=1)


def test_backward_reinitialization():
    y_0 = np.array([1.0])
    backward_solver = EulerSolver(style='backward', abs_tol=1e-8)
    y, t = backward_solver.solve(simple_function, y_0, 4.0)
    exact_solution = exact_simple_solution(t)
    np.testing.assert_almost_equal(y, exact_solution, decimal=1)

    y_0 = np.array([1.0, 0.0, 0.0])
    p = np.array([0.1, 0.2, 0.5])
    forward_solver = EulerSolver(style='forward', abs_tol=1e-6)
    y1, t1 = forward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p)
    y2, t2 = backward_solver.solve(complex_function_with_parameters, y_0, 60, parameters=p, out_times=t1)
    np.testing.assert_almost_equal(y1, y2, decimal=1)


def test_backward_a_b_c_reaction_and_dae():
    y_0 = np.array([0.05, 0.35])
    backward_solver = EulerSolver(style='backward', abs_tol=1e-6)
    y, t = backward_solver.solve(dae_function, y_0, 60, number_odes=1)
    # just solving this passes the test

