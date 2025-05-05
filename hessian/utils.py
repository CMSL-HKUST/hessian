import jax
import jax.numpy as np
import jax.flatten_util
import time


def tree_l2_norm_error(θ1, θ2):
    return np.sqrt(jax.tree_util.tree_reduce(lambda x, y: x + y,
        jax.tree_util.tree_map(lambda x, y: np.sum((x - y)**2), θ1, θ2)))


def compute_l2_norm_error(problem, sol_list_pred, sol_list_true):
    u_pred_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_pred[0]) # (num_cells, num_quads, vec)
    u_true_quad = problem.fes[0].convert_from_dof_to_quad(sol_list_true[0]) # (num_cells, num_quads, vec)
    l2_error = np.sqrt(np.sum((u_pred_quad - u_true_quad)**2 * problem.fes[0].JxW[:, :, None]))
    return l2_error


def timing_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start_time
        return result, time_elapsed
    return wrapper