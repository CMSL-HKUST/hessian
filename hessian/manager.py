import jax
import jax.numpy as np
import os
import time

from jax_fem.solver import ad_wrapper, linear_solver
from jax_fem import logger
from jax_fem.utils import save_sol

from hessian.hessp import forward_and_adjoint, incremental_forward_and_adjoint, forward_step
from hessian.utils import tree_l2_norm_error

class HessVecProduct:
    def __init__(self, problem, J_fn, θ_ini, solver_options, adjoint_solver_options, *args):
        logger.info(f"########################## hess_vec_prod.__init__ is called...")
        self.cached_vars = {'θ': None, 'u': None, 'λ': None, 'F_fn': None, 'A': None}
        self.problem = problem
        self.J_fn = J_fn
        self.solver_options = solver_options
        self.adjoint_solver_options = adjoint_solver_options
        self.state_linear_solver = lambda A, b: linear_solver(A, b, None, solver_options)
        self.adjoint_linear_solver = lambda A, b: linear_solver(A, b, None, adjoint_solver_options)
        self.fwd_pred = ad_wrapper(problem, solver_options, adjoint_solver_options)
        self.θ_ini = θ_ini
        self.θ_ini_flat, self.unflatten = jax.flatten_util.ravel_pytree(self.θ_ini)
        self.opt_step = 1
        self.hessp_option = 'rev_fwd'
        self.args = args
        self.counter = {'J': 0, 'grad': 0, 'hessp_full': 0, 'hessp_cached': 0}
        self.J_value = self.J(self.θ_ini_flat)
        self.counter['J'] -= 1 # No need to count this: offset by 1.
        self.J_values = [self.J_value]
        self.u_to_save = None
        logger.info(f"************************** hess_vec_prod.__init__ finished...\n\n")

    def J(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.J is called...")
        self.counter['J'] += 1
        θ = self.unflatten(θ_flat)
        u, F_fn = forward_step(self.problem, θ, self.solver_options)
        J_value = self.J_fn(u, θ)
        self.J_value = J_value
        self.u_to_save = u
        logger.info(f"J_value = {J_value}")
        logger.info(f"************************** hess_vec_prod.J finished...\n")
        return J_value

    def grad(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.grad is called...")
        self.counter['grad'] += 1
        θ = self.unflatten(θ_flat)
        def J(θ):
            return self.J_fn(self.fwd_pred(θ), θ)
        grad_J = jax.grad(J)(θ)
        logger.info(f"************************** hess_vec_prod.grad finished...\n")
        return jax.flatten_util.ravel_pytree(grad_J)[0]

    def hessp(self, θ_flat, θ_hat_flat):
        logger.info(f"########################## hess_vec_prod.hessp is called...")
        self.timer = []
        self.timer.append(time.perf_counter())
        θ = self.unflatten(θ_flat)
        θ_hat = self.unflatten(θ_hat_flat)
        tol = 1e-8
        if (self.cached_vars['θ'] is None) or tree_l2_norm_error(self.cached_vars['θ'], θ) > tol:
            logger.info(f"hessp NEEDs to solve forward and adjoint problem...")
            self.counter['hessp_full'] += 1
            u, λ, F_fn, A = forward_and_adjoint(self.problem, θ, self.J_fn, self.solver_options, self.adjoint_solver_options, self.timer)
            self.cached_vars['θ'] = θ
            self.cached_vars['u'] = u
            self.cached_vars['λ'] = λ
            self.cached_vars['F_fn'] = F_fn
            self.cached_vars['A'] = A
        else:
            logger.info(f"hessp DOES NOT NEED to solve forward and adjoint problem...")
            self.counter['hessp_cached'] += 1
            θ = self.cached_vars['θ']
            u = self.cached_vars['u']
            λ = self.cached_vars['λ']
            F_fn = self.cached_vars['F_fn']
            A = self.cached_vars['A']
            self.timer = self.timer + [self.timer[-1], self.timer[-1]] # No time spent for forward and adjoint problem

        dθ_dθ_J_θ_hat, self.profile_info = incremental_forward_and_adjoint(u, θ, λ, θ_hat, self.J_fn, F_fn, A, 
            self.state_linear_solver, self.adjoint_linear_solver, self.timer, self.hessp_option)
        logger.info(f"************************** hess_vec_prod.hessp finished...\n")
        return jax.flatten_util.ravel_pytree(dθ_dθ_J_θ_hat)[0]

    def callback(self, θ_flat):
        """
        Overwrite this function to define customized callback rule.
        """
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        self.opt_step += 1


def central_finite_difference_hessp(hess_vec_prod, θ_flat, θ_hat_flat, h):
    θ_minus_flat = θ_flat - h*θ_hat_flat
    θ_plus_flat  = θ_flat + h*θ_hat_flat
    value_plus = hess_vec_prod.grad(θ_plus_flat)
    value_minus = hess_vec_prod.grad(θ_minus_flat)
    dθ_dθ_J_θ_hat = jax.tree_util.tree_map(lambda x, y: (x - y)/(2*h), value_plus, value_minus)
    return dθ_dθ_J_θ_hat


def forward_finite_difference_hessp(hess_vec_prod, θ_flat, θ_hat_flat, h):
    θ_plus_flat  = θ_flat + h*θ_hat_flat
    value_plus = hess_vec_prod.grad(θ_plus_flat)
    value = hess_vec_prod.grad(θ_flat)
    dθ_dθ_J_θ_hat = jax.tree_util.tree_map(lambda x, y: (x - y)/(h), value_plus, value)
    return dθ_dθ_J_θ_hat
