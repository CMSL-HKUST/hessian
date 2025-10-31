import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import time

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem import logger

from hessian.manager import HessVecProduct
from hessian.utils import compute_l2_norm_error, timing_wrapper


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=10)

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


logger.setLevel(logging.INFO)

output_dir = os.path.join(os.path.dirname(__file__), f'output')
fwd_vtk_dir = os.path.join(output_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(output_dir, 'forward/numpy')
inv_vtk_dir = os.path.join(output_dir, 'inverse/vtk')
inv_numpy_dir = os.path.join(output_dir, 'inverse/numpy')
inv_pdf_dir = os.path.join(output_dir, 'inverse/pdf')
os.makedirs(fwd_numpy_dir, exist_ok=True)
os.makedirs(inv_numpy_dir, exist_ok=True)
os.makedirs(inv_pdf_dir, exist_ok=True)


class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x, θ: x

    def get_mass_map(self):
        def mass_map(u, x, θ):
            val = θ
            return np.array([val])
        return mass_map

    def set_params(self, θ):
        self.internal_vars = [θ]


class HessVecProductPoisson(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        if not self.timing_flag:
            print(f"solution size = {self.u_to_save[0].size}, parameter size = {θ_flat.size}")
            self.J_values.append(self.J_value)
            if self.opt_flag == 'cg_ad':
                inv_vtk_dir, = self.args
                # save_sol(self.problem.fes[0], self.u_to_save[0], os.path.join(inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'),
                #     cell_infos=[('θ', np.mean(self.unflatten(θ_flat), axis=-1))])
        self.opt_step += 1


def workflow():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.
    meshio_mesh = rectangle_mesh(Nx=64, Ny=64, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    def dirichlet_val_left(point):
        return 0.

    def dirichlet_val_right(point):
        return 0.

    location_fns = [left, right]
    value_fns = [dirichlet_val_left, dirichlet_val_right]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns, vecs, value_fns]
    problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
    
    fwd_pred = ad_wrapper(problem) 
    # (num_cells, num_quads, dim)
    quad_points = problem.fes[0].get_physical_quad_points()

    run_forward_flag = False
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*')) + glob.glob(os.path.join(fwd_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        θ_true = -10*np.exp(-(np.power(quad_points[:, :, 0] - 0.5, 2) + np.power(quad_points[:, :, 1] - 0.5, 2)) / 0.02)
        u_true = fwd_pred(θ_true)

        save_sol(problem.fes[0], u_true[0], os.path.join(fwd_vtk_dir, f'u.vtu'), cell_infos=[('θ', np.mean(θ_true, axis=-1))])
        np.save(os.path.join(fwd_numpy_dir, f'u.npy'), u_true[0])

    run_inverse_flag = True
    if run_inverse_flag:
        # files = glob.glob(os.path.join(inv_vtk_dir, f'*'))
        # for f in files:
        #     os.remove(f)

        u_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]

        def J_fn(u, θ):
            u_pred = u
            l2_error = compute_l2_norm_error(problem, u_pred, u_true)
            θ_vec = jax.flatten_util.ravel_pytree(θ)[0]
            return 1e3*l2_error**2 + 0*np.sum(θ_vec**2)

        θ_ini = 1*np.ones_like(quad_points)[:, :, 0]
        u_ini = fwd_pred(θ_ini)
        # save_sol(problem.fes[0], u_ini[0], os.path.join(inv_vtk_dir, f'u_{0:05d}.vtu'), cell_infos=[('θ', np.mean(θ_ini, axis=-1))])

        opt_flags = ['cg_ad'] # ['cg_ad', 'cg_fd', 'bgfs']
        for opt_flag in opt_flags:
            hess_vec_prod = HessVecProductPoisson(problem, J_fn, θ_ini, {}, {}, inv_vtk_dir)
            hess_vec_prod.timing_flag = False
            hess_vec_prod.opt_flag = opt_flag
            if opt_flag == 'cg_ad':
                # for recording memory only: maxiter = 5 - 1
                result, time_elapsed, peak_memory = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, hessp=hess_vec_prod.hessp, 
                    callback=hess_vec_prod.callback, options={'maxiter': 5, 'xtol': 1e-20}) # gtol and ftol not applicable
            elif opt_flag == 'cg_fd':
                # Does not converge
                result, time_elapsed, peak_memory = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 5, 'xtol': 1e-20})
            else:
                result, time_elapsed, peak_memory = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='L-BFGS-B', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 20, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 
            postprocess_results(hess_vec_prod, result, time_elapsed, peak_memory)                


def postprocess_results(hess_vec_prod, result, time_elapsed, peak_memory):
    print(result)
    print(f"Time elapsed is {time_elapsed}")
    print(f"J_values = {hess_vec_prod.J_values}")
    print(f"len(J_values) = {len(hess_vec_prod.J_values)}") 
    print(f"More information: {hess_vec_prod.counter}")  
    print(f"Peak memory: {peak_memory} MiB")

    # if hess_vec_prod.timing_flag:
    #     np.save(os.path.join(inv_numpy_dir, f'time_{hess_vec_prod.opt_flag}.npy'), time_elapsed)
    # else:
    #     # num_J, num_grad, num_hessp_full, num_hessp_cached 
    #     np.save(os.path.join(inv_numpy_dir, f'opt_info_{hess_vec_prod.opt_flag}.npy'), np.array(list(hess_vec_prod.counter.values())))
    #     np.save(os.path.join(inv_numpy_dir, f'obj_{hess_vec_prod.opt_flag}.npy'), hess_vec_prod.J_values)


def generate_figures():
    opt_flags = ['cg_ad', 'bgfs']
    labels = ['Newton-CG (AD)', 'L-BFGS-B']
    colors = ['red', 'blue']
    markers = ['o', 's']
    drops = [1, 0]
    plt.figure(figsize=(8, 6))
    # plt.figure()

    for i, opt_flag in enumerate(opt_flags):
        time_elapsed = np.load(os.path.join(inv_numpy_dir, f'time_{opt_flag}.npy'))
        num_J, num_grad, num_hessp_full, num_hessp_cached = np.load(os.path.join(inv_numpy_dir, f'opt_info_{opt_flag}.npy'))
        J_values = np.load(os.path.join(inv_numpy_dir, f'obj_{opt_flag}.npy'))
        opt_steps = len(J_values)

        total_time = np.linspace(0, time_elapsed, opt_steps)
        print(f"\nopt_flag = {opt_flag}")
        print(f"iteration steps = {opt_steps - drops[i]}")
        print(f"J_values = {J_values}")
        print(f"num_J = {num_J}, num_grad = {num_grad}, num_hessp_full = {num_hessp_full}, num_hessp_cached = {num_hessp_cached}")
        print(f"Time elapsed is {total_time[opt_steps - drops[i] - 1]}")
        print(f"Final value for obj = {J_values[opt_steps - drops[i] - 1]}")

        plt.plot(total_time[:opt_steps - drops[i]], J_values[:opt_steps - drops[i]], 
            linestyle='-', marker=markers[i], markersize=8, linewidth=2, color=colors[i], label=labels[i])        

    plt.xlabel(f"Execution time [s]", fontsize=16)
    plt.ylabel("Objective value", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16, frameon=False)   

    # plt.xscale('log')
    plt.yscale('log')

    plt.savefig(os.path.join(inv_pdf_dir, f'obj_y-log.pdf'), bbox_inches='tight')
    # plt.show()


if __name__=="__main__":
    # workflow()
    generate_figures()
