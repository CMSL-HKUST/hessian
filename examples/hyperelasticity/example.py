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
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh 
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
fwd_dir = os.path.join(output_dir, 'forward')
fwd_vtk_dir = os.path.join(output_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(output_dir, 'forward/numpy')
inv_vtk_dir = os.path.join(output_dir, 'inverse/vtk')
inv_numpy_dir = os.path.join(output_dir, 'inverse/numpy')
inv_pdf_dir = os.path.join(output_dir, 'inverse/pdf')
os.makedirs(fwd_numpy_dir, exist_ok=True)
os.makedirs(inv_numpy_dir, exist_ok=True)
os.makedirs(inv_pdf_dir, exist_ok=True)


class HyperElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        def psi(F):
            E = 1e6
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x, load_value):
            return np.array([0., -load_value, 0.])
        return [surface_map]

    def set_params(self, params):
        surface_params = params
        # Generally, [[surface1_params1, surface1_params2, ...], [surface2_params1, surface2_params2, ...], ...]
        self.internal_vars_surfaces = [[surface_params]] 


class HessVecProductTraction(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        self.opt_step += 1
 
        if not self.timing_flag:
            self.J_values.append(self.J_value)
            if self.opt_flag == 'cg_ad':
                inv_vtk_dir, = self.args
                save_sol(self.problem.fes[0], self.u_to_save[0], os.path.join(inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'))
                np.save(os.path.join(inv_numpy_dir, f'traction_{self.opt_step:05d}.npy'), self.unflatten(θ_flat))


def workflow():

    # Specify mesh-related information (first-order hexahedron element).
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly, Lz = 1., 1., 0.05
    meshio_mesh = box_mesh_gmsh(Nx=20, Ny=20, Nz=1, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=fwd_dir, ele_type=ele_type)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def zero_dirichlet_val(point):
        return 0.

    # Define boundary locations.
    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)

    dirichlet_bc_info = [[bottom]*3, [0, 1, 2], [zero_dirichlet_val]*3]
    location_fns = [top]

    # Create an instance of the problem.
    problem = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem) 
    # (num_selected_faces, num_face_quads, dim)
    surface_quad_points = problem.physical_surface_quad_points[0]

    run_forward_flag = True
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*')) + glob.glob(os.path.join(fwd_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        # traction_true = np.ones((surface_quad_points.shape[0], surface_quad_points.shape[1])) 
        traction_true = 1e5*np.exp(-(np.power(surface_quad_points[:, :, 0] - Lx/2., 2)) / (2.*(Lx/5.)**2))

        sol_list_true = fwd_pred(traction_true)

        save_sol(problem.fes[0], sol_list_true[0], os.path.join(fwd_vtk_dir, f'u.vtu'))
        os.makedirs(fwd_numpy_dir, exist_ok=True)
        np.save(os.path.join(fwd_numpy_dir, f'u.npy'), sol_list_true[0])
        np.save(os.path.join(fwd_numpy_dir, f'traction.npy'), np.stack([surface_quad_points[:, :, 0], traction_true]))

    run_inverse_flag = False
    if run_inverse_flag:
        files = glob.glob(os.path.join(inv_vtk_dir, f'*'))
        for f in files:
            os.remove(f)

        sol_list_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]

        def J_fn(u, θ):
            sol_list_pred = u
            l2_error = compute_l2_norm_error(problem, sol_list_pred, sol_list_true)
            θ_vec = jax.flatten_util.ravel_pytree(θ)[0]

            # A good implementation of the optimizer should not depend the scaling factor, but scipy Newton-CG does depend.
            return 1e10*l2_error**2 + 0*np.sum(θ_vec**2)
 
        traction_ini = 1e5*np.ones_like(surface_quad_points)[:, :, 0]
        sol_list_ini = fwd_pred(traction_ini)
        save_sol(problem.fes[0], sol_list_ini[0], os.path.join(inv_vtk_dir, f'u_{0:05d}.vtu'))
        np.save(os.path.join(inv_numpy_dir, f'traction_{0:05d}.npy'), traction_ini)

        opt_flags = ['cg_ad'] # ['cg_ad', 'cg_fd', 'bgfs']
        for opt_flag in opt_flags:
            hess_vec_prod = HessVecProductTraction(problem, J_fn, traction_ini, {}, {}, inv_vtk_dir)
            hess_vec_prod.timing_flag = False
            hess_vec_prod.opt_flag = opt_flag
            if opt_flag == 'cg_ad':
                result, time_elapsed = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, hessp=hess_vec_prod.hessp, 
                    callback=hess_vec_prod.callback, options={'maxiter': 10, 'xtol': 1e-20}) # gtol and ftol not applicable
            elif opt_flag == 'cg_fd':
                # Does not converge
                result, time_elapsed = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 10, 'xtol': 1e-20})
            else:
                result, time_elapsed = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='L-BFGS-B', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 15, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 
            postprocess_results(hess_vec_prod, result, time_elapsed)  



def postprocess_results(hess_vec_prod, result, time_elapsed):
    print(result)
    print(f"Time elapsed is {time_elapsed}")
    print(f"J_values = {hess_vec_prod.J_values}")
    print(f"More information: {hess_vec_prod.counter}")  
    if hess_vec_prod.timing_flag:
        np.save(os.path.join(inv_numpy_dir, f'time_{hess_vec_prod.opt_flag}.npy'), time_elapsed)
    else:
        # num_J, num_grad, num_hessp_full, num_hessp_cached 
        np.save(os.path.join(inv_numpy_dir, f'opt_info_{hess_vec_prod.opt_flag}.npy'), np.array(list(hess_vec_prod.counter.values())))
        np.save(os.path.join(inv_numpy_dir, f'obj_{hess_vec_prod.opt_flag}.npy'), hess_vec_prod.J_values)



def generate_figures():
    opt_flags = ['cg_ad', 'bgfs']
    labels = ['Newton-CG (AD)', 'L-BFGS-B']
    colors = ['red', 'blue']
    markers = ['o', 's']
    drops = [0, 7]
    num_opt_steps = []
    plt.figure(figsize=(8, 6))
    # plt.figure()

    for i, opt_flag in enumerate(opt_flags):
        time_elapsed = np.load(os.path.join(inv_numpy_dir, f'time_{opt_flag}.npy'))
        num_J, num_grad, num_hessp_full, num_hessp_cached = np.load(os.path.join(inv_numpy_dir, f'opt_info_{opt_flag}.npy'))
        J_values = np.load(os.path.join(inv_numpy_dir, f'obj_{opt_flag}.npy'))
        J_values = J_values/1e10
        opt_steps = len(J_values)
        num_opt_steps.append(opt_steps)

        print(f"\nopt_flag = {opt_flag}")
        print(f"Time elapsed is {time_elapsed}")
        print(f"J_values = {J_values}")
        print(f"num_J = {num_J}, num_grad = {num_grad}, num_hessp_full = {num_hessp_full}, num_hessp_cached = {num_hessp_cached}")
        print(f"Final value for obj = {J_values[opt_steps - drops[i] - 1]}")

        plt.plot(np.linspace(0, time_elapsed, opt_steps)[:opt_steps - drops[i]], J_values[:opt_steps - drops[i]], 
            linestyle='-', marker=markers[i], markersize=8, linewidth=2, color=colors[i], label=labels[i])        

    plt.xlabel(f"Execution time [s]", fontsize=18)
    plt.ylabel(r"Objective value [m$^5$]", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, frameon=False)   

    plt.savefig(os.path.join(inv_pdf_dir, f'obj.pdf'), bbox_inches='tight')


    fig = plt.figure(figsize=(8, 6)) 
    x_pos, traction = np.load(os.path.join(fwd_numpy_dir, f'traction.npy'))
    plt.plot(np.mean(x_pos, -1), np.mean(traction, -1)/1e6, color='black', linestyle='-', linewidth=2, label='Rreference')

    opt_steps_cg_ad = num_opt_steps[0]

    steps = [0, 1, 2, 6]
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['(I)', '(II)', '(III)', '(IV)']

    for i, step in enumerate(steps):
        traction = np.load(os.path.join(inv_numpy_dir, f'traction_{step:05d}.npy'))
        plt.plot(np.mean(x_pos, -1), np.mean(traction, -1)/1e6, color=colors[i], linestyle='none', 
            marker='o', markersize=8, label=labels[i])

    plt.xlabel(f"Physical coordinate (x-direction) [m]", fontsize=18)
    plt.ylabel(f"Traction force [MPa]", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18, frameon=False)
    plt.savefig(os.path.join(inv_pdf_dir, f'traction.pdf'), bbox_inches='tight')

    plt.show()


if __name__=="__main__":
    # workflow()
    generate_figures()

