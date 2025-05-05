import numpy as onp
import jax
import jax.numpy as np
import os
import meshio
import glob
import sys
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
                     precision=8)

logger.setLevel(logging.INFO)

output_dir = os.path.join(os.path.dirname(__file__), f'output')
fwd_vtk_dir = os.path.join(output_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(output_dir, 'forward/numpy')
inv_vtk_dir = os.path.join(output_dir, 'inverse/vtk')
inv_numpy_dir = os.path.join(output_dir, 'inverse/numpy')
inv_pdf_dir = os.path.join(output_dir, 'inverse/pdf')

input_dir = os.path.join(os.path.dirname(__file__), f'input')
fwd_mesh_dir = os.path.join(input_dir, 'forward/mesh')

os.makedirs(fwd_numpy_dir, exist_ok=True)
os.makedirs(inv_numpy_dir, exist_ok=True)
os.makedirs(inv_pdf_dir, exist_ok=True)


# Define global parameters (Never to be changed)
T0 = 293. # ambient temperature
E = 70e3
nu = 0.3
mu = E/(2.*(1. + nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
rho = 2700. # density
alpha = 2.31e-5 # thermal expansion coefficient
kappa = alpha*(2*mu + 3*lmbda)
k = 237e-6 # thermal conductivity


# Define the coupling problems.
class ThermalMechanical(Problem):
    def custom_init(self):
        self.fe_u = self.fes[0]
        self.fe_T = self.fes[1]
    
    def get_universal_kernel(self):
        def strain(u_grad):
            return 0.5 * (u_grad + u_grad.T)
        
        def stress(u_grad, T):
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon - kappa * T * np.eye(self.dim)
            return sigma
        
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            ## Split
            # [(num_nodes, vec), ...]
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol_u, cell_sol_T = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads_u, cell_shape_grads_T = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW_u, cell_v_grads_JxW_T = cell_v_grads_JxW_list
            cell_JxW_u, cell_JxW_T = cell_JxW[0], cell_JxW[1]
     
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec)
            T = np.sum(cell_sol_T[None,:,:] * self.fe_T.shape_vals[:,:,None],axis=1)
            # (num_quads, vec, dim)
            u_grads = np.sum(cell_sol_u[None,:,:,None] * cell_shape_grads_u[:,:,None,:], axis=1)

            ## Handles the term 'k * inner(grad(T_crt), grad(Q)) * dx'
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim) 
            # -> (num_quads, vec, dim)
            T_grads = np.sum(cell_sol_T[None,:,:,None] * cell_shape_grads_T[:,:,None,:], axis=1)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec) 
            val3 = np.sum(k * T_grads[:,None,:,:] * cell_v_grads_JxW_T,axis=(0,-1))
            
            ## Handles the term 'inner(sigma, grad(v)) * dx'
            u_physics = jax.vmap(stress)(u_grads, T)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec) 
            val4 = np.sum(u_physics[:,None,:,:] * cell_v_grads_JxW_u,axis=(0,-1))
        
            weak_form = [val4, val3]
            
            return jax.flatten_util.ravel_pytree(weak_form)[0]
        
        return universal_kernel


    def set_params(self, params):
        self.fes[1].vals_list[0] = params


class HessVecProductThermalMechanical(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        logger.info(f"params = {θ_flat}")
        self.opt_step += 1

        if not self.timing_flag:
            self.J_values.append(self.J_value)
            if self.opt_flag == 'cg_ad':
                inv_vtk_dir, J_fn_reg = self.args
                θ = self.unflatten(θ_flat)
                print(f"Regularization = {J_fn_reg(θ)}")
                disp_save = np.hstack((self.cached_vars['u'][0], np.zeros((len(self.cached_vars['u'][0]), 1))))
                T_save = self.cached_vars['u'][1]
                save_sol(self.problem.fes[0], disp_save, os.path.join(inv_vtk_dir, f'disp_{self.opt_step:05d}.vtu'))
                save_sol(self.problem.fes[1], T_save, os.path.join(inv_vtk_dir, f'T_{self.opt_step:05d}.vtu'))
        

def workflow():
    meshio_mesh = meshio.read(os.path.join(fwd_mesh_dir, 'theta_0.vtu'))
    ele_type = 'TRI3'
    cell_type = get_meshio_cell_type(ele_type)
    mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict[cell_type])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[1], 1., atol=1e-5)
   
    def hole(point):
        R = 0.1
        return np.isclose(point[0]**2+point[1]**2, R**2, atol=1e-3)

    def zero_dirichlet(point):
        return 0.

    # The actual hole boundary T will always be updated by T_params, not by this function.
    def T_hole(point):
        return 0.

    def T_top(point):
        return 0.

    def T_right(point):
        return 0.

    dirichlet_bc_info_u = [[hole, hole],[0, 1], [zero_dirichlet]*2]
    dirichlet_bc_info_T = [[hole, top, right],[0, 0, 0],[T_hole, T_top, T_right]]

    problem = ThermalMechanical([mesh, mesh], vec=[2, 1], dim=2, ele_type=[ele_type, ele_type], gauss_order=[1, 1],
                                      dirichlet_bc_info=[dirichlet_bc_info_u, dirichlet_bc_info_T])

    hole_boundary_node_inds = problem.fes[1].node_inds_list[0]
    hole_boundary_nodes = mesh.points[hole_boundary_node_inds]
    angles = onp.arctan2(hole_boundary_nodes[:, 1], hole_boundary_nodes[:, 0])
    sorted_indices = onp.argsort(angles)
    num_hole_boundary_nodes = len(hole_boundary_node_inds)
    corner_node_id = 3456 # Top right corner nodal index, obtained from visualization in Paraview

    fwd_pred = ad_wrapper(problem) 

    run_forward_flag = False
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*'))
        for f in files:
            os.remove(f)

        T_params = 100.*np.ones(num_hole_boundary_nodes)
        sol_list_true = fwd_pred(T_params)

        print(sol_list_true[0].shape)
    
        save_sol(problem.fes[0], np.hstack((sol_list_true[0], np.zeros((len(sol_list_true[0]), 1)))), os.path.join(fwd_vtk_dir, f'disp.vtu'))
        save_sol(problem.fes[1], sol_list_true[1], os.path.join(fwd_vtk_dir, f'T.vtu'))

        corner_disp_ref = sol_list_true[0][corner_node_id]
        print(f'disp of corner = {corner_disp_ref}')

    run_inverse_flag = True
    if run_inverse_flag:
        files = glob.glob(os.path.join(inv_vtk_dir, f'*')) 
        for f in files:
            os.remove(f)

        def J_fn_reg(θ):
            θ_sorted  = θ[sorted_indices]
            reg_term = np.sum(np.diff(θ_sorted)**2)
            reg_l1 = np.sum(np.abs(np.diff(θ_sorted))) # Total variation
            reg_l2 = np.sum(np.diff(θ_sorted)**2) # Tikhonov
            return 1*reg_term

        def J_fn(u, θ):
            sol_list_pred = u
            corner_disp_pred = sol_list_pred[0][corner_node_id]
            corner_disp_goal = np.array([0.001, -0.001])
            error = np.sum(((corner_disp_pred - corner_disp_goal)**2))
            return 1e10*error + J_fn_reg(θ)
 
        # T_params_ini = 100.*np.ones(num_hole_boundary_nodes)
        # sol_list_ini = fwd_pred(T_params_ini)
        # save_sol(problem.fes[0], np.hstack((sol_list_ini[0], np.zeros((len(sol_list_ini[0]), 1)))), os.path.join(inv_vtk_dir, f'disp_{0:05d}.vtu'))
        # save_sol(problem.fes[1], sol_list_ini[1], os.path.join(inv_vtk_dir, f'T_{0:05d}.vtu'))

        # option_petsc = {'petsc_solver': {'ksp_type': 'tfqmr', 'pc_type': 'lu'}}
        # option_umfpack = {'umfpack_solver': {}}
        # hess_vec_prod = HessVecProductThermalMechanical(problem, J_fn, T_params_ini, option_umfpack, option_umfpack, inv_vtk_dir, J_fn_reg)

        # # result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
        # #     hessp=hess_vec_prod.hessp, callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-30})

        # result = minimize(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, method='newton-cg', jac=hess_vec_prod.grad, 
        #     callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-30})

        # # result = minimize(hess_vec_prod.J, hess_vec_prod.θ_ini_flat, method='L-BFGS-B', jac=hess_vec_prod.grad, 
        # #     callback=hess_vec_prod.callback, options={'maxiter': 100, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 

        T_params_ini = 100.*np.ones(num_hole_boundary_nodes)
        sol_list_ini = fwd_pred(T_params_ini)
        save_sol(problem.fes[0], np.hstack((sol_list_ini[0], np.zeros((len(sol_list_ini[0]), 1)))), os.path.join(inv_vtk_dir, f'disp_{0:05d}.vtu'))
        save_sol(problem.fes[1], sol_list_ini[1], os.path.join(inv_vtk_dir, f'T_{0:05d}.vtu'))

        # opt_flags = ['cg_ad', 'cg_fd', 'bgfs']
        opt_flags = ['bgfs']
        for opt_flag in opt_flags:
            option_petsc = {'petsc_solver': {'ksp_type': 'tfqmr', 'pc_type': 'lu'}}
            option_umfpack = {'umfpack_solver': {}}        
            hess_vec_prod = HessVecProductThermalMechanical(problem, J_fn, T_params_ini, option_umfpack, option_umfpack, inv_vtk_dir, J_fn_reg)
            hess_vec_prod.timing_flag = True
            hess_vec_prod.opt_flag = opt_flag
            if opt_flag == 'cg_ad':
                result, time_elapsed = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, hessp=hess_vec_prod.hessp, 
                    callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-20}) # gtol and ftol not applicable
            elif opt_flag == 'cg_fd':
                result, time_elapsed = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-20})
            else:
                result, time_elapsed = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='L-BFGS-B', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 30, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 
            postprocess_results(hess_vec_prod, result, time_elapsed) 

            print(f"Final corner disp = {fwd_pred(hess_vec_prod.unflatten(result.x))[0][corner_node_id]}")


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
    opt_flags = ['cg_ad', 'cg_fd', 'bgfs']
    labels = ['Newton-CG (AD)', 'Newton-CG (FD)', 'L-BFGS-B']
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    # plt.figure(figsize=(10, 10))
    plt.figure()

    for i, opt_flag in enumerate(opt_flags):
        time_elapsed = np.load(os.path.join(inv_numpy_dir, f'time_{opt_flag}.npy'))
        num_J, num_grad, num_hessp_full, num_hessp_cached = np.load(os.path.join(inv_numpy_dir, f'opt_info_{opt_flag}.npy'))
        J_values = np.load(os.path.join(inv_numpy_dir, f'obj_{opt_flag}.npy'))

        print(f"opt_flag = {opt_flag}")
        print(f"Time elapsed is {time_elapsed}")
        print(f"J_values = {J_values}")
        print(f"num_J = {num_J}, num_grad = {num_grad}, num_hessp_full = {num_hessp_full}, num_hessp_cached = {num_hessp_cached}")

        plt.plot(np.linspace(0, time_elapsed, len(J_values)), J_values, 
            linestyle='-', marker=markers[i], markersize=10, linewidth=2, color=colors[i], label=labels[i])        

        plt.xlabel(f"Execution time [s]", fontsize=20)
        plt.ylabel("Objective value", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)   

    plt.savefig(os.path.join(inv_pdf_dir, f'obj.pdf'), bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    workflow()
    generate_figures()
