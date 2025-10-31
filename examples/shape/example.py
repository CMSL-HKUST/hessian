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

# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


output_dir = os.path.join(os.path.dirname(__file__), f'output')
fwd_dir = os.path.join(output_dir, 'forward')
fwd_vtk_dir = os.path.join(output_dir, 'forward/vtk')
inv_vtk_dir = os.path.join(output_dir, 'inverse/vtk')
inv_numpy_dir = os.path.join(output_dir, 'inverse/numpy')
inv_pdf_dir = os.path.join(output_dir, 'inverse/pdf')

os.makedirs(inv_numpy_dir, exist_ok=True)
os.makedirs(inv_pdf_dir, exist_ok=True)


def scaled_sigmoid(x, lower_lim, upper_lim, p=0.1):
    return lower_lim + (upper_lim - lower_lim)/((1. + np.exp(-x*p)))


def pore_fn(x, pore_center, L0, beta):
    beta = scaled_sigmoid(beta, -np.pi/4., np.pi/4., p=0.1)
    porosity = 0.5
    theta = np.arctan2(x[1] - pore_center[1], x[0] - pore_center[0]) 
    r = np.sqrt(np.sum((x - pore_center)**2))
    x_rel = r*np.cos(theta - beta)
    y_rel = r*np.sin(theta - beta)
    p = 200.
    rho = 1./(1. + np.exp(-(np.abs(x_rel) + np.abs(y_rel) - 0.9*L0/2)*p))
    return rho

pore_fn_vmap = jax.vmap(pore_fn, in_axes=(0, None, None, None))


class Elasticity(Problem):
    def custom_init(self, Lx, Ly, nx, ny):
        self.fe = self.fes[0]
        # (num_cells, num_quads, dim)
        physical_quad_points = self.fe.get_physical_quad_points()
        L0 = Lx/nx
        self.pore_center_list = []
        self.quad_inds_list = []
        for i in range(nx):
            for j in range(ny):
                pore_center = np.array([i*L0 + L0/2., j*L0 + L0/2.])
                self.pore_center_list.append(pore_center)
                # (num_selected_quad_points, 2)
                quad_inds = np.argwhere((physical_quad_points[:, :, 0] >= i*L0) &
                                        (physical_quad_points[:, :, 0] < (i + 1)*L0) & 
                                        (physical_quad_points[:, :, 1] >= j*L0) &
                                        (physical_quad_points[:, :, 1] < (j + 1)*L0))
                self.quad_inds_list.append(quad_inds)
        self.L0 = L0

    def get_tensor_map(self):
        def psi(F_2d, rho):
            # Plane strain
            F = np.array([[F_2d[0, 0], F_2d[0, 1], 0.], 
                          [F_2d[1, 0], F_2d[1, 1], 0.],
                          [0., 0., 1.]])
            Emax = 1e6  
            Emin = 1e-3*Emax
            E = Emin + (Emax - Emin)*rho
            nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, rho):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, rho)
            return P
        return first_PK_stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0, 1e4])
        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        beta = params
        rhos = np.ones((self.fe.num_cells, self.fe.num_quads))
        for i in range(len(self.pore_center_list)):
            quad_inds = self.quad_inds_list[i]
            # (num_selected_quad_points, dim)
            quad_points = self.physical_quad_points[quad_inds[:, 0], quad_inds[:, 1]]
            pore_center = self.pore_center_list[i]
            rho_vals = pore_fn_vmap(quad_points, pore_center, self.L0, beta[i])
            rhos = rhos.at[quad_inds[:, 0], quad_inds[:, 1]].set(rho_vals)
        self.internal_vars = [rhos]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


class HessVecProductPore(HessVecProduct):
    def callback(self, θ_flat):
        logger.info(f"########################## hess_vec_prod.callback is called for optimization step {self.opt_step} ...")
        if not self.timing_flag:
            print(f"solution size = {self.u_to_save[0].size}, parameter size = {θ_flat.size}")
            self.J_values.append(self.J_value)
            if self.opt_flag == 'cg_ad':
                θ = self.unflatten(θ_flat)
                sol_list = self.u_to_save
                rho = jax.lax.stop_gradient(self.problem.internal_vars[0])
                inv_vtk_dir, = self.args
                # save_sol(self.problem.fe, np.hstack((sol_list[0], np.zeros((len(sol_list[0]), 1)))), 
                #          os.path.join(inv_vtk_dir, f'u_{self.opt_step:05d}.vtu'), cell_infos=[('rho', np.mean(rho, axis=-1))])

                logger.info(f"########################## θ = \n{θ}")
        self.opt_step += 1
        

def workflow():
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 0.5
    nx, ny = 4, 2 # pore numbers along x-axis and y-axis
    meshio_mesh = rectangle_mesh(Nx=100, Ny=50, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load_location(point):
        return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

    def dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]

    location_fns = [load_location]

    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                         location_fns=location_fns, additional_info=(Lx, Ly, nx, ny))
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})

    run_forward_flag = False
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*'))
        for f in files:
            os.remove(f)

        params = np.array([0.])
        sol_list_true = fwd_pred(params)
        save_sol(problem.fe, np.hstack((sol_list_true[0], np.zeros((len(sol_list_true[0]), 1)))), 
            os.path.join(fwd_vtk_dir, f'u.vtu'), cell_infos=[('rho', np.mean(problem.internal_vars[0], axis=-1))])

    run_inverse_flag = True
    if run_inverse_flag:
        files = glob.glob(os.path.join(inv_vtk_dir, f'*'))
        for f in files:
            os.remove(f)

        def J_fn(u, θ):
            sol_list = u
            compliace = problem.compute_compliance(sol_list[0])
            return 1e5*compliace

        θ_ini = np.array([0.]*nx*ny)
        u_ini = fwd_pred(θ_ini)
        # save_sol(problem.fe, np.hstack((u_ini[0], np.zeros((len(u_ini[0]), 1)))), 
        #     os.path.join(inv_vtk_dir, f'u_{0:05d}.vtu'), cell_infos=[('rho', np.mean(problem.internal_vars[0], axis=-1))])

        opt_flags = ['cg_ad'] # ['cg_ad', 'cg_fd', 'bgfs']
        for opt_flag in opt_flags:
            option_umfpack = {'umfpack_solver': {}}
            hess_vec_prod = HessVecProductPore(problem, J_fn, θ_ini, option_umfpack, option_umfpack, inv_vtk_dir)
            hess_vec_prod.timing_flag = False
            hess_vec_prod.opt_flag = opt_flag
            if opt_flag == 'cg_ad':
                result, time_elapsed, peak_memory = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, hessp=hess_vec_prod.hessp, 
                    callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-20})
            elif opt_flag == 'cg_fd':
                result, time_elapsed, peak_memory = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='newton-cg', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 6, 'xtol': 1e-20})
            else:
                # for recording memory only: maxiter = 10 - 1
                result, time_elapsed, peak_memory = timing_wrapper(minimize)(fun=hess_vec_prod.J, x0=hess_vec_prod.θ_ini_flat, 
                    method='L-BFGS-B', jac=hess_vec_prod.grad, 
                    callback=hess_vec_prod.callback, options={'maxiter': 10, 'disp': True, 'gtol': 1e-20, 'xtol': 1e-30}) 
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
    opt_flags = ['cg_ad', 'cg_fd', 'bgfs']
    labels = ['Newton-CG (AD)', 'Newton-CG (FD)', 'L-BFGS-B']
    colors = ['red', 'green', 'blue']
    markers = ['o', '^', 's']
    drops = [0, 0, 1]
    num_opt_steps = []
    plt.figure(figsize=(8, 6))
    # plt.figure()

    for i, opt_flag in enumerate(opt_flags):
        time_elapsed = np.load(os.path.join(inv_numpy_dir, f'time_{opt_flag}.npy'))
        num_J, num_grad, num_hessp_full, num_hessp_cached = np.load(os.path.join(inv_numpy_dir, f'opt_info_{opt_flag}.npy'))
        J_values = np.load(os.path.join(inv_numpy_dir, f'obj_{opt_flag}.npy'))
        J_values = J_values/1e5
        opt_steps = len(J_values)
        num_opt_steps.append(opt_steps)

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
    plt.ylabel("Objective value [J]", fontsize=16)
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
