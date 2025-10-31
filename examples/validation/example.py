import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as onp
import jax
import jax.numpy as np

import glob
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import time
from memory_profiler import memory_usage
 
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem import logger

from hessian.manager import HessVecProduct, central_finite_difference_hessp
from hessian.utils import compute_l2_norm_error

platform = jax.lib.xla_bridge.get_backend().platform
print(f"platform = {platform}")


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


logger.setLevel(logging.DEBUG)

data_dir = os.path.join(os.path.dirname(__file__), f'output')
fwd_vtk_dir = os.path.join(data_dir, 'forward/vtk')
fwd_numpy_dir = os.path.join(data_dir, 'forward/numpy')
val_numpy_dir = os.path.join(data_dir, 'inverse/validation/numpy')
val_pdf_dir = os.path.join(data_dir, 'inverse/validation/pdf')
prof_numpy_dir = os.path.join(data_dir, 'inverse/profiling/numpy')
prof_pdf_dir = os.path.join(data_dir, 'inverse/profiling/pdf')


CPC_revision_dir = os.path.join(data_dir, 'CPC_revision')
CPC_q1_dir = os.path.join(CPC_revision_dir, 'q1')
os.makedirs(CPC_q1_dir, exist_ok=True)
CPC_q236_dir = os.path.join(CPC_revision_dir, 'q236')
os.makedirs(CPC_q236_dir, exist_ok=True)


class NonlinearPoisson(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_universal_kernel(self):
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, theta):
            # Handles the term `exp(theta*u) * inner(grad(u), grad(v)*dx`

            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            # theta: (num_quads,)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            # cell_sol_u: (num_nodes_u, vec)
            cell_sol_u, = cell_sol_list
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            # (num_quads, num_nodes_u, dim)
            cell_shape_grads_u, = cell_shape_grads_list
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            # (num_quads, num_nodes_u, 1, dim)
            cell_v_grads_JxW_u, = cell_v_grads_JxW_list

            # (1, num_nodes_u, vec_u) * (num_quads, num_nodes_u, 1) -> (num_quads, num_nodes_u, vec_u) -> (num_quads, vec_u)
            u = np.sum(cell_sol_u[None, :, :] * self.fe.shape_vals[:, :, None], axis=1)

            # (1, num_nodes_u, vec_u, 1) * (num_quads, num_nodes_u, 1, dim) -> (num_quads, num_nodes_u, vec_u, dim)
            u_grads = cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec_u, dim)

            # (num_quads, num_nodes_u, vec_u, dim) -> (num_nodes_u, vec_u)
            val = np.sum(np.exp(theta[:, None, None, None] * u[:, None, :, None]) * u_grads[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))
            weak_form = [val] # [(num_nodes, vec), ...]

            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return universal_kernel

    def get_mass_map(self):
        def mass_map(u, x, theta):
            val = -np.array([10*np.exp(-(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02)])
            return val
        return mass_map

    def get_surface_maps(self):
        def surface_map(u, x):
            return -np.array([np.sin(5.*x[0])])

        return [surface_map, surface_map]

    def set_params(self, theta):
        self.internal_vars = [theta]


def profile_hessp(hess_vec_prod):
    θ_flat = jax.random.normal(jax.random.key(1), hess_vec_prod.θ_ini_flat.shape)
    θ_hat_flat = jax.random.normal(jax.random.key(2), hess_vec_prod.θ_ini_flat.shape)

    hessp_options = ['fwd_rev', 'rev_fwd', 'rev_rev']
    num_loops = 11
    J_times = []
    F_times = []
    for index, hessp_option in enumerate(hessp_options):
        hess_vec_prod.hessp_option = hessp_option
        J_times.append([])
        F_times.append([])
        for i in range(num_loops):
            hess_vec_prod.hessp(θ_flat, θ_hat_flat)
            J_time, F_time = hess_vec_prod.profile_info
            J_times[-1].append(J_time)
            F_times[-1].append(F_time)
            print(f"option = {hessp_option}, J_time = {J_time}, F_time = {F_time}")

    profile_results = np.array([J_times, F_times])
    os.makedirs(prof_numpy_dir, exist_ok=True)
    np.save(os.path.join(prof_numpy_dir, f'profile_results_{time.perf_counter_ns()}.npy'), profile_results)


def hessian_validation(hess_vec_prod, h, seed_1=1, seed_2=2, seed_3=3):
    θ_flat = jax.random.normal(jax.random.key(seed_1), hess_vec_prod.θ_ini_flat.shape)
    θ_hat_flat = jax.random.normal(jax.random.key(seed_2), hess_vec_prod.θ_ini_flat.shape)
    θ_tilde_flat = jax.random.normal(jax.random.key(seed_3), hess_vec_prod.θ_ini_flat.shape)
    hess_v_ad = hess_vec_prod.hessp(θ_flat, θ_hat_flat)
    hess_v_fd = central_finite_difference_hessp(hess_vec_prod, θ_flat, θ_hat_flat, h)
    hess_v_ad_flat = jax.flatten_util.ravel_pytree(hess_v_ad)[0]
    hess_v_fd_flat = jax.flatten_util.ravel_pytree(hess_v_fd)[0]
    v_hess_v_ad = np.dot(θ_tilde_flat, hess_v_ad_flat)
    v_hess_v_fd = np.dot(θ_tilde_flat, hess_v_fd_flat)
    rel_err = np.linalg.norm(hess_v_ad_flat - hess_v_fd_flat)/np.linalg.norm(hess_v_ad_flat)
    logger.info(f"\n")
    logger.info(f"v_hess_v_ad = {v_hess_v_ad}, v_hess_v_fd = {v_hess_v_fd}")
    logger.info(f"\n")
    return v_hess_v_ad, v_hess_v_fd, rel_err


def taylor_remainder_test(hess_vec_prod, h, seed_1=1, seed_2=2):
    θ_flat = jax.random.normal(jax.random.key(seed_1), hess_vec_prod.θ_ini_flat.shape)
    θ_hat_flat = jax.random.normal(jax.random.key(seed_2), hess_vec_prod.θ_ini_flat.shape)

    f_val = hess_vec_prod.J(θ_flat)
    θ_plus_flat = θ_flat + h*θ_hat_flat
    f_plus_val = hess_vec_prod.J(θ_plus_flat)

    f_grad = hess_vec_prod.grad(θ_flat)
    f_grad_flat = jax.flatten_util.ravel_pytree(f_grad)[0]
    v_f_grad = h*np.dot(θ_hat_flat, f_grad_flat)

    hess_v = hess_vec_prod.hessp(θ_flat, θ_hat_flat)
    hess_v_flat = jax.flatten_util.ravel_pytree(hess_v)[0]
    v_hess_v = h**2./2.*np.dot(θ_hat_flat, hess_v_flat)

    return f_plus_val, f_val, v_f_grad, v_hess_v


def workflow(N=64, q236_num_seeds=100):
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1., 1.


    meshio_mesh = rectangle_mesh(Nx=N, Ny=N, domain_x=Lx, domain_y=Ly)
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

    location_fns_dirichlet = [left, right]
    value_fns = [dirichlet_val_left, dirichlet_val_right]
    vecs = [0, 0]
    dirichlet_bc_info = [location_fns_dirichlet, vecs, value_fns]

    location_fns = [bottom, top]
    problem = NonlinearPoisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    option_umfpack = {'umfpack_solver': {}}
    fwd_pred = ad_wrapper(problem, solver_options=option_umfpack, adjoint_solver_options=option_umfpack) 

    # (num_cells, num_quads, dim)
    quad_points = problem.fes[0].get_physical_quad_points()

    run_forward_flag = False
    if run_forward_flag:
        files = glob.glob(os.path.join(fwd_vtk_dir, f'*')) + glob.glob(os.path.join(fwd_numpy_dir, f'*'))
        for f in files:
            os.remove(f)

        theta_true =  np.ones_like(quad_points)[:, :, 0]
        sol_list_true = fwd_pred(theta_true)

        save_sol(problem.fes[0], sol_list_true[0], os.path.join(fwd_vtk_dir, f'u.vtu'), cell_infos=[('theta', np.mean(theta_true, axis=-1))])
        os.makedirs(fwd_numpy_dir, exist_ok=True)
        np.save(os.path.join(fwd_numpy_dir, f'u.npy'), sol_list_true[0])

    run_inverse_flag = False
    if run_inverse_flag:
        sol_list_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]
        def J_fn(u, θ):
            sol_list_pred = u
            l2_u = compute_l2_norm_error(problem, sol_list_pred, sol_list_true)
            return l2_u**2

        theta_ini = np.zeros_like(quad_points)[:, :, 0]
        hess_vec_prod = HessVecProduct(problem, J_fn, theta_ini, option_umfpack, option_umfpack, None)

        run_profiling_flag = True
        if run_profiling_flag:
            profile_hessp(hess_vec_prod)

        run_validation_flag = True
        if run_validation_flag:
            files = glob.glob(os.path.join(val_numpy_dir, f'*')) 
            for f in files:
                os.remove(f)

            hessp_options = ['fwd_rev', 'rev_fwd', 'rev_rev']
            for index, hessp_option in enumerate(hessp_options):
                hess_vec_prod.hessp_option = hessp_option

                # Do we need this line?
                hessian_validation(hess_vec_prod, h=1e-3)

                hs = [1e-1, 1e-2, 1e-3, 1e-4]

                taylor_results = []
                for h in hs:
                    f_plus_val, f_val, v_f_grad, v_hess_v = taylor_remainder_test(hess_vec_prod, h)
                    taylor_results.append([h, f_plus_val, f_val, v_f_grad, v_hess_v])

                taylor_results = np.array(taylor_results)
                os.makedirs(val_numpy_dir, exist_ok=True)
                np.save(os.path.join(val_numpy_dir, f'taylor_results_{hessp_option}.npy'), taylor_results)

                num_seeds = 100
                vHv_results = []
                for h in hs:
                    vHv_results.append([])
                    for i in range(num_seeds):
                        print(f"\n\n######################## Random testing {hessp_option}: h = {h}, seed = {i + 1} ")
                        seed_1 = i + 1
                        seed_2 = i + 1 + num_seeds
                        seed_3 = i + 1 + 2*num_seeds
                        v_hess_v_ad, v_hess_v_fd, rel_err = hessian_validation(hess_vec_prod, h, seed_1, seed_2, seed_3)
                        vHv_results[-1].append([v_hess_v_ad, v_hess_v_fd, rel_err])

                vHv_results = np.array(vHv_results)
                np.save(os.path.join(val_numpy_dir, f'vHv_results_{hessp_option}.npy'), vHv_results)

    run_CPC_flag = True
    if run_CPC_flag:
        sol_list_true = [np.load(os.path.join(fwd_numpy_dir, f'u.npy'))]
        def J_fn(u, θ):
            sol_list_pred = u
            l2_u = compute_l2_norm_error(problem, sol_list_pred, sol_list_true)
            return l2_u**2

        theta_ini = np.zeros_like(quad_points)[:, :, 0]
        hess_vec_prod = HessVecProduct(problem, J_fn, theta_ini, option_umfpack, option_umfpack, None)
        hessp_option = 'rev_fwd'
        hess_vec_prod.hessp_option = hessp_option

        run_q1_flag = False
        if run_q1_flag:
            # files = glob.glob(os.path.join(CPC_q1_dir, f'*')) 
            # for f in files:
            #     os.remove(f)

            hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
            num_seeds = 50
            seed_results = []
            for i in range(num_seeds):  
                print(f"\n\n######################## i = {i}, num_seeds = {num_seeds}")
                seed_results.append([])
                for h in hs:
                    f_plus_val, f_val, v_f_grad, v_hess_v = taylor_remainder_test(hess_vec_prod, h, seed_1=1, seed_2=2+i)
                    seed_results[-1].append([h, f_plus_val, f_val, v_f_grad, v_hess_v])

            seed_results = np.array(seed_results)
            np.save(os.path.join(CPC_q1_dir, f'seed_results_{hessp_option}.npy'), seed_results)

        run_q236_flag = True
        if run_q236_flag:
            num_seeds = q236_num_seeds
            time_profile_results = []
            θ_flat = jax.random.normal(jax.random.key(1), hess_vec_prod.θ_ini_flat.shape)

            for i in range(num_seeds):  
                print(f"\n\n######################## i = {i}, num_seeds = {num_seeds}")
                θ_hat_flat = jax.random.normal(jax.random.key(2 + i), hess_vec_prod.θ_ini_flat.shape)
                hess_vec_prod.hessp(θ_flat, θ_hat_flat)
                timer = np.diff(np.array(hess_vec_prod.timer))
                profile_info = np.array(hess_vec_prod.profile_info)
                print(f"timer = {timer}")
                print(f"profile_info = {profile_info}")
                # timer: 5 time intervals for the 5 steps in Hessian-vector products
                # profile_info: J and F time in Step 4
                time_profile_results.append(np.hstack((timer, profile_info)))

            time_profile_results = np.array(time_profile_results)
            print(f"time_profile_results.shape = {time_profile_results.shape}")
            np.save(os.path.join(CPC_q236_dir, f'time_profile_N_{N:05d}_seed_{q236_num_seeds}_{platform}.npy'), time_profile_results)


def size_test_CPC_revision():
    run_q236_flag = False
    if run_q236_flag:
        # q2
        # N = 256
        # q236_num_seeds = 101
        # peak_memory = memory_usage((workflow, (N, q236_num_seeds), {}), max_usage=True)
        # print(f"Peak memory: {peak_memory} MiB")
        # np.save(os.path.join(CPC_q236_dir, f'memory_profile_N_{N:05d}_seed_{q236_num_seeds}_{platform}.npy'), peak_memory)

        # q3 and q6
        Ns = [64, 128, 256, 512, 1024]
        Ns = [64]
        q236_num_seeds = 4
        for N in Ns:
            print(f"\n\n########################## N = {N}")
            peak_memory = memory_usage((workflow, (N, q236_num_seeds), {}), max_usage=True)
            print(f"Peak memory: {peak_memory} MiB")
            np.save(os.path.join(CPC_q236_dir, f'memory_profile_N_{N:05d}_seed_{q236_num_seeds}_{platform}.npy'), peak_memory)
     
    figures_q2_flag = False
    if figures_q2_flag: 
        # q2
        N = 64
        q236_num_seeds = 101
        time_profile_results = np.load(os.path.join(CPC_q236_dir, f'time_profile_N_{N:05d}_seed_{q236_num_seeds}_{platform}.npy'))
        print(f"Total time for the first call is {np.sum(time_profile_results[0, :5])}")
        print(f"Avg time for the steady call is {np.mean(np.sum(time_profile_results[1:, :5], axis=1))}")

        plt.figure(figsize=(10, 10))
        plt.plot(np.arange(5) + 1, time_profile_results[0][:5], linestyle='-', linewidth=1, 
            marker='o', markersize=8, color='blue', label="First call")
   
        for i in range(q236_num_seeds - 1):
            if i == 0:
                plt.plot(np.arange(5) + 1, time_profile_results[i + 1][:5], linestyle='-', linewidth=1, 
                    marker='o', markersize=8, color='black', label="Steady-state calls")
            else:
                plt.plot(np.arange(5) + 1, time_profile_results[i + 1][:5], linestyle='-', linewidth=1, 
                    marker='o', markersize=8, color='black')

        x_positions = np.arange(5) + 1  # [1, 2, 3, 4, 5]
        x_labels = ['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5']
        plt.xticks(ticks=x_positions, labels=x_labels)
        # plt.xlabel(r"Hessian-vector product step", fontsize=20)
        plt.ylabel("Time [s]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(os.path.join(CPC_q236_dir, f'q2_time_profile_N_{N:05d}.pdf'), bbox_inches='tight')

    figures_q3_flag = False
    if figures_q3_flag: 
        # q3
        q236_num_seeds = 4
        Ns = np.array([64, 128, 256, 512, 1024])

        first_call_time = []
        steady_call_time = []
        peak_memories = []

        for N in Ns:
            time_profile_results = np.load(os.path.join(CPC_q236_dir, f'time_profile_N_{N:05d}_seed_{q236_num_seeds}_{platform}.npy'))
            peak_memory = np.load(os.path.join(CPC_q236_dir, f'memory_profile_N_{N:05d}_seed_{q236_num_seeds}_{platform}.npy'))

            first_call_time.append(time_profile_results[0, :5])
            steady_call_time.append(np.mean(time_profile_results[1:, :5], axis=0))
            peak_memories.append(peak_memory)

        first_call_time = np.array(first_call_time)
        steady_call_time = np.array(steady_call_time)
        peak_memories = np.array(peak_memories)

        # Time profile plot
        plt.figure(figsize=(10, 10))
        colors = ['red', 'black', 'blue', 'green', 'orange']
        num_Hv_steps = 5
        for i in range(num_Hv_steps):
            plt.plot(4*Ns**2, first_call_time[:, i], linestyle='-', linewidth=1, 
                    marker='o', markersize=8, color=colors[i], label=f"First call - Step {i + 1}")

        for i in range(num_Hv_steps):
            if i > 1:
                plt.plot(4*Ns**2, steady_call_time[:, i], linestyle='--', linewidth=1, 
                        marker='s', markersize=8, color=colors[i], label=f"Steady call - Step {i + 1}")

        plt.xlabel(r"Size of parameter vector", fontsize=20)
        plt.ylabel("Time [s]", fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(os.path.join(CPC_q236_dir, f'q3_time_profile.pdf'), bbox_inches='tight')

        # Peak memory profile plot
        plt.figure(figsize=(10, 10))
        num_Hv_steps = 5
        plt.plot(4*Ns**2, peak_memories, linestyle='-', linewidth=1, 
                marker='o', markersize=8, color='black')

        plt.xlabel(r"Size of parameter vector", fontsize=20)
        plt.ylabel("Peak memory [MiB]", fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(os.path.join(CPC_q236_dir, f'q3_memory_profile.pdf'), bbox_inches='tight')


    figures_q6_flag = True
    if figures_q6_flag: 
        q236_num_seeds = 4
        Ns = np.array([64, 128, 256, 512, 1024])

        first_call_time_cpu = []
        steady_call_time_cpu = []
        first_call_time_gpu = []
        steady_call_time_gpu = []

        for N in Ns:
            time_profile_results_cpu = np.load(os.path.join(CPC_q236_dir, f'time_profile_N_{N:05d}_seed_{q236_num_seeds}_cpu.npy'))
            time_profile_results_gpu = np.load(os.path.join(CPC_q236_dir, f'time_profile_N_{N:05d}_seed_{q236_num_seeds}_gpu.npy'))

            first_call_time_cpu.append(time_profile_results_cpu[0, 5:])
            steady_call_time_cpu.append(np.mean(time_profile_results_cpu[1:, 5:], axis=0))
            first_call_time_gpu.append(time_profile_results_gpu[0, 5:])
            steady_call_time_gpu.append(np.mean(time_profile_results_gpu[1:, 5:], axis=0))
 
        first_call_time_cpu = np.array(first_call_time_cpu)
        steady_call_time_cpu = np.array(steady_call_time_cpu)
        first_call_time_gpu = np.array(first_call_time_gpu)
        steady_call_time_gpu = np.array(steady_call_time_gpu)

        plt.figure(figsize=(10, 10))
        plt.plot(4*Ns**2, first_call_time_cpu[:, 1], linestyle='-', linewidth=1, 
                 marker='o', markersize=8, color='blue', label=f"First call - CPU")
        plt.plot(4*Ns**2, first_call_time_gpu[:, 1], linestyle='-', linewidth=1, 
                 marker='o', markersize=8, color='red', label=f"First call - GPU")

        plt.plot(4*Ns**2, steady_call_time_cpu[:, 1], linestyle='--', linewidth=1, 
                 marker='s', markersize=8, color='blue', label=f"Stead call - CPU")
        plt.plot(4*Ns**2, steady_call_time_gpu[:, 1], linestyle='--', linewidth=1, 
                 marker='s', markersize=8, color='red', label=f"Stead call - GPU")

        plt.xlabel(r"Size of parameter vector", fontsize=20)
        plt.ylabel("Time [s]", fontsize=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)
        plt.savefig(os.path.join(CPC_q236_dir, f'q6_time_profile_F.pdf'), bbox_inches='tight')

        plt.show()

def generate_figures_CPC_revision():

    # Reviewer 2: Repeating the Taylor-remainder check over many random parameter directions
    hessp_option = 'rev_fwd' # Most efficient
    seed_results = np.load(os.path.join(CPC_q1_dir, f"seed_results_{hessp_option}.npy"))
    slopes = []

    for taylor_results in seed_results:

        hs, f_plus_val, f_val, v_f_grad, v_hess_v = taylor_results.T
        res_zero = np.abs(f_plus_val - f_val)
        res_first = np.abs(f_plus_val - f_val - v_f_grad)
        res_second = np.abs(f_plus_val - f_val - v_f_grad - v_hess_v)
        slopes.append([])
        for i in range(len(hs) - 1):
            # slope = np.log(res_second[0]/res_second[i + 1])/np.log(10.**(i + 1))
            slope = np.log(res_second[i]/res_second[i + 1])/np.log(10.**(1))
            slopes[-1].append(slope)
            # print(slope)
        # print("\n")

    plt.figure(figsize=(10, 10))

    for i in range(len(slopes)):
        plt.plot(np.arange(len(slopes[i])) + 1, slopes[i], linestyle='-', linewidth=0.5, color='blue')
    
    plt.plot(np.arange(len(slopes[0])) + 1, 3.*np.ones(len(slopes[0])), linestyle='-', linewidth=2, color='red', label="Reference")
    plt.xlabel(r"Line segment", fontsize=20)
    plt.ylabel("Slope", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   

    plt.savefig(os.path.join(CPC_q1_dir, f'slope.pdf'), bbox_inches='tight')

    taylor_results = seed_results[0]
    hs, f_plus_val, f_val, v_f_grad, v_hess_v = taylor_results.T
    res_zero = np.abs(f_plus_val - f_val)
    res_first = np.abs(f_plus_val - f_val - v_f_grad)
    res_second = np.abs(f_plus_val - f_val - v_f_grad - v_hess_v)

    ref_zero = [1/5.*res_zero[0]/hs[0] * h for h in hs]
    ref_first = [1/5.*res_first[0]/hs[0]**2 * h**2 for h in hs]
    ref_second = [1/5.*res_second[0]/hs[0]**3 * h**3 for h in hs]

    print(f"slope = {np.log(res_second[0]/res_second[3])/np.log(10.**3)} for {hessp_option}")

    plt.figure(figsize=(10, 10))
    plt.plot(hs, res_zero, linestyle='-', marker='o', markersize=10, linewidth=2, color='blue', label=r"$r_{\textrm{zeroth}}$")
    plt.plot(hs, ref_zero, linestyle='--', linewidth=2, color='blue', label='First order reference')
    plt.plot(hs, res_first, linestyle='-', marker='o', markersize=10, linewidth=2, color='red', label=r"$r_{\textrm{first}}$")
    plt.plot(hs, ref_first, linestyle='--', linewidth=2, color='red', label='Second order reference')
    plt.plot(hs, res_second, linestyle='-', marker='o', markersize=10, linewidth=2, color='green', label=r"$r_{\textrm{second}}$")
    plt.plot(hs, ref_second, linestyle='--', linewidth=2, color='green', label='Third order reference')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Scaling factor $\epsilon$", fontsize=20)
    plt.ylabel("Residual", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   

    plt.savefig(os.path.join(CPC_q1_dir, f'taylor_results_{hessp_option}_round_off.pdf'), bbox_inches='tight')

    plt.show()


def generate_figures():
    # Figure set 1: Taylor remainder test

    hessp_options = ['fwd_rev', 'rev_fwd', 'rev_rev']

    for index, hessp_option in enumerate(hessp_options):

        taylor_results = np.load(os.path.join(val_numpy_dir, f"taylor_results_{hessp_option}.npy"))

        hs, f_plus_val, f_val, v_f_grad, v_hess_v = taylor_results.T
        res_zero = np.abs(f_plus_val - f_val)
        res_first = np.abs(f_plus_val - f_val - v_f_grad)
        res_second = np.abs(f_plus_val - f_val - v_f_grad - v_hess_v)

        ref_zero = [1/5.*res_zero[0]/hs[0] * h for h in hs]
        ref_first = [1/5.*res_first[0]/hs[0]**2 * h**2 for h in hs]
        ref_second = [1/5.*res_second[0]/hs[0]**3 * h**3 for h in hs]

        print(f"slope = {np.log(res_second[0]/res_second[-1])/np.log(10.**3)} for {hessp_option}")

        plt.figure(figsize=(10, 10))
        plt.plot(hs, res_zero, linestyle='-', marker='o', markersize=10, linewidth=2, color='blue', label=r"$r_{\textrm{zeroth}}$")
        plt.plot(hs, ref_zero, linestyle='--', linewidth=2, color='blue', label='First order reference')
        plt.plot(hs, res_first, linestyle='-', marker='o', markersize=10, linewidth=2, color='red', label=r"$r_{\textrm{first}}$")
        plt.plot(hs, ref_first, linestyle='--', linewidth=2, color='red', label='Second order reference')
        plt.plot(hs, res_second, linestyle='-', marker='o', markersize=10, linewidth=2, color='green', label=r"$r_{\textrm{second}}$")
        plt.plot(hs, ref_second, linestyle='--', linewidth=2, color='green', label='Third order reference')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"Scaling factor $\epsilon$", fontsize=20)
        plt.ylabel("Residual", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)   

        val_pdf_dir_case = os.path.join(val_pdf_dir, hessp_option)
        os.makedirs(val_pdf_dir_case, exist_ok=True)
        plt.savefig(os.path.join(val_pdf_dir_case, f'taylor_results_{hessp_option}.pdf'), bbox_inches='tight')

        # Figure set 2: Random sampling Hv or vHv
        vHv_results = np.load(os.path.join(val_numpy_dir, f"vHv_results_{hessp_option}.npy"))

        num_bins = 30
        colors = ['red', 'blue', 'green', 'orange']
        labels = [f'h={hs[i]}' for i in range(len(hs))]

        # Set flag to 'Hv' or 'vHv'
        flags = ['Hv', 'vHv']
        for flag in flags:
            for i, h in enumerate(hs):
                plt.figure(figsize=(10, 6))
                relative_errors_Hv = vHv_results[i][:, 2]
                relative_errors_vHv = np.abs((vHv_results[i][:, 0] - vHv_results[i][:, 1])/vHv_results[i][:, 0])

                data = relative_errors_Hv if flag == 'Hv' else relative_errors_vHv

                plt.hist(data,
                         bins=num_bins,
                         color=colors[i],
                         alpha=0.5,  # Transparency for overlapping regions
                         edgecolor='black',
                         label=labels[i])

                # plt.title(f'Histogram of relative errors')
                plt.tick_params(labelsize=15)
                plt.xlabel('Relative difference', fontsize=20)
                plt.ylabel('Count', fontsize=20)
                plt.legend(fontsize=20, frameon=False)
                # plt.grid(axis='y', alpha=0.75)
                plt.savefig(os.path.join(val_pdf_dir_case, f'{flag}_{i:03d}_{hessp_option}.pdf'), bbox_inches='tight')

    # Figure set 3: Profiling
    profile_results = np.load(os.path.join(prof_numpy_dir, f"profile_results_33664168912.npy"))
    labels = ['fwd-rev', 'rev-fwd', 'rev-rev']

    J_times, F_times = profile_results
    F_means = np.mean(F_times[:, 1:], axis=1)
    F_stds =  np.std(F_times[:, 1:], axis=1)
    J_means = np.mean(J_times[:, 1:], axis=1)
    J_stds =  np.std(J_times[:, 1:], axis=1)

    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(labels))
    bar_width = 0.8

    bars = plt.bar(x_pos, F_means, 
                   yerr=F_stds, 
                   width=bar_width,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black',
                   error_kw=dict(elinewidth=2, ecolor='black', capsize=5))

    plt.ylabel('Execution time [s]', fontsize=20)
    plt.xticks(x_pos, labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(axis='y', alpha=0.75)
    os.makedirs(prof_pdf_dir, exist_ok=True)
    plt.savefig(os.path.join(prof_pdf_dir, f'F.pdf'), bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(labels))
    bar_width = 0.8

    # Create bars with error bars
    bars = plt.bar(x_pos, J_means, 
                   yerr=J_stds, 
                   width=bar_width,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black',
                   error_kw=dict(elinewidth=2, ecolor='black', capsize=5))

    # Customize plot
    plt.ylabel('Execution time [s]', fontsize=20)
    plt.xticks(x_pos, labels, fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(prof_pdf_dir, f'J.pdf'), bbox_inches='tight')

    # plt.show()


def relative_error_AD_modes():
    vHv_results_fwd_rev = np.load(os.path.join(val_numpy_dir, f"vHv_results_fwd_rev.npy"))[0][:, 0]
    vHv_results_rev_fwd = np.load(os.path.join(val_numpy_dir, f"vHv_results_rev_fwd.npy"))[0][:, 0]
    vHv_results_rev_rev = np.load(os.path.join(val_numpy_dir, f"vHv_results_rev_rev.npy"))[0][:, 0]
    rel_err1 = np.max(np.abs((vHv_results_rev_fwd - vHv_results_fwd_rev)/vHv_results_fwd_rev))
    rel_err2 = np.max(np.abs((vHv_results_rev_rev - vHv_results_fwd_rev)/vHv_results_fwd_rev))
    print(rel_err1)
    print(rel_err2)


if __name__=="__main__":
    # workflow()
    # relative_error_AD_modes()
    # generate_figures_CPC_revision()
    size_test_CPC_revision()

