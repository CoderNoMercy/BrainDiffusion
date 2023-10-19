'''
Module providing supporting utility function to solve the time-dependent PDE.

Solve the time-dependent PDE: dc(x)/dt = -k Div(T(x) grad(c(x))) with x as coordinate in brain domain, and T
  as anisotropic diffusion tensor computed from DTI data. The procedures are repeated for each region of interest.

Author: Zheyu Wen, Ali Ghafouri, George Biros
Version: Jul 17th, 2023

'''
import copy
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header)
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine)
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename)

def grad3D(params):

    ''' Convert data in real spatial domain to spectrum domain.

    :param params: params class
    :return: update gradient of x, y, z directions of variable c.
    '''

    c = params.c

    IOmega_x = params.IOmega_x
    IOmega_y = params.IOmega_y
    IOmega_z = params.IOmega_z

    c_hat = np.fft.fftn(c)

    params.grad_x = np.real(np.fft.ifftn(IOmega_x * c_hat))
    params.grad_y = np.real(np.fft.ifftn(IOmega_y * c_hat))
    params.grad_z = np.real(np.fft.ifftn(IOmega_z * c_hat))

    return params



def compute_div(params, c):

    ''' compute div(T(x) grad(c(x)))

    :param params: class params initialized in BrainDiffusion_cpu.py
    :param c: current concentration tensor c(x, t)
    :return: right hand side of the PDE equation.
    '''

    params.c = c.copy()
    N = params.N

    params = grad3D(params)

    IOmega_x = params.IOmega_x
    IOmega_y = params.IOmega_y
    IOmega_z = params.IOmega_z

    grad_x = params.grad_x
    grad_y = params.grad_y
    grad_z = params.grad_z


    Kxx = params.Kxx
    Kxy = params.Kxy
    Kxz = params.Kxz
    Kyy = params.Kyy
    Kyz = params.Kyz
    Kzz = params.Kzz

    tmp = Kxx * grad_x + Kxy * grad_y + Kxz * grad_z
    tmp = gaussian_filter(tmp.copy(), 1)

    x_hat = np.fft.fftn(tmp)
    x_hat *= IOmega_x

    tmp = Kxy*grad_y + Kyy*grad_y + Kyz*grad_z
    tmp = gaussian_filter(tmp.copy(), 1)

    y_hat = np.fft.fftn(tmp)
    y_hat *= IOmega_y

    tmp = Kxz*grad_x + Kyz*grad_y + Kzz*grad_z
    tmp = gaussian_filter(tmp.copy(), 1)

    z_hat = np.fft.fftn(tmp)
    z_hat *= IOmega_z

    div = np.fft.ifftn(x_hat) + np.fft.ifftn(y_hat) + np.fft.ifftn(z_hat)

    return np.real(div)


def applyA_lhs(params, c):

    ''' After Time discretization and Spatial discretization, here we rearange the term and compute the left hand side

    :param params: class params initialized in BrainDiffusion_cpu.py
    :param c: current concentration c(x, t)
    :return: left hand side of discretized PDE.
    '''

    dt = params.dt
    N = params.N
    c = c.reshape((N, N, N)).copy()
    c = gaussian_filter(c.copy(), 1)
    div = compute_div(params, c)

    Ac = c - dt * 0.5 * div
    return np.real(Ac.flatten())


def applyb_rhs(params, c):

    ''' After Time discretization and Spatial discretization, here we rearange the term and compute the left hand side

    :param params: class params initialized in BrainDiffusion_cpu.py
    :param c: current concentration c(x, t)
    :return: right hand side of discretized PDE.
    '''

    dt = params.dt
    N = params.N
    c = c.reshape((N,N,N)).copy()

    c = gaussian_filter(c.copy(), 1)
    div = compute_div(params, c)

    bc = c + dt * 0.5 * div

    return np.real(bc.flatten())


def apply_Pinv(params, c):

    ''' Apply the precondition matrix to the left/right hand side

    :param params: class params initialized in BrainDiffusion_cpu.py
    :param c: current concentration c(x, t)
    :return: left/right hand side of discretized PDE.
    '''

    N = params.N
    dt = params.dt
    c = c.reshape((N,N,N)).copy()

    c_hat = np.fft.fftn(c)

    IOmega_x = params.IOmega_x
    IOmega_y = params.IOmega_y
    IOmega_z = params.IOmega_z

    kp = params.kappa * params.kmean

    c_hat = c_hat / (1 - dt * 0.5 * kp * (IOmega_x**2 + IOmega_y**2 + IOmega_z**2))

    Pinv_c = np.real(np.fft.ifftn(c_hat))

    return np.real(Pinv_c.flatten())


def solve_cg(A, b, x0=None, maxiter=100, term_tol=1e-6, verbose=1):

    ''' CPU version for Conjugate Gradient to solve x in the following equation with known b and A.
        b = A(x)

    :param A: right hand side operator
    :param b: observation
    :param x0: initialization of solution x.
    :param maxiter: max iteration number to run.
    :param term_tol: quit condition by measuring the norm of residual
    :param verbose: whether report the error in each iteration
    :return: inverse problem solution.
    '''

    def one_loop(i, rTr, x, r, p):
        Ap = A(p)
        alpha = (rTr / np.sum((np.conj(p) * Ap)))
        alpha = alpha + 0.j
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = np.sum(np.conj(r) * r)
        beta = rTrNew / rTr
        beta = beta + 0.j
        p = r + beta * p
        return i+1, rTrNew, x, r, p

    if x0 is None:
        x = np.zeros(b.shape) + 0.j
    else:
        x = x0 + 0.j
    r = b - A(x) + 0.j
    i = 0
    p = r.copy()
    rTr = np.sum(np.conj(r) * r)
    tol = np.linalg.norm(b)
    norm_res = np.linalg.norm(r)
    norm_res_init = norm_res.copy()
    # if verbose:
    #   print('pcg iter: %d, residual = %.4e'%(0, norm_res))
    while norm_res > term_tol * tol:
        [i, rTr, x, r, p] = one_loop(i, rTr, x, r, p)
        norm_res = np.linalg.norm(r)
        if i >= maxiter:
            break
        if verbose:
            print('pcg iter: %d, residual = %.4e'%(i, norm_res/norm_res_init))
    return x



def solve_forward_cg(params, roi_id, subj_name):

    ''' Solve the diffusion PDE utilizing all functions mentioned above.

    :param params: class params initialized in BrainDiffusion_cpu.py
    :param roi_id: id of region of interest
    :return: class params with PDE solution referred as params.c1
    '''

    T = params.T
    Nt = params.Nt
    dt = T / Nt
    N = params.N

    kappa = params.kappa

    wm = params.wm_ref
    gm = params.gm_ref
    k_gm_wm = params.k_gm_wm

    gm_plus_wm = gm + wm
    '''
      below compute Kxx, Kxy ... for anisotropic diffusion in the directions of 
      xx, xy, xz, yy, yz, zz.
    '''

    Kxx = params.Kxx_ref * kappa * (k_gm_wm * gm + wm)
    Kxy = params.Kxy_ref * kappa * (k_gm_wm * gm + wm)
    Kxz = params.Kxz_ref * kappa * (k_gm_wm * gm + wm)
    Kyy = params.Kyy_ref * kappa * (k_gm_wm * gm + wm)
    Kyz = params.Kyz_ref * kappa * (k_gm_wm * gm + wm)
    Kzz = params.Kzz_ref * kappa * (k_gm_wm * gm + wm)

    tmp = np.zeros(6)
    tmp[0] = np.mean(Kxx)
    tmp[1] = np.mean(Kxy)
    tmp[2] = np.mean(Kxz)
    tmp[3] = np.mean(Kyy)
    tmp[4] = np.mean(Kyz)
    tmp[5] = np.mean(Kzz)
    params.kmean = np.mean(tmp)

    tmp = np.zeros(6)
    tmp[0] = np.amax(Kxx)
    tmp[1] = np.amax(Kxy)
    tmp[2] = np.amax(Kxz)
    tmp[3] = np.amax(Kyy)
    tmp[4] = np.amax(Kyz)
    tmp[5] = np.amax(Kzz)
    params.kmax = np.amax(tmp)

    params.Kxx = Kxx
    params.Kxy = Kxy
    params.Kxz = Kxz
    params.Kyy = Kyy
    params.Kyz = Kyz
    params.Kzz = Kzz


    c = np.zeros(Kxx.shape)
    c[params.labels == roi_id] = 1.0
    cinit = copy.deepcopy(c)
    cinit_sum = np.sum(cinit)
    c_inf = cinit_sum / (np.sum(gm_plus_wm))
    c_accum_result = np.zeros(Kxx.shape)

    params.c = c.copy()
    params.c0 = c.copy()

    '''
      construct left hand side and right hand side operator preparing solving the inverse problem by CG.
    '''
    f_A = lambda x: apply_Pinv(params, applyA_lhs(params, x))
    f_b = lambda x: apply_Pinv(params, applyb_rhs(params, x))

    params = compute_epicenter(params)
    far_reg = params.far_reg_dict[str(roi_id)]
    # fname = os.path.join(params.dat_dir, 'data/{}/brain_background.npy'.format(subj_name))
    # brain_background = np.load(fname)
    t_max = params.t_max
    '''
      Below solve the time dependent PDE each time step at a time. t_max is determined by time horizon and time step size.
    '''
    old_tar_mean_mass = 0
    for t in range(t_max):

        c = solve_cg(f_A, f_b(c.flatten()), c.flatten(), maxiter=params.maxiter, term_tol=params.term_tol, verbose=0)
        c = np.real(c.copy())
        c[c < 0.0] = 0.0
        # print('total mass at step: {} is {}'.format(t, np.sum(c)))

        c_3d = np.real(c.reshape((N, N, N)))
        # c_3d[np.where(brain_background==1)] = 0
        # c_3d[c_3d > c_inf] -= (np.sum(c_3d) - cinit_sum) / np.sum(c_3d[c_3d > c_inf]) * c_3d[c_3d > c_inf]

        # ## for eximining the trajectory
        # fname_1 = os.path.join(params.out_dir, 'c1_{}_t_{}.nii.gz'.format(roi_id, t)) # save the resulting solution.
        # writeNII(c_3d, fname_1, ref_image=params.affine)
        c_3d[c_3d * t_max < c_inf] = 0
        c_accum_result += c_3d
        tmp = c_3d[params.labels == far_reg]
        # tar_mass = cp.linalg.norm(tmp.flatten())
        # print("t = %d, Tar_mass (%d) = %4e / %.4e (%d)"%(t+1, far_reg, tar_mass, init_mass, roi_id))
        # if tar_mass > 1e-8 * init_mass:
        #   break

        tar_mean_mass = np.mean(tmp.flatten())
        # print("t = %d, Tar_mean_mass (%d) = %4e / %.4e (%d)"%(t+1, far_reg, tar_mean_mass, c_inf, roi_id))
        # if tar_mean_mass > c_inf * 1.1:
        if old_tar_mean_mass >= tar_mean_mass:
            break
        old_tar_mean_mass = copy.deepcopy(tar_mean_mass)

    c_3d = np.real(c.reshape((N, N, N)))
    c_3d = gaussian_filter(c_3d.copy(), 1)
    params.c1 = c_3d.copy()
    params.c1_accum_t = c_accum_result

    return params


def compute_epicenter(params):

    ''' Compute center of mass for each region of interest. The aim is to find the furthest region to the source region.
        Therefore, the quit condition can be set as when the furthest region receives sufficient concentration by diffusion.

    :param params: class params initialized in BrainDiffusion_cpu.py
    :return: updated params with epicenter of each region of interest.
    '''

    epi_dict = {}

    labels_list = params.labels_list
    epi_mat = np.zeros((len(labels_list),3))

    labels = params.labels
    tmp = np.arange(0, labels.shape[0])
    x, y, z = np.meshgrid(tmp, tmp, tmp)

    n = len(labels_list)
    # print(x.shape)
    # print(labels.shape)
    labels_list = np.array(labels_list)
    for (i, l) in enumerate(labels_list):

        epi_coord = np.zeros(3)

        x_set = x[labels == l]
        y_set = y[labels == l]
        z_set = z[labels == l]
        epi_coord[0] = int(np.round(np.mean(x_set), 0))
        epi_coord[1] = int(np.round(np.mean(y_set), 0))
        epi_coord[2] = int(np.round(np.mean(z_set), 0))

        epi_dict[str(l)] = epi_coord
        epi_mat[i, :] = epi_coord

    far_reg_dict = {}

    for (i,l) in enumerate(labels_list):

        rep = np.tile(epi_mat[i,:], (n, 1))
        diff = np.linalg.norm((epi_mat - rep), axis=1)

        assert(diff.shape == (n,))

        far_reg_dict[str(l)] = labels_list[np.argmax(diff)]


    params.far_reg_dict = far_reg_dict
    params.epi_dict = epi_dict


    return params




































