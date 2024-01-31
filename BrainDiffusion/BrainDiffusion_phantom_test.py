'''
Module to compute connectivity matrix from Diffusion Tensor Imaging (DTI) on phantom datasets.

DTI describe the strength of diffusivity on directions: xx, xy, xz, yy, yz, zz, for each voxel of brain.
Connectivity matrix is constructed following the procedure:
       1. For each region of interest, we set the whole region as source region with initial concentration one,
          and other regions as target regions with initial concentration zero.
       2. Solve the time-dependent PDE: dc(x)/dt = -k Div(T(x) grad(c(x))) with x as coordinate in brain domain, and T
          as anisotropic diffusion tensor computed from DTI data. The procedures are repeated for each region of interest.
       3. Each PDE solution provide one row of 2D connecitivity matrix by summing the resulting concentration of each
          region. Normalization is applied, and the resulting connectivity is a symmetric 2D real matrix.

Author: Zheyu Wen, Ali Ghafouri, George Biros
Version: Jan. 19th, 2024
'''

import os, argparse
import cupy as cp
import nibabel as nib
import numpy as np
import pandas as pd
from BrainDiffusion.operators_gpu import *
from cupyx.scipy.ndimage import gaussian_filter
import scipy.io as sio

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
from socket import gethostname
from sys import argv

class params:
    ''' Class describing brain diffusion process.
    __init__: describe the basic property of class, e.g. time stepsize, diffusion intensity kappa, termination condition,
              file paths, problem dimension.
    forward: the forward solution of PDE system.
    construct_segmentation: determine the segmentation of each patient brain from its Parcellation image.
    add_seg_diff_tensor: apply segmentation onto DTI matrix to specify the domain of interest.
    '''
    def __init__(self, device, path, diff_time, disco_id, resolution, snr):
        ''' Initialize the class.
        :param device: determine the device number if multiple are available.
        :param path: paths of files, e.g. template, DTI image
        :param dat_dir: directory that user specify their dataset.
        :param diff_time: time of diffusion the PDE solve. diff_time is recommended to set as 100. A small number can
                          be used as a quick test.
        '''
        with cp.cuda.Device(device):
            self.T = diff_time  # The time can be set to 100 for more diffusion
            self.dt = 0.1       # time step size of PDE discretization
            self.Nt = int(cp.ceil(self.T / self.dt))
            self.kappa = 10000  # scalar intensity for diffusion term.
            self.maxiter = 200  # max iteration in Conjugate gradient
            self.term_tol = 1e-5 # termination condition for PDE solver.
            self.t_max = int(self.T / self.dt) # maximum number of PDE dicretization time step.
            self.k_gm_wm = 0.0001  # hyper-parameter to adjust the diffusion between white matter and gray matter.
            self.gaussian_sigma = 1

            self.labels = cp.asarray(nib.load(path['labels']).get_fdata()) # template file
            self.ref_img = nib.load(path['labels'])
            labels = np.asarray(nib.load(path['labels']).get_fdata())
            self.labels_list = np.delete(np.unique(labels), 0)
            # self.Nroi = len(self.labels_list)
            self.out_dir = path['out_dir'] ## output path
            self.path = path
            self.disco_id = disco_id
            self.resolution = resolution
            self.snr = snr

            ##  below do FFT for coordinate in real space.

            N = 40
            self.N = N
            self.sigma = 2 * cp.pi / N

            dx = 2 * cp.pi / N
            tmp = 1j * cp.fft.fftfreq(N, dx)

            assert (max(tmp.shape) == N)

            self.IOmega_x, self.IOmega_y, self.IOmega_z = cp.meshgrid(tmp, tmp, tmp)

            assert (self.IOmega_x.shape == (N, N, N))
            assert (self.IOmega_y.shape == (N, N, N))
            assert (self.IOmega_z.shape == (N, N, N))

    def forward(self, roi_id, device):
        ''' execute forward of PDE

        :param roi_id: current roi_id as source region, and other regions as targets.
        :param device: GPU device number
        :return: initial condition c0 and final solution c1 (which used for connectivity)
        '''

        with cp.cuda.Device(device):
            params = solve_forward_cg(self, roi_id, '')

        return params.c1, params.c1_accum_t


    def add_seg_diff_tensor(self, device):
        ''' apply the segmentation to DTI tensor

        :param device: GPU device number
        :return: params class updates the resulting DTI tensor.
        '''

        with cp.cuda.Device(device):

            ## construct mask for each segmentation.

            tmp_gm = cp.asarray(nib.load(self.path['ROI-mask']).get_fdata())
            tmp_wm = cp.asarray(nib.load(self.path['mask']).get_fdata())
            tmp_bg = 1 - tmp_wm
            self.gm_ref = cp.array(tmp_gm.copy())
            self.wm_ref = cp.array(tmp_wm.copy()) - cp.array(tmp_gm.copy())

            ## apply gaussian filter to smooth the boundary of regions
            self.wm = gaussian_filter(cp.array(self.wm_ref .copy()), self.gaussian_sigma)
            self.gm = gaussian_filter(cp.array(self.gm_ref.copy()), self.gaussian_sigma)
            bg = gaussian_filter(cp.array(tmp_bg.copy()), self.gaussian_sigma)
            total_ = self.wm + self.gm + bg

            self.wm /= total_
            self.gm /= total_

            ## eigen value decomposition of DTI data. The largest eigen value will be used as the indication of
            ## strongest diffusivity direction.
            L1 = cp.array(nib.load(self.path['L1']).get_fdata())
            L2 = cp.array(nib.load(self.path['L2']).get_fdata())
            L3 = cp.array(nib.load(self.path['L3']).get_fdata())
            V1 = cp.array(nib.load(self.path['V1']).get_fdata())
            V2 = cp.array(nib.load(self.path['V2']).get_fdata())
            V3 = cp.array(nib.load(self.path['V3']).get_fdata())

            V1_T = cp.expand_dims(V1, axis=3)
            V2_T = cp.expand_dims(V2, axis=3)
            V3_T = cp.expand_dims(V3, axis=3)

            V1 = cp.expand_dims(V1, axis=4)
            V2 = cp.expand_dims(V2, axis=4)
            V3 = cp.expand_dims(V3, axis=4)

            L1 = cp.expand_dims(L1, axis=3)
            L1 = cp.expand_dims(L1, axis=4)
            L2 = cp.expand_dims(L2, axis=3)
            L2 = cp.expand_dims(L2, axis=4)
            L3 = cp.expand_dims(L3, axis=3)
            L3 = cp.expand_dims(L3, axis=4)

            L1 = cp.tile(L1, (1, 1, 1, 3, 3))
            L2 = cp.tile(L2, (1, 1, 1, 3, 3))
            L3 = cp.tile(L3, (1, 1, 1, 3, 3))

            T_par = L1 * cp.matmul(V1, V1_T)
            T_orth = L2 * cp.matmul(V2, V2_T) + L3 * cp.matmul(V3, V3_T)

            orth_par_ratio = 0.01
            Tot = T_par + orth_par_ratio * T_orth # deemphasize the orthogonal direction of diffusion.

            self.Kxx_ref = Tot[:, :, :, 0, 0]
            self.Kxy_ref = Tot[:, :, :, 0, 1]
            self.Kxz_ref = Tot[:, :, :, 0, 2]
            self.Kyy_ref = Tot[:, :, :, 1, 1]
            self.Kyz_ref = Tot[:, :, :, 1, 2]
            self.Kzz_ref = Tot[:, :, :, 2, 2]


def context_init(device):
    with cp.cuda.Device(device):
        c = cp.ones(10, dtype='complex128')


def eval_diffusion(dat_dir, output_path, args, diff_time=100):

    ''' Evaluate the PDE solution for all regions in brain. Parall computing for all regions.

    :param dti_path: deterimined by user for dir to contain data.
    :param template_file: template file path
    :param output_path: output path of PDE solution
    :param diff_time: time length of diffusion. recommended to 100.
    :return: save the resulting 3D PDE solution under output_path.
    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    if rank == 0:
        print('Starting MPI for {} jobs on each of the {} ranks\n'.format(int(cp.ceil(16/size)), size))
    comm.Barrier()
    if diff_time <= 1:
        print('---- Notice: Please use a larger [diff_time] if possible (100 recomanded). The default value of variable [diff_time] is for a quick example only.----')

    path = {}

    if args.resolution == 'high':
        dwi_path = os.path.join(dat_dir, 'DiSCo{}/high_resolution_40x40x40'.format(args.disco_id))
        prefix = ''
    else:
        dwi_path = os.path.join(dat_dir, 'DiSCo{}/low_resolution_20x20x20'.format(args.disco_id))
        prefix = 'lowRes_'

    if args.snr == 'inf':
        postfix = ''
    else:
        postfix = '_RicianNoise-snr{}'.format(args.snr)

    path['labels'] = os.path.join(dwi_path, '{}DiSCo{}_ROIs.nii.gz'.format(prefix, args.disco_id))
    #path['dti'] = os.path.join(dwi_path, 'dti{}_tensor.nii'.format(postfix))
    path['V1'] = os.path.join(dwi_path, 'dti_mrtrix{}_V1.nii'.format(postfix))
    path['V2'] = os.path.join(dwi_path, 'dti_mrtrix{}_V2.nii'.format(postfix))
    path['V3'] = os.path.join(dwi_path, 'dti_mrtrix{}_V3.nii'.format(postfix))
    path['L1'] = os.path.join(dwi_path, 'dti_mrtrix{}_L1.nii'.format(postfix))
    path['L2'] = os.path.join(dwi_path, 'dti_mrtrix{}_L2.nii'.format(postfix))
    path['L3'] = os.path.join(dwi_path, 'dti_mrtrix{}_L3.nii'.format(postfix))
    path['ROI-mask'] = os.path.join(dwi_path, '{}DiSCo{}_ROIs-mask.nii.gz'.format(prefix, args.disco_id))
    path['mask'] = os.path.join(dwi_path, '{}DiSCo{}_mask.nii.gz'.format(prefix, args.disco_id))

    path['out_dir'] = output_path
    os.makedirs(path['out_dir'], exist_ok=True)
    labels = np.asarray(nib.load(path['labels']).get_fdata()) # template file
    labels_list = np.delete(np.unique(labels), 0)
    num_regions = len(labels_list)

    '''
    MPI here to parallely compute for four regions of interests. Therefore, four classes will be initialized, and 
    executed on four GPUs.
    '''

    roi_groups = []
    for i in range(int(cp.ceil(num_regions / size))):
        roi_groups.append(labels_list[size * i: (i + 1) * size])

    context_init(rank % 4)
    test_params = params(rank % 4, path, diff_time, args.disco_id, args.resolution, args.snr)
    test_params.add_seg_diff_tensor(rank % 4)

    if not os.path.exists(test_params.out_dir):
        os.mkdir(test_params.out_dir)
    for (i_group, roi_group) in enumerate(roi_groups):

        if rank >= len(roi_group):
            continue
        roi_id = roi_group[rank]
        # print('rank: {}, roi_id: {}'.format(rank, roi_id))
        fname_1 = os.path.join(test_params.out_dir, 'c1_td_%d.nii.gz' % roi_id)  # save the resulting solution.
        fname_2 = os.path.join(test_params.out_dir, 'c1_td_accum_%d.nii.gz' % roi_id)
        # if os.path.exists(fname_1):
        #     continue

        print('Now executing ROI group: ', roi_group, ' ---progress: {}/{} in MPI rank {}'.format(i_group+1, len(roi_groups), rank))
        result = test_params.forward(roi_id, rank % 4)

        writeNII(result[0].get(), fname_1, ref_image=test_params.ref_img)
        writeNII(result[1].get(), fname_2, ref_image=test_params.ref_img)

    comm.Barrier()


def update_conn_mtx(labels, c, source_roi, roi_list, conn_mtx):

    ''' update connecitiy matrix row by row from each PDE solution file.
        This code is a subsequent step after [eval_diffusion]

    :param labels: patient parcellation file
    :param c: PDE solution (3D tensor) for source_roi.
    :param source_roi: source region of resulting PDE solution c.
    :param roi_list: list of all region of intersts in parcellation
    :param conn_mtx: connecitivity matrix that is under update.
    :return: updated connectivity matrix.
    '''
    # print('roi list: ', roi_list)
    roi_idx = list(roi_list).index(source_roi)
    for other_roi in list(roi_list):
        if other_roi != source_roi:
            other_roi_idx = list(roi_list).index(other_roi)
            sum_concentration = c[np.where(labels == other_roi)].sum()
            conn_mtx[roi_idx, other_roi_idx] += sum_concentration

    return conn_mtx


def gen_conn_from_diffusion(template_file,
                            diff_output_path,
                            conn_output_fname):

    ''' user interfaced function to construct connectivity matrix from PDE solution

    :param template_file: parcellation file path.
    :param diff_output_path: dir of PDE solution, equivalent to 'output_path' in function [eval_diffusion].
    :param conn_output_fname: file name of output connectivity matrix under format .mat
    :return: save connectivity matrix.
    '''

    print('----Starting generation of connectivity matrix from diffusion result----')

    labels = np.asarray(nib.load(template_file).get_fdata()).astype('int')
    labels_list = np.delete(np.unique(labels), 0)
    conn_mat = np.zeros([len(labels_list), len(labels_list)])
    for (i_roi, roi_id) in enumerate(labels_list):
        c1 = np.asarray(nib.load(os.path.join(diff_output_path, 'c1_td_{}.nii.gz'.format(roi_id))).get_fdata())
        conn_mat = update_conn_mtx(labels, c1, roi_id, labels_list, conn_mat)
        # print('finish ROI: {}/{}'.format(i_roi, len(labels_list)))
        if ((i_roi+1)%4==0):
            print('Generation of connectivity Progress: {}/{}'.format(i_roi+1, len(labels_list)))
    for (diag_idx, diag_roi) in enumerate(labels_list):
        conn_mat[diag_idx, diag_idx] = conn_mat[diag_idx, :].sum()
    conn_mat /= np.max(conn_mat, axis=1).reshape([-1, 1])
    conn_mat = (conn_mat + conn_mat.T) / 2
    sio.savemat(conn_output_fname, {'data': conn_mat})


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dwi_process')
    parser.add_argument('--disco_id', default='2') ## example 1, 2, 3
    parser.add_argument('--resolution', default='high')
    parser.add_argument('--snr', default='inf')
    args = parser.parse_args()
    dat_dir = '/scratch1/08171/zheyw1/phantom_DWI'

    if args.resolution == 'high':
        dwi_path = os.path.join(dat_dir, 'DiSCo{}/high_resolution_40x40x40'.format(args.disco_id))
        prefix = ''
    else:
        dwi_path = os.path.join(dat_dir, 'DiSCo{}/low_resolution_20x20x20'.format(args.disco_id))
        prefix = 'lowRes_'

    if args.snr == 'inf':
        postfix = ''
    else:
        postfix = '_RicianNoise-snr{}'.format(args.snr)

    output_path = '/scratch1/08171/zheyw1/phantom_DWI_result/{}DiSCo{}{}'.format(prefix, args.disco_id, postfix)
    params = eval_diffusion(dat_dir, output_path, args, diff_time=100)


    ### construct connectivity
    template_file = os.path.join(dwi_path, '{}DiSCo{}_ROIs.nii.gz'.format(prefix, args.disco_id))
    conn_output_fname = '/scratch1/08171/zheyw1/phantom_DWI_result/{}DiSCo{}{}/conn_mtx_td.mat'.format(prefix, args.disco_id, postfix)
    gen_conn_from_diffusion(template_file, output_path, conn_output_fname)
