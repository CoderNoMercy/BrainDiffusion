'''
Module to compute connectivity matrix from Diffusion Tensor Imaging (DTI).

DTI describe the strength of diffusivity on directions: xx, xy, xz, yy, yz, zz, for each voxel of brain.
Connectivity matrix is constructed following the procedure:
       1. For each region of interest, we set the whole region as source region with initial concentration one,
          and other regions as target regions with initial concentration zero.
       2. Solve the time-dependent PDE: dc(x)/dt = -k Div(T(x) grad(c(x))) with x as coordinate in brain domain, and T
          as anisotropic diffusion tensor computed from DTI data. The procedures are repeated for each region of interest.
       3. Each PDE solution provide one row of 2D connecitivity matrix by summing the resulting concentration of each
          region. Normalization is applied, and the resulting connectivity is a symmetric 2D real matrix.

Author: Zheyu Wen, Ali Ghafouri, George Biros
Version: Jul 17th, 2023
'''

import os, argparse
import cupy as cp
import nibabel as nib
import numpy as np
import pandas as pd
from BrainDiffusion.operators import *
from cupyx.scipy.ndimage import gaussian_filter
import joblib
import scipy.io as sio


class params:
    ''' Class describing brain diffusion process.
    __init__: describe the basic property of class, e.g. time stepsize, diffusion intensity kappa, termination condition,
              file paths, problem dimension.
    forward: the forward solution of PDE system.
    construct_segmentation: determine the segmentation of each patient brain from its Parcellation image.
    add_seg_diff_tensor: apply segmentation onto DTI matrix to specify the domain of interest.
    '''
    def __init__(self, device, path, dat_dir, diff_time):
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
            self.kappa = 10     # scalar intensity for diffusion term.
            self.maxiter = 100  # max iteration in Conjugate gradient
            self.term_tol = 1e-5 # termination condition for PDE solver.
            self.t_max = int(self.T / self.dt) # maximum number of PDE dicretization time step.
            df = pd.read_csv(path['muse_dictionary']) # template information in csv file. e.g. regions name, segmentation..
            self.labels_list = df.loc[df['TISSUE_SEG'] == 'GM']['ROI_INDEX'].to_numpy()
            self.num_regions = len(self.labels_list)
            self.k_gm_wm = 0.01  # hyper-parameter to adjust the diffusion between white matter and gray matter.
            self.Nroi = len(self.labels_list)

            self.labels = cp.asarray(nib.load(path['labels']).get_fdata()) # template file
            self.out_dir = path['out_dir'] ## output path
            self.path = path
            self.dat_dir = dat_dir

            ##  below do FFT for coordinate in real space.

            N = 256
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
            params = solve_forward_cg(self, roi_id)

        return params.c0, params.c1_accum_t

    def construct_segmentation(self, subj_name):
        ''' construct segmentation from patient parcellation.

        :param subj_name: subj_name
        :return: write segmentation file in data dir.
        '''

        dic_muse = pd.read_csv(self.path['muse_dictionary'])
        roi_list_all = dic_muse['ROI_INDEX'].to_numpy()
        template = nib.load(self.path['labels'])
        template_data = np.asarray(template.get_fdata())
        segmentation_result = np.zeros_like(template_data)
        for roi_id in roi_list_all:
            seg_type = dic_muse.loc[dic_muse['ROI_INDEX']==roi_id]['TISSUE_SEG'].item()
            if seg_type == 'GM':
                segmentation_result[np.where(template_data==roi_id)] = 5
            elif seg_type == 'WM':
                segmentation_result[np.where(template_data == roi_id)] = 6
            elif seg_type == 'CSF':
                segmentation_result[np.where(template_data == roi_id)] = 7
            elif seg_type == 'VN':
                segmentation_result[np.where(template_data == roi_id)] = 8
            else:
                segmentation_result[np.where(template_data == roi_id)] = 0
        fname = self.dat_dir + '/data/{}/segmentation_from_labels.nii.gz'.format(subj_name)
        writeNII(segmentation_result, fname, ref_image=template)

    def add_seg_diff_tensor(self, device):
        ''' apply the segmentation to DTI tensor

        :param device: GPU device number
        :return: params class updates the resulting DTI tensor.
        '''

        with cp.cuda.Device(device):

            ## construct mask for each segmentation.
            wm = nib.load(self.path['seg'])
            self.affine = wm
            tmp = wm.get_fdata().copy()
            tmp[tmp != 6] = 0
            tmp[tmp == 6] = 1
            tmp2 = wm.get_fdata().copy()
            tmp2[tmp2 != 5] = 0
            tmp2[tmp2 == 5] = 1

            tmp3 = wm.get_fdata().copy()
            tmp3[tmp3 != 7] = 0
            tmp3[tmp3 == 7] = 1

            tmp4 = wm.get_fdata().copy()
            tmp4[tmp4 != 8] = 0
            tmp4[tmp4 == 8] = 1

            tmp5 = wm.get_fdata().copy()
            tmp5[tmp5 != 0] = 2
            tmp5[tmp5 == 0] = 1
            tmp5[tmp5 != 1] = 0

            self.wm_ref = cp.array(tmp.copy())
            self.gm_ref = cp.array(tmp2.copy())

            ## apply gaussian filter to smooth the boundary of regions
            self.wm = gaussian_filter(cp.array(tmp.copy()), 1)
            self.gm = gaussian_filter(cp.array(tmp2.copy()), 1)
            csf = gaussian_filter(cp.array(tmp3.copy()), 1)
            vt = gaussian_filter(cp.array(tmp4.copy()), 1)
            bg = gaussian_filter(cp.array(tmp5.copy()), 1)
            total_ = self.wm + self.gm + csf + vt + bg

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


def eval_diffusion(use_default_data,
                   dti_path,
                   template_file,
                   subj_name,
                   output_path, diff_time=100):
    ''' Evaluate the PDE solution for all regions in brain. Parall computing for all regions.

    :param use_default_data: bool. if true. an example default data will be downloaded to dti_path.
    :param dti_path: deterimined by user for dir to contain data.
    :param template_file: template file path
    :param subj_name: subject name
    :param output_path: output path of PDE solution
    :param diff_time: time length of diffusion. recommended to 100.
    :return: save the resulting 3D PDE solution under output_path.
    '''

    if use_default_data:
        print('Downloading default DTI file...')
        filename = dti_path + '/data.zip'
        os.system('gdown 1xVmNZqsyFT_1Hbg-zikSIAntQm23m4Iq -O {}'.format(filename))
        os.system('unzip {}'.format(filename))
        os.system('rm {}'.format(filename))

    if diff_time <= 1:
        print('---- Notice: Please use a larger [diff_time] if possible (100 recomanded). The default value of variable [diff_time] is for a quick example only.----')

    path = {}
    dat_dir = dti_path
    path['labels'] = template_file
    path['dti'] = os.path.join(dat_dir, 'data/{}/dti_0r_tensor_aff2jakob.nii'.format(subj_name))
    path['V1'] = os.path.join(dat_dir, 'data/{}/V1_aff2jakob.nii.gz'.format(subj_name))
    path['V2'] = os.path.join(dat_dir, 'data/{}/V2_aff2jakob.nii.gz'.format(subj_name))
    path['V3'] = os.path.join(dat_dir, 'data/{}/V3_aff2jakob.nii.gz'.format(subj_name))
    path['L1'] = os.path.join(dat_dir, 'data/{}/L1_aff2jakob.nii.gz'.format(subj_name))
    path['L2'] = os.path.join(dat_dir, 'data/{}/L2_aff2jakob.nii.gz'.format(subj_name))
    path['L3'] = os.path.join(dat_dir, 'data/{}/L3_aff2jakob.nii.gz'.format(subj_name))

    path['out_dir'] = output_path
    os.makedirs(path['out_dir'], exist_ok=True)
    path['muse_dictionary'] = os.path.join(dat_dir, 'data/muse/MUSE Template - Dictionary_ROI_Hierarchy.csv')
    df = pd.read_csv(path['muse_dictionary'])
    labels_list = df.loc[df['TISSUE_SEG'] == 'GM']['ROI_INDEX'].to_numpy()
    num_regions = len(labels_list)

    '''
    joblib here to parallely compute for four regions of interests. Therefore, four classes will be initialized, and 
    executed on four GPUs.
    '''
    pool = joblib.Parallel(n_jobs=4, prefer="threads", verbose=1)
    pool(joblib.delayed(context_init)(device) for device in [0, 1, 2, 3])
    test_params = pool(joblib.delayed(params)(device, path, dat_dir, diff_time) for device in [0, 1, 2, 3])
    test_params[0].construct_segmentation(subj_name)
    path['seg'] = os.path.join(dat_dir, 'data/{}/segmentation_from_labels.nii.gz'.format(subj_name))
    pool(joblib.delayed(test_params[device].add_seg_diff_tensor)(device) for device in [0, 1, 2, 3])

    roi_groups = []
    for i in range(int(np.ceil(num_regions / 4))):
        roi_groups.append([test_params[0].labels_list[4 * i: (i + 1) * 4]])

    for (i_group, roi_group) in enumerate(roi_groups):

        checkpoint_flag = 1
        for (roi_ind, roi_id) in enumerate(roi_group[0]):
            fname_1 = os.path.join(test_params[roi_ind].out_dir, 'c1_%d.nii.gz' % roi_id)
            if not os.path.exists(fname_1):
                checkpoint_flag = 0
        if checkpoint_flag:
            continue

        print('Now executing ROI group: ', roi_group, ' ---progress: {}/{}'.format(i_group+1, len(roi_groups)))
        result = pool(joblib.delayed(test_params[device].forward)(roi_id, device) for roi_id, device in [(roi_group[0][i], i) for i in range(len(roi_group[0]))])

        roi_ind = 0
        for roi_id in roi_group[0]:
            if not os.path.exists(test_params[roi_ind].out_dir):
                os.mkdir(test_params[roi_ind].out_dir)
            fname_1 = os.path.join(test_params[roi_ind].out_dir, 'c1_%d.nii.gz' % roi_id) # save the resulting solution.
            writeNII(result[roi_ind][1].get(), fname_1, ref_image=test_params[roi_ind].affine)
            roi_ind += 1


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

    roi_idx = list(roi_list).index(source_roi)
    for other_roi in list(roi_list):
        if other_roi != source_roi:
            other_roi_idx = list(roi_list).index(other_roi)
            sum_concentration = 0
            sum_concentration += c[np.where(labels == other_roi)].sum()
            conn_mtx[roi_idx, other_roi_idx] += sum_concentration

    return conn_mtx


def gen_conn_from_diffusion(dti_path,
                            template_file,
                            diff_output_path,
                            conn_output_fname):

    ''' user interfaced function to construct connectivity matrix from PDE solution

    :param dti_path: path that contain DTI data.
    :param template_file: parcellation file path.
    :param diff_output_path: dir of PDE solution, equivalent to 'output_path' in function [eval_diffusion].
    :param conn_output_fname: file name of output connectivity matrix under format .mat
    :return: save connectivity matrix.
    '''

    print('----Starting generation of connectivity matrix from diffusion result----')
    muse_dictionary = os.path.join(dti_path, 'data/muse/MUSE Template - Dictionary_ROI_Hierarchy.csv')
    df = pd.read_csv(muse_dictionary)
    labels_list = df.loc[df['TISSUE_SEG'] == 'GM']['ROI_INDEX'].to_numpy()[5:]
    labels = np.asarray(nib.load(template_file).get_fdata()).astype('int')
    conn_mat = np.zeros([len(labels_list), len(labels_list)])
    for (i_roi, roi_id) in enumerate(labels_list):
        c1 = np.asarray(nib.load(os.path.join(diff_output_path, 'c1_{}.nii.gz'.format(roi_id))).get_fdata())
        conn_mat = update_conn_mtx(labels, c1, roi_id, labels_list, conn_mat)
        # print('finish ROI: {}/{}'.format(i_roi, len(labels_list)))
        if ((i_roi+1)%10==0):
            print('Generation of connectivity Progress: {}/{}'.format(i_roi+1, len(labels_list)))
    for (diag_idx, diag_roi) in enumerate(labels_list):
        conn_mat[diag_idx, diag_idx] = conn_mat[diag_idx, :].sum()
    conn_mat /= np.max(conn_mat, axis=1).reshape([-1, 1])
    conn_mat = (conn_mat + conn_mat.T) / 2
    sio.savemat(conn_output_fname, {'data': conn_mat})

def evd_dti(dti_path, subj_name):

    ''' do eigen value decomposition for DTI file with shape [256, 256, 256, 6]

    :param dti_path: path to DTI file folder
    :param subj_name: patient id.
    :return: save EVD result in .nii.gz file and sved under path of {dti_path}/data/
    '''

    dti_file = nib.load(os.path.join(dti_path, 'data/{}/dti_0r_tensor_aff2jakob.nii'.format(subj_name)))
    dti_tensor = np.asarray(dti_file.get_fdata())
    dti_complete_dim = np.zeros_like([256, 256, 256, 3, 3])
    dti_complete_dim[:, :, :, 0, 0] = dti_tensor[:, :, :, 0]
    dti_complete_dim[:, :, :, 0, 1] = dti_tensor[:, :, :, 1]
    dti_complete_dim[:, :, :, 1, 0] = dti_tensor[:, :, :, 1]
    dti_complete_dim[:, :, :, 0, 2] = dti_tensor[:, :, :, 2]
    dti_complete_dim[:, :, :, 2, 0] = dti_tensor[:, :, :, 2]
    dti_complete_dim[:, :, :, 1, 1] = dti_tensor[:, :, :, 3]
    dti_complete_dim[:, :, :, 1, 2] = dti_tensor[:, :, :, 4]
    dti_complete_dim[:, :, :, 2, 1] = dti_tensor[:, :, :, 4]
    dti_complete_dim[:, :, :, 2, 2] = dti_tensor[:, :, :, 5]
    eval, evec = np.linalg.eig(dti_complete_dim)

    fname = os.path.join(dti_path, 'data/{}/L1_aff2jakob.nii.gz'.format(subj_name))
    writeNII(eval[:, :, :, 0], fname, ref_image=dti_file)
    fname = os.path.join(dti_path, 'data/{}/L2_aff2jakob.nii.gz'.format(subj_name))
    writeNII(eval[:, :, :, 1], fname, ref_image=dti_file)
    fname = os.path.join(dti_path, 'data/{}/L3_aff2jakob.nii.gz'.format(subj_name))
    writeNII(eval[:, :, :, 2], fname, ref_image=dti_file)

    fname = os.path.join(dti_path, 'data/{}/V1_aff2jakob.nii.gz'.format(subj_name))
    writeNII(evec[:, :, :, :, 0], fname, ref_image=dti_file)
    fname = os.path.join(dti_path, 'data/{}/V2_aff2jakob.nii.gz'.format(subj_name))
    writeNII(evec[:, :, :, :, 1], fname, ref_image=dti_file)
    fname = os.path.join(dti_path, 'data/{}/V3_aff2jakob.nii.gz'.format(subj_name))
    writeNII(evec[:, :, :, :, 2], fname, ref_image=dti_file)






























