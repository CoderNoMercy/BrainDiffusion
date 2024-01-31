import os, argparse
import pandas as pd
import numpy as np

def MRtrix3_conn_mtx_gen(dwi_path, output_dir, prefix, postfix, disco_id):


    fsldir = '/work2/08171/zheyw1/frontera/fsl'

    os.environ["FSLOUTPUTTYPE"] = "NIFTI"
    os.environ["FSLDIR"] = fsldir
    os.environ["FSLMULTIFILEQUIT"] = "TRUE"
    os.environ["PATH"] += os.pathsep + os.path.join('%s/bin' % fsldir)

    parcellation = os.path.join(dwi_path, '{}DiSCo{}_ROIs.nii.gz'.format(prefix, disco_id))
    curr_path = os.getcwd()
    lookup_tabel = os.path.join(curr_path, 'phantom_label.txt')
    dwi = os.path.join(dwi_path, '{}DiSCo{}_DWI{}.nii.gz'.format(prefix, disco_id, postfix))
    bvecs = os.path.join(dwi_path, '../../diffusion_gradients/DiSCo_gradients_fsl.bvecs')
    bvals = os.path.join(dwi_path, '../../diffusion_gradients/DiSCo_gradients.bvals')
    mask = os.path.join(dwi_path, '{}DiSCo{}_mask.nii.gz'.format(prefix, disco_id))

    ## connectome generation
    num_streamlines = '1M'
    labelconvert = lambda : 'labelconvert {} {} {} {}/nodes.mif'.format(parcellation, lookup_tabel, lookup_tabel, output_dir)
    tckgen = lambda : 'tckgen {} {}.tck -algorithm Tensor_Det -fslgrad {} {} -seed_image {} -select {}'.format(dwi, output_dir+'/'+ prefix+num_streamlines+postfix+'_'+disco_id, bvecs, bvals, mask, num_streamlines)
    tck2connectome = lambda : 'tck2connectome {}.tck {}/nodes.mif {}/connectome_{}.csv'.format(output_dir+'/'+ prefix+num_streamlines+postfix+'_'+disco_id, output_dir, output_dir, prefix+num_streamlines+postfix+'_'+disco_id)

    os.system(labelconvert())
    os.system(tckgen())
    os.system(tck2connectome())
    print('completed the third step: connectome generation')
    return output_dir + '/connectome_{}.csv'.format(prefix+num_streamlines+postfix+'_'+disco_id)

def change_mtx_format(conn_mtx_csv_file, output_path):

    conn_mtx_csv = pd.read_csv(conn_mtx_csv_file)
    nregion = len(conn_mtx_csv.keys())
    conn_mtx = np.zeros([nregion, nregion])
    first_row = np.asarray(conn_mtx_csv.keys())
    conn_mtx[0, :] = first_row
    conn_mtx[1:, :] = np.asarray(conn_mtx_csv.values)
    np.fill_diagonal(conn_mtx, 0)
    conn_mtx += conn_mtx.T
    np.savez(output_path, conn_mtx=conn_mtx)


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

    output_path = '/scratch1/08171/zheyw1/phantom_DWI_result/MRtrix3_{}DiSCo{}{}'.format(prefix, args.disco_id, postfix)
    os.makedirs(output_path, exist_ok=True)
    conn_mtx_csv_path = MRtrix3_conn_mtx_gen(dwi_path, output_path, prefix, postfix, args.disco_id)
    #change_mtx_format(conn_mtx_csv_path, output_path + 'conn_mtx.npz')


