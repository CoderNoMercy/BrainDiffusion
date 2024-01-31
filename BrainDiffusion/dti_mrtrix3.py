import os
import glob
import argparse
import nibabel as nib
import numpy as np
# Auxiliary function to execute an OS command.
def excmd(cmd,skip=False):
    print(cmd)
    if not skip:
        os.system(cmd)

def writeNII(img, filename):
  '''
  function to write a nifti image, creates a new nifti object
  '''
  data = nib.Nifti1Image(img, np.eye(4))
  nib.save(data, filename)

def EVD_diffucivity(dti, dim, dwi_path, postfix):

  total_diffu = np.zeros([dim, dim, dim, 3, 3])
  total_diffu[:, :, :, 0, 0] = dti[:, :, :, 0]
  total_diffu[:, :, :, 0, 1] = dti[:, :, :, 3] # special for MRtrix3 output
  total_diffu[:, :, :, 1, 0] = dti[:, :, :, 3]
  total_diffu[:, :, :, 1, 1] = dti[:, :, :, 1]
  total_diffu[:, :, :, 2, 0] = dti[:, :, :, 4]
  total_diffu[:, :, :, 0, 2] = dti[:, :, :, 4]
  total_diffu[:, :, :, 1, 2] = dti[:, :, :, 5]
  total_diffu[:, :, :, 2, 1] = dti[:, :, :, 5]
  total_diffu[:, :, :, 2, 2] = dti[:, :, :, 2]
  val, vec = np.linalg.eig(total_diffu)
  writeNII(val[:, :, :, 0].squeeze(), os.path.join(dwi_path, 'dti_mrtrix{}_L1.nii'.format(postfix)))
  writeNII(val[:, :, :, 1].squeeze(), os.path.join(dwi_path, 'dti_mrtrix{}_L2.nii'.format(postfix)))
  writeNII(val[:, :, :, 2].squeeze(), os.path.join(dwi_path, 'dti_mrtrix{}_L3.nii'.format(postfix)))

  writeNII(vec[:, :, :, :, 0].squeeze(), os.path.join(dwi_path, 'dti_mrtrix{}_V1.nii'.format(postfix)))
  writeNII(vec[:, :, :, :, 1].squeeze(), os.path.join(dwi_path, 'dti_mrtrix{}_V2.nii'.format(postfix)))
  writeNII(vec[:, :, :, :, 2].squeeze(), os.path.join(dwi_path, 'dti_mrtrix{}_V3.nii'.format(postfix)))
  print('finish eigen decomposition')

parser = argparse.ArgumentParser(description='dwi_process')
parser.add_argument('--DiSCo_ID', default=2) ## example 1, 2, 3
parser.add_argument('--resolution', default='high')
parser.add_argument('--snr', default='inf')
args = parser.parse_args()

fsldir = '/work/08171/zheyw1/frontera/fsl'

os.environ["FSLOUTPUTTYPE"]="NIFTI"
os.environ["FSLDIR"]=fsldir
os.environ["FSLMULTIFILEQUIT"]="TRUE"
os.environ["PATH"]+=os.pathsep + os.path.join('%s/bin'%fsldir)


data_path = '/scratch1/08171/zheyw1/phantom_DWI'

if args.resolution == 'high':
    dwi_path = os.path.join(data_path, 'DiSCo{}/high_resolution_40x40x40'.format(args.DiSCo_ID))
    prefix = ''
else:
    dwi_path = os.path.join(data_path, 'DiSCo{}/low_resolution_20x20x20'.format(args.DiSCo_ID))
    prefix = 'lowRes_'

if args.snr == 'inf':
    postfix = ''
else:
    postfix = '_RicianNoise-snr{}'.format(args.snr)

bvec_path = os.path.join(data_path, 'diffusion_gradients/DiSCo_gradients_fsl.bvecs')
bval_path = os.path.join(data_path, 'diffusion_gradients/DiSCo_gradients.bvals')
mask = os.path.join(dwi_path, prefix + 'DiSCo{}_mask.nii.gz'.format(args.DiSCo_ID))

print(" \t Fitting DTI tensor")

dwi2tensor = lambda imgin, imgout, bvec, bval: 'dwi2tensor -fslgrad {} {} -mask {} -force {} {}'.format(bvec, bval, mask, imgin, imgout)
imgin = os.path.join(dwi_path, 'DiSCo{}_DWI{}.nii.gz'.format(args.DiSCo_ID, postfix))
imgout = os.path.join(dwi_path, 'dti_mrtrix{}.nii.gz'.format(postfix))
cmd = dwi2tensor(imgin, imgout, bvec_path, bval_path)
excmd(cmd)
imgout_arr = np.asarray(nib.load(imgout).get_fdata()).astype('float')
EVD_diffucivity(imgout_arr, 40, dwi_path, postfix)

