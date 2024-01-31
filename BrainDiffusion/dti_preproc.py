import os
import glob
import argparse

# Auxiliary function to execute an OS command.
def excmd(cmd,skip=False):
    print(cmd)
    if not skip:
        os.system(cmd)


parser = argparse.ArgumentParser(description='dwi_process')
parser.add_argument('--DiSCo_ID') ## example 1, 2, 3
parser.add_argument('--resolution')
parser.add_argument('--snr')
args = parser.parse_args()

fsldir = '/work/08171/zheyw1/frontera/fsl'

os.environ["FSLOUTPUTTYPE"]="NIFTI"
os.environ["FSLDIR"]=fsldir
os.environ["FSLMULTIFILEQUIT"]="TRUE"
os.environ["PATH"]+=os.pathsep + os.path.join('%s/bin'%fsldir)


split = lambda imgin, prefix : 'fslsplit %s %s'%(imgin, prefix)
bet = lambda imgin, imgout: 'bet %s %s -m' %(imgin, imgout)
merge = lambda imgout, list_imgs: 'fslmerge -t %s %s '%(imgout, list_imgs)   
mcflirt = lambda imgin, imgout : 'mcflirt -in %s -out %s '%(imgin, imgout)
extract = lambda imgin, imgout, tmin, tsize : 'fslroi %s %s %d %d'%(imgin, imgout, tmin, tsize)
topup = lambda imgin, acq, out, imgout : 'topup --imain=%s --datain=%s --out=%s --iout=%s'%(imgin, acq, out, imgout)

eddy = lambda imgin, mask, acqparams, index, bvecs, bvals, tu_out, imgout : \
       "eddy --imain=%s --mask=%s --acqp=%s --index=%s --bvecs=%s --bvals=%s --topup=%s --out=%s --cnr_maps --repol --estimate_move_by_susceptibility --verbose"%(imgin, mask, acqparams, index, bvecs, bvals, tu_out, imgout)

dtifit = lambda imgin, imgout, mask, bvecs, bvals : "dtifit --data=%s --mask=%s --out=%s --bvecs=%s --bvals=%s --save_tensor"%(imgin, mask, imgout, bvecs, bvals)


print("\t Skull stripping")


data_path = '/scratch/08171/zheyw1/phantom_DWI'

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

acqp = "0 -1 0 %.3f\n"%(0.102)
acqp_path = os.path.join(data_path, 'acqp_b0.txt')
if not os.path.exists(acqp_path):
    with open(acqp_path, 'w') as f:
        for i in range(1):
            f.write(acqp)

acqp_eddy_path = os.path.join(data_path, 'acqp_eddy.txt')
if not os.path.exists(acqp_eddy_path):
    with open(acqp_eddy_path, 'w') as f:
        for i in range(1):
            f.write(acqp)

idx=""
for i in range(364):
    idx+=" 1"
index_path = os.path.join(data_path, 'index.txt')
if not os.path.exists(index_path):
    with open(index_path, 'w') as f:
        f.write(idx)


# dwi_process_path = os.path.join(data_path, 'DiSCo1_processed')
# if not os.path.exists(dwi_process_path):
#   os.makedirs(dwi_process_path, exist_ok=True)

# imgin = os.path.join(dwi_path, 'DiSCo1_DWI.nii.gz')
# prefix = os.path.join(dwi_process_path, 'dwi_v_')
# print(" \t Spliting into 3D images")
# cmd = split(imgin, prefix)
# excmd(cmd)


# list_str_out = ''
# list_str_out_mask = ''
# for v in range(35):
#   imgin = prefix+'%04d.nii'%v
#   imgout = prefix+'%04d_str.nii'%v
#   list_str_out += imgout+' '
#   list_str_out_mask += prefix+'%04d_str_mask.nii'%v +' '
#   cmd = bet(imgin, imgout)
#   excmd(cmd)
#
# print(" \t Merging into 4D and remove unnecessary files")
# imgout = os.path.join(res_path, 'dwi_%d_str.nii.gz'%t)
# cmd = merge(imgout, list_str_out)
# excmd(cmd)

# imgout = os.path.join(res_path, 'dwi_str_mask_all.nii.gz')
# cmd = merge(imgout, list_str_out_mask)
# excmd(cmd)

# imgin = os.path.join(res_path, 'dwi_%d_str_mask_all.nii.gz'%t)
# imgout = os.path.join(res_path, 'dwi_%d_str_mask_b0_all.nii.gz'%t)
# cmd = extract(imgin, imgout, 0, 5)
# excmd(cmd)

# imgin = os.path.join(res_path, 'dwi_%d_str_mask_b0_all.nii.gz'%t)
# cmd = "fslmaths "+imgin+" -Tmedian "+os.path.join(res_path, 'dwi_%d_str_mask.nii.gz'%t)
# excmd(cmd)

# cmd = 'rm '+os.path.join(res_path, 'dwi_%d_v_*'%t)
# excmd(cmd)

'''
print(" \t Motion correction")

imgin = os.path.join(res_path, 'dwi_%d_str.nii.gz'%t)
imgout = os.path.join(res_path, 'dwi_%d_str_mc.nii.gz'%t)
cmd = mcflirt(imgin, imgout)
excmd(cmd)
'''

# print(" \t Extracting first images are with b=0")
# imgin = os.path.join(dwi_path, prefix+'DiSCo{}_DWI{}.nii.gz'.format(args.DiSCo_ID, postfix))
# imgout = os.path.join(dwi_path, 'DiSCo{}_DWI_b0{}.nii.gz'.format(args.DiSCo_ID, postfix))
# cmd = extract(imgin, imgout, 0, 1)
# excmd(cmd)
#
# print(" \t Susceptibility correction")
# imgin = os.path.join(dwi_path, 'DiSCo{}_DWI_b0{}.nii.gz'.format(args.DiSCo_ID, postfix))
# imgout = os.path.join(dwi_path, 'DiSCo{}_DWI_b0_tu{}.nii.gz'.format(args.DiSCo_ID, postfix))
# tuout = os.path.join(dwi_path, 'out_tu_{}'.format(postfix))
# cmd = topup(imgin, acqp_path, tuout, imgout)
# excmd(cmd)
#
# print(" \t Eddy current correction")
#
# imgin = os.path.join(dwi_path, prefix + 'DiSCo{}_DWI{}.nii.gz'.format(args.DiSCo_ID, postfix))
# imgout = os.path.join(dwi_path, 'DiSCo{}_DWI_ec{}.nii.gz'.format(args.DiSCo_ID, postfix))
bvec_path = os.path.join(data_path, 'diffusion_gradients/DiSCo_gradients_fsl.bvecs')
bval_path = os.path.join(data_path, 'diffusion_gradients/DiSCo_gradients.bvals')
mask = os.path.join(dwi_path, prefix + 'DiSCo{}_mask.nii.gz'.format(args.DiSCo_ID))
# cmd = eddy(imgin, mask, acqp_eddy_path, index_path, bvec_path, bval_path, tuout, imgout)
# excmd(cmd)


print(" \t Fitting DTI tensor")

imgin = os.path.join(dwi_path, 'DiSCo{}_DWI{}.nii.gz'.format(args.DiSCo_ID, postfix))
imgout_base = os.path.join(dwi_path, 'dti{}'.format(postfix))
cmd = dtifit(imgin, imgout_base, mask, bvec_path, bval_path)
excmd(cmd)

    
    
     
    

    
    
    

    








