import cupy as cp
import os
# import h5py
import cupy.linalg
import nibabel as nib
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy.sparse.linalg import cg
from cupyx.scipy.sparse.linalg import LinearOperator
import scipy.io as sio
import proplot as pplt

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
  
  c = params.c.copy()
  
  IOmega_x = params.IOmega_x.copy()
  IOmega_y = params.IOmega_y.copy()
  IOmega_z = params.IOmega_z.copy()
  
  c_hat = cp.fft.fftn(c)
  
  params.grad_x = cp.real(cp.fft.ifftn(IOmega_x * c_hat))
  params.grad_y = cp.real(cp.fft.ifftn(IOmega_y * c_hat))
  params.grad_z = cp.real(cp.fft.ifftn(IOmega_z * c_hat))


  return params


def weierstrass_smoother(c, N, sigma):


  x = cp.arange(0, N) * 2 * cp.pi / N
  y = x.copy()
  z = x.copy()

  h = 2 * cp.pi / N
  sigma2 = sigma ** 2

  filter = cp.zeros((N, N, N))

  for k in range(N):
    filter[:, :, k] = cp.exp(-0.5 * (y/sigma)**2) * cp.exp(-0.5 * (x/sigma)**2) * cp.exp(-0.5 * (z[k]/sigma)**2) \
                      + cp.exp(-0.5 * (y-2*cp.pi)**2) * cp.exp(-0.5 * (x-2*cp.pi)**2) * cp.exp(-0.5 * (z[k]-2*cp.pi)**2)

  filter /= cp.sum(filter.flatten())/(h**3)
  filter_hat = cp.fft.fftn(filter)

  c_hat = cp.fft.fftn(c)
  dummy = c_hat * filter_hat

  c_smooth = h*h*h * cp.fft.ifftn(dummy)

  return cp.real(c_smooth)


def X_mask(params, roi_id):
  
  Xmask = cp.zeros(params.wm.shape)
  
  Xmask[params.labels == roi_id] = 1
  Xmask = gaussian_filter(Xmask, params.gm_sigma)
  # Xmask = weierstrass_smoother(Xmask, params.N, params.sm_sigma)
  params.Xmask = Xmask.copy()
  
  
  return params


def compute_div(params, c, i):
  # dt = params.dt
  N = params.N
  c = c.reshape((N, N, N)).copy()
  if i % params.nskip_cg == 0 and i!=0:
    c = gaussian_filter(c.copy(), params.gm_sigma)
    # c = weierstrass_smoother(c.copy(), N, params.sm_sigma)
  #if i % 10 == 0:
     # fname = params.out_dir + '/P_FC2EHP/debug/' + 'c_iter{}.nii.gz'.format(i)
     # writeNII(cp.real(c).get(), fname, ref_image=params.affine)
  params.c = c.copy()
  params = grad3D(params)

  # Kxx = params.Kxx
  # Kxy = params.Kxy
  # Kxz = params.Kxz
  # Kyy = params.Kyy
  # Kyz = params.Kyz
  # Kzz = params.Kzz

  grad_x = params.grad_x.copy()
  grad_y = params.grad_y.copy()
  grad_z = params.grad_z.copy()


  # wm = params.wm.copy()

  # tmp = Kxx * grad_x + Kxy * grad_y + Kxz * grad_z
  tmp = params.T_par[:, :, :, 0, 0].squeeze() * grad_x + params.T_par[:, :, :, 0, 1].squeeze() * grad_y + params.T_par[:, :, :, 0, 2].squeeze() * grad_z
  tmp += 0.01 * (params.T_orth[:, :, :, 0, 0].squeeze() * grad_x + params.T_orth[:, :, :, 0, 1].squeeze() * grad_y + params.T_orth[:, :, :, 0, 2].squeeze() * grad_z)

  # tmp = params.kappa * grad_x
  if i % params.nskip_cg == 0:
    tmp = gaussian_filter(tmp.copy(), params.gm_sigma)
    # tmp = weierstrass_smoother(tmp.copy(), N, params.sm_sigma)

  #if i % 10 == 0:
    #fname = params.out_dir + '/P_FC2EHP/debug/' + 'k_gradcx_iter{}.nii.gz'.format(i)
   # writeNII(cp.real(tmp).get(), fname, ref_image=params.affine)

  x_hat = cp.fft.fftn(tmp.copy())
  x_hat *= params.IOmega_x

  # tmp = Kxy * grad_x + Kyy * grad_y + Kyz * grad_z
  tmp = params.T_par[:, :, :, 1, 0].squeeze() * grad_x + params.T_par[:, :, :, 1, 1].squeeze() * grad_y + params.T_par[:, :, :, 1, 2].squeeze() * grad_z
  tmp += 0.01 * (params.T_orth[:, :, :, 1, 0].squeeze() * grad_x + params.T_orth[:, :, :, 1, 1].squeeze() * grad_y + params.T_orth[:, :, :, 1, 2].squeeze() * grad_z)

  # tmp = params.kappa * grad_y
  if i % params.nskip_cg == 0:
    tmp = gaussian_filter(tmp.copy(), params.gm_sigma)
    # tmp = weierstrass_smoother(tmp.copy(), N, params.sm_sigma)
  #if i % 10 == 0:
    #fname = params.out_dir + '/P_FC2EHP/debug/' + 'k_gradcy_iter{}.nii.gz'.format(i)
   # writeNII(cp.real(tmp).get(), fname, ref_image=params.affine)

  y_hat = cp.fft.fftn(tmp.copy())
  y_hat *= params.IOmega_y

  # tmp = Kxz * grad_x + Kyz * grad_y + Kzz * grad_z
  tmp = params.T_par[:, :, :, 2, 0].squeeze() * grad_x + params.T_par[:, :, :, 2, 1].squeeze() * grad_y + params.T_par[:, :, :, 2, 2].squeeze() * grad_z
  tmp += 0.01 * (params.T_orth[:, :, :, 2, 0].squeeze() * grad_x + params.T_orth[:, :, :, 2, 1].squeeze() * grad_y + params.T_orth[:, :, :, 2, 2].squeeze() * grad_z)

  # tmp = params.kappa * grad_z
  if i % params.nskip_cg == 0:
    tmp = gaussian_filter(tmp.copy(), params.gm_sigma)
    # tmp = weierstrass_smoother(tmp.copy(), N, params.sm_sigma)
  #if i % 10 == 0:
   # fname = params.out_dir + '/P_FC2EHP/debug/' + 'k_gradcz_iter{}.nii.gz'.format(i)
   #writeNII(cp.real(tmp).get(), fname, ref_image=params.affine)
  z_hat = cp.fft.fftn(tmp.copy())
  z_hat *= params.IOmega_z

  x = cp.real(cp.fft.ifftn(x_hat))
  y = cp.real(cp.fft.ifftn(y_hat))
  z = cp.real(cp.fft.ifftn(z_hat))

  # div_hat = x_hat + y_hat + z_hat
  div = x + y + z
  return div

def applyA_lhs(params, c):

  #div = cp.fft.ifftn(div_hat)
  i = 1
  div = compute_div(params, c, i)
  # div = gaussian_filter(div.copy(), 1)
  Ac = -div + params.beta * params.Xmask * params.c
  # Ac = gaussian_filter(Ac.copy(), 1)
  return Ac.flatten()



def applyb_rhs(params, c):
  
  N = params.N
  bc = params.Xmask * params.beta
  
  return bc.flatten()
  


def apply_Pinv(params, c):
  
  N = params.N
  
  c = c.reshape((N,N,N)).copy()
  c_hat = cp.fft.fftn(c)

  kmean = params.kappa * params.kmean
   
  IOmega_x = params.IOmega_x
  IOmega_y = params.IOmega_y
  IOmega_z = params.IOmega_z
  
  c_hat = c_hat / ((kmean) * (- IOmega_x**2 - IOmega_y**2 - IOmega_z**2) + params.beta)
  
  Pinv_c = cp.real(cp.fft.ifftn(c_hat))

  return Pinv_c.flatten()


def solve_cg_cupy(A, b, x0=None, maxiter=100, term_tol=1e-6, params=None):
  # gwm_mask = (params.wm + 0.01 * params.gm).flatten().copy()
  def one_loop(i, rTr, x, r, p):
      Ap = A(p, i)
      alpha = (rTr / cp.sum((cp.conj(p) * Ap)))
      alpha = alpha + 0.j
      x = x + alpha * p
      r = r - alpha * Ap
      rTrNew = cp.sum(cp.conj(r) * r)
      beta = rTrNew / rTr
      beta = beta + 0.j
      p = r + beta * p
      return i+1, rTrNew, x, r, p
    
  if x0 is None:
      x = cp.zeros(b.shape) + 0.j
  else:
      x = x0 + 0.j

  r = b - A(x, 0) + 0.j
  i = 1
  p = r.copy()
  rTr = cp.sum(cp.conj(r) * r)
  tol = cp.linalg.norm(b)
  norm_res = cp.linalg.norm(r)
  norm_res_init = norm_res.copy()
  while norm_res > term_tol * tol:

    [i, rTr, x, r, p] = one_loop(i, rTr, x, r, p)
    norm_res = cp.linalg.norm(r)
    if i >= maxiter:
      break
    if (i-1) % 100 == 0:
      print('pcg iter: %d, relative residual = %.4e'%(i, norm_res/norm_res_init))
  return x



def solve_forward_cg(params, roi_id):
  

  N = params.N
  wm = params.wm_ref.copy()
  gm = params.gm_ref.copy()

  params.k_gm_wm = 0.0001

  params.T_par = params.T_par * params.kappa * cupy.expand_dims(wm + params.k_gm_wm * gm, axis=(3, 4))
  params.T_orth = params.T_orth * params.kappa * cupy.expand_dims(wm + params.k_gm_wm * gm, axis=(3, 4))

  params.kmean = (cp.mean(params.T_par + 0.01 * params.T_orth))

  c = cp.zeros(wm.shape)
  c[params.labels == roi_id] = 1
 
  params = X_mask(params, roi_id)
 
  c0_sm = gaussian_filter(c, params.gm_sigma).copy()
  params.c = c.copy() 
  params.c0 = c.copy()

  #f_A = lambda x, i : apply_Pinv(params, applyA_lhs(params, x, i))
  #f_b = lambda x : apply_Pinv(params, applyb_rhs(params, x))

  f_A = lambda x : applyA_lhs(params, x)
  f_b = lambda x : applyb_rhs(params, x)

  #c = solve_cg_cupy(f_A, f_b(c0_sm.copy()), x0=c0_sm.flatten().copy(), maxiter = params.maxiter, term_tol=9e-4, params=params)
  f_A_op = LinearOperator((c0_sm.copy().flatten().shape[0],f_b(c0_sm.copy()).shape[0]), matvec=f_A)
  c, _ = cg(f_A_op, f_b(c0_sm.copy()), maxiter=params.maxiter, tol=1e-4)
  c[c < 0.0] = 0
  c = gaussian_filter(cp.real(c.reshape((N,N,N))), params.gm_sigma)
  # c = weierstrass_smoother(cp.real(c.reshape((N,N,N))), N, params.sm_sigma)


  params.c1 = c.copy()
  
  return params


def main_objective_func(params, c):

    div = compute_div(params, c)
    return cupy.linalg.norm(div) / 2

def regularity_func(params, c):

    return 1 / 2 * cupy.linalg.norm(params.Xmask * (c - 1))


def find_surface_wm(roi_voxels, wm_voxels):
    collection_vox = []
    voxels = []
    for i in range(len(roi_voxels[0])):
      voxels.append((roi_voxels[0][i].item(), roi_voxels[1][i].item(), roi_voxels[2][i].item()))
    for vox in voxels:
        for i in [-1, 0, 1]:
          for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
              collection_vox.append(tuple(np.array(vox) + np.array((i, j, k))))
    surface_wm = set(collection_vox).intersection(set(wm_voxels))
    return surface_wm

def find_surface_wm_fast(params, subject_name):

    x_wm, y_wm, z_wm = np.where(params.wm_ref == 1)
    wm_loc = np.zeros([len(x_wm), 3])
    wm_loc[:, 0] = x_wm.get()
    wm_loc[:, 1] = y_wm.get()
    wm_loc[:, 2] = z_wm.get()

    for roi in params.labels_list:
      x, y, z = np.where(params.labels == roi)
      x = x.get()
      y = y.get()
      z = z.get()

      surface_3D = np.zeros(params.wm_ref.shape)
      for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
          for k in [-1, 0, 1]:
              x_new = x + i
              y_new = y + j
              z_new = z + k
              surface_3D[(x_new.astype('int'), y_new.astype('int'), z_new.astype('int'))] = 1

      surface_3D *= params.wm_ref.get()
      writeNII(surface_3D, '/scratch1/08171/zheyw1/conn_mat_data_real/' + subject_name + '/surface_roi{}.nii.gz'.format(roi), ref_image=params.affine)
      print('roi {} saved'.format(roi))



def update_conn_mtx(params, c, source_roi, roi_list, conn_mtx):
    # c = cupy.asnumpy(c)
    roi_idx = list(roi_list).index(source_roi)
    for other_roi in list(roi_list):
        if other_roi != source_roi:
          other_roi_idx = list(roi_list).index(other_roi)
          surface_roi_mask = nib.load('/scratch1/08171/zheyw1/conn_mat_data/surface_roi{}.nii'.format(other_roi)).get_fdata()

          sum_concentration = c[np.where(surface_roi_mask == 1)].sum()
          # sum_concentration += c[np.where(params.labels.get() == other_roi)].sum()
          del surface_roi_mask
          conn_mtx[roi_idx, other_roi_idx] += sum_concentration / 2
          conn_mtx[other_roi_idx, roi_idx] += sum_concentration / 2
        # print('finish roi {}'.format(other_roi))
    return conn_mtx

def get_rois_voxel_wm_voxel(roi_list, parcellation, wm_mask):

  rois_voxel = {}
  wm_voxel = []
  for roi in roi_list:
      x, y, z = np.where(parcellation == roi)
      rois_voxel[roi] = (x, y, z)
  x, y, z = np.where(wm_mask == 1)
  for i in range(len(x)):
      wm_voxel.append((x[i].item(), y[i].item(), z[i].item()))
  return rois_voxel, wm_voxel

def plot_conn_mat(params, conn_mtx):

  fig = pplt.figure(share=False, refwidth='30cm')
  axs = fig.subplots()
  m = axs.imshow(conn_mtx, vmin=0, vmax=100)
  axs.set_xticks(np.arange(0, len(params.labels_list)))
  axs.set_xticklabels(params.labels_name, fontsize=5, rotation=90)
  axs.set_yticks(np.arange(0, len(params.labels_list)))
  axs.set_yticklabels(params.labels_name, fontsize=5)
  axs.colorbar(m, ticks=[0, 1])
  fig.savefig('/scratch1/08171/zheyw1/conn_mat_data/conn_mat.pdf', format='pdf')

def EVD_diffucivity(params, subj_name):

  total_diffu = np.zeros([256, 256, 256, 3, 3])
  total_diffu[:, :, :, 0, 0] = params.Kxx_ref
  total_diffu[:, :, :, 0, 1] = params.Kxy_ref
  total_diffu[:, :, :, 1, 0] = params.Kxy_ref
  total_diffu[:, :, :, 1, 1] = params.Kyy_ref
  total_diffu[:, :, :, 2, 0] = params.Kxz_ref
  total_diffu[:, :, :, 0, 2] = params.Kxz_ref
  total_diffu[:, :, :, 1, 2] = params.Kyz_ref
  total_diffu[:, :, :, 2, 1] = params.Kyz_ref
  total_diffu[:, :, :, 2, 2] = params.Kzz_ref
  val, vec = np.linalg.eig(total_diffu)
  writeNII(val[:, :, :, 0].squeeze(), '/scratch1/08171/zheyw1/conn_mat_data_real/' + subj_name + '/aff2jakob/L1_aff2jakob.nii.gz')
  writeNII(val[:, :, :, 1].squeeze(), '/scratch1/08171/zheyw1/conn_mat_data_real/' + subj_name + '/aff2jakob/L2_aff2jakob.nii.gz')
  writeNII(val[:, :, :, 2].squeeze(), '/scratch1/08171/zheyw1/conn_mat_data_real/' + subj_name + '/aff2jakob/L3_aff2jakob.nii.gz')

  writeNII(vec[:, :, :, :, 0].squeeze(), '/scratch1/08171/zheyw1/conn_mat_data_real/' + subj_name + '/aff2jakob/V1_aff2jakob.nii.gz')
  writeNII(vec[:, :, :, :, 1].squeeze(), '/scratch1/08171/zheyw1/conn_mat_data_real/' + subj_name + '/aff2jakob/V2_aff2jakob.nii.gz')
  writeNII(vec[:, :, :, :, 2].squeeze(), '/scratch1/08171/zheyw1/conn_mat_data_real/' + subj_name + '/aff2jakob/V3_aff2jakob.nii.gz')
  print('finish subj {}'.format(subj_name))
  # L1 = np.expand_dims((val[:, :, :, 0]).squeeze(), axis=(3, 4))
  # L2 = np.expand_dims((val[:, :, :, 1]).squeeze(), axis=(3, 4))
  # L3 = np.expand_dims((val[:, :, :, 2]).squeeze(), axis=(3, 4))
  #
  # V1_T = np.expand_dims((vec[:, :, :, :, 0]).squeeze(), axis=3)
  # V2_T = np.expand_dims((vec[:, :, :, :, 1]).squeeze(), axis=3)
  # V3_T = np.expand_dims((vec[:, :, :, :, 2]).squeeze(), axis=3)
  #
  # V1 = np.expand_dims((vec[:, :, :, :, 0]).squeeze(), axis=4)
  # V2 = np.expand_dims((vec[:, :, :, :, 1]).squeeze(), axis=4)
  # V3 = np.expand_dims((vec[:, :, :, :, 2]).squeeze(), axis=4)
  #
  # T_par = L1 * np.matmul(V1, V1_T)
  # T_orth = L2 * np.matmul(V2, V2_T) + L3 * np.matmul(V3, V3_T)
  #
  # T_toal = T_par + T_orth
  # print(np.linalg.norm(T_toal[:, :, :, 0, 0].squeeze() - params.Kxx_ref))
  # print(np.linalg.norm(T_toal[:, :, :, 0, 1].squeeze() - params.Kxy_ref))
  # print(np.linalg.norm(T_toal[:, :, :, 0, 2].squeeze() - params.Kxz_ref))
  # print(np.linalg.norm(T_toal[:, :, :, 1, 1].squeeze() - params.Kyy_ref))
  # print(np.linalg.norm(T_toal[:, :, :, 1, 2].squeeze() - params.Kyz_ref))
  # print(np.linalg.norm(T_toal[:, :, :, 2, 2].squeeze() - params.Kzz_ref))
   
  




  
  
  
  
