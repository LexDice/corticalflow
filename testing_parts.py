import nibabel
import pyvista as pv
import numpy as np
import trimesh
import torch

def mri_reader(path, hemisphere=None):
    nib_mri = nibabel.load(path)
    # if hemisphere: nib_mri = {'L': nib_mri.slicer[128:256,15:303,0:240], 'R': nib_mri.slicer[9:105,12:204,10:170]}[hemisphere]
    mri_header, mri_vox, mri_affine = nib_mri.header, np.array(nib_mri.get_fdata()).astype(np.float32), nib_mri.affine.astype(np.float32)
    # nibabel voxel to world cordinates affine
    return mri_header, mri_vox, mri_affine


def mesh_reader(path):
    file_extension = path.split('.')[-1]
    if file_extension == 'vtk' or file_extension == 'vtp':
        mesh = pv.read(path)
        vertices = np.array(mesh.points).astype(np.float32)
        faces = mesh.faces.reshape(-1, 4)
        # remove padding of pv polygons(first column)
        faces = np.array(faces[:, 1:]).astype(np.int32)
    else:
        mesh = trimesh.load(path)
        vertices = np.array(mesh.vertices).astype(np.float32)
        faces = np.array(mesh.faces).astype(np.int32)

    return vertices, faces


nii_filename = '/mnt/yaplab/data/junjiez/HCP_TRT/433839/T1w/T1w_acpc_dc_restore.nii'
mesh_filename = '/mnt/yaplab/data/junjiez/HCP_TRT/433839/T1w/Native/433839.L.white.surf.vtk'
header, mri_vox, mri_affine = mri_reader(nii_filename, 'L')
verts, faces = mesh_reader(mesh_filename)

# map world coordinates to mri voxels
pred = torch.cat([torch.from_numpy(verts), torch.ones((verts.shape[0], 1))], dim=-1)
pred = torch.matmul(torch.from_numpy(np.linalg.inv(mri_affine)), pred.transpose(1, 0)).transpose(1, 0)[:, :-1]
coords = np.rint(pred.numpy()).astype(np.int32)

# coords = np.sort(coords, axis=0)
aa = coords.transpose()
mri_vox[aa[0], aa[1], aa[2]] = 1500
new = nibabel.Nifti1Image(mri_vox, mri_affine, header)
nibabel.save(new, 'mesh_voxel.nii')
