import logging, os, random
from torch.utils import data
import numpy as np
import pandas as pd
import torch
import trimesh, nibabel
import pyvista as pv
from pytorch3d.structures import Meshes


logger = logging.getLogger(__name__)


class CSRDataset(data.Dataset):
    def __init__(self, subjects, mris, surfaces=None, hemisphere=None, transforms=None):
        assert (len(subjects) == len(mris)) 
        assert (not surfaces) or (len(subjects) == len(surfaces))
        assert (not hemisphere) or (hemisphere in ['L', 'R'])
        
        self.subjects, self.mris, self.surfaces = subjects, mris, surfaces        
        self.hemisphere, self.transforms = hemisphere, transforms
        
    def __len__(self):               
        return len(self.subjects)

    def __getitem__(self, idx):
        subject, mri_path = self.subjects[idx], self.mris[idx]
        surf_path = self.surfaces[idx] if self.surfaces else None
        data = {}

        data['subject'] = subject

        # read MRI
        _, mri_vox, mri_affine = mri_reader(mri_path, hemisphere=self.hemisphere)
        data['mri_vox'], data['mri_affine'] = mri_vox, mri_affine

        # read surfaces
        if surf_path != None:
            mesh_vertices, mesh_faces = mesh_reader(surf_path)
            data['surf_verts'], data['surf_faces'] = mesh_vertices, mesh_faces

        # apply field specific transform
        if self.transforms:
            for field in ['mri', 'surf']:
                if field in self.transforms: data = self.transforms[field](data)

        return data


def csr_dataset_factory(dataset_type, hemisphere, transforms, **kwargs):    
    list_subjects, list_mris, list_surfaces = [], [], []            
    
    if dataset_type == 'file':
        # read '\t' separated file with columns subject_id, mri_path
        with open(kwargs['input_file'], 'r') as mrilist_file:            
            for subj_mri_surf in mrilist_file:
                subj_mri_surf = subj_mri_surf.split('\t')
                list_subjects.append(subj_mri_surf[0].strip()); list_mris.append(subj_mri_surf[1].strip());                     

    elif dataset_type == 'list': 
        list_subjects, list_mris = kwargs['subjects'], kwargs['mris']
    
    elif dataset_type == 'formatted':
        data_desc = pd.read_csv(kwargs['split_file'], dtype={'subject': object, 'split': object})
        data_desc = data_desc[data_desc['split'] == kwargs['split_name']]            
        list_subjects = list(data_desc.subject.values)
        list_mris = list(data_desc.subject.apply(lambda subj: os.path.join(kwargs['dataset_path'], subj,  'T1w/T1w_acpc_dc_restore.nii')).values)
        if kwargs['surface_name']:
            list_surfaces = list(data_desc.subject.apply(lambda subj: os.path.join(kwargs['dataset_path'], subj, 'T1w/Native', str(subj) + '.{}.surf.vtk'.format(kwargs['surface_name']))).values)
            assert hemisphere == kwargs['surface_name'].split('.')[0]
    else:
        raise ValueError('dataset type option is not supported')

    return CSRDataset(list_subjects, list_mris, list_surfaces, hemisphere, transforms)


def collate_CSRData_fn(batch_list):

    batch_data = dict()        
    batch_data['subject'] = [data['subject'] for data in batch_list]
    batch_data['mri_vox'] = torch.from_numpy(np.stack([data['mri_vox'] for data in batch_list], axis=0))
    batch_data['mri_affine'] = torch.from_numpy(np.stack([data['mri_affine'] for data in batch_list], axis=0))
    if all(['surf_verts' in data.keys() for data in batch_list]):
        batch_data['py3d_meshes'] = Meshes(verts=[torch.from_numpy(data['surf_verts']).float() for data in batch_list], faces=[torch.from_numpy(data['surf_faces']).long() for data in batch_list])
    return batch_data


def pack_meshes(vertices_list, faces_list):
    packed_vertices, packed_faces, packed_lenghts, vertices_cumsum = [], [], [], 0.0
    assert len(vertices_list) == len(faces_list)
    for verts, faces in zip(vertices_list, faces_list):
        packed_vertices.append(verts)
        packed_faces.append(faces +  vertices_cumsum)
        packed_lenghts.append([verts.shape[0], faces.shape[0]])
        vertices_cumsum += verts.shape[0]    
    return np.concatenate(packed_vertices, axis=0), np.concatenate(packed_faces, axis=0), np.array(packed_lenghts)


# READERS
def mri_reader(path, hemisphere=None):
    nib_mri = nibabel.load(path)
    if hemisphere: nib_mri = {'L': nib_mri.slicer[124:236,31:287,48:224], 'R': nib_mri.slicer[16:128,31:287,48:224]}[hemisphere]
    # if hemisphere: nib_mri = {'L': nib_mri.slicer[128:256,15:303,0:240], 'R': nib_mri.slicer[9:105,12:204,10:170]}[hemisphere]
    mri_header, mri_vox, mri_affine = nib_mri.header, nib_mri.get_fdata().astype(np.float32), nib_mri.affine.astype(np.float32)
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

def transform_reader(path):
    transform = np.loadtxt(path)
    return transform


# TRANSFORMS
class NormalizeMRIVoxels(object):
    '''
    Normalize input voxel data
    '''
    def __init__(self, norm_type='mean_std', **kwargs):
        """
        normalize transform for inputs
        :param norm_type: type of normalization mean-std or min-max
        :param kwargs: parameters for normalization
        """
        super(NormalizeMRIVoxels, self).__init__()

        self.norm_type = norm_type
        self.args_dict = kwargs

    def __call__(self, data):
        """
        Apply normalization to data
        :param data: data point
        :return: data points with normalized inputs
        """
        if self.norm_type == 'mean_std':            
            mean = float(self.args_dict.get('mean', data['mri_vox'].mean()))
            std = float(self.args_dict.get('std', data['mri_vox'].std()))
            data['mri_vox'] = (data['mri_vox'] - mean) / std                        

        elif self.norm_type == 'min_max':
            min, max = float(data['mri_vox'].min()), float(data['mri_vox'].max())
            scale = float(self.args_dict.get('scale', 1.0))
            data['mri_vox'] = ((data['mri_vox'] - min) / (max - min)) * scale

        else:
            raise ValueError('{} normalization is not supported'.format(self.norm_type))

        return data


class InvertAffine(object):
    '''
    Invert Affine
    '''
    def __init__(self, affine_data_key):
        super(InvertAffine, self).__init__()

        self.affine_data_key = affine_data_key        

    def __call__(self, data):
        data[self.affine_data_key] = np.linalg.inv(data[self.affine_data_key]).astype(np.float32)
        return data
