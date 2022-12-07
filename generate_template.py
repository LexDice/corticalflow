import scipy.spatial
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pyvista as pv
import numpy as np
import pandas as pd
import os

surface_name = 'R.white'

df = pd.read_csv('./resources/hcp_split.csv', dtype={'subject': object, 'split': object})
df_train = df[df['split'] == 'train']
train_subjects = list(df_train.subject.values)

reader = vtk.vtkPolyDataReader()
for i, subject_id in enumerate(train_subjects):
    surf_fn = subject_id + '.' + surface_name + '.surf.vtk'
    mesh_path = os.path.join('/mnt/yaplab/data/junjiez/HCP_TRT', subject_id, 'T1w/Native', surf_fn)
    reader.SetFileName(mesh_path)
    reader.Update()
    cur_vertices = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    if i == 0:
        surf_verts = cur_vertices
    else:
        surf_verts = np.append(surf_verts, cur_vertices, axis=0)


all_points = pv.wrap(surf_verts)
# my_surf = all_points.reconstruct_surface()
# hull = scipy.spatial.ConvexHull(surf_verts)

# tri = scipy.spatial.Delaunay(surf_verts[hull.vertices])
# cloud = pv.PolyData(surf_verts[hull.vertices])
# surf = cloud.delaunay_3d()

pset = pv.PolyData(surf_verts)
pset.save('all_points.ply')
# pset = pv.PointSet(surf_verts[hull.vertices])
# pset.save('hull_points.xyz')
# faces = np.insert(hull.simplices, 0, 3, axis=1)
# faces = faces.flatten()

# aa = pv.PolyData(surf_verts[hull.vertices], faces)
# aa.save('aa.vtk')
print('aa')