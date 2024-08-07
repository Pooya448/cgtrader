import numpy as np
import trimesh
from skimage import measure
import torch
from models.VoxelVAE import VoxelVAE
from pathlib import Path
import matplotlib.pyplot as plt


def crop_voxels(voxels):

    non_empty_voxels = np.argwhere(voxels)
    min_coords = non_empty_voxels.min(axis=0)
    max_coords = non_empty_voxels.max(axis=0)

    cropped_voxels = voxels[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]
    return cropped_voxels


def save_voxel_as_mesh(voxel, file_path):
    vertices, faces, normals, _ = measure.marching_cubes(voxel, level=0.5)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")
