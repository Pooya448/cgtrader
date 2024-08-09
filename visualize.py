import numpy as np
import trimesh
from skimage import measure
import torch
from models.VoxelVAE import VoxelVAE
from pathlib import Path
import matplotlib.pyplot as plt


### Runs marching cubes on a voxel grid and saves the generated mesh
def save_voxel_as_mesh(voxel, file_path):
    vertices, faces, normals, _ = measure.marching_cubes(voxel, level=0.5)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    mesh.export(file_path)
    print(f"Mesh saved to {file_path}")
