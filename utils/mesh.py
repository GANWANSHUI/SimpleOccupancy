"""
@file   mesh.py
@author WANSHUI GAN adapted from Jianfei Guo, Shanghai AI Lab
@brief  Mesh utilities.
"""

__all__ = ['extract_simpleocc_mesh']

import os
import sys
import time
import skimage
import skimage.measure
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from typing import Callable, List, Union

import plyfile
import pdb

import torch


def extract_simpleocc_mesh(
    query_sdf_fn: Callable[[torch.Tensor], torch.Tensor], 
    query_color_fn: Callable[[torch.Tensor], torch.Tensor]=None, Voxel_feat = None, 
    filepath: str='/home/wsgan/project/bev/S3DO/surface.ply', 
    level: float=0.0, N: Union[int, List[int]]=512, chunk: int=16 * 1024, 
    include_color=False, show_progress=True, 
    bmin: Union[List, np.ndarray]=[-1., -1., -1.], bmax: Union[List, np.ndarray]=[1., 1., 1.],
    offset: np.ndarray=None, scale: np.ndarray=None, transform: np.ndarray=None, 
    device: torch.device=torch.device('cuda')):
    """ Extract mesh via running Marching Cubes

    Args:
        query_sdf_fn (Callable[[torch.Tensor], torch.Tensor]): Query function with x input and SDF output
        query_color_fn (Callable[[torch.Tensor], torch.Tensor], optional): Query function with x input and RGB output. Defaults to None.
        filepath (str, optional): Filepath to save the extracted mesh. Defaults to './surface.ply'.
        level (float, optional): Levelset to extract. Defaults to 0.0.
        N (Union[int, List[int]], optional): Resolution of the AABB.
            If a single integer is given, it represents the resolution of the shortest edge of the AABB. Defaults to 512.
        chunk (int, optional): Chunkify function querying. Defaults to 16*1024.
        include_color (bool, optional): Whether to include colors to extracted meshes. 
            Setting to True will require giving `query_color_fn`. Defaults to False.
        show_progress (bool, optional): Whether to show querying progress. Defaults to True.
        bmin (Union[List, np.ndarray], optional): Minimum boundary. Defaults to [-1., -1., -1.].
        bmax (Union[List, np.ndarray], optional): Maximum boundary. Defaults to [1., 1., 1.].
        scale (np.ndarray, optional): Optional scale applied to mesh vertices (`new_vert` = `old_vert` * `scale`). Defaults to None.
        offset (np.ndarray, optional): Optional offset applied to mesh vertices (`new_vert` = `old_vert` - `offset`). Defaults to None.
        transform (np.ndarray, optional): Optional transform applied to mesh vertices (`new_vert` = `transform` @ `old_vert`). Defaults to None.
        device (torch.device, optional): Given torch.device to gather query output. Defaults to torch.device('cuda').

    """
    
    start_time = time.time()

    
    bmin = np.array(bmin.to('cpu'))
    bmax = np.array(bmax.to('cpu'))
    bmax[2] = bmax[2] - 1

    # pdb.set_trace()

    volume_size: np.ndarray = bmax - bmin

    if isinstance(N, int) or len(N) == 1:
        N = (volume_size / volume_size.min() * N).astype(np.int32)
    else:
        N = np.array(N)


    print(f"Extract mesh on {N[0]}x{N[1]}x{N[2]} grid.")
    
    grid_spacing: np.ndarray = volume_size / (N - 1)
    
    if scale is not None:
        scale = np.array(scale, dtype=np.float32)
    if offset is not None:
        offset = np.array(offset, dtype=np.float32)
    if transform is not None:
        transform = np.array(transform, dtype=np.float32)

    # pdb.set_trace()
    # array([2632, 1817,  697],
    xyz = torch.stack(torch.meshgrid([torch.linspace(bmin[0], bmax[0], N[0]), torch.linspace(bmin[1], bmax[1], N[1]), torch.linspace(bmin[2], bmax[2], N[2]),], indexing="ij"), -1).reshape(-1, 3)


    def batchify_surf(query_fn, inputs: torch.Tensor, chunk: int=chunk) -> np.ndarray:

        out = np.empty((inputs.shape[0],))

        # pdb.set_trace()

        for i in trange(0, inputs.shape[0], chunk, disable=not show_progress,
                        desc="Querying surface", leave=False):
            
            x = inputs[i:i + chunk].to(device)
            out_i = query_fn(x, Voxel_feat)
            out[i:i + chunk] = out_i.detach().cpu().numpy()

        return out


    def batchify_color(query_fn, inputs: np.ndarray, normals: np.ndarray, chunk: int=chunk) -> np.ndarray:
        colors = np.empty((inputs.shape[0], 3), dtype=np.uint8)
        for i in trange(0, inputs.shape[0], chunk, disable=not show_progress,
                        desc="Querying color", leave=False):
            x = torch.tensor(inputs[i:i+chunk], device=device)
            v = torch.tensor(normals[i:i+chunk].copy(), device=device)
            out_i = query_fn(x, v)
            colors[i:i+chunk] = (out_i * 255.).to(torch.uint8).cpu().numpy()
        return colors



    out = batchify_surf(query_sdf_fn, xyz)

    print("Running marching cubes...")

    verts, faces, normals, _ = skimage.measure.marching_cubes(
        out.reshape(*N), level=level, spacing=grid_spacing)
    
    verts += bmin
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    print(f"Marching cubes done. Extracted mesh contains {num_verts} vertices and {num_faces} faces.")

    if include_color:
        colors = batchify_color(query_color_fn, verts.astype(np.float32), normals.astype(np.float32))

    # Apply additional offset and scale
    if scale is not None:
        verts = verts * scale
    if offset is not None:
        verts = verts - offset
    if transform is not None:
        verts = verts @ transform[:3, :3].T + transform[:3, 3]

    

    if include_color:
        verts_tuple = np.zeros((num_verts,), dtype=[(
            "x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(verts[i, :].tolist() + colors[i].tolist())
    else:
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(verts[i, :])

    # face_list
    faces_list = []
    for i in range(0, num_faces):
        faces_list.append(((faces[i, :].tolist(),)))

    faces_tuple = np.array(faces_list, dtype=[("vertex_indices", "i4", (3,))])


    # write
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
    ply_data = plyfile.PlyData([el_verts, el_faces])


    # pdb.set_trace()
    return ply_data

    # ply_data.write('/home/wsgan/project/bev/S3DO/surface3.ply') 
    # ply_data.write('/home/wsgan/project/bev/S3DO/surface_mlp_sdf_4.ply') 

    print(f"=> Saving mesh to {str(filepath)}")
    ply_data.write(filepath)

    print(f"converting to ply format and writing to file took {time.time() - start_time} s")
    
