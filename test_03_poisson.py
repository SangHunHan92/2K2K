import argparse
import os
import pymeshlab
import trimesh
from tqdm import tqdm
import numpy as np
import cv2

# pymeshlab version 2022.2 is most fast

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./result', help='path to save folder')
parser.add_argument('--proj_name', type=str, default='test3', help='path to save folder')
args = parser.parse_args()

def main(save_path, proj_name):

    pc_path = os.path.join(save_path, proj_name, 'output_plys')
    obj_path = os.path.join(save_path, proj_name, 'output_objs')
    # pc_path = os.path.join(save_path, proj_name, 'output_plys_c')
    # obj_path = os.path.join(save_path, proj_name, 'output_objs_c')
    os.makedirs(obj_path, exist_ok=True)
    os.chmod(obj_path, 0o777)

    pc_list = os.listdir(pc_path)
    pc_list.sort()

    for pc_name in tqdm(pc_list):
        print(pc_name)
        name = pc_name[:-4]
        pc = trimesh.load(os.path.join(pc_path, pc_name))
        xyz = pc.vertices
        rgba = pc.visual.vertex_colors.astype(float)/255.
        ms = pymeshlab.Mesh(vertex_matrix = xyz, v_color_matrix=rgba)
        mss = pymeshlab.MeshSet()
        mss.add_mesh(ms)
        mss.apply_filter('compute_normals_for_point_sets', k=70)
        # mss.apply_filter('compute_normal_for_point_clouds', k=70))
        mss.apply_filter('surface_reconstruction_screened_poisson', depth=10)
        # mss.apply_filter('generate_surface_reconstruction_screened_poisson', depth=10)        
        # mss.save_current_mesh(os.path.join(obj_path, name+'.obj'))        
        
        m  = mss.mesh(1)
        v  = m.vertex_matrix()
        vn = m.vertex_normal_matrix()
        vc = m.vertex_color_matrix()
        f  = m.face_matrix()        
        x_min, y_min, z_min = np.argmin(v, axis=0)
        x_max, y_max, z_max = np.argmax(v, axis=0)
        if sum([vn[x_min][0]>0, vn[y_min][1]>0, vn[z_min][2]>0, vn[x_max][0]<0, vn[y_max][1]<0, vn[z_max][2]<0]) > 3:
            vn *= -1.
        ex_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=vn, vertex_colors=vc)
        save_path = os.path.join(obj_path, name+'.obj')
        ex_mesh.export(save_path)

if __name__ == '__main__':    
    main(args.save_path, args.proj_name)
    