import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from render_utils import load_obj_mesh, param_to_tensor, rotate_mesh, \
    pers_get_depth_maps, get_depth_maps, pers_add_lights, add_lights
from tqdm import tqdm
import numpy as np
import pickle
import smplx
import cv2
import torch
from scipy.spatial.transform import Rotation as R_
import argparse
import trimesh

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str,         default='/workspace/code_github/data/',  help='path to mesh data')
parser.add_argument('--data_path', type=str,         default='/workspace/dataset/data/',  help='path to mesh data')

parser.add_argument('--data_name', type=str,         default='2K2K', help='folder name of rendered dataset')
parser.add_argument('--smpl_model_path', type=none_or_str,   default='None', help='path to smplx model')
parser.add_argument('--render_ORTH', type=bool, default=False, help='render orthgonal images')
parser.add_argument('--file', type=str, default='ply', help='obj or ply')

# parser.add_argument('--data_name', type=str,         default='RP', help='folder name of rendered dataset')
# parser.add_argument('--smpl_model_path', type=none_or_str,   default='None', help='path to smplx model')
# parser.add_argument('--file', type=str, default='obj', help='only obj for RP')
# parser.add_argument('--render_ORTH', type=bool, default=False, help='render orthgonal images')


# parser.add_argument('--data_name', type=str,         default='THuman2', help='folder name of rendered dataset')
# parser.add_argument('--smpl_model_path', type=str,   default='/workspace/code_github/render/smpl_related/models', help='path to smplx model')
# parser.add_argument('--file', type=str, default='obj', help='only obj for THuman2')
# parser.add_argument('--render_ORTH', type=bool, default=False, help='render orthgonal images')
args = parser.parse_args()

def make_train_list(data_path, data_name, angle_min_x, angle_max_x, interval_x, axis_x, axis_y):

    os.makedirs(os.path.join(data_path, 'list'), exist_ok=True)
    list_file = os.path.join(data_path, 'list', data_name+'_all.txt')

    list_name_degree = []
    for x in range(angle_min_x, angle_max_x + 1, interval_x):
        list_name_degree.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))

    if data_name=="THuman2":
        data = sorted(os.listdir(os.path.join(data_path, 'obj', data_name, 'data')))
    else:
        data = sorted(os.listdir(os.path.join(data_path, 'obj', data_name)))

    if os.path.isfile(list_file):
        os.remove(list_file)
    with open(list_file, "a") as f:
        for d in data: # ['rp_aaron_posed_013_OBJ']    
            if data_name=="RP":
                item_name = d[:-4] # 'rp_aaron_posed_013'
            elif data_name=="THuman2" or data_name=="2K2K":
                item_name = d
            else:
                raise Exception("Only for RenderPeople, THuman2, and 2K2K dataset.")
            for name_degree in list_name_degree: # ['RP_0_y_-30_x', 'RP_0_y_-20_x', 'RP_0_y_-10_x', 'RP_0_y_0_x', 'RP_0_y_10_x', 'RP_0_y_20_x', 'RP_0_y_30_x']
                line = '/PERS/COLOR/SHADED/{0}/{1}_front.png /PERS/COLOR/NOSHADING/{0}/{1}_front.png /PERS/DEPTH/{0}/{1}_front.png\n'.format(name_degree, item_name)
                f.write(line)
    os.chmod(list_file, 0o777)

def render_mesh(data_path, # '/workspace/code_github/data/'
                data_name, # 'RP', 'THuman2'
                f='ply',
                cnt=0,
                fov=50,
                cam_res=2048,
                angle_min_x=0,
                angle_max_x=0,
                interval_x=5,
                angle_min_y=0,
                angle_max_y=0,
                interval_y=3,
                axis_x='x',
                axis_y='y',
                shad_num=1,
                save_img=True,
                smpl_model_path=None,
                render_orth=False,
                device=torch.device("cuda:0")):
    
    make_train_list(data_path, data_name, angle_min_x, angle_max_x, interval_x, axis_x, axis_y)

    PERS_COLOR_ROOT = os.path.join(data_path, 'PERS', 'COLOR', 'NOSHADING')
    PERS_SHAD_COLOR_ROOT = os.path.join(data_path, 'PERS', 'COLOR', 'SHADED')
    PERS_DEPTH_ROOT = os.path.join(data_path, 'PERS', 'DEPTH') # '/workspace/code/data/PERS/DEPTH/'
    ORTH_COLOR_ROOT = os.path.join(data_path, 'ORTH', 'COLOR', 'NOSHADING')
    ORTH_SHAD_COLOR_ROOT = os.path.join(data_path, 'ORTH', 'COLOR', 'SHADED')
    ORTH_DEPTH_ROOT = os.path.join(data_path, 'ORTH', 'DEPTH')

    os.makedirs(PERS_COLOR_ROOT, exist_ok=True)
    os.makedirs(PERS_SHAD_COLOR_ROOT, exist_ok=True)
    os.makedirs(PERS_DEPTH_ROOT, exist_ok=True)
    if render_orth:
        os.makedirs(ORTH_COLOR_ROOT, exist_ok=True)
        os.makedirs(ORTH_SHAD_COLOR_ROOT, exist_ok=True)
        os.makedirs(ORTH_DEPTH_ROOT, exist_ok=True)

    folder_pers_shad_color = []
    folder_orth_shad_color = []
    folder_pers_color = []
    folder_orth_color = []
    folder_pers_depth = []
    folder_orth_depth = []
    rot_angle_x = []
    rot_angle_y = []

    if smpl_model_path is not None:
        smpl = smplx.create(model_path = smpl_model_path,
                            model_type = 'smplx',
                            gender     = 'male', # 'neutral',
                            num_pca_comps = 12,
                            # use_pca    = True,
                            # use_face_contour = True,
                            ).to(device)

    # for y in range(angle_min_y, angle_max_y + 1, interval_y):
    for x in range(angle_min_x, angle_max_x + 1, interval_x):
        folder_pers_shad_color.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))
        folder_orth_shad_color.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))
        folder_pers_color.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))
        folder_orth_color.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))
        folder_pers_depth.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))
        folder_orth_depth.append('{}_{}_{}_{}_{}'.format(data_name, 0, axis_y, x, axis_x))
        rot_angle_y.append(0)
        rot_angle_x.append(x)

    for k in range(len(folder_pers_shad_color)):
        dir_pers_shad_color = os.path.join(PERS_SHAD_COLOR_ROOT, folder_pers_shad_color[k])
        dir_orth_shad_color = os.path.join(ORTH_SHAD_COLOR_ROOT, folder_orth_shad_color[k])
        dir_pers_color = os.path.join(PERS_COLOR_ROOT, folder_pers_color[k])
        dir_orth_color = os.path.join(ORTH_COLOR_ROOT, folder_orth_color[k])
        dir_pers_depth = os.path.join(PERS_DEPTH_ROOT, folder_pers_depth[k])
        dir_orth_depth = os.path.join(ORTH_DEPTH_ROOT, folder_orth_depth[k])

        if os.path.isdir(dir_pers_shad_color) is False and save_img is True:
            os.mkdir(dir_pers_shad_color)
        if os.path.isdir(dir_pers_color) is False and save_img is True:
            os.mkdir(dir_pers_color)
        if os.path.isdir(dir_pers_depth) is False and save_img is True:
            os.mkdir(dir_pers_depth)
        if render_orth:
            if os.path.isdir(dir_orth_shad_color) is False and save_img is True:
                os.mkdir(dir_orth_shad_color)
            if os.path.isdir(dir_orth_color) is False and save_img is True:
                os.mkdir(dir_orth_color)
            if os.path.isdir(dir_orth_depth) is False and save_img is True:
                os.mkdir(dir_orth_depth)

    if data_name=="THuman2":
        data = sorted(os.listdir(os.path.join(data_path, 'obj', data_name, 'data')))
    else:
        data = sorted(os.listdir(os.path.join(data_path, 'obj', data_name)))

    for d in tqdm(data):                
        if data_name=="RP":
            item_name = d[:-4] # 'rp_aaron_posed_013'
            obj_name = d[:-4]+'_100k.obj'
            obj_path = os.path.join(data_path, 'obj', data_name, d, obj_name)
            if not os.path.exists(obj_path):
                obj_path = os.path.join(data_path, 'obj', data_name, d, obj_name)[:-3]+'OBJ'
            if not os.path.exists(obj_path):
                obj_path =  os.path.join(data_path, 'obj', data_name, d, obj_name)[:-8]+'200k.obj'
            tex_path = os.path.join(data_path, 'obj', data_name, d, 'tex', d[:-3]+'dif_8k.jpg')
            if not os.path.exists(tex_path):
                tex_path = os.path.join(data_path, 'obj', data_name, d, 'tex', d[:-3]+'dif.jpg')
            
            mesh = load_obj_mesh(obj_path, tex_path)  
            if not os.path.exists(obj_path):
                print('ERROR: obj file does not exist!!', obj_path)
                return
            glob_rotation = np.array([0., 0., 0.], dtype=np.float32)
            
        elif data_name=="2K2K":
            item_name = d
            if f=="obj":
                obj_path = os.path.join(data_path, 'obj', data_name, d, d+'.obj')
                tex_path = os.path.join(data_path, 'obj', data_name, d, d+'.png')
                mesh = load_obj_mesh(obj_path, tex_path)
                if not os.path.exists(obj_path):
                    print('ERROR: obj file does not exist!!', obj_path)
                    return
            elif f=="ply":
                ply_path = os.path.join(data_path, 'obj', data_name, d, d+'.ply')
                mesh = trimesh.load(ply_path)
                if not os.path.exists(ply_path):
                    print('ERROR: ply file does not exist!!', ply_path)
                    return
            glob_rotation = np.array([0., 0., 0.], dtype=np.float32)

        elif data_name=="THuman2":
            item_name = d
            obj_name = d+'.obj'
            obj_path  = os.path.join(data_path, 'obj', data_name, 'data', d, obj_name)            
            tex_path  = os.path.join(data_path, 'obj', data_name, 'data', d, 'material0.jpeg')            
            pose_path = os.path.join(data_path, 'obj', data_name, 'smplx', d, 'smplx_param.pkl')
            if not os.path.isfile(pose_path):
                pose_path = None

            mesh = load_obj_mesh(obj_path, tex_path)
            if not os.path.exists(obj_path):
                print('ERROR: obj file does not exist!!', obj_path)
                return

            # SMPLX
            glob_rotation = np.array([0., 0., 0.], dtype=np.float32)
            if pose_path is not None:
                with open(pose_path, 'rb') as smplx_file:
                    smpl_param = pickle.load(smplx_file, encoding='latin1')
                glob_rotation[1] = smpl_param['global_orient'][0][1]

                smpl_param = param_to_tensor(smpl_param, device)

                smpl_mesh = smpl(
                    betas = smpl_param['betas'],
                    expression = smpl_param['expression'],
                    # transl = smpl_param['transl'],
                    global_orient = smpl_param['global_orient'],
                    body_pose = smpl_param['body_pose'],
                    jaw_pose = smpl_param['jaw_pose'],
                    left_hand_pose = smpl_param['left_hand_pose'], # [15,3]
                    right_hand_pose = smpl_param['right_hand_pose'],
                    leye_pose = smpl_param['leye_pose'],
                    reye_pose = smpl_param['reye_pose'],
                    return_verts=True,
                )
                smpl_mesh.vertices *= smpl_param['scale']
                smpl_mesh.vertices += smpl_param['translation']

        cnt += 1

        for p in range(len(rot_angle_x)):
            pers_depth_name = os.path.join(PERS_DEPTH_ROOT, folder_pers_depth[p], item_name)
            pers_img_name = os.path.join(PERS_COLOR_ROOT, folder_pers_color[p], item_name)
            pers_shad_img_name = os.path.join(PERS_SHAD_COLOR_ROOT, folder_pers_shad_color[p], item_name)
            orth_depth_name = os.path.join(ORTH_DEPTH_ROOT, folder_orth_depth[p], item_name)
            orth_img_name = os.path.join(ORTH_COLOR_ROOT, folder_orth_color[p], item_name)
            orth_shad_img_name = os.path.join(ORTH_SHAD_COLOR_ROOT, folder_orth_shad_color[p], item_name)

            pers_color_front_name      = pers_img_name + '_front.png' 
            pers_color_back_name       = pers_img_name + '_back.png'
            pers_depth_front_name      = pers_depth_name + '_front.png'
            pers_depth_back_name       = pers_depth_name + '_back.png' 
            pers_shad_color_front_name = pers_shad_img_name + '_front.png'
            # pers_shad_color_back_name  = pers_shad_img_name + '_back.png'

            orth_color_front_name      = orth_img_name + '_front.png' 
            orth_color_back_name       = orth_img_name + '_back.png'
            orth_depth_front_name      = orth_depth_name + '_front.png'
            orth_depth_back_name       = orth_depth_name + '_back.png' 
            orth_shad_color_front_name = orth_shad_img_name + '_front.png'
            # orth_shad_color_back_name  = orth_shad_img_name + '_back.png'

            vertices = (mesh.vertices - mesh.centroid)
            vertices_np = np.array(vertices)
            val = np.maximum(np.max(vertices_np), np.abs(np.min(vertices_np)))
            vertices /= val * 2.8
            
            # For RenderPeople Dataset
            turn_right =  size =  0
            if d in [
                'rp_wendy_posed_002_OBJ',
                'rp_toshiro_posed_021_OBJ',
                'rp_scott_posed_037_OBJ',
                'rp_pamela_posed_012_OBJ',
                'rp_oliver_posed_029_OBJ',
                'rp_noah_posed_011_OBJ',
                'rp_mira_posed_001_OBJ',
                'rp_luke_posed_008_OBJ',
                'rp_luke_posed_007_OBJ',
                'rp_jessica_posed_006_OBJ',
                'rp_helen_posed_038_OBJ',
                'rp_eve_posed_003_OBJ',
                'rp_eric_posed_036_OBJ',
                'rp_eric_posed_007_OBJ',
                'rp_emma_posed_025_OBJ',
                'rp_dennis_posed_008_OBJ',
                'rp_chloe_posed_004_OBJ',
                'rp_anna_posed_001_OBJ',
                'rp_andrew_posed_004_OBJ',
                'rp_maya_posed_027_OBJ',
                'rp_petra_posed_006_OBJ',
            ]:
                turn_right = 1
            if d in [
                'rp_michael_posed_019_OBJ',
                'rp_mei_posed_007_OBJ',
                'rp_joel_posed_006_OBJ',
                'rp_ethan_posed_003_OBJ',
                'rp_elena_posed_013_OBJ',
                'rp_dennis_posed_001_OBJ',
                'rp_christine_posed_017_OBJ',
                'rp_beatrice_posed_034_OBJ',
                'rp_andrew_posed_007_OBJ',
            ]:
                size = 1

            if turn_right:
                vertices = vertices
                rotation_axis = np.array([0, 1, 0])
                rotation_degrees = -90
                rotation_radians = np.radians(rotation_degrees)
                rotation_vector = rotation_radians * rotation_axis
                rotation = R_.from_rotvec(rotation_vector)
                rot_max = rotation.as_matrix()
                vertices = np.einsum('ij,Bj ->Bi', rot_max, vertices)
            if size:
                vertices = vertices * 0.9


            angle_y = rot_angle_y[p]
            angle_x = rot_angle_x[p]

            mesh_local = rotate_mesh(vertices, angle_x, glob_rotation, mesh.faces, mesh.visual.vertex_colors, axis=axis_x)
            scene = mesh_local.scene()
            scene.camera.resolution = [cam_res, cam_res]

            pers_color_front, pers_depth_front, pers_depth_back, pers_color_back = \
                pers_get_depth_maps(mesh_local, scene, cam_res, fov, item_name=item_name)            
            if render_orth:
                orth_depth_front, orth_depth_back, orth_color_front, orth_color_back = \
                    get_depth_maps(mesh_local, scene, cam_res, fov, 'front', item_name=item_name)

            cv2.imwrite(pers_color_front_name, (pers_color_front * 255).astype(np.int64))
            cv2.imwrite(pers_color_back_name, (pers_color_back * 255).astype(np.int64))
            if render_orth:
                cv2.imwrite(orth_color_front_name, (orth_color_front * 255).astype(np.int64))
                cv2.imwrite(orth_color_back_name, (orth_color_back * 255).astype(np.int64))

            cv2.imwrite(pers_depth_front_name, (pers_depth_front * 32.0).astype(np.uint16))
            cv2.imwrite(pers_depth_back_name, (pers_depth_back * 32.0).astype(np.uint16))
            if render_orth:
                cv2.imwrite(orth_depth_front_name, (orth_depth_front * 32.0).astype(np.uint16))
                cv2.imwrite(orth_depth_back_name, (orth_depth_back * 32.0).astype(np.uint16))

            # orthogonal-projection with shading
            for sh in range(shad_num):
                pers_shad_color_front = pers_add_lights(mesh_local, cam_res, rot_angle_x, scene, fov)
                pers_shad_color_front[pers_depth_front == 0, :] = [0, 0, 0]
                cv2.imwrite(pers_shad_color_front_name, (pers_shad_color_front))
                if render_orth:
                    orth_shad_color_front = add_lights(mesh_local, cam_res, rot_angle_x, scene, fov)
                    orth_shad_color_front[orth_depth_front == 0, :] = [0, 0, 0]
                    cv2.imwrite(orth_shad_color_front_name, (orth_shad_color_front))
        print('')



if __name__ == '__main__':
    
    folders = sorted(os.listdir(args.data_path))

    # folder_num = 0
    cnt_x = 0
    cnt_y = 0
    cnt_z = 0
    os.makedirs(os.path.join(args.data_path, 'PERS', 'COLOR'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'PERS', 'DEPTH'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'ORTH', 'COLOR'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'ORTH', 'DEPTH'), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    render_mesh(args.data_path, 
                args.data_name,
                f=args.file,
                cnt=cnt_y,
                fov=50,
                cam_res=2048,
                # cam_res=256,
                angle_min_x=-30,
                angle_max_x=30,
                interval_x=10,
                angle_min_y=0,
                angle_max_y=0,
                interval_y=0,
                axis_x='x',
                axis_y='y',
                shad_num=1, 
                smpl_model_path=args.smpl_model_path,
                render_orth=args.render_ORTH,
                device=device)