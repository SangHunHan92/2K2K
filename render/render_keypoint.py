from json import load
import matplotlib
matplotlib.use('Agg')
import os
import math
import numpy as np
import cv2
import torch
from tqdm import tqdm
import random
import trimesh
import json
from scipy.spatial.transform import Rotation as R_
import pickle
import argparse

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

parser.add_argument('--data_name', type=str,         default='RP', help='folder name of rendered dataset')
parser.add_argument('--smpl_model_path', type=none_or_str,   default='None', help='path to smplx model')
parser.add_argument('--render_ORTH', type=bool, default=False, help='path to smplx model')
# parser.add_argument('--data_name', type=str,         default='THuman2', help='folder name of rendered dataset')
# parser.add_argument('--smpl_model_path', type=str,   default='/workspace/code_github/render/smpl_related/models', help='path to smplx model')
# parser.add_argument('--render_ORTH', type=bool, default=False, help='path to smplx model')
args = parser.parse_args()


def keypoint_projection(data_path, data_name, fov=50, cam_res=2048,                
                angle_min_x=-30, angle_max_x=30, interval_x=10,
                angle_min_y=0,   angle_max_y=0,  interval_y=0,
                smpl_model_path=None,
                device=torch.device("cuda:0")):
        
    if data_name=="THuman2":
        mesh_path    = os.path.join(data_path, 'obj', data_name, 'data')         
    else:
        mesh_path    = os.path.join(data_path, 'obj', data_name) # mesh path '/workspace/dataset/RenderPeople/'
    joint_path       = os.path.join(data_path, 'Joint3D', data_name) # '/workspace/dataset/RP_2048/Joint3D/'
    proj_result_path = os.path.join(data_path, 'PERS', 'keypoint') # '/workspace/dataset/RP_2048/PERS/keypoint/'
    pers_path        = os.path.join(data_path, 'PERS', 'COLOR', 'NOSHADING') # '/workspace/dataset/RP_2048/PERS/COLOR/NOSHADING/'
    sample_path      = os.path.join(data_path, 'PERS', 'keypoint_sample') #'/workspace/dataset/RP_2048/PERS/keypoint_sample/'
    x_angles         = list(range(angle_min_x, angle_max_x + 1, interval_x)) # [-30, -20, -10, 0, 10, 20, 30]
    
    if not os.path.isdir(proj_result_path):
        os.makedirs(proj_result_path, exist_ok=True)
    if not os.path.isdir(sample_path):
        os.makedirs(sample_path, exist_ok=True)

    if data_name == 'THuman2': 
        import smplx        
        smpl = smplx.create(model_path = smpl_model_path,
                            model_type = 'smplx',
                            gender     = 'male', #'neutral',
                            num_pca_comps = 12,
                            # use_pca    = True,
                            # use_face_contour = True,
                            ).to(device)

    data = os.listdir(mesh_path) 
    data.sort()

    if not os.path.isdir(proj_result_path):
        os.makedirs(proj_result_path)

    for angle in x_angles:
        if not os.path.isdir( os.path.join(proj_result_path, data_name+'_0_y_{}_x'.format(angle)) ):
            os.makedirs( os.path.join(proj_result_path, data_name+'_0_y_{}_x'.format(angle)) )
        if not os.path.isdir( os.path.join(sample_path, data_name+'_0_y_{}_x'.format(angle)) ):
            os.makedirs( os.path.join(sample_path, data_name+'_0_y_{}_x'.format(angle)) )

    # K, R, t
    K = np.identity(3)
    K[0,0] = K[1,1] = cam_res / (np.tan(np.radians(fov) / 2.0) * 2) # 2195.9750866
    K[0,2] = K[1,2] = cam_res / 2

    RT = np.zeros((3, 4))
    C  = np.array([0, 0, 1])
    R  = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1],])
    T  = -np.matmul(R, C)
    RT[:, :3] = R
    RT[:,  3] = T

    KRT = np.matmul(K, RT)

    for d in tqdm(data):
        if data_name == "RP":
            obj_name = d[:-3]+'100k.obj'
            obj_path = os.path.join(mesh_path, d, obj_name)
            if not os.path.exists(obj_path):
                obj_path = os.path.join(mesh_path, d, obj_name)[:-3]+'OBJ'
            if not os.path.exists(obj_path):
                obj_path =  os.path.join(mesh_path, d, obj_name)[:-8]+'200k.obj'
            key_path = os.path.join(joint_path, d[:-4]+'.json')
            if not os.path.isfile(key_path):
                continue
        elif data_name == 'THuman2':
            obj_name = d+'.obj'
            obj_path = os.path.join(mesh_path, d, obj_name)
            key_path = os.path.join(joint_path, d+'.json')
            smpl_path = os.path.join(mesh_path[:-4], 'smplx', d, 'smplx_param.pkl')

        mesh = trimesh.load(obj_path)
        vertices = (mesh.vertices - mesh.centroid)
        vertices_np = np.array(vertices)
        val = np.maximum(np.max(vertices_np), np.abs(np.min(vertices_np)))
        vertices /= val * 2.8

        with open(key_path, "r") as j:
            keypoint = json.load(j)
        for k, v in keypoint.items():
            keypoint[k] = np.array(v)

        filter_indices = np.concatenate(
                [ np.array(range(23)), np.array([96, 100, 104, 108, 117, 121, 125, 129]) ]
                )        

        turn_right = size = 0
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
        
        # if turn_right == 0 and size == 0:
        #     continue

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
        
        filter_indices = np.concatenate(
                [ np.array(range(23)), np.array([96, 100, 104, 108, 117, 121, 125, 129]) ]
                )
        visible_ = keypoint['visible'][filter_indices]

        # smplx finger
        if data_name == 'THuman2':
            with open(smpl_path, 'rb') as smplx_file:
                smpl_param = pickle.load(smplx_file, encoding='latin1')
            smpl_param = param_to_tensor(smpl_param, device)

            if np.where(visible_ == 0)[0].size > 0:
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

                # scale, translation, global_orient
                smpl_joint = smpl_mesh.joints
                smpl_joint *= smpl_param['scale']
                smpl_joint += smpl_param['translation']        
            
            else: 
                smpl_joint = None
        else:
            smpl_joint = None

        keypoint = refine(d, keypoint, smpl_joint)
        
        filter_indices = np.concatenate(
                [ np.array(range(23)), np.array([96, 100, 104, 108, 117, 121, 125, 129]) ]
                )
        pc = keypoint['joint3d'][filter_indices]
        visible = keypoint['visible'][filter_indices]

        pc = pc - mesh.centroid
        pc /= val * 2.8

        if turn_right:
            pc = pc
            rotation_axis = np.array([0, 1, 0])
            rotation_degrees = -90
            rotation_radians = np.radians(rotation_degrees)
            rotation_vector = rotation_radians * rotation_axis
            rotation = R_.from_rotvec(rotation_vector)
            rot_max = rotation.as_matrix()
            pc = np.einsum('ij,Bj ->Bi', rot_max, pc)
        if size:
            pc = pc * 0.9

        if data_name == 'THuman2':
            glob_rotation = np.array([0., 0., 0.], dtype=np.float32)
            glob_rotation[1] = smpl_param['global_orient'][0][1]

            pc = pc
            rotation_axis = np.array([0, 1, 0])
            rotation_degrees = 0        
            rotation_radians = np.radians(rotation_degrees)
            rotation_vector = rotation_radians * rotation_axis
            rotation_vector -= glob_rotation
            rotation = R_.from_rotvec(rotation_vector)
            rot_max = rotation.as_matrix()
            pc = np.einsum('ij,Bj ->Bi', rot_max, pc)

        glob_rotation = np.array([0., 0., 0.], dtype=np.float32)

        for angle in x_angles:
            # angle = 0
            cam_vert = rotate_pc(vertices, angle, glob_rotation)
            cam_vert *= 1.037
            cam_vert = np.append(cam_vert, np.ones((cam_vert.shape[0], 1)), axis=1)
            cam_vert = np.einsum('ij,Bj ->Bi', KRT, cam_vert)
            cam_vert = np.divide(cam_vert, cam_vert[:,2:])

            img_path = os.path.join(pers_path, data_name+'_0_y_{}_x'.format(angle), d[:-3]+'front.png')

            cam_pc = rotate_pc(pc, angle, glob_rotation)
            cam_pc *= 1.037
            cam_pc = np.append(cam_pc, np.ones((cam_pc.shape[0], 1)), axis=1)
            cam_pc = np.einsum('ij,Bj ->Bi', KRT, cam_pc)
            cam_pc = np.divide(cam_pc, cam_pc[:,2:])

            cam_pc[visible==0] = 0

            if data_name == "RP":
                save_path = os.path.join(proj_result_path, data_name+'_0_y_{}_x'.format(angle), d[:-4]+'_front.npy')
            elif data_name == "THuman2":
                save_path = os.path.join(proj_result_path, data_name+'_0_y_{}_x'.format(angle), d+'_front.npy')
            np.save(save_path, cam_pc)
            os.chmod(save_path, 0o777)
        
            # test
            if data_name == "RP":
                img_path = os.path.join(pers_path, data_name+'_0_y_{}_x'.format(angle), d[:-3]+'front.png')
            elif data_name == "THuman2":
                img_path = os.path.join(pers_path, data_name+'_0_y_{}_x'.format(angle), d+'_front.png')
            img_cv = cv2.imread(img_path)
            for k in range(cam_pc.shape[0]):
                x = int(cam_pc[k, 0])
                y = int(cam_pc[k, 1])
                if k in [23, 24, 25, 26]:
                    img_cv = cv2.circle(img_cv, (x, y), radius=0, color=(255, 0, 0), thickness=int(3))
                elif k in [27, 28, 29, 30]:
                    img_cv = cv2.circle(img_cv, (x, y), radius=0, color=(0, 255, 0), thickness=int(3))
                else:
                    img_cv = cv2.circle(img_cv, (x, y), radius=0, color=(0, 0, 255), thickness=int(3))
            if data_name == "RP":
                sam_path = os.path.join(sample_path, data_name+'_0_y_{}_x'.format(angle), d[:-4]+'.png')
            elif data_name == "THuman2":
                sam_path = os.path.join(sample_path, data_name+'_0_y_{}_x'.format(angle), d+'.png')
            cv2.imwrite(sam_path, img_cv) 

def rotate_pc(vertices, angle, global_angle, axis='x'):
    vertices_re = (np.zeros_like(vertices))
    if axis == 'y':  # pitch
        rotation_axis = np.array([1, 0, 0])
    elif axis == 'x':  # yaw
        rotation_axis = np.array([0, 1, 0])
    elif axis == 'z':  # roll
        rotation_axis = np.array([0, 0, 1])
    else:  # default is x (yaw)
        rotation_axis = np.array([0, 1, 0])

    for i in range(vertices.shape[0]):
        vec = vertices[i, :]
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)

        rotation_vector = rotation_radians * rotation_axis
        rotation_vector -= global_angle
        rotation = R_.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(vec)
        vertices_re[i, :] = rotated_vec

    return vertices_re

def param_to_tensor(param, device):
    for key in param.keys():
        try :
            # whatever numpy or tensor, working well
            param[key] = torch.reshape(torch.as_tensor(param[key], device=device), (1, -1))
            param[key].requires_grad=True
        except :
            continue
    return param

def refine(d, keypoint, smpl_joint=None):
    # [96, 100, 104, 108, 117, 121, 125, 129]
    if d[:-4] == 'rp_alvin_posed_016':
        keypoint['joint3d'][8] = np.array([1.0385541517671586,143.8048885415659,20.073939913489461])
        keypoint['joint3d'][10] = np.array([24.25403167, 151.65561945,  6.71324646])
        keypoint['joint3d'][96] = keypoint['joint3d'][100]
        keypoint['joint3d'][117]= np.array([35.25403167, 151.65561945,  6.71324646])
        keypoint['joint3d'][121]= np.array([35.25403167, 151.65561945,  6.71324646])
        keypoint['joint3d'][125]= np.array([35.25403167, 151.65561945,  6.71324646])
        keypoint['joint3d'][129]= np.array([35.25403167, 151.65561945,  6.71324646])
        keypoint['visible'][[8, 10, 96, 117, 121, 125, 129]] = 1
    elif d[:-4] == 'rp_felice_posed_004':
        keypoint['joint3d'][15] = np.array([4.,  3.,   4.])
        keypoint['joint3d'][19] = np.array([4.,  3.,   4.])
        keypoint['joint3d'][16] = np.array([-13., 3.,   4.])
        keypoint['joint3d'][22] = np.array([-13.,  3.,   4.])
        keypoint['joint3d'][17] = np.array([4.,  3.,   20.])
        keypoint['joint3d'][18] = np.array([4.,  3.,   20.])
        keypoint['joint3d'][20] = np.array([-13.,  3.,   20.])
        keypoint['joint3d'][21] = np.array([-13.,  3.,   20.])
        keypoint['visible'][[15, 19, 16, 22, 17, 18, 20, 21]] = 1
    elif d[:-4] == 'rp_holly_posed_010':
        keypoint['joint3d'][7] = np.array([12.,  150.,   -0.8])
        keypoint['joint3d'][104] = keypoint['joint3d'][108]
        keypoint['joint3d'][117] = keypoint['joint3d'][121]
        keypoint['visible'][[7, 104, 117]] = 1
    elif d[:-4] == 'rp_joyce_posed_021':
        keypoint['joint3d'][8] = np.array([14., 149.,  4.])
        keypoint['joint3d'][10] = np.array([34., 155.,  -11.])
        keypoint['joint3d'][96]  = np.array([16., 149.,  6.])
        keypoint['joint3d'][100] = np.array([16., 149.,  6.])
        keypoint['joint3d'][104] = np.array([16., 149.,  6.])
        keypoint['joint3d'][108] = np.array([16., 149.,  6.])
        keypoint['joint3d'][117] = np.array([44., 155.,  -16.])
        keypoint['joint3d'][121] = np.array([44., 155.,  -16.])
        keypoint['joint3d'][125] = np.array([44., 155.,  -16.])
        keypoint['joint3d'][129] = np.array([44., 155.,  -16.])
        keypoint['visible'][[8, 10, 96, 100, 104, 108, 117, 121, 125, 129]] = 1
    elif d[:-4] == 'rp_tenzin_posed_001':
        keypoint['joint3d'][0] = np.array([14., 149.,  4.])
        keypoint['joint3d'][1] = np.array([14., 149.,  4.])
        keypoint['joint3d'][2] = np.array([14., 149.,  4.])
        keypoint['joint3d'][3] = np.array([14., 149.,  4.])
        keypoint['joint3d'][4] = np.array([14., 149.,  4.])
        keypoint['joint3d'][8] = np.array([14., 149.,  4.])
        keypoint['joint3d'][20] = np.array([14., 149.,  4.])
        keypoint['joint3d'][21] = np.array([14., 149.,  4.])
        keypoint['joint3d'][22] = np.array([14., 149.,  4.])
        keypoint['visible'][[0, 1, 2, 3, 4, 8, 20, 21, 22]] = 1
    # [96, 100, 104, 108, 117, 121, 125, 129]
    elif d == '0016':
        keypoint['joint3d'][16] = np.array([-0.10702853,  0.01278133,  0.00621173])
        keypoint['joint3d'][19] = keypoint['joint3d'][17]
        keypoint['joint3d'][20] = keypoint['joint3d'][117]
        keypoint['joint3d'][21] = keypoint['joint3d'][117]
        keypoint['joint3d'][22] = np.array([-0.10702853,  0.01278133,  0.00621173])
        keypoint['visible'][[16, 19, 20, 21, 22]] = 1
        pass
    elif d == '0022':
        keypoint['joint3d'][3] = np.array([0.04409288, 0.21141951, 0.002662  ])
        keypoint['joint3d'][22] = keypoint['joint3d'][16]
        keypoint['joint3d'][96] = keypoint['joint3d'][1]
        keypoint['joint3d'][100] = keypoint['joint3d'][1]
        keypoint['joint3d'][104] = keypoint['joint3d'][1]
        keypoint['joint3d'][108] = keypoint['joint3d'][1]
        keypoint['visible'][[3, 22, 96, 100, 104, 108]] = 1
    elif d == '0040':
        keypoint['joint3d'][9] = np.array([-0.16802483, -0.25595808,  0.03412504])
        keypoint['joint3d'][21] = keypoint['joint3d'][20]
        keypoint['joint3d'][22] = keypoint['joint3d'][16]
        keypoint['joint3d'][96] = np.array([-0.2410311 , -0.27075141,  -0.01])
        keypoint['joint3d'][100] = np.array([-0.2410311 , -0.27075141,  -0.01])
        keypoint['joint3d'][104] = np.array([-0.2410311 , -0.27075141,  -0.01])
        keypoint['joint3d'][108] = np.array([-0.2410311 , -0.27075141,  -0.01])
        keypoint['visible'][[9, 21, 22, 96, 100, 104, 108]] = 1
    elif d == '0058':
        # 117, 121, 125, 129
        keypoint['joint3d'][96]  = smpl_joint[0, 67, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        keypoint['joint3d'][100] = smpl_joint[0, 68, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        keypoint['joint3d'][104] = smpl_joint[0, 69, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        keypoint['joint3d'][108] = smpl_joint[0, 70, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        keypoint['joint3d'][117] = np.array([ -0.04098062,  0.3442096, -0.04529726]) 
        keypoint['joint3d'][121] = np.array([ -0.04098062,  0.3442096, -0.04529726]) 
        keypoint['joint3d'][125] = np.array([ -0.04098062,  0.3442096, -0.04529726]) 
        keypoint['joint3d'][129] = np.array([ -0.04098062,  0.3442096, -0.04529726])
        keypoint['visible'][[96, 100, 104, 108, 117, 121, 125, 129]] = 1
    elif d == '0061':
        keypoint['joint3d'][16] = keypoint['joint3d'][10] + np.array([0.03, -0.02, 0])
        keypoint['joint3d'][20] = keypoint['joint3d'][16] + np.array([0.02, -0.02, -0.05 ])
        keypoint['joint3d'][21] = keypoint['joint3d'][16] + np.array([0.02, -0.02, -0.05 ])
        keypoint['joint3d'][22] = keypoint['joint3d'][16]
        keypoint['visible'][[16, 20, 21, 22]] = 1
    elif d == '0317':
        # keypoint['joint3d'][96]  = smpl_joint[0, 67, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        # keypoint['joint3d'][100] = smpl_joint[0, 68, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        # keypoint['joint3d'][104] = smpl_joint[0, 69, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        # keypoint['joint3d'][108] = smpl_joint[0, 70, :].cpu().detach().numpy() + np.array([0, -0.028, 0])
        # keypoint['joint3d'][117] = np.array([ -0.04098062,  0.3442096, -0.04529726]) 
        # keypoint['joint3d'][121] = np.array([ -0.04098062,  0.3442096, -0.04529726]) 
        # keypoint['joint3d'][125] = np.array([ -0.04098062,  0.3442096, -0.04529726]) 
        # keypoint['joint3d'][129] = np.array([ -0.04098062,  0.3442096, -0.04529726])
        keypoint['visible'][[96, 100, 104, 108, 117, 121, 125, 129]] = 0
    
    if keypoint['visible'][19] == 0:
        if keypoint['visible'][15] == 1:
            keypoint['joint3d'][19] = keypoint['joint3d'][15]
            keypoint['visible'][19] = 1
    if keypoint['visible'][17] == 0 or keypoint['visible'][18] == 0:
        if keypoint['visible'][17] == 0 and keypoint['visible'][18] == 1:
            keypoint['joint3d'][17] = keypoint['joint3d'][18]
            keypoint['visible'][17] = 1
        elif keypoint['visible'][17] == 1 and keypoint['visible'][18] == 0:
            keypoint['joint3d'][18] = keypoint['joint3d'][17]
            keypoint['visible'][18] = 1
        else: # both not visible
            pass
            
    if keypoint['visible'][22] == 0:
        if keypoint['visible'][16] == 1:
            keypoint['joint3d'][22] = keypoint['joint3d'][16]
            keypoint['visible'][22] = 1
    if keypoint['visible'][20] == 0 or keypoint['visible'][21] == 0:
        if keypoint['visible'][20] == 0 and keypoint['visible'][21] == 1:
            keypoint['joint3d'][20] = keypoint['joint3d'][21]
            keypoint['visible'][20] = 1
        elif keypoint['visible'][20] == 1 and keypoint['visible'][21] == 0:
            keypoint['joint3d'][21] = keypoint['joint3d'][20]
            keypoint['visible'][21] = 1
        else: # both not visible
            pass
    if smpl_joint is not None:
        # joints idx
        # https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py

        if keypoint['visible'][0] == 0:
            keypoint['joint3d'][0] = smpl_joint[0, 55, :].cpu().detach().numpy()
        if keypoint['visible'][1] == 0:
            keypoint['joint3d'][1] = smpl_joint[0, 57, :].cpu().detach().numpy()
        if keypoint['visible'][2] == 0:
            keypoint['joint3d'][2] = smpl_joint[0, 56, :].cpu().detach().numpy()

        # ear        
        if keypoint['visible'][3] == 0:
            keypoint['joint3d'][3] = smpl_joint[0, 59, :].cpu().detach().numpy()
        if keypoint['visible'][4] == 0:
            keypoint['joint3d'][4] = smpl_joint[0, 58, :].cpu().detach().numpy()
        
        if keypoint['visible'][5] == 0:
            keypoint['joint3d'][5] = smpl_joint[0, 16, :].cpu().detach().numpy()
        if keypoint['visible'][6] == 0:
            keypoint['joint3d'][6] = smpl_joint[0, 17, :].cpu().detach().numpy()
        if keypoint['visible'][7] == 0:
            keypoint['joint3d'][7] = smpl_joint[0, 18, :].cpu().detach().numpy()
        if keypoint['visible'][8] == 0:
            keypoint['joint3d'][8] = smpl_joint[0, 19, :].cpu().detach().numpy()
        if keypoint['visible'][9] == 0:
            keypoint['joint3d'][9] = smpl_joint[0, 20, :].cpu().detach().numpy()
        if keypoint['visible'][10] == 0:
            keypoint['joint3d'][10] = smpl_joint[0, 21, :].cpu().detach().numpy()
        if keypoint['visible'][11] == 0:
            keypoint['joint3d'][11] = smpl_joint[0, 1, :].cpu().detach().numpy() # 3? 1?
        if keypoint['visible'][12] == 0:
            keypoint['joint3d'][12] = smpl_joint[0, 2, :].cpu().detach().numpy() # 7? 2?
        if keypoint['visible'][13] == 0:
            keypoint['joint3d'][13] = smpl_joint[0, 4, :].cpu().detach().numpy()
        if keypoint['visible'][14] == 0:
            keypoint['joint3d'][14] = smpl_joint[0, 5, :].cpu().detach().numpy()
        if keypoint['visible'][15] == 0:
            keypoint['joint3d'][15] = smpl_joint[0, 7, :].cpu().detach().numpy()
        if keypoint['visible'][16] == 0:
            keypoint['joint3d'][16] = smpl_joint[0, 8, :].cpu().detach().numpy()
        keypoint['visible'][[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = 1

        # left foot
        if keypoint['visible'][17] == 0:
            keypoint['joint3d'][17] = smpl_joint[0, 60, :].cpu().detach().numpy()
        if keypoint['visible'][18] == 0:
            keypoint['joint3d'][18] = smpl_joint[0, 61, :].cpu().detach().numpy()
        if keypoint['visible'][19] == 0:
            keypoint['joint3d'][19] = smpl_joint[0, 62, :].cpu().detach().numpy()
        # right foot
        if keypoint['visible'][20] == 0:
            keypoint['joint3d'][20] = smpl_joint[0, 63, :].cpu().detach().numpy()
        if keypoint['visible'][21] == 0:
            keypoint['joint3d'][21] = smpl_joint[0, 64, :].cpu().detach().numpy()
        if keypoint['visible'][22] == 0:
            keypoint['joint3d'][22] = smpl_joint[0, 65, :].cpu().detach().numpy()

        # left hand
        if keypoint['visible'][96] == 0:
            keypoint['joint3d'][96]  = smpl_joint[0, 67, :].cpu().detach().numpy()
        if keypoint['visible'][100] == 0:
            keypoint['joint3d'][100] = smpl_joint[0, 68, :].cpu().detach().numpy()
        if keypoint['visible'][104] == 0:
            keypoint['joint3d'][104] = smpl_joint[0, 69, :].cpu().detach().numpy()
        if keypoint['visible'][108] == 0:
            keypoint['joint3d'][108] = smpl_joint[0, 70, :].cpu().detach().numpy()

        # right hand
        if keypoint['visible'][117] == 0:
            keypoint['joint3d'][117] = smpl_joint[0, 72, :].cpu().detach().numpy()
        if keypoint['visible'][121] == 0:
            keypoint['joint3d'][121] = smpl_joint[0, 73, :].cpu().detach().numpy()
        if keypoint['visible'][125] == 0:
            keypoint['joint3d'][125] = smpl_joint[0, 74, :].cpu().detach().numpy()
        if keypoint['visible'][129] == 0:
            keypoint['joint3d'][129] = smpl_joint[0, 75, :].cpu().detach().numpy()

        keypoint['visible'][[17, 18, 19, 20, 21, 22]] = 1
        keypoint['visible'][[96, 100, 104, 108, 117, 121, 125, 129]] = 1
    return keypoint

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    keypoint_projection(args.data_path, 
                        args.data_name, 
                        fov=50,
                        cam_res=2048,
                        # cam_res=256,
                        angle_min_x=-30,
                        angle_max_x=30,
                        angle_min_y=0,
                        angle_max_y=0,
                        interval_y=0,
                        interval_x=10,
                        smpl_model_path=args.smpl_model_path,
                        device=device)

            
