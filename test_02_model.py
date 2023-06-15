import models
from utils.ReconDataset import init_affine_2048, inv_aff, pers2pc
from utils.train_utils import init_variables
from models.loss_builder import get_normal
import numpy as np
import collections
import argparse
import os
from torchvision import transforms
import trimesh
from tqdm import tqdm
import time 
import random
import cv2
import torch
import json
from rembg import remove
from rembg.session_factory import new_session

# import sys
# sys.path.append('/workspace/code/openpose/build/python');
# from openpose import pyopenpose as op

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=1, help='workers')
parser.add_argument('--load_ckpt',  type=str, default='ckpt_bg_mask.pth.tar', help='somewhere in your PC') 
parser.add_argument('--data_path', type=str, default='./test/', help='path to dataset')
parser.add_argument('--checkpoints_load_path', type=str, default='./checkpoints/', help='path to save checkpoints')
parser.add_argument('--save_path', type=str, default='./result', help='path to save folder')
parser.add_argument('--save_name', type=str, default='test', help='name of save folder inside save_path')
parser.add_argument('--phase', type=int, default=2, help='set training phase')
args = parser.parse_args()

def main():
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

    # 1. GPUs settings
    torch.cuda.empty_cache()
    # cudnn.benchmark = True
    # cudnn.fastest = True
       
    args.local_rank = 0  # indicates designated gpu id.
        
    # load a model to the designated GPUs
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda:{}".format(args.local_rank))    

    # 2. Dataset settings
    
    args.proj_name = args.load_ckpt[:-8]
    args.model_name = 'Model_2K2K'

    res = 2048 
    
    # 3. Test Model
    
    data_path = args.data_path
    os.makedirs(args.save_path + '/' + args.save_name + '/depth/', exist_ok=True)
    os.makedirs(args.save_path + '/' + args.save_name + '/cons/', exist_ok=True)
    os.makedirs(args.save_path + '/' + args.save_name + '/normal/', exist_ok=True)
    os.makedirs(args.save_path + '/' + args.save_name + '/normal_parts/', exist_ok=True)    
    os.makedirs(args.save_path + '/' + args.save_name + '/output_plys/', exist_ok=True)
    os.makedirs(args.save_path + '/' + args.save_name + '/output_plys_c/', exist_ok=True)
    os.makedirs(args.save_path + '/' + args.save_name + '/images/', exist_ok=True)
    # os.makedirs(args.save_path + '/' + args.save_name + '/test/', exist_ok=True)    
    # os.makedirs(args.save_path + '/' + args.save_name + '/images/', exist_ok=True)
    # os.makedirs(args.save_path + '/' + args.save_name + '/512_images/', exist_ok=True)
    os.chmod   (args.save_path + '/' + args.save_name , 0o777)

    print("Start Loading Model ...")

    args.phase = 2
    model = getattr (models, args.model_name)(args.phase, args.device) 
    
    # load checkpoint if required
    if args.load_ckpt:
        ckpt = torch.load(args.checkpoints_load_path + args.load_ckpt)
        model_state_dict = collections.OrderedDict( {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()} )
        model.load_state_dict(model_state_dict, strict=True)

    model.to(args.device)
    model.eval()        

    print("Model Loaded !!")
    
    h = w = res # 2048, 1024, 512
    transform_final = transforms.Compose ([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_human = transforms.Compose ([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=(0.85, 0.85), contrast=(0.85, 0.85), saturation=(0.85, 0.85)),
        transforms.ToTensor(),
    ])

    data_list = [x for x in os.listdir(data_path) if ".png" in x or ".jpg" in x or ".JPG" in x]
    data_list.sort()

    with torch.no_grad():
        for f_i_name in tqdm(data_list): 
            
            img_name = f_i_name.split(".")[0]
            img_path = os.path.join(data_path, f_i_name)

            # openpose json -> numpy
            op_path = os.path.join(data_path, img_name + "_keypoints.json")
            pose = op_json_to_numpy(op_path)

            image = cv2.imread (img_path, cv2.IMREAD_COLOR) #uint8
            image, pose = center_padding(image, pose, 2048)

            start = time.time()
            img_rembg = remove(image, post_process_mask=True, session=new_session("u2net"))
            print(time.time() - start)


            mask = img_rembg[:,:,3].astype(bool)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image[~mask, :] = 0
            image = transform_human(image)

            if len(pose.shape) == 3:
                pose = pose[0]
            if pose.shape[1] == 31:
                pose = pose.T
            pose = pose[:, :2]
            if image.shape[1] != 2048:
                pose = pose / 2
            pose[np.where(pose == 0)] = 1024
            pose[:, 0] = 2*pose[:, 0]/2048 - 1
            pose[:, 1] = 2*pose[:, 1]/2048 - 1
            
            init_affine = init_affine_2048(pose)
            init_affine = torch.Tensor(init_affine)
            
            R = init_affine[:, :2, :2]                     # [b * n, 2, 2]
            try:
                inv_R = torch.linalg.inv(R)              # [b * n, 2, 2]
            except :
                continue

            init_affine = inv_aff(init_affine) # [n, 2, 3]            
            image = transform_final(image)            
            image, _, _, _, init_affine = \
                init_variables(image, None, None, None, init_affine, device=args.device)                
            image = image.unsqueeze(dim=0)
            init_affine = init_affine.unsqueeze(dim=0)

            end = time.time()            
            pred_var = model (image, init_affine, 0)            
            print(time.time() - end)
            
            p_d_f = pred_var['pred_depth_front'].detach()
            p_d_b = pred_var['pred_depth_back'].detach()
            p_n_f = pred_var['pred_normal_front'].detach()
            p_n_b = pred_var['pred_normal_back'].detach()

            p_n_f_face       = pred_var['pred_face_normal_front'].detach()  
            p_n_b_face       = pred_var['pred_face_normal_back'].detach()  
            p_n_f_upper      = pred_var['pred_upper_normal_front'].detach()  
            p_n_b_upper      = pred_var['pred_upper_normal_back'].detach()  
            p_n_f_arm        = pred_var['pred_arm_normal_front'].detach()  
            p_n_b_arm        = pred_var['pred_arm_normal_back'].detach()  
            p_n_f_leg        = pred_var['pred_leg_normal_front'].detach()  
            p_n_b_leg        = pred_var['pred_leg_normal_back'].detach()  
            p_n_f_shoe       = pred_var['pred_shoe_normal_front'].detach()  
            p_n_b_shoe       = pred_var['pred_shoe_normal_back'].detach()  
            p_n_f_down       = pred_var['pred_down_normal_front'].detach()  
            p_n_b_down       = pred_var['pred_down_normal_back'].detach()  
            p_d_f_down       = pred_var['pred_down_depth_front'].detach()  
            p_d_b_down       = pred_var['pred_down_depth_back'].detach()  
            p_d_f_down_mask  = (pred_var['pred_down_depth_back'].detach()>0).float()
            p_d_f_mask       = (pred_var['pred_depth_back'].detach()>0).float()

            # remove front_depth, back_depth artifact
            arti = p_d_f > p_d_b
            p_d_f[arti] = 0
            p_d_b[arti] = 0
            
            s_n_f = get_normal(p_d_f)
            s_n_b = get_normal(p_d_b)

            front_depth = p_d_f[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 512 * 32
            front_depth = front_depth.astype(np.uint16)
            back_depth = p_d_b[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 512 * 32
            back_depth = back_depth.astype(np.uint16)
            front_cons = s_n_f[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            back_cons = s_n_b[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            front_cons = cv2.cvtColor(front_cons, cv2.COLOR_BGR2RGB)
            back_cons = cv2.cvtColor(back_cons, cv2.COLOR_BGR2RGB)
            front_cons = s_n_f[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            back_cons = s_n_b[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            front_cons = cv2.cvtColor(front_cons, cv2.COLOR_BGR2RGB)
            back_cons = cv2.cvtColor(back_cons, cv2.COLOR_BGR2RGB)            
            front_normal = p_n_f[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            back_normal = p_n_b[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            front_normal = cv2.cvtColor(front_normal, cv2.COLOR_BGR2RGB)
            back_normal = cv2.cvtColor(back_normal, cv2.COLOR_BGR2RGB)

            p_n_f_face = p_n_f_face[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_b_face = p_n_b_face[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_f_face = cv2.cvtColor(p_n_f_face, cv2.COLOR_BGR2RGB)
            p_n_b_face = cv2.cvtColor(p_n_b_face, cv2.COLOR_BGR2RGB)

            p_n_f_upper = p_n_f_upper[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_b_upper = p_n_b_upper[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_f_upper = cv2.cvtColor(p_n_f_upper, cv2.COLOR_BGR2RGB)
            p_n_b_upper = cv2.cvtColor(p_n_b_upper, cv2.COLOR_BGR2RGB)

            p_n_f_arm = p_n_f_arm[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_b_arm = p_n_b_arm[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_f_arm = cv2.cvtColor(p_n_f_arm, cv2.COLOR_BGR2RGB)
            p_n_b_arm = cv2.cvtColor(p_n_b_arm, cv2.COLOR_BGR2RGB)

            p_n_f_leg = p_n_f_leg[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_b_leg = p_n_b_leg[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_f_leg = cv2.cvtColor(p_n_f_leg, cv2.COLOR_BGR2RGB)
            p_n_b_leg = cv2.cvtColor(p_n_b_leg, cv2.COLOR_BGR2RGB)

            p_n_f_shoe = p_n_f_shoe[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_b_shoe = p_n_b_shoe[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_f_shoe = cv2.cvtColor(p_n_f_shoe, cv2.COLOR_BGR2RGB)
            p_n_b_shoe = cv2.cvtColor(p_n_b_shoe, cv2.COLOR_BGR2RGB)

            p_n_f_down = p_n_f_down[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_b_down = p_n_b_down[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 255
            p_n_f_down = cv2.cvtColor(p_n_f_down, cv2.COLOR_BGR2RGB)
            p_n_b_down = cv2.cvtColor(p_n_b_down, cv2.COLOR_BGR2RGB)

            p_d_f_down = p_d_f_down[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 512 * 32
            p_d_f_down = p_d_f_down.astype(np.uint16)
            p_d_b_down = p_d_b_down[0].detach ().cpu ().numpy ().transpose(1, 2, 0) * 512 * 32
            p_d_b_down = p_d_b_down.astype(np.uint16)

            p_d_f_down_mask = p_d_f_down_mask[0].detach ().cpu ().numpy ().transpose(1, 2, 0) 
            p_d_f_down_mask = p_d_f_down_mask.astype(np.uint16)
            p_d_f_mask = p_d_f_mask[0].detach ().cpu ().numpy ().transpose(1, 2, 0)
            p_d_f_mask = p_d_f_mask.astype(np.uint16)
        
            # this
            cv2.imwrite((args.save_path + '/' + args.save_name + '/depth/' + img_name+'_front.png'), front_depth)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/depth/' + img_name+'_back.png'), back_depth)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/cons/' + img_name+'_front.png'), front_cons)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/cons/' + img_name+'_back.png'), back_cons)                 
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal/' + img_name+'_front.png'), front_normal)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal/' + img_name+'_back.png'), back_normal)

            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_face.png'), p_n_f_face)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_face.png'),  p_n_b_face)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_upper.png'), p_n_f_upper)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_upper.png'),  p_n_b_upper)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_arm.png'), p_n_f_arm)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_arm.png'),  p_n_b_arm)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_leg.png'), p_n_f_leg)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_leg.png'),  p_n_b_leg)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_shoe.png'), p_n_f_shoe)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_shoe.png'),  p_n_b_shoe)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_down.png'), p_n_f_down)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_down.png'),  p_n_b_down)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_front_down_depth.png'), p_d_f_down)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_back_down_depth.png'),  p_d_b_down)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_down_mask.png'), p_d_f_down_mask*255)
            cv2.imwrite((args.save_path + '/' + args.save_name + '/normal_parts/' + img_name+'_mask.png'), p_d_f_mask*255)

            cv2.imwrite((args.save_path + '/' + args.save_name + '/images/' + img_name+'.png'), img_rembg)

            kernel = np.ones((3, 3), 'uint8')
            mask = cv2.erode(front_depth, kernel, iterations=3)>0
            back_depth[:, :, 0] = back_depth[:, :, 0] * mask
            
            corr1_depth = cv2.dilate(back_depth, kernel, iterations=1)
            corr2_depth = cv2.dilate(back_depth, kernel, iterations=2)
            corr3_depth = cv2.dilate(back_depth, kernel, iterations=3)

            mask_1 = 1*(cv2.erode(front_depth, kernel, iterations=4)>0)
            mask0 = 1*(back_depth[:, :, 0]>0)
            mask1 = 1*(corr1_depth>0)
            mask2 = 1*(corr2_depth>0)
            mask3 = 1*(corr3_depth>0)

            mask_10 = ~((mask0-mask_1)>0)
            mask01 = ~((mask1-mask0)>0)
            mask12 = ~((mask2-mask1)>0)
            mask23 = ~((mask3-mask2)>0)
            
            corr_depth10 = (5*front_depth[:, :, 0].astype(np.float64)/6 + corr1_depth.astype(np.float64)/6)
            corr_depth10[mask01] = 0.0
            corr_depth10 = np.expand_dims(corr_depth10, axis=2)

            corr_depth20 = (3*front_depth[:, :, 0].astype(np.float64)/5 + 2*corr2_depth.astype(np.float64)/5)
            corr_depth20[mask12] = 0.0
            corr_depth20 = np.expand_dims(corr_depth20, axis=2)

            corr_depth30 = (front_depth[:, :, 0].astype(np.float64)/4 + 3*corr3_depth.astype(np.float64)/4)
            corr_depth30[mask23] = 0.0
            corr_depth30 = np.expand_dims(corr_depth30, axis=2)


            corr_depth05 = (6.5*front_depth[:, :, 0].astype(np.float64)/7 + 0.5*back_depth[:, :, 0].astype(np.float64)/7)
            corr_depth05[mask_10] = 0.0
            corr_depth05 = np.expand_dims(corr_depth05, axis=2)

            corr_depth15 = (4.5*front_depth[:, :, 0].astype(np.float64)/6 + 1.5*corr1_depth.astype(np.float64)/6)
            corr_depth15[mask01] = 0.0
            corr_depth15 = np.expand_dims(corr_depth15, axis=2)

            corr_depth25 = (2.5*front_depth[:, :, 0].astype(np.float64)/5 + 2.5*corr2_depth.astype(np.float64)/5)
            corr_depth25[mask12] = 0.0
            corr_depth25 = np.expand_dims(corr_depth25, axis=2)

            corr_depth35 = (0.5*front_depth[:, :, 0].astype(np.float64)/4 + 3.5*corr3_depth.astype(np.float64)/4)
            corr_depth35[mask23] = 0.0
            corr_depth35 = np.expand_dims(corr_depth35, axis=2)

            if res==512:
                front_depth_temp = np.zeros((2048, 2048, 1), dtype=np.uint16)
                back_depth_temp = np.zeros((2048, 2048, 1), dtype=np.uint16)
                front_depth_temp[0::4, 0::4, :] = front_depth
                back_depth_temp[0::4, 0::4, :] = back_depth
                front_depth = front_depth_temp
                back_depth = back_depth_temp
            elif res==1024:
                front_depth_temp = np.zeros((2048, 2048, 1), dtype=np.uint16)
                back_depth_temp = np.zeros((2048, 2048, 1), dtype=np.uint16)
                front_depth_temp[0::2, 0::2, :] = front_depth
                back_depth_temp[0::2, 0::2, :] = back_depth
                front_depth = front_depth_temp
                back_depth = back_depth_temp
                        
            
            image_front = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)            
            image_front, pose = center_padding(image_front, pose, 2048)
            image_back = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
            image_back, pose = center_padding(image_back, pose, 2048)
            if image_front.shape[0] != 2048 or image_front.shape[1] != 2048:
                image_front = cv2.resize(image_front, (2048, 2048), interpolation=cv2.INTER_CUBIC)
            if image_back.shape[0] != 2048 or image_back.shape[1] != 2048:
                image_back = cv2.resize(image_back, (2048, 2048), interpolation=cv2.INTER_CUBIC)
            image_front = cv2.cvtColor(image_front, cv2.COLOR_BGR2RGB)
            image_back  = cv2.cvtColor(image_back , cv2.COLOR_BGR2RGB)
            if image_front.shape[0] != 2048:
                image_front = cv2.resize(image_front, (2048, 2048), interpolation=cv2.INTER_CUBIC)
                image_back = cv2.resize(image_back, (2048, 2048), interpolation=cv2.INTER_CUBIC)
            xyz_f, rgb_f = pers2pc(image_front, front_depth.astype(np.float64) / 32.0, 2048, 50)
            xyz_b, rgb_b = pers2pc(image_back, back_depth.astype(np.float64) / 32.0, 2048, 50)
            
            xyz = np.concatenate((xyz_f, xyz_b), axis=0)
            rgb = np.concatenate((rgb_f, rgb_b), axis=0)
            xyz[:, 2] -= 1
            xyz[:, 1] *= -1
            xyz[:, 2] *= -1
            xyz /= 1.0367394227574303
            pc = trimesh.points.PointCloud(vertices=xyz, colors=rgb)
            pc.export(args.save_path + '/' + args.save_name + '/output_plys/' + img_name+'.ply')

            if res==2048:
                xyz_c1,  rgb_c1  = pers2pc(None, corr_depth10.astype(np.float64) / 32.0, 2048, 50)
                xyz_c2,  rgb_c2  = pers2pc(None, corr_depth20.astype(np.float64) / 32.0, 2048, 50)
                xyz_c3,  rgb_c3  = pers2pc(None, corr_depth30.astype(np.float64) / 32.0, 2048, 50)
                xyz_c05, rgb_c05 = pers2pc(None, corr_depth05.astype(np.float64) / 32.0, 2048, 50)
                xyz_c15, rgb_c15 = pers2pc(None, corr_depth15.astype(np.float64) / 32.0, 2048, 50)
                xyz_c25, rgb_c25 = pers2pc(None, corr_depth25.astype(np.float64) / 32.0, 2048, 50)
                xyz_c35, rgb_c35 = pers2pc(None, corr_depth35.astype(np.float64) / 32.0, 2048, 50)

                xyz = np.concatenate((xyz_f, xyz_b, xyz_c1, xyz_c2, xyz_c3, xyz_c05, xyz_c15, xyz_c25, xyz_c35), axis=0)
                rgb = np.concatenate((rgb_f, rgb_b, rgb_c1, rgb_c2, rgb_c3, rgb_c05, rgb_c15, rgb_c25, rgb_c35), axis=0)
                xyz[:, 2] -= 1
                xyz[:, 1] *= -1
                xyz[:, 2] *= -1
                xyz /= 1.0367394227574303
                pc = trimesh.points.PointCloud(vertices=xyz, colors=rgb)
                pc.export(args.save_path + '/' + args.save_name + '/output_plys_c/' + img_name+'.ply')


def op_json_to_numpy(op_path):
    
    pose_npy = np.zeros((31, 3))
    key_path = op_path  
    with open(key_path, "r") as f:
        jfile = json.load(f)
    pose_j  = jfile['people'][0]['pose_keypoints_2d']    
    lhand_j = jfile['people'][0]['hand_left_keypoints_2d']
    rhand_j = jfile['people'][0]['hand_right_keypoints_2d']
    
    pose_npy[0] = pose_j[0*3:0*3+3]
    pose_npy[1] = pose_j[16*3:16*3+3]
    pose_npy[2] = pose_j[15*3:15*3+3]
    pose_npy[3] = pose_j[18*3:18*3+3]
    if pose_j[18*3:18*3+3][0] == 0 and pose_j[18*3:18*3+3][1] == 0:
        pose_npy[3] = pose_j[16*3:16*3+3]
    pose_npy[4] = pose_j[17*3:17*3+3]
    if pose_j[17*3:17*3+3][0] == 0 and pose_j[17*3:17*3+3][1] == 0:
        pose_npy[4] = pose_j[15*3:15*3+3]
    pose_npy[5] = pose_j[5*3:5*3+3]
    pose_npy[6] = pose_j[2*3:2*3+3]
    pose_npy[7] = pose_j[6*3:6*3+3]
    pose_npy[8] = pose_j[3*3:3*3+3]
    pose_npy[9] = pose_j[7*3:7*3+3]
    pose_npy[10] = pose_j[4*3:4*3+3]
    
    pose_npy[11] = pose_j[12*3:12*3+3] # [11, 12, 13, 14, 15, 16] = [L hip, R hip, L knee, R knee, L ankle, R ankle]
    pose_npy[12] = pose_j[9*3:9*3+3]
    pose_npy[13] = pose_j[13*3:13*3+3]
    pose_npy[14] = pose_j[10*3:10*3+3]
    pose_npy[15] = pose_j[14*3:14*3+3]
    pose_npy[16] = pose_j[11*3:11*3+3]
    
    pose_npy[17] = pose_j[19*3:19*3+3] # [17, 18, 19, 20, 21, 22] = [L big toe, L little toe, L sole, R big toe, R little toe, R sole]
    pose_npy[18] = pose_j[20*3:20*3+3]
    pose_npy[19] = pose_j[21*3:21*3+3]
    pose_npy[20] = pose_j[22*3:22*3+3]
    pose_npy[21] = pose_j[23*3:23*3+3]
    pose_npy[22] = pose_j[24*3:24*3+3]
    
    pose_npy[23] = lhand_j[6 *3:6 *3+3] # [23, 24, 25, 26, 27, 28, 29, 30] = [L finger 2, 3, 4, 5, R finger 2, 3, 4, 5]
    pose_npy[24] = lhand_j[10*3:10*3+3]
    pose_npy[25] = lhand_j[14*3:14*3+3]
    pose_npy[26] = lhand_j[18*3:18*3+3]    
    pose_npy[27] = rhand_j[6 *3:6 *3+3]
    pose_npy[28] = rhand_j[10*3:10*3+3]
    pose_npy[29] = rhand_j[14*3:14*3+3]
    pose_npy[30] = rhand_j[18*3:18*3+3]

    return pose_npy

def center_padding(img, pose, set_size):
    h,w,c = img.shape
    if h < w:
        top = (w-h)//2
        bottom = (w-h) - (w-h)//2
        img = cv2.copyMakeBorder(img,top,bottom,0,0,cv2.BORDER_REPLICATE) # top, bottom, left, right
        pose[:,1] += top
    elif h > w:
        left = (h-w)//2
        right = (h-w) - (h-w)//2
        img = cv2.copyMakeBorder(img,0,0,left,right,cv2.BORDER_REPLICATE) # top, bottom, left, right
        pose[:,0] += left
    
    if img.shape[0] != set_size or img.shape[1] != set_size:
        img = cv2.resize(img, (set_size, set_size), interpolation=cv2.INTER_CUBIC)
        pose[:,:2] *= set_size / max(h, w)

    return img, pose


if __name__ == '__main__':    
    main()
