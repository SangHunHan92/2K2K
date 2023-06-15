import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import trimesh

def init_affine_2048(pose):
    """
    [0, 1, 2, 3, 4] = [nose, L eye, R eye, L ear, R ear]
    [5, 6, 7, 8, 9, 10] = [L shoudler, R shoudler, L elbow, R elbow, L wrist, R wrist]
    [11, 12, 13, 14, 15, 16] = [L hip, R hip, L knee, R knee, L ankle, R ankle]
    [17, 18, 19, 20, 21, 22] = [L big toe, L little toe, L sole, R big toe, R little toe, R sole]
    [23, 24, 25, 26, 27, 28, 29, 30] = [L finger 2, 3, 4, 5, R finger 2, 3, 4, 5]
    """        
    limbSeq = [[4, 3], [6, 5, 12, 11], # face, body
                [5, 7], [7, 9, 23, 24, 25, 26], [6, 8], [8, 10, 27, 28, 29, 30], # arm
                [11, 13], [13, 15], [12, 14], [14, 16], #leg
                [17, 18, 19, 15], [20, 21, 22, 16]] # [17, 18, 19], [20, 21, 22]] # foot
    affine = np.zeros((len(limbSeq), 2, 3), dtype=np.float32) # [10, 2, 3]
    for i, part in enumerate(limbSeq):
        subset = np.array(part)  # 16, 17

        if i == 3 or i == 5: # for empty finger keypoint
            p_temp = pose[subset] # [2, 2] or [4, 2]
            A = np.zeros((2*3, 4))
            b = np.zeros((2*3, 1))
            p = np.zeros((3,   2))

            key = [] # delete empty keypoint
            for j, a in enumerate(p_temp[2:]):
                if a[0] != 0 and a[1] != 0:
                    key.append(j+2)
            
            p[0] = p_temp[0]
            p[1] = p_temp[1]
            p[2] = np.average(p_temp[key], axis=0)

            for j in range(3):
                A[2*j][0]=p[j][0]            
                A[2*j][1]=-p[j][1]
                A[2*j][2]=1
                A[2*j+1][0]=p[j][1]
                A[2*j+1][1]=p[j][0]
                A[2*j+1][3]=1
        else:                
            p = pose[subset] # [2, 2] or [4, 2]
            A = np.zeros((2*len(p), 4))
            b = np.zeros((2*len(p), 1))
            for j in range(len(p)):
                A[2*j][0]=p[j][0]            
                A[2*j][1]=-p[j][1]
                A[2*j][2]=1
                A[2*j+1][0]=p[j][1]
                A[2*j+1][1]=p[j][0]
                A[2*j+1][3]=1

        if i==0: # 0 head
            length = np.sqrt( (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2 )
            thre = 1/6
            if length < thre:
                b = np.array([-1/12, 1/24, 1/12, 1/24]) * np.array([length/thre])
            else:
                b = np.array([-1/12, 1/24, 1/12, 1/24])
            
        elif i==1: # 1 upper
            b = np.array([-1/6, -1/4, 1/6, -1/4, -1/8, 1/4, 1/8, 1/4]) * np.array([2/3])
        
        elif i==2 or i==4 : # 2 3  upper arm
            length = np.sqrt( (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2 )
            thre = 1/4
            if length < thre:
                b = np.array([0., (-1/8-1/24-1/48), 0., (1/8-1/24-1/48)])  * np.array([length/thre])
            else:
                b = np.array([0., -1/8-1/24-1/48, 0., 1/8-1/24-1/48]) 

        elif i==3 or i==5 : # 4 5 lower arm
            length = np.sqrt( (p[0][0]-p[2][0])**2 + (p[0][1]-p[2][1])**2 )
            thre = (1/6+1/24) * 4/3
            if length < thre:
                b = np.array([0., -1/12-2/24, 0., 1/12-2/24, 0., 1/12-1/24]) * 4/3 * np.array([length/thre])
            else:
                b = np.array([0., -1/12-2/24, 0., 1/12-2/24, 0., 1/12-1/24]) * 4/3
        
        elif i==6 : # L upper leg
            length = np.sqrt( (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2 )
            thre = 1/4
            if length < thre:
                b = np.array([-1/32+1/128, -1/4+3/32, -1/32+1/128, 3/32]) * np.array([length/thre])
            else:
                b = np.array([-1/32+1/128, -1/4+3/32, -1/32+1/128, 3/32])
        elif i == 8: # R upper leg
            length = np.sqrt( (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2 )
            thre = 1/4
            if length < thre:
                b = np.array([1/32-1/128, -1/4+3/32, 1/32-1/128, 3/32]) * np.array([length/thre])
            else:
                b = np.array([1/32-1/128, -1/4+3/32, 1/32-1/128, 3/32])

        elif i == 7 : # L lower leg
            length = np.sqrt( (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2 )
            thre = 6/16*1/2*3/2
            if length < thre:
                b = np.array([-1/32+1/128, -3/16*1/2*3/2, -1/32+1/128, 3/16*1/2*3/2]) * np.array([length/thre])
            else:
                b = np.array([-1/32+1/128, -3/16*1/2*3/2, -1/32+1/128, 3/16*1/2*3/2])
        elif i == 9 : # R lower leg
            length = np.sqrt( (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2 )
            thre = 6/16*1/2*3/2
            if length < thre:
                b = np.array([1/32-1/128, -3/16*1/2*3/2, 1/32-1/128, 3/16*1/2*3/2]) * np.array([length/thre])
            else:
                b = np.array([1/32-1/128, -3/16*1/2*3/2, 1/32-1/128, 3/16*1/2*3/2])

        elif i==10 or i==11 : # 12, 13 foot
            if not np.any(p[0]):
                p[0] = p[1]
            if not np.any(p[1]):
                p[1] = p[0]
            if not np.any(p[2]):
                p[2] = p[3]
            # length = np.sqrt( (p[0][0]-p[2][0])**2 + (p[0][1]-p[2][1])**2 )
            # thre = 1/12
            # if length < thre:
                # b = np.array([-1/24, -1/12, 1/24, -1/12, 0, 1/12])
                # b = np.array([0, 1/24, 0, 1/24, 0, -1/24, 0, -1/24]) * np.array([length/thre])
            # else:
            b = np.array([0, 1/24, 0, 1/24, 0, -1/24, 0, -1/24])
        
        x = np.dot(np.linalg.pinv(A), b)
        y = np.array([[x[0], -x[1], x[2]],
                        [x[1],  x[0], x[3]]])
        affine[i] = y
    return affine

def inv_aff(theta):
    # Input : [b * n, 2, 3]
    R = theta[:, :2, :2]                     # [b * n, 2, 2]
    T = theta[:, :2, 2:]                     # [b * n, 2, 1]
    inv_R = torch.linalg.inv(R)              # [b * n, 2, 2]
    inv_T = - inv_R @ T                      # [b * n, 2, 1]
    return torch.cat((inv_R, inv_T), dim=2)  # [b * n, 2, 3]

def pers2pc(pers_color, pers_depth, res, fov):
    
    focal = res / (2 * np.tan(np.radians(fov) / 2.0))
    pers_depth[pers_depth>0] = (pers_depth[pers_depth>0] - res / 2) / focal + 1
    pers_depth = pers_depth.reshape(-1,1)
    pers_depth = np.tile(pers_depth, reps=[1, 3])

    temp = trimesh.scene.Scene()
    temp.camera.resolution = [res,res]
    temp.camera.fov = [fov,fov]
    pers_origins, pers_vectors, pers_pixels = temp.camera_rays() 
    pers_vectors = np.rot90(pers_vectors.reshape(res,res,3), 1).reshape(-1,3)
    pers_vectors[:,2] *= -1

    xyz = pers_depth * pers_vectors
    xyz = xyz[xyz[:,2] > 0.3, :]    

    if pers_color is not None:
        pers_color = torch.Tensor(pers_color).permute(2, 0, 1).unsqueeze(0)
        pers_color = color2pix(pers_color).squeeze()
        pers_color = torch.masked_select(pers_color, torch.from_numpy(pers_depth[:,2]) > 0.3).view(3, -1).permute(1,0).numpy()
    else:
        pers_color = np.ones_like(xyz) * 128

    return xyz, pers_color

def color2pix(color_map):
    b, _, h, w = color_map.size()
    pixel_coords_vec = color_map.reshape(b, 3, -1)

    return pixel_coords_vec

class ReconDataset_2048(Dataset):
    def __init__(self,
                 data_path='/workspace/dataset/DATASET',
                 data_list='train_all',
                 is_training=True,
                 bg_path='/workspace/dataset/IndoorCVPR09',
                 bg_list=None,
                 res=2048):

        # initialize variables here
        self.data_path = data_path  
        self.data_list = data_list 
        self.is_training = is_training 
        self.bg_path = bg_path
        self.bg_list = bg_list
        self.res = res

        self.depth_div = 32.0
        self.h = self.res
        self.w = self.res
        self.zz = np.zeros(self.h * self.w)
        self.x = np.linspace(0, self.w-1, self.w)
        self.y = np.linspace(0, self.h-1, self.h)
        self.xx, self.yy = np.meshgrid(self.x,self.y)
        self.xxyy = np.c_[self.xx.ravel(), self.yy.ravel()]

        self.transform_human = transforms.Compose ([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.7, 1.0), contrast=(0.7, 1.0), saturation=(0.7, 1.0)),
            transforms.ToTensor(),
        ])
        self.transform_bg = transforms.Compose ([
            transforms.ToPILImage(),
            transforms.RandomCrop((self.h, self.w)),
            transforms.ToTensor ()
        ])
        self.transform_final = transforms.Compose ([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        file_name = os.path.join(self.data_path, 'list', self.data_list + '.txt')
        self.__init_list__(file_name)

        if self.bg_list is not None:
            bg_name = os.path.join (self.data_path, 'list', self.bg_list + '.txt')
            self.__init_bg__(bg_name)

    # 2. initialize paths to the entire training samples.
    def __init_list__(self, file_name):
        self.input = []
        self.back_image = []
        self.front_depth = []
        self.back_depth = []
        self.pose = []
        self.mask = torch.zeros(self.h, self.w)

        with open (file_name) as f:
            for line in f:
                # non-shaded + shaded 
                self.input.append(line.strip().split(" ")[0]) 
                self.back_image.append(line.strip().split(" ")[0].replace('front', 'back').replace('SHADED', 'NOSHADING'))
                self.front_depth.append(line.strip().split(" ")[2]) 
                self.back_depth.append(line.strip().split(" ")[2].replace('front', 'back')) 
                self.pose.append(line.strip().split(" ")[2].replace('DEPTH', 'keypoint').replace('png', 'npy')) 
                
                # self.input.append(line.strip().split(" ")[0].replace('SHADED', 'NOSHADING'))
                # self.back_image.append(line.strip().split(" ")[0].replace('front', 'back').replace('SHADED', 'NOSHADING'))
                # self.front_depth.append(line.strip().split(" ")[2]) 
                # self.back_depth.append(line.strip().split(" ")[2].replace('front', 'back')) 
                # self.pose.append(line.strip().split(" ")[2].replace('DEPTH', 'keypoint').replace('png', 'npy'))
                

    def __fetch_data__(self, idx):
        
        image = cv2.imread (self.data_path + self.input[idx], cv2.IMREAD_COLOR) #uint8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        front_depth = cv2.imread (self.data_path + self.front_depth[idx], cv2.IMREAD_ANYDEPTH).astype(np.float64) / self.depth_div #uint16 -> float64
        back_depth  = cv2.imread (self.data_path + self.back_depth[idx] , cv2.IMREAD_ANYDEPTH).astype(np.float64) / self.depth_div #uint16 -> float64
        mask = cv2.imread (self.data_path + self.front_depth[idx], cv2.IMREAD_ANYDEPTH)
        pose = np.load(self.data_path + self.pose[idx]).astype(np.float32)
        if pose.shape[1] == 31:
            pose = pose.T
        
        front_depth = np.expand_dims (front_depth, axis=2)
        front_depth = torch.Tensor (front_depth).permute (2, 0, 1)
        back_depth = np.expand_dims (back_depth, axis=2)
        back_depth = torch.Tensor (back_depth).permute (2, 0, 1)
        mask = np.where(mask > 0.0, 1.0, 0.0)
        mask = np.expand_dims (mask, axis=2)
        mask = torch.Tensor (mask).permute (2, 0, 1)        

        pose = pose[:, :2]
        pose[:, 0] = 2*pose[:, 0]/self.w - 1
        pose[:, 1] = 2*pose[:, 1]/self.h - 1

        init_affine = init_affine_2048(pose)
        init_affine = torch.Tensor(init_affine)
        init_affine = inv_aff(init_affine) # [n, 2, 3]
                        
        front_depth  = front_depth / 512
        back_depth = back_depth / 512
        image = self.transform_human(image) 

        # human mask
        self.mask[:,:] = 0.0
        self.mask[front_depth[0] > 0] = 1
        
        image = self.composite_image_blur (image, mask=self.mask) # image should be tensor        
        image = self.transform_final(image) # mean std = 0.5
        data_path = self.data_path + self.input[idx]

        return image, front_depth, back_depth, mask, init_affine, data_path

    # 4. get a training sample as the form of a dictionary
    def __getitem__(self, idx):
        image, front_depth, back_depth, mask, init_affine, data_path = self.__fetch_data__(idx)

        return {'image': image, 'front_depth': front_depth, 'back_depth': back_depth, 'mask': mask, 'init_affine': init_affine, 'data_path': data_path}

    # 5. return the size of the training data
    def __len__(self):
        return len(self.input) # all of input

    def __init_bg__(self, f_name):        
        bg_list = []
        with open (f_name) as f:
            lines = f.readlines ()           

            for line in lines:
                line = line.strip().split("\n")[0]
                line = os.path.join('IndoorCVPR09', line.replace('\\', '/'))
                bg_list.append(line)
        
        self.bg_list = bg_list
        self.bg_total = len(self.bg_list)
        self.black_bg = np.zeros((2048, 2048, 3), dtype=np.uint8)

    def composite_image(self, image, mask=None):

        image_idx = random.randrange(0, self.bg_total, 1) # randomly sample background_image_idx
        if random.random() > 0.1:
            bg_file = self.bg_list[image_idx]
            bg_img = cv2.imread(os.path.join(self.bg_path, bg_file), cv2.IMREAD_COLOR)
            bg_img = cv2.resize(bg_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        else:
            bg_img = self.black_bg
        bg_img = self.transform_bg(bg_img)
        condition = mask[:, :] > 0 # mask = depth_map
        image[0:3, condition == False] = bg_img[0:3, condition == False]
        return image
        
    def composite_image_blur(self, image, mask=None):
    
        image_idx = random.randrange(0, self.bg_total, 1) # randomly sample background_image_idx
        if random.random() > 0.1:
            bg_file = self.bg_list[image_idx]
            bg_img = cv2.imread(os.path.join(self.bg_path, bg_file), cv2.IMREAD_COLOR)
            bg_img = cv2.resize(bg_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        else:
            bg_img = self.black_bg
        bg_img = self.transform_bg(bg_img) # random crop, totensor, normalize
        condition = mask[:, :] > 0 # mask = depth_map

        blur = cv2.GaussianBlur(mask.numpy(), (3, 3), 0)
        b_idx = (blur > 0.2) * (blur < 0.8) # (2048, 2048)

        image[0:3, condition == False] = bg_img[0:3, condition == False]
        image = image.permute(1, 2, 0).numpy()
        filtered_front = cv2.medianBlur(image, 3)
        filtered_front = cv2.GaussianBlur(filtered_front, (5, 5), 0)
        image[b_idx, :] = filtered_front[b_idx, :]
        image = torch.Tensor(image).permute(2, 0, 1)

        return image

    @staticmethod
    def fetch_output(datum):
        image = datum['image']
        front_depth = datum['front_depth']
        back_depth = datum['back_depth']
        mask = datum['mask']
        init_affine = datum['init_affine']
        data_path = datum['data_path']

        return image, front_depth, back_depth, mask, init_affine, data_path


if __name__ == '__main__':
    print('')