from __future__ import print_function
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as T
import math

class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int): #512 512 256
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU (inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)         #256
        x1 = self.W_x(x)         #256
        psi = self.relu(g1 + x1) #256
        psi = self.psi(psi)      #1

        return x * psi           #512

class Part_2048(nn.Module):
    def __init__(self, device=None):
        # Input : Image, Openpose Skeleton Part Gaussian Map(Heatmap)
        # Output : Affine Images, Affine Occupancy Grid Map, Part Affine(Similarity) matrix, Part Inverse Affine matrix
        # Parameter : Downsampling height, width
        super(Part_2048, self).__init__()

        self.device = device

    # Get inverse of affine matrix(theta)
    def inv_aff(self, theta):
        # Input : [b * n, 2, 3]
        R = theta[:, :2, :2]                     # [b * n, 2, 2]
        T = theta[:, :2, 2:]                     # [b * n, 2, 1]
        inv_R = torch.linalg.inv(R)              # [b * n, 2, 2]
        inv_T = - inv_R @ T                      # [b * n, 2, 1]
        return torch.cat((inv_R, inv_T), dim=2)  # [b * n, 2, 3]

    def get_sim_mat(self, param):
        param = param.view(-1, 4)                                        # [b * n, 4]
        rot_mat = torch.stack([torch.stack([param[:,0], -param[:,1]]),
                               torch.stack([param[:,1],  param[:,0]])]).permute (2, 0, 1)  # [b * n, 2, 2] 
        trans_mat = param[:,2:].unsqueeze(dim=2)                         # [b * n, 2, 1]
        sim_mat = torch.cat((rot_mat, trans_mat), dim=2)                 # [b * n, 2, 3]
        return sim_mat
    
    def affine_bmm(self, x, y):
        A = torch.bmm(x[:, :, :2], y[:, :, :2])
        b = torch.bmm(x[:, :, :2], y[:, :, 2:]) + x[:, :, 2:]
        return torch.cat((A, b), dim=2)

    def forward(self, x, init_affine):
        # 2d Occupancy Grid Map
        b, _, ho, wo = list(x.size())
        _, n,  _,  _ = list(init_affine.size())
        o = torch.ones(b, n, ho, wo).to(x.device)
        z = init_affine.view(-1, 2, 3)

        # Affine Images
        target_size = list(x.size())
        target_size[0] = target_size[0] * n
        grid = F.affine_grid(z, target_size)                       # [b*n, 512, 512, 2]
        x = x.unsqueeze(dim=1).expand(-1, n, -1, -1, -1)           # [b, n, 3, 512, 512]
        x = x.reshape(-1, 3, ho, wo)                               # [b*n, 3, 512, 512]
        x = F.grid_sample(x, grid)                                 # [b*n, 3, 512, 512]
        
        # Affine Occupancy Grid Map
        o = o.view(-1, 1, ho, wo)                                  # [b*n, 1, 512, 512]
        grid = F.affine_grid(z, o.size())                          # [b*n, 512, 512, 2]
        o = F.grid_sample(o, grid)                                 # [b*n, 1, 512, 512]

        # Inverse Affine Matrix
        inv_z = self.inv_aff(z)                                    # [b*n, 2, 3]
        
        # Divide Batch and Parts
        x     = x    .view(-1, n, 3, ho, wo)                       # [b, n, 3, 512, 512]
        o     = o    .view(-1, n, 1, ho, wo)                       # [b, n, 1, 512, 512], int
        z     = z    .view(-1, n, 2, 3)                            # [b, n, 2, 3]
        inv_z = inv_z.view(-1, n, 2, 3)                            # [b, n, 2, 3]

        return x, o, z, inv_z

class conv_block_1x1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ATUNet_small(nn.Module):
    def __init__(self, in_ch=3, out_ch=6):
        super(ATUNet_small, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_ch, ch_out=64) # in_ch, 64 # in_ch
        self.Conv2 = conv_block(ch_in=64, ch_out=128) # 64, 128
        self.Conv3 = conv_block(ch_in=128, ch_out=256) # 128, 256
        self.Conv4 = conv_block(ch_in=256, ch_out=512) # 256, 512

        self.Up4 = up_conv(ch_in=512, ch_out=256) # 512, 256
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128) # 256, 256, 128
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256) # 512, 256

        self.Up3 = up_conv(ch_in=256, ch_out=128) # 256, 128
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64) #128, 128, 64
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128) # 256, 128

        self.Up2 = up_conv(ch_in=128, ch_out=64) # 128, 64
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32) # 64, 64, 32
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64) # 128, out_ch

        self.Conv_1x1 = conv_block_1x1(ch_in=64, ch_out=out_ch)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)      # 64,  H,   W

        x2 = self.Maxpool(x1)   # 64,  H/2, W/2
        x2 = self.Conv2(x2)     # 128, H/2, W/2

        x3 = self.Maxpool(x2)   # 128, H/4, W/4
        x3 = self.Conv3(x3)     # 256, H/4, W/4

        x4 = self.Maxpool(x3)   # 256, H/8, W/8
        x4 = self.Conv4(x4)     # 512, H/8, W/8

        d4 = self.Up4(x4) # d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)          # 256,  H/4, W/4

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)          # 128,  H/2, W/2

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)          # 64,   H,   W

        d1 = self.Conv_1x1(d2)          # out_c,H,   W

        return d1                       # out_c,   H,   W

class ATUNetME(nn.Module):
    def __init__(self, in_ch1=6, in_ch2=6, out_ch=2):
        super(ATUNetME, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(ch_in=in_ch1, ch_out=32)
        self.Conv2_1 = conv_block(ch_in=32, ch_out=64)
        self.Conv3_1 = conv_block(ch_in=64, ch_out=128)
        self.Conv4_1 = conv_block(ch_in=128, ch_out=256)
        self.Conv5_1 = conv_block(ch_in=256, ch_out=512)

        self.Conv1_2 = conv_block (ch_in=in_ch2, ch_out=32)
        self.Conv2_2 = conv_block (ch_in=32, ch_out=64)
        self.Conv3_2 = conv_block (ch_in=64, ch_out=128)
        self.Conv4_2 = conv_block (ch_in=128, ch_out=256)
        self.Conv5_2 = conv_block (ch_in=256, ch_out=512)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)
        # self.Conv_1x1 = conv_block_1x1(ch_in=64, ch_out=out_ch)

    def forward(self, x, y): # x : [3, 256, 256], # y : [6, 256, 256]
        # encoding path
        x1 = self.Conv1_1(x)  # [ 32, 256, 256]
        x2 = self.Maxpool(x1) # [ 32, 128, 128]
        x2 = self.Conv2_1(x2) # [ 64, 128, 128]
        x3 = self.Maxpool(x2) # [ 64,  64,  64]
        x3 = self.Conv3_1(x3) # [128,  64,  64]
        x4 = self.Maxpool(x3) # [128,  32,  32]
        x4 = self.Conv4_1(x4) # [256,  32,  32]
        x5 = self.Maxpool(x4) # [256,  16,  16]
        x5 = self.Conv5_1(x5) # [512,  16,  16]

        y1 = self.Conv1_2(y)  # [ 32, 256, 256]
        y2 = self.Maxpool(y1) # [ 32, 128, 128]
        y2 = self.Conv2_2(y2) # [ 64, 128, 128]
        y3 = self.Maxpool(y2) # [ 64,  64,  64]
        y3 = self.Conv3_2(y3) # [128,  64,  64]
        y4 = self.Maxpool(y3) # [128,  32,  32]
        y4 = self.Conv4_2(y4) # [256,  32,  32]
        y5 = self.Maxpool(y4) # [256,  16,  16]
        y5 = self.Conv5_2(y5) # [512,  16,  16]

        d_shared = torch.cat([x5, y5], dim=1)                   # [1024,  16,  16]

        d5_1 = self.Up5(d_shared)                               # [ 512,  32,  32]
        x4_1 = self.Att5(g=d5_1, x=torch.cat([x4, y4], dim=1))  # [ 512,  32,  32]
        d5_1 = torch.cat([x4_1, d5_1], dim=1)                   # [1024,  32,  32]
        d5_1 = self.Up_conv5(d5_1)                              # [ 512,  32,  32]

        d4_1 = self.Up4(d5_1)                                   # [ 256,  64,  64]
        x3_1 = self.Att4(g=d4_1, x=torch.cat([x3, y3], dim=1))  # [ 256,  64,  64]
        d4_1 = torch.cat([x3_1, d4_1], dim=1)                   # [ 512,  64,  64]
        d4_1 = self.Up_conv4(d4_1)                              # [ 256,  64,  64]

        d3_1 = self.Up3(d4_1)                                   # [ 128, 128, 128]
        x2_1 = self.Att3(g=d3_1, x=torch.cat([x2, y2], dim=1))
        d3_1 = torch.cat([x2_1, d3_1], dim=1)
        d3_1 = self.Up_conv3(d3_1)                              # [ 128, 128, 128]

        d2_1 = self.Up2(d3_1)                                   # [  64, 256, 256]
        x1_1 = self.Att2(g=d2_1, x=torch.cat([x1, y1], dim=1))
        d2_1 = torch.cat([x1_1, d2_1], dim=1)
        d2_1 = self.Up_conv2(d2_1)                              # [  64, 256, 256]

        d1_1 = self.Conv_1x1(d2_1)                              # [   2, 256, 256]

        return d1_1

class Refine_2048(nn.Module):
    def __init__(self, in_ch1=2, in_ch2=6, out_ch=3):
        super(Refine_2048, self).__init__()
        self.depth1   = up_conv(ch_in=in_ch1, ch_out=32) 
        self.depth2   = up_conv(ch_in=64, ch_out=32)
        self.depth3   = up_conv(ch_in=64, ch_out=32)

        self.normal1  = single_conv(ch_in=in_ch2, ch_out=32) 
        self.normal2  = single_conv(ch_in=in_ch2, ch_out=32)
        self.normal3  = single_conv(ch_in=in_ch2, ch_out=32)
        
        self.merge1_1 = conv_block(ch_in=64, ch_out=64)
        self.merge2_1 = conv_block(ch_in=64, ch_out=64)
        self.merge3_1 = conv_block(ch_in=64, ch_out=64)

        self.merge1_2 = conv_block_1x1(ch_in=64, ch_out=64) 
        self.merge2_2 = conv_block_1x1(ch_in=64, ch_out=64)
        self.merge3_2 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, d, n1, n2, n3, n4, n5, o, inv_z):
        # d : 256 resolution depth
        # n1, n2, n3, n4, n5 = part normal    

        with torch.no_grad ():
            b, _, _, _ = list(d.size())
            ho, wo = [2048, 2048]            

            # merge part normal
            n1 = self.centerpadding(n1 , ho, wo)          # [b,   6, 1024, 1024]
            n2 = self.centerpadding(n2 , ho, wo)          # [b,   6, 1024, 1024]
            n3 = self.centerpadding(n3 , ho, wo)          # [b*4, 6, 1024, 1024]
            n4 = self.centerpadding(n4 , ho, wo)          # [b*4, 6, 1024, 1024] #1024:13477 #512:10023
            n5 = self.centerpadding(n5 , ho, wo)          # [b*4, 6, 1024, 1024] #1024:13477 #512:10023

            grid = F.affine_grid(inv_z[:, 0,:,:], [b, 6, ho, wo])
            n1  = F.grid_sample(n1, grid, mode='bilinear').view([-1, 1, 6, ho, wo])    # [b,  1, 6, 1024, 1024]
            grid = F.affine_grid(inv_z[:, 1,:,:], [b, 6, ho, wo])
            n2  = F.grid_sample(n2, grid, mode='bilinear').view([-1, 1, 6, ho, wo])   # [b,  1, 6, 1024, 1024]
            grid = F.affine_grid(inv_z[:, 2:6,:,:].reshape(-1, 2, 3), [b*4, 6, ho, wo])
            n3  = F.grid_sample(n3, grid, mode='bilinear').view([-1, 4, 6, ho, wo])     # [b,  4, 6, 1024, 1024]
            grid = F.affine_grid(inv_z[:, 6:10,:,:].reshape(-1, 2, 3), [b*4, 6, ho, wo])
            n4  = F.grid_sample(n4, grid, mode='bilinear').view([-1, 4, 6, ho, wo])
            grid = F.affine_grid(inv_z[:, 10:12,:,:].reshape(-1, 2, 3), [b*2, 6, ho, wo])
            n5  = F.grid_sample(n5, grid, mode='bilinear').view([-1, 2, 6, ho, wo])

            # o : [b, 12, 2048, 2048]
            n = torch.cat((n1, n2, n3, n4, n5), dim=1)            # [b, 12, 6, 2048, 2048]
            o = torch.unsqueeze(o, dim=2)
            o = o.expand(-1, -1, 6, -1, -1)
            n = n * o
            n = torch.sum(n, dim=1, keepdim=False)            # [b, 6, 2048, 2048]
            n1 = n[:, :, 1::4, 1::4]
            n2 = n[:, :, 0::2, 0::2]
        
        d1 = self.depth1(d)
        n1 = self.normal1(n1)
        d1 = torch.cat((d1, n1), dim=1)
        d1 = self.merge1_1(d1)
        d1 = self.merge1_2(d1)

        d2 = self.depth2(d1)
        n2 = self.normal2(n2)
        d2 = torch.cat((d2, n2), dim=1)
        d2 = self.merge2_1(d2)
        d2 = self.merge2_2(d2)

        d3 = self.depth3(d2)
        n3 = self.normal3(n)
        d3 = torch.cat((d3, n3), dim=1)
        d3 = self.merge3_1(d3)
        d3 = self.merge3_2(d3)

        return n, d3

    def centerpadding(self, data, h, w):
        ch, cw = list(data.size())[-2], list(data.size())[-1]
        p2d = ( (w-cw)//2, (w-cw)//2, (h-ch)//2, (h-ch)//2 )
        return F.pad(data, p2d, 'constant', 0)

class Model_2K2K (nn.Module):
    def __init__(self, phase=1, device=None):
        super (Model_2K2K, self).__init__ ()
        self.phase = phase
        self.device = device
        self.img2img = []
        self.normal2depth = []
        self.refine_lr = []
        self.pers2orth = []
        self.use = False
        self.k_size = 15

        self.stn                  = Part_2048 (device=self.device)
        if self.phase == 1 :
            self.img2normal_face  = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_upper = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_arm   = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_leg   = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_shoe  = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_down  = ATUNet_small (in_ch=3, out_ch=6)
            self.ImgNorm_to_Dep   = ATUNetME (in_ch1=3, in_ch2=6, out_ch=3)
            self.Sigmoid          = nn.Sigmoid()
            self.relu6            = nn.ReLU6()
        elif self.phase == 2:
            self.img2normal_face  = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_upper = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_arm   = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_leg   = ATUNet_small (in_ch=6, out_ch=6)        
            self.img2normal_shoe  = ATUNet_small (in_ch=6, out_ch=6)
            self.img2normal_down  = ATUNet_small (in_ch=3, out_ch=6)
            self.ImgNorm_to_Dep   = ATUNetME (in_ch1=3, in_ch2=6, out_ch=3)
            self.Sigmoid          = nn.Sigmoid()
            self.relu6            = nn.ReLU6()
            self.transform        = T.GaussianBlur(kernel_size=(self.k_size, self.k_size), sigma=5.0)
            self.refine           = Refine_2048 (in_ch1=2, in_ch2=6, out_ch=3)
        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def centercrop(self, data, h, w):
        ch, cw = list(data.size())[-2]//2, list(data.size())[-1]//2
        return data[:, :, ch-h//2:ch+h//2, cw-w//2:cw+w//2]
    
    def centerpadding(self, data, h, w):
        ch, cw = list(data.size())[-2], list(data.size())[-1]
        p2d = ( (w-cw)//2, (w-cw)//2, (h-ch)//2, (h-ch)//2 )
        return F.pad(data, p2d, 'constant', 0)
    
    def inv_aff(self, theta):
        # Input : [b * n, 2, 3]
        R = theta[:, :2, :2]                     # [b * n, 2, 2]
        T = theta[:, :2, 2:]                     # [b * n, 2, 1]
        inv_R = torch.linalg.inv(R)              # [b * n, 2, 2]
        inv_T = - inv_R @ T                      # [b * n, 2, 1]
        return torch.cat((inv_R, inv_T), dim=2)  # [b * n, 2, 3]

    def forward(self, x, init_affine, epoch, x2=None, x3=None):
        b, _, ho, wo = list(x.size())                                # [b, _, 1024, 1024] 
        _, h,  _,  _ = list(init_affine.size())
        
        f_n_f, f_n_b = [None, None]
        u_n_f, u_n_b = [None, None]
        a_n_f, a_n_b = [None, None]
        l_n_f, l_n_b = [None, None]
        s_n_f, s_n_b = [None, None]
        d_n_f, d_n_b = [None, None]
        n_f  , n_b   = [None, None]
        d_f  , d_b,   d_m   = [None, None, None]
        d_d_f, d_d_b, d_d_m = [None, None, None]
        o_show, z, inv_z = [None, None, None]

        f_c = [368, 320] 
        u_c = [528, 336] 
        a_c = [352, 224] 
        l_c = [272, 256] 
        s_c = [176, 128] 

        if self.phase == 1: # Phase 1 
            with torch.no_grad ():
                x1, o, z, inv_z = self.stn(x, init_affine)         

                # 2D Occupancy Grid Map
                o_face  = o[:, 0, :, :, :]                            # [b, 1, 1024, 1024]
                o_upper = o[:, 1, :, :, :]                            # [b, 1, 1024, 1024]
                o_arm   = o[:, 2:6, :, :, :].reshape(-1, 1, ho, wo)   # [b, 4, 1, 1024, 1024]
                o_leg   = o[:, 6:10, :, :, :].reshape(-1, 1, ho, wo)    # [b, 4, 1, 1024, 1024]
                o_foot  = o[:, 10:12, :, :, :].reshape(-1, 1, ho, wo)    # [b, 2, 1, 1024, 1024]

                o_face  = self.centercrop(o_face , f_c[0], f_c[1])          # [b, 1, 176, 160]
                o_upper = self.centercrop(o_upper, u_c[0], u_c[1])          # [b, 1, 320, 256]
                o_arm   = self.centercrop(o_arm  , a_c[0], a_c[1])
                o_leg   = self.centercrop(o_leg  , l_c[0], l_c[1])
                o_foot  = self.centercrop(o_foot , s_c[0], s_c[1])
                o_arm   = o_arm.reshape(-1, 1,  a_c[0], a_c[1])              # [b*4, 1, 200,  96]
                o_leg   = o_leg.reshape(-1, 1,  l_c[0], l_c[1])              # [b*4, 1, 224, 112]
                o_foot  = o_foot.reshape(-1, 1, s_c[0], s_c[1])              # [b*4, 1, 224, 112]

                o_face  = self.centerpadding(o_face , ho, wo)         # [b, 1, 1024, 1024]
                o_upper = self.centerpadding(o_upper , ho, wo)        # [b, 1, 1024, 1024]
                o_arm   = self.centerpadding(o_arm , ho, wo)          # [b*4, 1, 1024, 1024]
                o_leg   = self.centerpadding(o_leg , ho, wo)          # [b*4, 1, 1024, 1024]
                o_foot  = self.centerpadding(o_foot , ho, wo)          # [b*4, 1, 1024, 1024]

                grid    = F.affine_grid(inv_z[:,0,:,:], [b,1,ho,wo])
                o_face  = F.grid_sample(o_face, grid, mode='bilinear')                      # [b, 1, 1024, 1024] 
                grid    = F.affine_grid(inv_z[:,1,:,:], [b,1,ho,wo])
                o_upper = F.grid_sample(o_upper, grid, mode='bilinear')                     # [b, 1, 1024, 1024]
                grid    = F.affine_grid(inv_z[:,2:6,:,:].reshape(-1, 2, 3), [b*4, 1,ho,wo])
                o_arm   = F.grid_sample(o_arm, grid, mode='bilinear').view([-1, 4, ho, wo]) # [b, 4, 1024, 1024]
                grid    = F.affine_grid(inv_z[:,6:10,:,:].reshape(-1, 2, 3),  [b*4, 1,ho,wo])
                o_leg   = F.grid_sample(o_leg, grid, mode='bilinear').view([-1, 4, ho, wo]) # [b, 4, 1024, 1024]
                grid    = F.affine_grid(inv_z[:,10:12,:,:].reshape(-1, 2, 3),  [b*2, 1,ho,wo])
                o_foot  = F.grid_sample(o_foot, grid, mode='bilinear').view([-1, 2, ho, wo]) # [b, 4, 1024, 1024]

                o = torch.cat((o_face, o_upper, o_arm, o_leg, o_foot), dim=1)    # [b, 10, 1024, 1024]
                o = torch.max(o, dim=1, keepdim=True).values             # [b, 1, 1024, 1024] 
                o = torch.squeeze(o, dim=1)                                     # [b, 1024, 1024]

                # Crop & Downsample Image 
                f_i_f = x1[:, 0, :, :, :]                           # [b,   3, 1024, 1024]
                u_i_f = x1[:, 1, :, :, :]                           # [b,   3, 1024, 1024]
                a_i_f = x1[:, 2:6, :, :, :].reshape(-1, 3, ho, wo)  # [b*4, 3, 1024, 1024]
                l_i_f = x1[:, 6:10, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 1024, 1024]
                s_i_f = x1[:, 10:12, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 1024, 1024]

                f_i_f = self.centercrop(f_i_f, f_c[0], f_c[1])          # [b,   3, 176, 160]    
                u_i_f = self.centercrop(u_i_f, u_c[0], u_c[1])          # [b,   3, 320, 256]
                a_i_f = self.centercrop(a_i_f, a_c[0], a_c[1])          # [b*4, 3, 200,  96]
                l_i_f = self.centercrop(l_i_f, l_c[0], l_c[1])          # [b*4, 3, 224, 112]    
                s_i_f = self.centercrop(s_i_f, s_c[0], s_c[1])          # [b*4, 3, 224, 112]   
                d_i_f = F.interpolate(x, [ho//8, wo//8], mode='bilinear')  # [b,   3, 256, 256] 
                
                o_temp = torch.cat((o_face, o_upper, o_arm, o_leg, o_foot), dim=1)    # [b, 10, 1024, 1024]
                o_temp = torch.sum(o_temp, dim=1, keepdim=True)             # [b, 1, 1024, 1024] 
                o_temp = torch.squeeze(o_temp, dim=1)                                     # [b, 1024, 1024]

            # pred down normal
            d_n = self.img2normal_down  (d_i_f)

            with torch.no_grad ():
                # pred part normal & down depth
                o_f  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048] # bilinear
                grid = F.affine_grid(z[:,0,:,:].reshape(-1, 2, 3), [b*1, 1,ho,wo])              
                o_f  = o_f[:,:3, :,:].unsqueeze(dim=1).expand(-1, 1, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_f  = F.grid_sample(o_f, grid)
                o_f  = self.centercrop(o_f, f_c[0], f_c[1])

                o_u  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048] # bilinear
                grid = F.affine_grid(z[:,1,:,:].reshape(-1, 2, 3), [b*1, 1,ho,wo])              
                o_u  = o_u[:,:3, :,:].unsqueeze(dim=1).expand(-1, 1, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_u  = F.grid_sample(o_u, grid)
                o_u  = self.centercrop(o_u, u_c[0], u_c[1])

                o_a  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048] # bilinear
                grid = F.affine_grid(z[:,2:6,:,:].reshape(-1, 2, 3), [b*4, 1,ho,wo])              
                o_a  = o_a[:,:3, :,:].unsqueeze(dim=1).expand(-1, 4, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_a  = F.grid_sample(o_a, grid)
                o_a  = self.centercrop(o_a, a_c[0], a_c[1])

                o_l  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048] # bilinear
                grid = F.affine_grid(z[:,6:10,:,:].reshape(-1, 2, 3), [b*4, 1,ho,wo])              
                o_l  = o_l[:,:3, :,:].unsqueeze(dim=1).expand(-1, 4, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_l  = F.grid_sample(o_l, grid)
                o_l  = self.centercrop(o_l, l_c[0], l_c[1])

                o_s  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048] # bilinear
                grid = F.affine_grid(z[:,10:12,:,:].reshape(-1, 2, 3), [b*2, 1,ho,wo])              
                o_s  = o_s[:,:3, :,:].unsqueeze(dim=1).expand(-1, 2, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_s  = F.grid_sample(o_s, grid)
                o_s  = self.centercrop(o_s, s_c[0], s_c[1])
                
                f_i_f = torch.cat((f_i_f, o_f), dim=1)
                u_i_f = torch.cat((u_i_f, o_u), dim=1)
                a_i_f = torch.cat((a_i_f, o_a), dim=1)
                l_i_f = torch.cat((l_i_f, o_l), dim=1)
                s_i_f = torch.cat((s_i_f, o_s), dim=1)

            f_n = self.img2normal_face  (f_i_f)
            u_n = self.img2normal_upper (u_i_f)
            a_n = self.img2normal_arm   (a_i_f)
            l_n = self.img2normal_leg   (l_i_f)
            s_n = self.img2normal_shoe  (s_i_f)
            d_d = self.ImgNorm_to_Dep   (d_i_f, d_n)

            f_n_f, f_n_b = torch.chunk (f_n, chunks=2, dim=1)     # [b,   3,  144,  128]
            u_n_f, u_n_b = torch.chunk (u_n, chunks=2, dim=1)     # [b,   3,  320,  256]
            a_n_f, a_n_b = torch.chunk (a_n, chunks=2, dim=1)     # [b*4, 3,  160,   80]
            l_n_f, l_n_b = torch.chunk (l_n, chunks=2, dim=1)     # [b*4, 3,  224,  112]
            s_n_f, s_n_b = torch.chunk (s_n, chunks=2, dim=1)     # [b*4, 3,  224,  112]
            d_n_f, d_n_b = torch.chunk (d_n, chunks=2, dim=1)     # [b,   3,  512,  512]
            d_d_f, d_d_b, d_d_m = torch.chunk (d_d, chunks=3, dim=1)
            d_d_f = self.relu6(d_d_f)
            d_d_b = self.relu6(d_d_b)
            d_d_m = self.Sigmoid(d_d_m)

        elif self.phase == 2:
            with torch.no_grad ():
                x1, o, z, inv_z = self.stn(x, init_affine)         # 1024:4545MB

                # 2D Occupancy Grid Map
                o_face  = o[:, 0, :, :, :]                            # [b, 1, 1024, 1024]
                o_upper = o[:, 1, :, :, :]                            # [b, 1, 1024, 1024]
                o_arm   = o[:, 2:6, :, :, :].reshape(-1, 1, ho, wo)   # [b, 4, 1, 1024, 1024]
                o_leg   = o[:, 6:10, :, :, :].reshape(-1, 1, ho, wo)    # [b, 4, 1, 1024, 1024]
                o_foot  = o[:, 10:12, :, :, :].reshape(-1, 1, ho, wo)    # [b, 2, 1, 1024, 1024]

                o_face  = self.centercrop(o_face , f_c[0]-self.k_size, f_c[1]-self.k_size)          # [b, 1, 176, 160]
                o_upper = self.centercrop(o_upper, u_c[0]-self.k_size, u_c[1]-self.k_size)          # [b, 1, 320, 256]
                o_arm   = self.centercrop(o_arm  , a_c[0]-self.k_size, a_c[1]-self.k_size)
                o_leg   = self.centercrop(o_leg  , l_c[0]-self.k_size, l_c[1]-self.k_size)
                o_foot  = self.centercrop(o_foot , s_c[0]-self.k_size, s_c[1]-self.k_size)
                o_arm   = o_arm.reshape (-1, 1, a_c[0]-self.k_size-1, a_c[1]-self.k_size-1)              # [b*4, 1, 200,  96]
                o_leg   = o_leg.reshape (-1, 1, l_c[0]-self.k_size-1, l_c[1]-self.k_size-1)              # [b*4, 1, 224, 112]
                o_foot  = o_foot.reshape(-1, 1, s_c[0]-self.k_size-1, s_c[1]-self.k_size-1)              # [b*4, 1, 224, 112]

                o_face  = self.centerpadding(o_face , ho, wo)         # [b, 1, 1024, 1024]
                o_upper = self.centerpadding(o_upper , ho, wo)        # [b, 1, 1024, 1024]
                o_arm   = self.centerpadding(o_arm , ho, wo)          # [b*4, 1, 1024, 1024]
                o_leg   = self.centerpadding(o_leg , ho, wo)          # [b*4, 1, 1024, 1024]
                o_foot  = self.centerpadding(o_foot , ho, wo)          # [b*4, 1, 1024, 1024]

                o_face  = self.transform(o_face)
                o_upper = self.transform(o_upper)
                o_arm   = self.transform(o_arm)
                o_leg   = self.transform(o_leg)
                o_foot  = self.transform(o_foot)

                grid    = F.affine_grid(inv_z[:,0,:,:], [b,1,ho,wo])
                o_face  = F.grid_sample(o_face, grid, mode='bilinear')                      # [b, 1, 1024, 1024]   
                grid    = F.affine_grid(inv_z[:,1,:,:], [b,1,ho,wo])
                o_upper = F.grid_sample(o_upper, grid, mode='bilinear')                     # [b, 1, 1024, 1024]
                grid    = F.affine_grid(inv_z[:,2:6,:,:].reshape(-1, 2, 3), [b*4, 1,ho,wo])
                o_arm   = F.grid_sample(o_arm, grid, mode='bilinear').view([-1, 4, ho, wo]) # [b, 4, 1024, 1024]
                grid    = F.affine_grid(inv_z[:,6:10,:,:].reshape(-1, 2, 3),  [b*4, 1,ho,wo])
                o_leg   = F.grid_sample(o_leg, grid, mode='bilinear').view([-1, 4, ho, wo]) # [b, 4, 1024, 1024]
                grid    = F.affine_grid(inv_z[:,10:12,:,:].reshape(-1, 2, 3),  [b*2, 1,ho,wo])
                o_foot  = F.grid_sample(o_foot, grid, mode='bilinear').view([-1, 2, ho, wo]) # [b, 4, 1024, 1024]

                o = torch.cat((o_face, o_upper, o_arm, o_leg, o_foot), dim=1)    # [b, 10, 1024, 1024]
                o_div = torch.sum(o, dim=1, keepdim=True)
                o_div[o_div == 0] = 1
                o = o / o_div
                
                o_show = torch.sum(o, dim=1, keepdim=True)             # [b, 1, 1024, 1024] 
                o_show = torch.squeeze(o_show, dim=1)    

                # Crop & Downsample Image 
                f_i_f = x1[:, 0, :, :, :]                           # [b,   3, 1024, 1024]
                u_i_f = x1[:, 1, :, :, :]                           # [b,   3, 1024, 1024]
                a_i_f = x1[:, 2:6, :, :, :].reshape(-1, 3, ho, wo)  # [b*4, 3, 1024, 1024]
                l_i_f = x1[:, 6:10, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 1024, 1024]
                s_i_f = x1[:, 10:12, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 1024, 1024]

                f_i_f = self.centercrop(f_i_f, f_c[0], f_c[1])          # [b,   3, 176, 160]    
                u_i_f = self.centercrop(u_i_f, u_c[0], u_c[1])          # [b,   3, 320, 256]
                a_i_f = self.centercrop(a_i_f, a_c[0], a_c[1])          # [b*4, 3, 200,  96]
                l_i_f = self.centercrop(l_i_f, l_c[0], l_c[1])          # [b*4, 3, 224, 112]    
                s_i_f = self.centercrop(s_i_f, s_c[0], s_c[1])          # [b*4, 3, 224, 112]   
                d_i_f = F.interpolate(x, [ho//8, wo//8], mode='bilinear')  # [b,   3, 256, 256] 
                
                d_n = self.img2normal_down  (d_i_f)

                o_f  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048]
                grid = F.affine_grid(z[:,0,:,:].reshape(-1, 2, 3), [b*1, 1,ho,wo])              
                o_f  = o_f[:,:3, :,:].unsqueeze(dim=1).expand(-1, 1, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_f  = F.grid_sample(o_f, grid)
                o_f  = self.centercrop(o_f, f_c[0], f_c[1])

                o_u  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048]
                grid = F.affine_grid(z[:,1,:,:].reshape(-1, 2, 3), [b*1, 1,ho,wo])              
                o_u  = o_u[:,:3, :,:].unsqueeze(dim=1).expand(-1, 1, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_u  = F.grid_sample(o_u, grid)
                o_u  = self.centercrop(o_u, u_c[0], u_c[1])

                o_a  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048]
                grid = F.affine_grid(z[:,2:6,:,:].reshape(-1, 2, 3), [b*4, 1,ho,wo])              
                o_a  = o_a[:,:3, :,:].unsqueeze(dim=1).expand(-1, 4, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_a  = F.grid_sample(o_a, grid)
                o_a  = self.centercrop(o_a, a_c[0], a_c[1])

                o_l  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048]
                grid = F.affine_grid(z[:,6:10,:,:].reshape(-1, 2, 3), [b*4, 1,ho,wo])              
                o_l  = o_l[:,:3, :,:].unsqueeze(dim=1).expand(-1, 4, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_l  = F.grid_sample(o_l, grid)
                o_l  = self.centercrop(o_l, l_c[0], l_c[1])

                o_s  = F.interpolate(d_n, [ho, wo], mode='bilinear') # [b,6,2048,2048]
                grid = F.affine_grid(z[:,10:12,:,:].reshape(-1, 2, 3), [b*2, 1,ho,wo])              
                o_s  = o_s[:,:3, :,:].unsqueeze(dim=1).expand(-1, 2, -1, -1, -1).reshape(-1, 3, ho, wo)
                o_s  = F.grid_sample(o_s, grid)
                o_s  = self.centercrop(o_s, s_c[0], s_c[1])

                f_i_f = torch.cat((f_i_f, o_f), dim=1)
                u_i_f = torch.cat((u_i_f, o_u), dim=1)
                a_i_f = torch.cat((a_i_f, o_a), dim=1)
                l_i_f = torch.cat((l_i_f, o_l), dim=1)
                s_i_f = torch.cat((s_i_f, o_s), dim=1)

                f_n = self.img2normal_face  (f_i_f)
                u_n = self.img2normal_upper (u_i_f)
                a_n = self.img2normal_arm   (a_i_f)
                l_n = self.img2normal_leg   (l_i_f)
                s_n = self.img2normal_shoe  (s_i_f)
                d_d = self.ImgNorm_to_Dep   (d_i_f, d_n) # [B, C, H, W]
                    
                f_n_f, f_n_b = torch.chunk (f_n, chunks=2, dim=1)     # [b,   3,  144,  128]
                u_n_f, u_n_b = torch.chunk (u_n, chunks=2, dim=1)     # [b,   3,  320,  256]
                a_n_f, a_n_b = torch.chunk (a_n, chunks=2, dim=1)     # [b*4, 3,  160,   80]
                l_n_f, l_n_b = torch.chunk (l_n, chunks=2, dim=1)     # [b*4, 3,  224,  112]
                s_n_f, s_n_b = torch.chunk (s_n, chunks=2, dim=1)     # [b*4, 3,  224,  112]
                d_n_f, d_n_b = torch.chunk (d_n, chunks=2, dim=1)     # [b,   3,  512,  512]
                d_d_f, d_d_b, d_d_m = torch.chunk (d_d, chunks=3, dim=1)
                d_d_f = self.relu6(d_d_f)
                d_d_b = self.relu6(d_d_b)
                d_d_m = self.Sigmoid(d_d_m)
                
                d_d_m_temp = (d_d_m>0.5).float()
                d_d_f[d_d_m_temp==0] = 6
                d_d_b = d_d_b * d_d_m_temp 
                d_d_fb = torch.cat((d_d_f, d_d_b), dim=1)

            n, d = self.refine(d_d_fb, f_n, u_n, a_n, l_n, s_n, o, inv_z)
            
            n_f,   n_b            = torch.chunk (n, chunks=2, dim=1)
            d_f,   d_b  , d_m     = torch.chunk (d, chunks=3, dim=1)
            d_f = self.relu6(d_f)
            d_b = self.relu6(d_b)
            d_m = self.Sigmoid(d_m)
            
            # d_m_temp = (d_m>0.5).float()
            # d_f[d_m_temp==0] = 6
            # d_b = d_b * d_m_temp             

        return {'pred_face_normal_front'  : f_n_f, 'pred_face_normal_back'  : f_n_b, 
                'pred_upper_normal_front' : u_n_f, 'pred_upper_normal_back' : u_n_b, 
                'pred_arm_normal_front'   : a_n_f, 'pred_arm_normal_back'   : a_n_b, 
                'pred_leg_normal_front'   : l_n_f, 'pred_leg_normal_back'   : l_n_b, 
                'pred_shoe_normal_front'  : s_n_f, 'pred_shoe_normal_back'  : s_n_b, 
                'pred_down_normal_front'  : d_n_f, 'pred_down_normal_back'  : d_n_b, 
                'pred_normal_front'       : n_f,   'pred_normal_back'       : n_b,
                'pred_depth_front'        : d_f,   'pred_depth_back'        : d_b, 'pred_depth_mask' : d_m, 
                'pred_down_depth_front'   : d_d_f, 'pred_down_depth_back'   : d_d_b, 'pred_down_depth_mask' : d_d_m, 
                'occupancy' : o_show, 'z' : z, 'inv_z' : inv_z,
                }

def weight_init_basic(m):
    if isinstance (m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_ (0, math.sqrt (2. / n))
    elif isinstance (m, nn.BatchNorm2d):
        m.weight.data.fill_ (1)
        m.bias.data.zero_ ()
    elif isinstance (m, nn.Linear):
        m.bias.data.zero_()
    return m

if __name__ == '__main__':
    print('')