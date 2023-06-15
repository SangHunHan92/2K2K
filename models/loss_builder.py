import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM

class LossBank:
    def __init__(self, device=None):  # set loss criteria and options
        super (LossBank).__init__ ()

        self.criterion_l1 = torch.nn.L1Loss ()
        self.criterion_sl1 = torch.nn.SmoothL1Loss ()
        self.criterion_l2 = torch.nn.MSELoss ()
        self.criterion_bce = torch.nn.BCEWithLogitsLoss ()
        self.criterion_ce = torch.nn.CrossEntropyLoss ()
        self.criterion_huber = torch.nn.SmoothL1Loss ()
        self.criterion_ssim_ch1 = SSIM (data_range=1.0, size_average=True, nonnegative_ssim=True, channel=1, win_size=5)
        self.criterion_ssim_ch3 = SSIM (data_range=1.0, size_average=True, nonnegative_ssim=True, channel=3, win_size=5)

        if device is not None and torch.cuda.is_available ():
            self.criterion_l1 = self.criterion_l1.to (device)
            self.criterion_sl1 = self.criterion_sl1.to (device)
            self.criterion_l2 = self.criterion_l2.to (device)
            self.criterion_bce = self.criterion_bce.to (device)
            self.criterion_ce = self.criterion_ce.to (device)
            self.criterion_huber = self.criterion_huber.to (device)
            self.criterion_ssim_ch1 = self.criterion_ssim_ch1.to (device)
            self.criterion_ssim_ch3 = self.criterion_ssim_ch3.to (device)

    # l1 loss
    def get_l1_loss(self, pred, target):
        loss = self.criterion_l1 (pred, target)
        return loss

    def get_sl1_loss(self, pred, target):
        loss = self.criterion_sl1 (pred, target)
        return loss

    # Huber loss
    def get_huber_loss(self, pred, target):
        loss = self.criterion_huber (pred, target)
        return loss

    # l1 loss
    def get_l2_loss(self, pred, target):
        loss = self.criterion_l2 (pred, target)
        return loss

    # binary cross entropy
    def get_bce_loss(self, pred, target):
        loss = self.criterion_bce (pred, target)
        return loss

    def get_ce_loss(self, pred, target):
        loss = self.criterion_ce (pred, target)
        return loss

    def get_ssim_loss(self, pred, target):
        if pred.shape[1] == 1:
            ssim_loss = 1 - self.criterion_ssim_ch1 (pred, target)
        else:
            ssim_loss = 1 - self.criterion_ssim_ch3 (pred, target)
        return ssim_loss

    def get_exist_loss(self, pred, target):
        return self.get_bce_loss (pred, target)

class LossBuilderHuman_2048 (LossBank):
    def __init__(self, device=None):
        LossBank.__init__ (self, device=device)
        self.device = device
        self.build = self.build_loss

        self.loss_phase = [[], []]
        self.loss_phase[0] = ['part_normal', 'down_normal', 'down_depth', 'down_con', 'down_depth_mask']
        self.loss_phase[1] = ['full_depth' , 'full_con', 'full_depth_mask']

        self.show          = [[], []]
        self.show[1]       = ['down_depth', 'down_normal', 'down_con']

    # generator
    def build_loss(self, model, image, front_depth, back_depth, mask, init_affine, phase, epoch, data_path):
        
        b, _, ho, wo = list(image.size())
        _, n,  _,  _ = list(init_affine.size())
        pred_var = model (image, init_affine, epoch, x2 = front_depth, x3=back_depth)

        f_c = [368, 320] #
        u_c = [528, 336] #
        a_c = [352, 224] #
        l_c = [272, 256] #
        s_c = [176, 128] #

        # 1. parsing predictions.
        p_f_n_f = pred_var['pred_face_normal_front']
        p_f_n_b = pred_var['pred_face_normal_back']
        p_u_n_f = pred_var['pred_upper_normal_front']
        p_u_n_b = pred_var['pred_upper_normal_back']
        p_a_n_f = pred_var['pred_arm_normal_front']
        p_a_n_b = pred_var['pred_arm_normal_back']
        p_l_n_f = pred_var['pred_leg_normal_front']
        p_l_n_b = pred_var['pred_leg_normal_back']
        p_s_n_f = pred_var['pred_shoe_normal_front']
        p_s_n_b = pred_var['pred_shoe_normal_back']
        
        p_d_n_f = pred_var['pred_down_normal_front']
        p_d_n_b = pred_var['pred_down_normal_back']
        p_n_f   = pred_var['pred_normal_front']       
        p_n_b   = pred_var['pred_normal_back']

        p_d_f   = pred_var['pred_depth_front']
        p_d_b   = pred_var['pred_depth_back']
        p_d_m   = pred_var['pred_depth_mask']
        p_d_d_f = pred_var['pred_down_depth_front']
        p_d_d_b = pred_var['pred_down_depth_back']
        p_d_d_m = pred_var['pred_down_depth_mask']

        o       = pred_var['occupancy']
        z       = pred_var['z']
        inv_z   = pred_var['inv_z']
                
        # 2. generate GT normal maps and depth maps.
        # Full Depth
        if 'part_normal' in self.loss_phase[phase-1] or 'full_normal' in self.loss_phase[phase-1] or \
           'part_con'    in self.loss_phase[phase-1] or 'full_con'    in self.loss_phase[phase-1] or \
            'part_normal' in self.show[phase-1]:
            t_n_f = get_normal (front_depth)
            if back_depth is not None:
                t_n_b = get_normal (back_depth)
        
        # part_normal calculate
        if 'part_normal' in self.loss_phase[phase-1] or 'part_con' in self.loss_phase[phase-1] or \
            'part_normal' in self.show[phase-1]:
            _t_n_f = t_n_f.unsqueeze(dim=1).expand(-1, n, -1, -1, -1)
            _t_n_f = _t_n_f.reshape(-1, 3, ho, wo)    # [b*n, 3, 512, 512]
            if back_depth is not None:
                _t_n_b = t_n_b.unsqueeze(dim=1).expand(-1, n, -1, -1, -1)
                _t_n_b = _t_n_b.reshape(-1, 3, ho, wo)

            # Affine Target Normal Maps
            grid  = F.affine_grid(z.view(-1, 2, 3), [b*n, 3, ho, wo])   # [b*n, 512, 512, 2]
            _t_n_f = F.grid_sample(_t_n_f, grid, mode = 'bilinear')     # [b*n, 3, 512, 512]
            _t_n_b = F.grid_sample(_t_n_b, grid, mode = 'bilinear')     # [b*n, 3, 512, 512]
            _t_n_f = _t_n_f.view(-1, n, 3, ho, wo)                      # [b, n, 3, 512, 512]
            _t_n_b = _t_n_b.view(-1, n, 3, ho, wo)                      # [b, n, 3, 512, 512]

            # Crop Target Normal Maps
            t_f_n_f  = _t_n_f[:, 0, :, :, :]                           # [b, 3, 512, 512]
            t_u_n_f  = _t_n_f[:, 1, :, :, :]                           # [b, 3, 512, 512]
            t_a_n_f  = _t_n_f[:, 2:6, :, :, :].reshape(-1, 3, ho, wo)  # [b*4, 3, 512, 512]
            t_l_n_f  = _t_n_f[:, 6:10, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 512, 512]
            t_s_n_f  = _t_n_f[:, 10:12, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 512, 512]

            # Target Part Normal
            t_f_n_f  = self.centercrop(t_f_n_f, f_c[0], f_c[1])             # [b, 3, 128, 128]
            t_u_n_f  = self.centercrop(t_u_n_f, u_c[0], u_c[1])             # [b, 3, 256, 192]
            t_a_n_f  = self.centercrop(t_a_n_f, a_c[0], a_c[1])             # [b*4, 3, 200,  96]
            t_l_n_f  = self.centercrop(t_l_n_f, l_c[0], l_c[1])             # [b*4, 3, 208, 112]
            t_s_n_f  = self.centercrop(t_s_n_f, s_c[0], s_c[1])             # [b*4, 3, 208, 112]

            t_f_n_b  = _t_n_b[:, 0, :, :, :]                           # [b, 3, 512, 512]
            t_u_n_b  = _t_n_b[:, 1, :, :, :]                           # [b, 3, 512, 512]
            t_a_n_b  = _t_n_b[:, 2:6, :, :, :].reshape(-1, 3, ho, wo)  # [b*4, 3, 512, 512]
            t_l_n_b  = _t_n_b[:, 6:10, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 512, 512]
            t_s_n_b  = _t_n_b[:, 10:12, :, :, :].reshape(-1, 3, ho, wo)   # [b*4, 3, 512, 512]

            t_f_n_b  = self.centercrop(t_f_n_b, f_c[0], f_c[1])             # [b, 3, 128, 128]
            t_u_n_b  = self.centercrop(t_u_n_b, u_c[0], u_c[1])             # [b, 3, 256, 192]
            t_a_n_b  = self.centercrop(t_a_n_b, a_c[0], a_c[1])             # [b*4, 3, 200,  96]
            t_l_n_b  = self.centercrop(t_l_n_b, l_c[0], l_c[1])             # [b*4, 3, 208, 112]
            t_s_n_b  = self.centercrop(t_s_n_b, s_c[0], s_c[1])             # [b*4, 3, 208, 112]

        # Down Depth
        if 'down_depth' in self.loss_phase[phase-1] or \
           'down_normal' in self.loss_phase[phase-1] or \
           'down_con' in self.loss_phase[phase-1] or \
           'down_depth' in self.show[phase-1] or \
           'down_normal' in self.show[phase-1] or \
           'down_con' in self.show[phase-1]:
            t_d_d_f = front_depth[:, :, 3::8, 3::8]
            t_d_d_b = back_depth[:, :, 3::8, 3::8]
        
        # Down Normal
        if 'down_normal' in self.loss_phase[phase-1] or 'down_con' in self.loss_phase[phase-1] or \
            'down_normal' in self.show[phase-1] or 'down_con' in self.show[phase-1]:
            t_d_n_f = get_normal(t_d_d_f)
            t_d_n_b = get_normal(t_d_d_b)

        # Mask for full depth
        if 'depth_mask' in self.loss_phase[phase-1] or 'full_depth_mask' in self.loss_phase[phase-1]:
            t_d_m   = (mask > 0).float() #check

        # Mask for pred depth
        if 'down_depth' in self.loss_phase[phase-1] or 'down_depth_mask' in self.loss_phase[phase-1]:
            t_d_d_m = (t_d_d_f > 0).float()
        
        # Depth Consistance
        if 'down_con' in self.loss_phase[phase-1] or 'down_con' in self.show[phase-1]:
            s_d_n_f = get_normal(p_d_d_f)
            s_d_n_b = get_normal(p_d_d_b)
            con_mask = (t_d_d_f == 0.0).expand(-1, 3, -1, -1)
            s_d_n_f[con_mask] = 0.0
            s_d_n_b[con_mask] = 0.0
        if 'full_depth' in self.loss_phase[phase-1] or 'full_con' in self.loss_phase[phase-1]:
            s_n_f   = get_normal(p_d_f)
            s_n_b   = get_normal(p_d_b)
            con_mask = (mask == 0.0).expand(-1, 3, -1, -1)
            s_n_f[con_mask] = 0.0
            s_n_b[con_mask] = 0.0            
            ts_n_f = t_n_f
            ts_n_b = t_n_b
            ts_n_f[con_mask] = 0.0
            ts_n_b[con_mask] = 0.0
            
        if 'down_depth' in self.loss_phase[phase-1] or \
           'down_normal' in self.loss_phase[phase-1] or \
           'down_con' in self.loss_phase[phase-1] or \
           'down_depth' in self.show[phase-1] or \
           'down_normal' in self.show[phase-1] or \
           'down_con' in self.show[phase-1]:
            t_d_d_f[t_d_d_f == 0.0] = 6.0
        
        if 'full_depth' in self.loss_phase[phase-1]:
            front_depth[front_depth == 0.0] = 6.0

        # 3. build losses.
        if 'part_normal' in self.loss_phase[phase-1] and 'down_normal' in self.loss_phase[phase-1] and 'full_normal' in self.loss_phase[phase-1]:
            pred_normal   = [ p_f_n_f, p_u_n_f, p_a_n_f, p_l_n_f, p_s_n_f, p_d_n_f, p_n_f,
                              p_f_n_b, p_u_n_b, p_a_n_b, p_l_n_b, p_s_n_b, p_d_n_b, p_n_b ]
            target_normal = [ t_f_n_f, t_u_n_f, t_a_n_f, t_l_n_f, t_s_n_f, t_d_n_f, t_n_f,
                              t_f_n_b, t_u_n_b, t_a_n_b, t_l_n_b, t_s_n_b, t_d_n_b, t_n_b ]
        elif 'part_normal' in self.loss_phase[phase-1] and 'full_normal' in self.loss_phase[phase-1]:
            pred_normal   = [ p_f_n_f, p_u_n_f, p_a_n_f, p_l_n_f, p_s_n_f, p_n_f,
                              p_f_n_b, p_u_n_b, p_a_n_b, p_l_n_b, p_s_n_b, p_n_b ]
            target_normal = [ t_f_n_f, t_u_n_f, t_a_n_f, t_l_n_f, t_s_n_f, t_n_f,
                              t_f_n_b, t_u_n_b, t_a_n_b, t_l_n_b, t_s_n_b, t_n_b ]
        elif 'part_normal' in self.loss_phase[phase-1] and 'down_normal' in self.loss_phase[phase-1]:
            pred_normal   = [ p_f_n_f, p_u_n_f, p_a_n_f, p_l_n_f, p_s_n_f, p_d_n_f, 
                              p_f_n_b, p_u_n_b, p_a_n_b, p_l_n_b, p_s_n_b, p_d_n_b ]
            target_normal = [ t_f_n_f, t_u_n_f, t_a_n_f, t_l_n_f, t_s_n_f, t_d_n_f,
                              t_f_n_b, t_u_n_b, t_a_n_b, t_l_n_b, t_s_n_b, t_d_n_b ]
        elif 'part_normal' in self.loss_phase[phase-1]:
            pred_normal   = [ p_f_n_f, p_u_n_f, p_a_n_f, p_l_n_f, p_s_n_f, 
                              p_f_n_b, p_u_n_b, p_a_n_b, p_l_n_b, p_s_n_b ]
            target_normal = [ t_f_n_f, t_u_n_f, t_a_n_f, t_l_n_f, t_s_n_f,
                              t_f_n_b, t_u_n_b, t_a_n_b, t_l_n_b, t_s_n_b ]
        elif 'down_normal' in self.loss_phase[phase-1]:
            pred_normal   = [ p_d_n_f, p_d_n_b ]
            target_normal = [ t_d_n_f, t_d_n_b ]
        elif 'full_normal' in self.loss_phase[phase-1]:
            pred_normal   = [ p_n_f, p_n_b ]
            target_normal = [ t_n_f, t_n_b ]
        else:
            pred_normal   = []
            target_normal = []

        if 'down_depth' in self.loss_phase[phase-1] and 'full_depth' in self.loss_phase[phase-1]:
            pred_depth   = [ p_d_d_f, p_d_d_b, p_d_f,       p_d_b     ]
            target_depth = [ t_d_d_f, t_d_d_b, front_depth, back_depth ]
        elif 'down_depth' in self.loss_phase[phase-1]:
            pred_depth   = [ p_d_d_f, p_d_d_b, None,        None ]
            target_depth = [ t_d_d_f, t_d_d_b, None,        None ]
        elif 'full_depth' in self.loss_phase[phase-1]:
            pred_depth   = [ None,    None,    p_d_f,       p_d_b      ]
            target_depth = [ None,    None,    front_depth, back_depth ]
        else:
            pred_depth   = [ None,    None,    None,    None  ]
            target_depth = [ None,    None,    None,    None  ]
        
        if 'down_con' in self.loss_phase[phase-1] and 'full_con' in self.loss_phase[phase-1]:
            pred_con     = [ s_d_n_f, s_d_n_b,  s_n_f,  s_n_b ]
            target_con   = [ t_d_n_f, t_d_n_b, ts_n_f, ts_n_b ]
        elif 'down_con' in self.loss_phase[phase-1]:
            pred_con     = [ s_d_n_f, s_d_n_b ]
            target_con   = [ t_d_n_f, t_d_n_b ]
        elif 'full_con' in self.loss_phase[phase-1]:
            pred_con     = [  s_n_f,  s_n_b ]
            target_con   = [ ts_n_f, ts_n_b ]
        else:            
            pred_con     = []
            target_con   = []

        pred_mask = []
        target_mask = []
        if 'down_depth_mask' in self.loss_phase[phase-1]:
            pred_mask.append(p_d_d_m)
            target_mask.append(t_d_d_m)
        if 'full_depth_mask' in self.loss_phase[phase-1]:
            pred_mask.append(p_d_m)
            target_mask.append(t_d_m)

        # 4. calculate losses.
        N_ratio = 0.5
        lambda_N_l1, lambda_N_ssim = [0.85, 0.15]
        lambda_D_l1, lambda_C_l1   = [0.85, 0.15]
        lambda_M_ce = 1.0
        
        loss, lossN, lossD, lossC, lossM = [0, 0, 0, 0, 0]
        if 'part_normal' in self.loss_phase[phase-1] or 'down_normal' in self.loss_phase[phase-1] or 'full_normal' in self.loss_phase[phase-1]:
            for pred, target in zip(pred_normal, target_normal):
                loss_n_l1 = self.get_losses (pred, target, loss_type='l1')
                lossN += loss_n_l1 * lambda_N_l1 * N_ratio
                loss_n_ssim = self.get_ssim_loss (pred, target)
                lossN += loss_n_ssim * lambda_N_ssim * N_ratio
            loss += lossN
        
        if 'down_depth' in self.loss_phase[phase-1] or 'full_depth' in self.loss_phase[phase-1] or 'part_depth' in self.loss_phase[phase-1]:
            for pred, target in zip(pred_depth, target_depth):
                if pred == None: # not pred
                    continue
                loss_d_l1 = self.get_losses (pred, target, loss_type='sl1')
                lossD += loss_d_l1 * lambda_D_l1
            loss += lossD

        if 'down_con' in self.loss_phase[phase-1] or 'full_con' in self.loss_phase[phase-1] or 'part_con' in self.loss_phase[phase-1]:
            for pred, target in zip(pred_con, target_con):
                loss_c_l1 = self.get_losses (pred, target, loss_type='sl1')
                lossC += loss_c_l1 * lambda_C_l1
            loss += lossC

        if 'down_depth_mask' in self.loss_phase[phase-1] or 'full_depth_mask' in self.loss_phase[phase-1]:
            for pred, target in zip(pred_mask, target_mask):
                if pred == None: # not pred
                    pass
                else:
                    loss_m_ce = self.get_losses (pred, target, loss_type='bce')
                    lossM += loss_m_ce * lambda_M_ce
            loss += lossM

        # target show        
        target_show = {}
        target_show['t_f_n_f'], target_show['t_f_n_b'] = [None, None]
        target_show['t_u_n_f'], target_show['t_u_n_b'] = [None, None]
        target_show['t_a_n_f'], target_show['t_a_n_b'] = [None, None]
        target_show['t_l_n_f'], target_show['t_l_n_b'] = [None, None]
        target_show['t_s_n_f'], target_show['t_s_n_b'] = [None, None]

        target_show['t_d_n_f'], target_show['t_d_n_b'] = [None, None]
        target_show['t_d_d_f'], target_show['t_d_d_b'] = [None, None]
        target_show['s_d_n_f'], target_show['s_d_n_b'] = [None, None]

        target_show['t_n_f'],   target_show['t_n_b'] = [None, None]
        target_show['t_d_f'],   target_show['t_d_b'] = [None, None]
        target_show['s_n_f'],   target_show['s_n_b'] = [None, None]

        target_show['t_1_n_f'], target_show['t_1_n_b']   = [None, None]
        target_show['t_1_d_f'], target_show['t_1_d_b']   = [None, None]        
        target_show['s_1_n_f'], target_show['s_1_n_b']   = [None, None]
        target_show['t_2_n_f'], target_show['t_2_n_b']   = [None, None]
        target_show['t_2_d_f'], target_show['t_2_d_b']   = [None, None]
        target_show['s_2_n_f'], target_show['s_2_n_b']   = [None, None]
        
        target_show['down_depth_mask'], target_show['full_depth_mask'] = [None, None]
        target_show['mask'] = mask

        if 'part_normal' in self.loss_phase[phase-1] or 'part_normal' in self.show[phase-1]:
            target_show['t_f_n_f'], target_show['t_f_n_b'] = [t_f_n_f, t_f_n_b]
            target_show['t_u_n_f'], target_show['t_u_n_b'] = [t_u_n_f, t_u_n_b]
            target_show['t_a_n_f'], target_show['t_a_n_b'] = [t_a_n_f, t_a_n_b]
            target_show['t_l_n_f'], target_show['t_l_n_b'] = [t_l_n_f, t_l_n_b]
            target_show['t_s_n_f'], target_show['t_s_n_b'] = [t_s_n_f, t_s_n_b]
        if 'down_normal' in self.loss_phase[phase-1] or 'down_normal' in self.show[phase-1] or \
           'down_con' in self.loss_phase[phase-1] or 'down_con' in self.show[phase-1]:
            target_show['t_d_n_f'], target_show['t_d_n_b'] = [t_d_n_f, t_d_n_b]
        if 'full_normal' in self.loss_phase[phase-1] or 'full_normal' in self.show[phase-1] or \
           'full_con' in self.loss_phase[phase-1] or 'full_con' in self.show[phase-1]:
            target_show['t_n_f'],   target_show['t_n_b']   = [t_n_f,   t_n_b]
        if 'down_depth' in self.loss_phase[phase-1] or 'down_depth' in self.show[phase-1]:
            target_show['t_d_d_f'], target_show['t_d_d_b'] = [t_d_d_f, t_d_d_b]
            target_show['s_d_n_f'], target_show['s_d_n_b'] = [s_d_n_f, s_d_n_b]
        if 'full_depth' in self.loss_phase[phase-1] or 'full_depth' in self.show[phase-1]:
            target_show['t_d_f'],   target_show['t_d_b']   = [front_depth, back_depth]
            target_show['s_n_f'],   target_show['s_n_b']   = [s_n_f, s_n_b]
        if 'down_depth_mask' in self.loss_phase[phase-1] or 'down_depth_mask' in self.show[phase-1]:
            target_show['down_depth_mask'] = t_d_d_m
        if 'full_depth_mask' in self.loss_phase[phase-1] or 'full_depth_mask' in self.show[phase-1]:
            target_show['full_depth_mask'] = t_d_m        

        losses     = { 'lossN'         : lossN, 
                       'lossD'         : lossD,
                       'lossC'         : lossC, 
                       'lossM'         : lossM, }
        input_var  = { 'image'         : image }
        
        return loss, losses, input_var, pred_var, target_show  # for visualization

    # custom loss functions here.
    def get_loss(self, pred, target, weight=None, loss_type='l1', sigma=None):
        if loss_type == 'l1':
            loss = self.get_l1_loss (pred, target)
        elif loss_type == 'sl1':
            loss = self.get_sl1_loss (pred, target)  
        elif loss_type == 'bce':
            loss = self.get_bce_loss (pred, target)
        elif loss_type == 'l2':
            loss = self.get_l2_loss (pred, target)
        elif loss_type == 'ssim':
            loss = self.get_ssim_loss (pred, target)
        elif loss_type == 'seg':
            loss = self.get_exist_loss (pred, target)
        else:
            loss = self.get_l1_loss (pred, target)
        return loss

    def get_losses(self, pred, target, weight=None, loss_type='l1', sigma=None, sigma_weight=0.1):
        loss = 0
        for i, p in enumerate (pred):
            if p is not None and target[i] is not None:
                if sigma is None:
                    if weight is None:
                        loss += self.get_loss (p, target[i], loss_type=loss_type)
                    else:
                        loss += self.get_loss (p, target[i], weight[i], loss_type=loss_type)
                else:
                    loss += self.get_loss (p, target[i], loss_type=loss_type, sigma=sigma[i], weight=sigma_weight)
        return loss
    
    def centercrop(self, data, h, w):
        ch, cw = list(data.size())[-2]//2, list(data.size())[-1]//2
        return data[:, :, ch-h//2:ch+h//2, cw-w//2:cw+w//2]        

# input depth x is normalized by 255.0
def get_normal(x, normalize=True, cut_off=0.2):
    def gradient_x(img):
        img = torch.nn.functional.pad (img, (0, 0, 1, 0), mode="replicate")  # pad a column to the end
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(img):
        img = torch.nn.functional.pad (img, (0, 1, 0, 0), mode="replicate")  # pad a row on the bottom
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    if len(x.shape) == 3:
        x = x.unsqueeze(1)

    if x is None:
        return None

    x = x.float()
    grad_x = gradient_x(x)
    grad_y = gradient_y(x)
    grad_z = torch.ones_like(grad_x) / 255.0
    n = torch.sqrt(torch.pow(grad_x, 2) + torch.pow (grad_y, 2) + torch.pow (grad_z, 2))
    normal = torch.cat ((grad_y / n, grad_x / n, grad_z / n), dim=1)

    normal += 1
    normal /= 2
    if normalize is False:  # false gives 0~255, otherwise 0~1.
        normal *= 255

    # remove normals along the object discontinuities and outside the object.
    normal[x.repeat(1, 3, 1, 1) < cut_off] = 0

    return normal #[batch, 3, x, y]
    
if __name__ == '__main__':
    a = [1, 2, 3]
    b = [4, 5, 6]
    for k, p in enumerate (a):
        print (k)
