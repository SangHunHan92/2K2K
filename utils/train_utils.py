import os
import torch.distributed as dist
import torch
import shutil
import torchvision
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def init_variables(image, front_depth, back_depth, mask, init_affine, device=None):
    if device is not None:
        if image is not None:
            image = image.to(device)
        if front_depth is not None:
            front_depth = front_depth.to(device)
        if back_depth is not None:
            back_depth = back_depth.to(device)
        if mask is not None:
            mask = mask.to(device)
        if init_affine is not None:
            init_affine = init_affine.to(device)

        if image is not None:
            image = torch.autograd.Variable(image)
        if front_depth is not None:
            front_depth = torch.autograd.Variable(front_depth)
        if back_depth is not None:
            back_depth = torch.autograd.Variable(back_depth)
        if mask is not None:
            mask = torch.autograd.Variable(mask)
        if init_affine is not None:
            init_affine = torch.autograd.Variable(init_affine)

    return image, front_depth, back_depth, mask, init_affine

# ddp related functions.
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    # initialize the process group
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    
def ddp_cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, current_epoch, best_loss, is_best, 
                    ckpt_path='/workspace/code/checkpoints/save_path/', 
                    model_name='human_recon', exp_name='', use_dp=False, use_ddp=False):
    # model_name = args.model_name.lower () + args.exp_name
    sub_dir = model_name.lower () + '_' + exp_name

    # check directories.
    if not os.path.exists(os.path.join(ckpt_path, sub_dir)):
        os.makedirs(os.path.join(ckpt_path, sub_dir))
        os.chmod(ckpt_path, 0o777)
        os.chmod(os.path.join(ckpt_path, sub_dir), 0o777)

    if torch.cuda.device_count() > 1 and (use_dp or use_ddp) :
        state = {'epoch': current_epoch, 'model': model_name, 'best_loss': best_loss, 
        'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    else : 
        state = {'epoch': current_epoch, 'model': model_name, 'best_loss': best_loss, 
        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}

    filename = os.path.join(ckpt_path, sub_dir,
                            '%s_model_epoch%03d_loss%0.4f.pth.tar' % (exp_name, current_epoch, best_loss))
    torch.save(state, filename)
    os.chmod(filename, 0o777)

    # save the best results within the directory.
    if is_best is True:
        if not os.path.exists (ckpt_path):
            os.makedirs (ckpt_path)
        if not os.path.exists (os.path.join (ckpt_path, sub_dir)):
            os.makedirs (os.path.join (ckpt_path, sub_dir))

        best_name = os.path.join(ckpt_path, sub_dir, '%s_model_best.pth.tar' % (exp_name) )
        shutil.copyfile(filename, best_name)


# write logs for tensorboard.
def write_summary(logger, loss_builder, loss, input_var, pred_var, target_show, data_path, phase,
                  epoch, index, is_train=True, full_logging=True, lr=None, loss_conf='all'):

    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD  = [0.5, 0.5, 0.5]

    if is_train is True:
        mode = 'train'
    else:
        mode = 'val'

    logger.add_scalar(mode + '/loss', loss.data, epoch)

    text = ' '
    for path in data_path:
        text += path + '\n'
    logger.add_text(mode + '/data_path', text, epoch * index)

    if lr is not None:
        logger.add_scalar(mode + '/lr', lr, epoch)

    show_list  = loss_builder.loss_phase[phase-1]
    show_list2 = loss_builder.show[phase-1]
    view = False

    image = input_var['image']
    mask  = target_show['mask']
    pred_depth_mask        = pred_var['pred_depth_mask']
    pred_down_depth_mask   = pred_var['pred_down_depth_mask']
    target_down_depth_mask = target_show['down_depth_mask']
    target_depth_mask      = target_show['full_depth_mask']

    # Pred
    pred_face_normal_front  = pred_var['pred_face_normal_front']
    pred_face_normal_back   = pred_var['pred_face_normal_back']
    pred_upper_normal_front = pred_var['pred_upper_normal_front']
    pred_upper_normal_back  = pred_var['pred_upper_normal_back']
    pred_arm_normal_front   = pred_var['pred_arm_normal_front']
    pred_arm_normal_back    = pred_var['pred_arm_normal_back']
    pred_leg_normal_front   = pred_var['pred_leg_normal_front']
    pred_leg_normal_back    = pred_var['pred_leg_normal_back']
    pred_shoe_normal_front  = pred_var['pred_shoe_normal_front']
    pred_shoe_normal_back   = pred_var['pred_shoe_normal_back']
    pred_down_normal_front  = pred_var['pred_down_normal_front']
    pred_down_normal_back   = pred_var['pred_down_normal_back']
    pred_normal_front       = pred_var['pred_normal_front']       # phase 2
    pred_normal_back        = pred_var['pred_normal_back']        # phase 2

    pred_face_depth_front   = None
    pred_face_depth_back    = None
    pred_upper_depth_front  = None
    pred_upper_depth_back   = None
    pred_arm_depth_front    = None
    pred_arm_depth_back     = None
    pred_leg_depth_front    = None
    pred_leg_depth_back     = None
    pred_shoe_depth_front   = None
    pred_shoe_depth_back    = None
    pred_down_depth_front   = pred_var['pred_down_depth_front']
    pred_down_depth_back    = pred_var['pred_down_depth_back']
    pred_down_depth_mask    = pred_var['pred_down_depth_mask']
    pred_depth_front        = pred_var['pred_depth_front']
    pred_depth_back         = pred_var['pred_depth_back']
    pred_depth_mask         = pred_var['pred_depth_mask']
    pred_1_depth_front      = None
    pred_1_depth_back       = None
    pred_2_depth_front      = None
    pred_2_depth_back       = None

    pred_face_image_back    = None
    pred_upper_image_back   = None
    pred_arm_image_back     = None
    pred_leg_image_back     = None
    pred_down_image_back    = None

    occupancy               = pred_var['occupancy']

    # Target
    # target_normal = target_var['target_normal']
    # target_depth  = target_var['target_depth']
    # target_img    = target_var['target_img']
    # mask          = target_var['mask']
    # pred_con      = target_var['pred_con']
    # target_con    = target_var['target_con']
    # show          = target_var['show']

    target_face_normal_front  = target_show['t_f_n_f']
    target_upper_normal_front = target_show['t_u_n_f']
    target_arm_normal_front   = target_show['t_a_n_f']
    target_leg_normal_front   = target_show['t_l_n_f']
    target_shoe_normal_front  = target_show['t_s_n_f']
    target_down_normal_front  = target_show['t_d_n_f']
    target_face_normal_back   = target_show['t_f_n_b']
    target_upper_normal_back  = target_show['t_u_n_b']
    target_arm_normal_back    = target_show['t_a_n_b']
    target_leg_normal_back    = target_show['t_l_n_b']
    target_shoe_normal_back   = target_show['t_s_n_b']
    target_down_normal_back   = target_show['t_d_n_b']
    target_normal_front       = target_show['t_n_f']
    target_normal_back        = target_show['t_n_b']
    target_1_normal_front     = None
    target_1_normal_back      = None
    target_2_normal_front     = None
    target_2_normal_back      = None

    # not use
    target_face_depth_front   = None
    target_upper_depth_front  = None
    target_arm_depth_front    = None
    target_leg_depth_front    = None
    target_shoe_depth_front   = None
    target_face_depth_back    = None
    target_upper_depth_back   = None
    target_arm_depth_back     = None
    target_leg_depth_back     = None
    target_shoe_depth_back    = None 

    target_down_depth_front   = target_show['t_d_d_f']
    target_down_depth_back    = target_show['t_d_d_b']
    target_depth_front        = target_show['t_d_f']
    target_depth_back         = target_show['t_d_b']
    target_1_depth_front      = None
    target_1_depth_back       = None
    target_2_depth_front      = None
    target_2_depth_back       = None

    # not use
    con_face_normal_front     = None
    con_upper_normal_front    = None
    con_arm_normal_front      = None
    con_leg_normal_front      = None
    con_shoe_depth_front      = None
    con_face_normal_back      = None
    con_upper_normal_back     = None
    con_arm_normal_back       = None
    con_leg_normal_back       = None
    con_shoe_depth_back       = None
    
    con_down_normal_front     = target_show['s_d_n_f']
    con_down_normal_back      = target_show['s_d_n_b']
    con_normal_front          = target_show['s_n_f']
    con_normal_back           = target_show['s_n_b']
    con_1_normal_front        = None
    con_1_normal_back         = None
    con_2_normal_front        = None
    con_2_normal_back         = None

    # not use
    target_face_image_back    = None
    target_upper_image_back   = None
    target_arm_image_back     = None
    target_leg_image_back     = None
    target_down_image_back    = None
        
    im_num = image.shape[0]

    # image = torch.unsqueeze(image, 1)
    image = torchvision.utils.make_grid (image[0:im_num, :, :, :], normalize=False, scale_each=True)
    image = (image.detach().cpu().numpy() * 0.5 + 0.5) 
    # image = ((np.transpose(image.detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0).astype(int) # RGB to BGR
    # image = image.cpu ().detach ()
    # image = image * torch.Tensor(RGB_STD).view(3, 1, 1) + torch.Tensor(RGB_MEAN).view(3, 1, 1)
    # image = image.numpy ()
    # image = image.transpose(1,2,0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.transpose(2,0,1)
    logger.add_image (mode + '/image', image, epoch * index)

    # Mask
    if mask is not None:
        # mask = torch.unsqueeze(mask, 1)
        mask = torchvision.utils.make_grid (mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        mask = mask.cpu ().detach ().numpy ()
        logger.add_image (mode + '/mask', mask, epoch * index)

    if pred_depth_mask is not None:
        # mask = torch.unsqueeze(pred_depth_mask, 1)
        mask = torchvision.utils.make_grid (pred_depth_mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        mask = mask.cpu ().detach ().numpy ()
        logger.add_image (mode + '/pred_depth_mask', mask, epoch * index)
    
    if pred_down_depth_mask is not None:
        # mask = torch.unsqueeze(pred_down_depth_mask, 1)
        mask = torchvision.utils.make_grid (pred_down_depth_mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        mask = mask.cpu ().detach ().numpy ()
        logger.add_image (mode + '/pred_down_depth_mask', mask, epoch * index)
    
    if target_down_depth_mask is not None:
        # mask = torch.unsqueeze(target_down_depth_mask, 1)
        mask = torchvision.utils.make_grid (target_down_depth_mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        mask = mask.cpu ().detach ().numpy ()
        logger.add_image (mode + '/target_down_depth_mask', mask, epoch * index)
    
    if target_depth_mask is not None:
        # mask = torch.unsqueeze(target_depth_mask, 1)
        mask = torchvision.utils.make_grid (target_depth_mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        mask = mask.cpu ().detach ().numpy ()
        logger.add_image (mode + '/target_depth_mask', mask, epoch * index)
   
    if occupancy is not None:
        occupancy = torch.unsqueeze(occupancy, 1)
        occupancy = torchvision.utils.make_grid (occupancy[0:im_num, :, :, :], normalize=True, scale_each=True)
        occupancy = occupancy.cpu ().detach ().numpy ()
        logger.add_image (mode + '/occupancy', occupancy, epoch * index)

    # Normal, pred
    if pred_face_normal_front is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_face_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_normal_front_pred', pred_normal_grid, epoch * index)

    if pred_upper_normal_front is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_upper_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_normal_front_pred', pred_normal_grid, epoch * index)

    if pred_arm_normal_front is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_arm_normal_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_normal_front_pred', pred_normal_grid, epoch * index)

    if pred_leg_normal_front is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_leg_normal_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_normal_front_pred', pred_normal_grid, epoch * index)

    # if pred_hand_normal_front is not None and 'part_normal' in show_list:
    #     pred_normal_grid = torchvision.utils.make_grid (pred_hand_normal_front[0:im_num*2, :, :, :], normalize=True, scale_each=True)
    #     pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
    #     logger.add_image (mode + '/hand_normal_front_pred', pred_normal_grid, epoch * index)

    if pred_shoe_normal_front is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_shoe_normal_front[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_normal_front_pred', pred_normal_grid, epoch * index)
    
    if pred_down_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (pred_down_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_normal_front_pred', pred_normal_grid, epoch * index)


    if pred_face_normal_back is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_face_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_normal_back_pred', pred_normal_grid, epoch * index)

    if pred_upper_normal_back is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_upper_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_normal_back_pred', pred_normal_grid, epoch * index)

    if pred_arm_normal_back is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_arm_normal_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_normal_back_pred', pred_normal_grid, epoch * index)

    if pred_leg_normal_back is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_leg_normal_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_normal_back_pred', pred_normal_grid, epoch * index)

    # if pred_hand_normal_back is not None and 'part_normal' in show_list:
    #     pred_normal_grid = torchvision.utils.make_grid (pred_hand_normal_back[0:im_num*2, :, :, :], normalize=True, scale_each=True)
    #     pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
    #     logger.add_image (mode + '/hand_normal_back_pred', pred_normal_grid, epoch * index)

    if pred_shoe_normal_back is not None and ('part_normal' in show_list or 'part_normal' in show_list2):
        pred_normal_grid = torchvision.utils.make_grid (pred_shoe_normal_back[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_normal_back_pred', pred_normal_grid, epoch * index)
    
    if pred_down_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (pred_down_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_normal_back_pred', pred_normal_grid, epoch * index)

    if pred_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (pred_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_front_pred', pred_normal_grid, epoch * index)

    if pred_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (pred_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_back_pred', pred_normal_grid, epoch * index)
    

    # Normal, target
    if target_face_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_face_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_normal_front_target', target_normal_grid, epoch * index)
    
    if target_upper_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_upper_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_normal_front_target', target_normal_grid, epoch * index)
    
    if target_arm_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_arm_normal_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_normal_front_target', target_normal_grid, epoch * index)
    
    if target_leg_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_leg_normal_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_normal_front_target', target_normal_grid, epoch * index)

    # if target_hand_normal_front is not None and 'part_normal' in show_list:
    #     target_normal_grid = torchvision.utils.make_grid (target_hand_normal_front[0:im_num*2, :, :, :], normalize=True, scale_each=True)
    #     target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
    #     logger.add_image (mode + '/hand_normal_front_target', target_normal_grid, epoch * index)

    if target_shoe_normal_front is not None and 'part_normal' in show_list:
        target_normal_grid = torchvision.utils.make_grid (target_shoe_normal_front[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_normal_front_target', target_normal_grid, epoch * index)
    
    if target_down_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_down_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_normal_front_target', target_normal_grid, epoch * index)
    

    if target_face_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_face_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_normal_back_target', target_normal_grid, epoch * index)
    
    if target_upper_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_upper_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_normal_back_target', target_normal_grid, epoch * index)
    
    if target_arm_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_arm_normal_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_normal_back_target', target_normal_grid, epoch * index)
    
    if target_leg_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_leg_normal_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_normal_back_target', target_normal_grid, epoch * index)

    # if target_hand_normal_back is not None and 'part_normal' in show_list:
    #     target_normal_grid = torchvision.utils.make_grid (target_hand_normal_back[0:im_num*2, :, :, :], normalize=True, scale_each=True)
    #     target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
    #     logger.add_image (mode + '/hand_normal_back_target', target_normal_grid, epoch * index)

    if target_shoe_normal_back is not None and 'part_normal' in show_list:
        target_normal_grid = torchvision.utils.make_grid (target_shoe_normal_back[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_normal_back_target', target_normal_grid, epoch * index)
    
    if target_down_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_down_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_normal_back_target', target_normal_grid, epoch * index)
    

    if target_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_front_target', target_normal_grid, epoch * index)
    
    if target_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_back_target', target_normal_grid, epoch * index)

    if target_1_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_1_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_1_front_target', target_normal_grid, epoch * index)
    
    if target_1_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_1_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_1_back_target', target_normal_grid, epoch * index)        

    if target_2_normal_front is not None:
        target_normal_grid = torchvision.utils.make_grid (target_2_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_2_front_target', target_normal_grid, epoch * index)
    
    if target_2_normal_back is not None:
        target_normal_grid = torchvision.utils.make_grid (target_2_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        target_normal_grid = target_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_2_back_target', target_normal_grid, epoch * index)

    # Depth, pred
    if pred_face_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_face_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_depth_front_pred', pred_grid, epoch * index)

    if pred_upper_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_upper_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_depth_front_pred', pred_grid, epoch * index)

    if pred_arm_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_arm_depth_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_depth_front_pred', pred_grid, epoch * index)

    if pred_leg_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_leg_depth_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_depth_front_pred', pred_grid, epoch * index)

    if pred_shoe_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_shoe_depth_front[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_depth_front_pred', pred_grid, epoch * index)

    if pred_down_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_down_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_depth_front_pred', pred_grid, epoch * index)
        

    if pred_face_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_face_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_depth_back_pred', pred_grid, epoch * index)

    if pred_upper_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_upper_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_depth_back_pred', pred_grid, epoch * index)

    if pred_arm_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_arm_depth_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_depth_back_pred', pred_grid, epoch * index)

    if pred_leg_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_leg_depth_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_depth_back_pred', pred_grid, epoch * index)    

    if pred_shoe_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_shoe_depth_back[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_depth_back_pred', pred_grid, epoch * index)

    if pred_down_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_down_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_depth_back_pred', pred_grid, epoch * index)

    if pred_down_depth_mask is not None:
        pred_grid = torchvision.utils.make_grid (pred_down_depth_mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_depth_mask_pred', pred_grid, epoch * index)


    if pred_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_front_pred', pred_grid, epoch * index)
    
    if pred_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_back_pred', pred_grid, epoch * index)

    if pred_depth_mask is not None:
        pred_grid = torchvision.utils.make_grid (pred_depth_mask[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_mask_pred', pred_grid, epoch * index)

    if pred_1_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_1_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_1_front_pred', pred_grid, epoch * index)
    
    if pred_1_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_1_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_1_back_pred', pred_grid, epoch * index)

    if pred_2_depth_front is not None:
        pred_grid = torchvision.utils.make_grid (pred_2_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_2_front_pred', pred_grid, epoch * index)
    
    if pred_2_depth_back is not None:
        pred_grid = torchvision.utils.make_grid (pred_2_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_grid = pred_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_2_back_pred', pred_grid, epoch * index)



    if target_face_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_face_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_depth_front_target', target_grid, epoch * index)
    
    if target_upper_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_upper_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_depth_front_target', target_grid, epoch * index)
        
    if target_arm_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_arm_depth_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_depth_front_target', target_grid, epoch * index)
        
    if target_leg_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_leg_depth_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_depth_front_target', target_grid, epoch * index)

    if target_shoe_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_shoe_depth_front[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_depth_front_target', target_grid, epoch * index)
        
    if target_down_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_down_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_depth_front_target', target_grid, epoch * index)
    

    if target_face_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_face_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_depth_back_target', target_grid, epoch * index)
        
    if target_upper_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_upper_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_depth_back_target', target_grid, epoch * index)
    
    if target_arm_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_arm_depth_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_depth_back_target', target_grid, epoch * index)
    
    if target_leg_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_leg_depth_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_depth_back_target', target_grid, epoch * index)

    if target_shoe_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_shoe_depth_back[0:im_num*2, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/shoe_depth_back_target', target_grid, epoch * index)
    
    if target_down_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_down_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_depth_back_target', target_grid, epoch * index)

    
    if target_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_front_target', target_grid, epoch * index)
    
    if target_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_back_target', target_grid, epoch * index)

    if target_1_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_1_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_1_front_target', target_grid, epoch * index)
    
    if target_1_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_1_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_1_back_target', target_grid, epoch * index)
    
    if target_2_depth_front is not None:
        target_grid = torchvision.utils.make_grid (target_2_depth_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_2_front_target', target_grid, epoch * index)
    
    if target_2_depth_back is not None:
        target_grid = torchvision.utils.make_grid (target_2_depth_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        target_grid = target_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/depth_2_back_target', target_grid, epoch * index)
        

    # Consistance, pred
    if con_face_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_face_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_normal_front_con', pred_normal_grid, epoch * index)
        
    if con_upper_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_upper_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_normal_front_con', pred_normal_grid, epoch * index)
        
    if con_arm_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_arm_normal_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_normal_front_con', pred_normal_grid, epoch * index)
        
    if con_leg_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_leg_normal_front[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_normal_front_con', pred_normal_grid, epoch * index)
        
    if con_down_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_down_normal_front[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_normal_front_con', pred_normal_grid, epoch * index)
        
    if con_face_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_face_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/face_normal_back_con', pred_normal_grid, epoch * index)
        
    if con_upper_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_upper_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/upper_normal_back_con', pred_normal_grid, epoch * index)
        
    if con_arm_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_arm_normal_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/arm_normal_back_con', pred_normal_grid, epoch * index)
        
    if con_leg_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_leg_normal_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/leg_normal_back_con', pred_normal_grid, epoch * index)
        
    if con_down_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_down_normal_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/down_normal_back_con', pred_normal_grid, epoch * index)
        
    if con_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_front_con', pred_normal_grid, epoch * index)
        
    if con_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_back_con', pred_normal_grid, epoch * index)

    if con_1_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_1_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_1_front_con', pred_normal_grid, epoch * index)
        
    if con_1_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_1_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_1_back_con', pred_normal_grid, epoch * index)
    
    if con_2_normal_front is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_2_normal_front[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_2_front_con', pred_normal_grid, epoch * index)
        
    if con_2_normal_back is not None:
        pred_normal_grid = torchvision.utils.make_grid (con_2_normal_back[0:im_num, :, :, :], normalize=False, scale_each=True)
        pred_normal_grid = pred_normal_grid.cpu ().detach ().numpy ()
        logger.add_image (mode + '/normal_2_back_con', pred_normal_grid, epoch * index)
        
    # Image
    if target_face_image_back is not None:
        image = torchvision.utils.make_grid (target_face_image_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/face_image_back_target', image, epoch * index)
        image = torchvision.utils.make_grid (pred_face_image_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/face_image_back_pred', image, epoch * index)

    if target_upper_image_back is not None:
        image = torchvision.utils.make_grid (target_upper_image_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/upper_image_back_target', image, epoch * index)
        image = torchvision.utils.make_grid (pred_upper_image_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/upper_image_back_pred', image, epoch * index)

    if target_arm_image_back is not None:
        image = torchvision.utils.make_grid (target_arm_image_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/arm_image_back_target', image, epoch * index)
        image = torchvision.utils.make_grid (pred_arm_image_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/arm_image_back_pred', image, epoch * index)

    if target_leg_image_back is not None:
        image = torchvision.utils.make_grid (target_leg_image_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/leg_image_back_target', image, epoch * index)
        image = torchvision.utils.make_grid (pred_leg_image_back[0:im_num*4, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/leg_image_back_pred', image, epoch * index)

    if target_down_image_back is not None:
        image = torchvision.utils.make_grid (target_down_image_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/down_image_back_target', image, epoch * index)
        image = torchvision.utils.make_grid (pred_down_image_back[0:im_num, :, :, :], normalize=True, scale_each=True)
        image = image.cpu ().detach ().numpy ()
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2,0,1)
        logger.add_image (mode + '/down_image_back_pred', image, epoch * index)


