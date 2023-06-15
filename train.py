import argparse
import numpy as np
import os
import collections
import models
import utils
import time
import sys
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.train_utils import *
from utils.scheduler import CosineAnnealingWarmUpRestarts
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if sys.platform == 'win32':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=8, help='workers')
parser.add_argument('--print_freq', type=int, default=6, help='num epoch to train')
parser.add_argument('--num_epoch', type=int, default=500, help='num epoch to train')
parser.add_argument('--log_freq', type=int, default=300, help='num epoch to train')
parser.add_argument('--start_epoch', type=int, default=1, help='# to the first epoch')
parser.add_argument('--logs_dir', type=str, default='./logs', help='path to tensorboard')
parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
parser.add_argument('--use_ddp', type=bool, default=False, help='utilize multi-gpus')
parser.add_argument('--use_dp', type=bool, default=True, help='utilize multi-gpus')
parser.add_argument('--is_master', type=bool, default=True, help='indicate whether the currnent is master')

parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='adam learning rate')
# parser.add_argument('--load_ckpt',  type=bool, default=False, help='somewhere in your PC') 
parser.add_argument('--load_ckpt',  type=str, default='ckpt_alldataset_phase2.pth.tar', help='somewhere in your PC') 
# parser.add_argument('--load_ckpt',  type=str, default='ckpt_rp_phase2.pth.tar', help='somewhere in your PC') 
# parser.add_argument('--load_ckpt',  type=str, default='ckpt_thuman2_phase2.pth.tar', help='somewhere in your PC') 
parser.add_argument('--data_path', type=str, default='/workspace/dataset/DATA_2048', help='path to dataset')
# parser.add_argument('--bg_path',   type=str, default='/workspace/dataset/DATA_2048', help='path to dataset')
# parser.add_argument('--data_path', type=str, default='/workspace/dataset/RP_2048', help='path to dataset')
parser.add_argument('--checkpoints_load_path', type=str, default='./checkpoints/', help='path to save checkpoints')
parser.add_argument('--checkpoints_save_path', type=str, default='./checkpoints/save_path/', help='path to save checkpoints')
parser.add_argument('--exp_name', type=str, default='AllData', help='checkpoint name to be saved')
parser.add_argument('--phase', type=int, default=1, help='set training phase')
args = parser.parse_args()

print("Training Options Initialized...")

def train(data_loader, dataset, model, loss_builder, optimizer, scheduler, epoch, 
          is_train=True, phase=1, summary_dir=None, log_freq=40, is_master=True, device=None):

    # set variables.
    loss_batch   = AverageMeter()
    loss_batch_N = AverageMeter()
    loss_batch_D = AverageMeter()
    loss_batch_C = AverageMeter()
    loss_batch_M = AverageMeter()
    batch_time   = AverageMeter()
    data_time    = AverageMeter()
    loss_sum     = 0
    
    if is_train is not True:
        model.eval()
    else:
        model.train()

    # putting log files outside the shared directory (the size becomes huge!)
    if summary_dir is not None:
        logger = SummaryWriter(summary_dir)
        os.chmod(summary_dir, 0o777)

    data_end = time.time()
    iters = len(data_loader)

    with tqdm(enumerate(data_loader)) as pbar:
        for i, datum in pbar:
            # set timers.
            data_time.update(time.time() - data_end)
            batch_end = time.time()

            # fetch images from the loader
            image, front_depth, back_depth, mask, init_affine, data_path = dataset.fetch_output(datum)            

            # initialize variables (in case of multiple images, they are returned as a tuple).
            image, front_depth, back_depth, mask, init_affine = \
                init_variables(image, front_depth, back_depth, mask, init_affine, device=device)

            # compute and update losses.
            loss, losses, input_var, pred_var, target_show \
                = loss_builder.build_loss (model, image, front_depth, back_depth, mask, init_affine, phase, epoch, data_path)

            lossN = losses['lossN']            
            lossD = losses['lossD']
            lossC = losses['lossC']
            lossM = losses['lossM']

            loss_batch.update (loss.data, image.shape[0])
            if lossN:
                loss_batch_N.update (lossN.data, image.shape[0])
            if lossD:
                loss_batch_D.update (lossD.data, image.shape[0])
            if lossC:
                loss_batch_C.update (lossC.data, image.shape[0])
            if lossM:
                loss_batch_M.update (lossM.data, image.shape[0])
            loss_sum = loss_sum + loss_batch.val

            # proceed one step
            if is_train is True:
                optimizer.zero_grad ()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch=(epoch-1 + i/iters))

            # update the batch time
            batch_time.update(time.time() - batch_end)

            if is_master:
                pbar.set_description('[{0}][{1}/{2}] loss: {loss:.4f}, '
                                     'lossN: {lossN:.4f}, lossD: {lossD:.4f}, lossC: {lossC:.4f}, lossM: {lossM:.4f}, lr: {lr:0.10f}'
                                     .format(epoch, i, iters,
                                            loss=loss_batch.val,
                                            lossN=loss_batch_N.val,                                                 
                                            lossD=loss_batch_D.val,
                                            lossC=loss_batch_C.val,
                                            lossM=loss_batch_M.val,
                                            lr=optimizer.param_groups[-1]['lr'],))
            batch_time.reset()
            data_time.reset()
            loss_batch.reset()
            pbar.update(i/iters)

            # save results for tensorboard
            if summary_dir is not None and is_master and (i % log_freq == 0 or i in [0, 1]):
                write_summary (logger, loss_builder, loss, input_var, pred_var, target_show, data_path, 
                                phase, epoch, i, is_train=is_train, lr=optimizer.param_groups[-1]['lr'])
                
    return loss_sum/len(data_loader)

def main():
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

    # 1. Training & GPU settings
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    cudnn.fastest = True    
    
    if args.use_ddp:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if args.local_rank != 0:
            args.is_master = False
    else:
        args.local_rank = 0  # indicates designated gpu id.
        
    torch.cuda.set_device(args.local_rank) # default -1?
    args.device = torch.device("cuda:{}".format(args.local_rank))
    world_size = torch.cuda.device_count ()
    local_batch = args.batch_size
    if args.use_dp:
        local_batch = local_batch // world_size
        
    args.train_list = 'train_test'
    args.val_list = 'val'
    args.bg_list = 'train_split_indoor09_1024'
    args.model_name = 'Model_2K2K'
    args.res = 2048
        
    # 2. Training Model
    model = getattr (models, args.model_name)(args.phase, args.device) #for ATUNet
    dataset = getattr(utils, 'ReconDataset_2048') #
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=5, T_warmup=0.01, decay=0.5)
    scheduler.step(args.start_epoch - 1)
    
    # load checkpoint if required
    if args.load_ckpt and args.is_master:
        ckpt = torch.load(args.checkpoints_load_path + args.load_ckpt) # model : single / ckpt : single & multi
        model_state_dict = collections.OrderedDict( {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()} )
        model.load_state_dict(model_state_dict, strict=False)
        
    if args.phase == 1 :
        pass
    if args.phase == 2 :        
        for param in model.parameters():
            param.requires_grad = False      
        for param in model.refine.parameters():
            param.requires_grad = True
            
    if world_size > 1:
        if args.use_ddp:
            if not torch.distributed.is_initialized():
                ddp_setup (args.local_rank, world_size)
            model.to(args.device)
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        elif args.use_dp:  # data parallel.
            gpu_ids = [k for k in range(world_size)]
            model = DP(model, device_ids=gpu_ids, output_device=gpu_ids[0])
            model.to(args.device)
    else:
        model.to(args.device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                
    loss_builder = getattr(models, 'LossBuilderHuman_2048') (device=args.device)
    
    print('dataset initialize end...')

    train_dataset = dataset(data_path=args.data_path,
                 data_list=args.train_list,
                 is_training=True,
                 bg_path=args.data_path,
                 bg_list=args.bg_list,
                 res=args.res)
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False  # already shuffled.
    else:
        train_sampler = None
        shuffle = True
    
    summary_root = os.path.join (args.logs_dir, args.model_name + '_' + args.exp_name)
    
    best_loss = np.inf
    print('training start')
    for current_epoch in range(args.start_epoch, args.num_epoch):
        
        train_loader = torch.utils.data.DataLoader (
            train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
            sampler=train_sampler, pin_memory=False, drop_last=True)

        if args.use_ddp:
            train_sampler.set_epoch(current_epoch)

        current_loss = train(train_loader, dataset, model, loss_builder, optimizer, scheduler, current_epoch, 
              is_train=True, phase=args.phase, summary_dir=summary_root, log_freq=args.log_freq, is_master=args.is_master, device=args.device) 

        is_best = False
        
        if current_epoch % 1 == 0 or current_epoch == args.num_epoch:
            if args.is_master:
                if best_loss > current_loss:
                    best_loss = current_loss
                is_best = True

                save_checkpoint(model, optimizer, current_epoch, current_loss, is_best, 
                                ckpt_path=args.checkpoints_save_path, 
                                model_name=args.model_name.split('_')[0], exp_name=args.exp_name,
                                use_dp=args.use_dp, use_ddp=args.use_ddp)

    if args.use_ddp:
        ddp_cleanup()

if __name__ == '__main__':
    main()