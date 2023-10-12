

import argparse
import torch

def Configs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, 
                        default='provid the path to the dataset', 
                        help='path to dataset')
    parser.add_argument('--dataset', dest='dataset', type=str, default='csiq',
                        help='Support datasets: clive|koniq|fblive|live|csiq|tid2013')
    parser.add_argument('--svpath', dest='svpath', type=str,
                        default='path to save the results',
                        help='the path to save the info')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=50, 
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=50, 
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--lrratio', dest='lrratio', type=float, default=1, 
                        help='Constant to decrease lr')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, 
                        help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=3, 
                        help='Epochs for training')
    parser.add_argument('--seed', dest='seed', type=int, default=2021, 
                        help='for reproducing the results')
    parser.add_argument('--vesion', dest='vesion', type=int, default=1,
                        help='vesion number for saving')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, 
                        help='Crop size for training & testing image patches')
    parser.add_argument('--droplr', dest='droplr', type=int, default=5, 
                        help='drop lr by every x iteration')   
    parser.add_argument('--gpunum', dest='gpunum', type=str, default='0',
                        help='the id for the gpu that will be used')
    parser.add_argument('--network', dest='network', type=str, default='resnet50',
                        help='the resnet backbone to use')
    parser.add_argument('--nheadt', dest='nheadt', type=int, default=16,
                        help='nheadt in the transformer')
    parser.add_argument('--num_encoder_layerst', dest='num_encoder_layerst', type=int, default=2,
                        help='num encoder layers in the transformer')
    parser.add_argument('--k', dest='k', type=int, default=3,
                        help='num of neighbours to augment network')
    parser.add_argument('--multi_return', dest='multi_return', type=int,
                        help='return k+1 labels or 1')
    parser.add_argument('--resume', dest='resume', type=int, default=0,
                        help='resume current checkpoint')
    parser.add_argument('--stateSnapshot', dest='stateSnapshot', type=str,  
                        help='path to checkpoint if resumed')
    parser.add_argument('--finetune', dest='finetune', type=int,
                        help='finetune head and legs or retrain full model')
    parser.add_argument('--ckpt', dest='ckpt', type=str, 
                        default='', 
                        help='path to ckpt if finetune')
    parser.add_argument('--multi_ranking', dest='multi_ranking', type=int, default=0,
                        help='relative ranking loss for all outputs')
    parser.add_argument('--full_finetune', dest='full_finetune', type=int, default=0,
                        help='if finetune, do we want to change all the internal weights')
    parser.add_argument('--single_channel', dest='single_channel', type=int,
                        help='if want to fuse before net, but not with unet')
    parser.add_argument('--unet', dest='unet', type=int,
                        help='if want to fuse before net, but not with 1x1-conv')
    parser.add_argument('--sin', dest='sin', type=int,
                        help='sin fuser + 1x1 conv')
    parser.add_argument('--before_conv_in_sin', dest='sin', type=int, default=0,
                        help='sum pics with label embeddings before 1x1 conv or not')
    parser.add_argument('--dim_feedforwardt', dest='dim_feedforwardt', type=int, default=64,
                        help='dim feedforward in the transformer')
    parser.add_argument('--resnet_path', dest='resnet_path', type=str, 
                        help='path to resnet50 default weights')
    parser.add_argument('--scheduler', dest='scheduler', type=str, 
                        help='log=multiply by const step every droplr epoch; cosine annealing')
    parser.add_argument('--optimizer', dest='optimizer', type=str, 
                        help='sgd, adam, radam')
    parser.add_argument('--nesterov', dest='nesterov', type=int, default=0,
                        help='Enable or disable Nesterov momentum in sgd')
    parser.add_argument('--conv_bias', dest='conv_bias', type=int, default=0,
                        help='Enable or disable bias term in 1x1 convolution')
    parser.add_argument('--attention_resolutions', dest='attention_resolutions', nargs="+", type=int, default=20,
                        help='at which downsampling add attention')
    parser.add_argument('--scaling_factors', dest='scaling_factors', nargs="+", type=int, default=[1,2,2,2],
                        help='scaling factors')
    parser.add_argument('--channel_mult', dest='channel_mult', nargs="+", type=float, default=[1,2,4],
                        help='channel multipliers schedule')
    parser.add_argument('--model_channels', dest='model_channels', type=int, default=32,
                        help='initial number of channel after first conv')
    parser.add_argument('--dump_cosine', dest='dump_cosine', type=float, default=-1,
                        help='if > 0, then make upper and lower bounds of cosine lr be multiplied by dump_cosine')
    parser.add_argument('--middle_attention', dest='middle_attention', type=int, default=1,
                        help='Enable or disable attention in the middle block of unet')
    parser.add_argument('--first_conv_resample', dest='first_conv_resample', type=int, default=0,
                        help='Enable or disable conv in the initial downsample')
    parser.add_argument('--num_heads', dest='num_heads', type=int, default=1,
                        help='Number of heads in attention.')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.0,
                        help='Momentum value for sgd')
    parser.add_argument('--T_max', dest='T_max', type=int, default=5, 
                        help='if scheduler is cosine, number of epochs till minimum lr')
    parser.add_argument('--eta_min', dest='eta_min', type=float, 
                        help='if scheduler is cosine, number of epochs till minimum lr')
    return parser.parse_args()
    
    
if __name__ == '__main__':
    config = Configs()
    for arg in vars(config):
        print(arg, getattr(config, arg))
        


    