import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor
from fusers import unet, sin_fuser
import nirvana_dl
from distutils.dir_util import copy_tree

import numpy as np
from scipy import stats
from tqdm import tqdm
import os
import math
import csv
import copy
import json
from typing import Optional, List

from transformers_tres import Transformer
import data_loader
from posencode import PositionEmbeddingSine

# linear combination between tensors of arbitrary shape
class LinearComb(torch.nn.Module):
	def __init__(self, n: int, len: int):
		super().__init__()
		# to multiply whole tensor by the vector of scalars
		# you need to fill the vector with 1s according to the
		# number of tensor's dims
		# example: [n, d] * [n, 1] will multiply each of the n d-dimensionals vectors by one of the n-s scalars
		# example: [n, s, h, w] * [n, 1, 1, 1]
		to_add = [1 for _ in range(len)]
		self.linear = torch.nn.Parameter(torch.randn([n, *to_add]))
	def forward(self, x):
		return (x*self.linear).sum(dim=1)


class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()
	
class Net(nn.Module):
	def __init__(self,cfg,device):
		super(Net, self).__init__()
		
		self.device = device
		self.cfg = cfg
		self.L2pooling_l1 = L2pooling(channels=256)
		self.L2pooling_l2 = L2pooling(channels=512)
		self.L2pooling_l3 = L2pooling(channels=1024)
		self.L2pooling_l4 = L2pooling(channels=2048)

		if cfg.middle_fuse and cfg.double_branch:
			self.L2pooling_l1_2 = L2pooling(channels=256)
			self.L2pooling_l2_2 = L2pooling(channels=512)
			self.L2pooling_l3_2 = L2pooling(channels=1024)
			self.L2pooling_l4_2 = L2pooling(channels=2048)
		
		if cfg.single_channel and not cfg.finetune:
			if cfg.unet:
				self.initial_fuser = unet.IQAUNetModel(
					image_size=(cfg.patch_size, cfg.patch_size),
					in_channels= 3*(cfg.k+1),
					model_channels=cfg.model_channels,
					out_channels=3,
					k = cfg.k,
					num_res_blocks=1,
					attention_resolutions= cfg.attention_resolutions,
					scaling_factors=cfg.scaling_factors,
					num_heads=1,
					resblock_updown=False,
					conv_resample=True,
    				first_conv_resample=cfg.first_conv_resample,
					channel_mult=cfg.channel_mult,
					middle_attention=cfg.middle_attention
				)
			elif cfg.sin:
				self.initial_fuser = sin_fuser.SinFuser(
					k = cfg.k,
					before_initial_conv=cfg.before_conv_in_sin
				)
			else:
				self.initial_fuser = nn.Conv2d(3*(cfg.k+1), 3, kernel_size=(1, 1), bias=cfg.conv_bias)


			
		if cfg.network =='resnet50':
			from resnet_modify  import resnet50 as resnet_modifyresnet
			dim_modelt = 3840
			# load default ResNet50 weights on Nirvana
			modelpretrain = models.resnet50()
			modelpretrain.load_state_dict(torch.load(cfg.resnet_path), strict=True)

			# multichannel input to TReS instead of 3-channeled
			if not cfg.single_channel:
				if cfg.k > 0 and not cfg.finetune:
					modelpretrain.conv1 = nn.Conv2d(3*(cfg.k+1), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		# Don't use this on Nirvana
		elif cfg.network =='resnet34':
			from resnet_modify  import resnet34 as resnet_modifyresnet
			modelpretrain = models.resnet34(weights="DEFAULT")
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)
		elif cfg.network == 'resnet18':
			from resnet_modify  import resnet18 as resnet_modifyresnet
			modelpretrain = models.resnet18(weights="DEFAULT")
			dim_modelt = 960
			self.L2pooling_l1 = L2pooling(channels=64)
			self.L2pooling_l2 = L2pooling(channels=128)
			self.L2pooling_l3 = L2pooling(channels=256)
			self.L2pooling_l4 = L2pooling(channels=512)


		torch.save(modelpretrain.state_dict(), 'modelpretrain')
		
		if not cfg.single_channel:
			if not cfg.finetune:
				self.model = resnet_modifyresnet(k=cfg.k)
			else:
				self.model = resnet_modifyresnet(k=0)
		else:
			self.model = resnet_modifyresnet(k=0)
			if cfg.middle_fuse and cfg.double_branch:
				self.model_2 = resnet_modifyresnet(k=0)
		self.model.load_state_dict(torch.load('modelpretrain'), strict=True)
		if cfg.middle_fuse and cfg.double_branch:
			self.model_2.load_state_dict(torch.load('modelpretrain'), strict=True)

		self.dim_modelt = dim_modelt

		os.remove("modelpretrain")
		
		nheadt=cfg.nheadt
		num_encoder_layerst=cfg.num_encoder_layerst
		dim_feedforwardt=cfg.dim_feedforwardt
		ddropout=0.5
		normalize =True
			
			
		self.transformer = Transformer(d_model=dim_modelt,nhead=nheadt,
									   num_encoder_layers=num_encoder_layerst,
									   dim_feedforward=dim_feedforwardt,
									   normalize_before=normalize,
									   dropout = ddropout)
		if cfg.middle_fuse and cfg.double_branch:
			self.transformer_2 = Transformer(d_model=dim_modelt,nhead=nheadt,
									   num_encoder_layers=num_encoder_layerst,
									   dim_feedforward=dim_feedforwardt,
									   normalize_before=normalize,
									   dropout = ddropout)

		self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)
		

		self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features) 

		if not cfg.single_channel:
			if not cfg.finetune:
				if cfg.multi_return:
					self.fc = nn.Linear(self.model.fc.in_features*2, cfg.k+1)
				else:
					self.fc = nn.Linear(self.model.fc.in_features*2, 1)
			else:
				self.fc = nn.Linear(self.model.fc.in_features*2, 1)
		else:
			self.fc = nn.Linear(self.model.fc.in_features*2, 1)

		# 16/10/2023 version of late fuse
		if cfg.late_fuse:
			self.final_fuser = nn.Sequential(
				nn.Linear(2, 8),
				nn.SiLU(),
				nn.Linear(8, 1)
			)

		if cfg.weight_before_late_fuse:
			self.weighter = nn.Sequential(
				nn.Linear(cfg.k_late, cfg.k_late*2),
				nn.SiLU(),
				nn.Linear(cfg.k_late*2, 1)
			)
		
		if cfg.middle_fuse:
			if cfg.attention_in_middle_fuse:
				self.first_middle_fuser = torch.nn.MultiheadAttention(embed_dim=3840, num_heads=4, kdim=3840, vdim=3840, batch_first=True)
				self.second_middle_fuser = torch.nn.MultiheadAttention(embed_dim=self.model.fc.in_features, num_heads=4, kdim=self.model.fc.in_features, vdim=self.model.fc.in_features, batch_first=True)
			else:	
				self.first_middle_fuser = LinearComb(cfg.k + 1, 1)
				self.second_middle_fuser = LinearComb(cfg.k + 1, 1)
			self.consist1_fuser = [
				LinearComb(cfg.k + 1, 3).to(device),
				LinearComb(cfg.k + 1, 3).to(device)
			]
			self.consist2_fuser = [
				LinearComb(cfg.k + 1, 3).to(device),
				LinearComb(cfg.k + 1, 3).to(device)
			]
		

		self.avg7 = nn.AvgPool2d((7, 7))
		self.avg8 = nn.AvgPool2d((8, 8))
		self.avg4 = nn.AvgPool2d((4, 4))
		self.avg2 = nn.AvgPool2d((2, 2))		   
		
		self.drop2d = nn.Dropout(p=0.1)

		if cfg.middle_fuse and cfg.double_branch:
			self.avg7_2 = nn.AvgPool2d((7, 7))
			self.avg8_2 = nn.AvgPool2d((8, 8))
			self.avg4_2 = nn.AvgPool2d((4, 4))
			self.avg2_2 = nn.AvgPool2d((2, 2))		   
			
			self.drop2d_2 = nn.Dropout(p=0.1)

		self.consistency = nn.L1Loss()
		

	def forward(self, x, t=0):
		self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(self.device))
		if self.cfg.middle_fuse:
			if self.cfg.double_branch:
				self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()
				self.pos_enc_2 = self.pos_enc_1.repeat(x.shape[0] * (self.cfg.k),1,1,1).contiguous()
			else:
				self.pos_enc = self.pos_enc_1.repeat(x.shape[0] * (self.cfg.k + 1),1,1,1).contiguous()
		else:
			self.pos_enc = self.pos_enc_1.repeat(x.shape[0],1,1,1).contiguous()
		batch_size = x.shape[0]

	
		if self.cfg.single_channel:
			if self.cfg.unet or self.cfg.sin:
				# unet and sin fusers eat [b, 3*(k+1), h, w] shaped tensors with labels
				# and outputs [b, 3, h, w]
				# here k must be equal to k_late
				x = self.initial_fuser(x,t)

			elif self.cfg.middle_fuse:
				if self.cfg.double_branch:
					# double branch middle fuse needs two input separate tensors:
					# 1) with original pic 2) with k neighbours pics
					# [b, 3*(k+1), h, w] -> [b, 3, h, w] , [b*k, 3, h, w]
					x_1 = x[::, :3, ::, :: ] #  -> [b, 3, h, w]
					x_2 = x[::, 3:, ::, :: ].reshape([batch_size * (self.cfg.k), 3, self.cfg.patch_size, self.cfg.patch_size]) # -> [b*k, 3, h, w]
				else:
					# single branch middle fuse proccess original pic and its neighbours
					# in parallel, which is obtained using big batch
					# [b, 3*(k+1), h, w] -> [b*(k+1), 3, h, w] 
					x = x.reshape([batch_size * (self.cfg.k + 1), 3, self.cfg.patch_size, self.cfg.patch_size])
			else:
				# 1x1 conv fuser. Takes [b, 3*(k+1), h, w] with no labels, outputs [b, 3, h, w]
				x = self.initial_fuser(x)

		# double branch handler
		if self.cfg.double_branch:
			# main branch
			_,layer1,layer2,layer3,layer4_1 = self.model(x_1) 

			layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
			layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
			layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
			layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4_1,dim=1, p=2)))
			layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)
			out_t_c_1 = self.transformer(layers,self.pos_enc)
			out_t_o_1 = torch.flatten(self.avg7(out_t_c_1),start_dim=1)
			layer4_o = self.avg7(layer4_1)
			layer4_o_1 = torch.flatten(layer4_o,start_dim=1)

			# neighbours branch
			_,layer1,layer2,layer3,layer4_2 = self.model_2(x_2) 

			layer1_t = self.avg8_2(self.drop2d_2(self.L2pooling_l1_2(F.normalize(layer1,dim=1, p=2))))
			layer2_t = self.avg4_2(self.drop2d_2(self.L2pooling_l2_2(F.normalize(layer2,dim=1, p=2))))
			layer3_t = self.avg2_2(self.drop2d_2(self.L2pooling_l3_2(F.normalize(layer3,dim=1, p=2))))
			layer4_t =           self.drop2d_2(self.L2pooling_l4_2(F.normalize(layer4_2,dim=1, p=2)))
			layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)
			out_t_c_2 = self.transformer_2(layers,self.pos_enc_2)
			out_t_o_2 = torch.flatten(self.avg7_2(out_t_c_2),start_dim=1)
			layer4_o = self.avg7_2(layer4_2)
			layer4_o_2 = torch.flatten(layer4_o,start_dim=1)

			# concat branches to [b*(k+1), 3, h, w] shape
			out_t_o = torch.cat([out_t_o_1, out_t_o_2], dim=0)
			out_t_c = torch.cat([out_t_c_1, out_t_c_2], dim=0)
			layer4_o = torch.cat([layer4_o_1, layer4_o_2], dim=0)
			layer4 = torch.cat([layer4_1, layer4_2], dim=0)

		# standard single branch
		else:
			_,layer1,layer2,layer3,layer4 = self.model(x) 

			layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
			layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
			layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
			layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4,dim=1, p=2)))
			layers = torch.cat((layer1_t,layer2_t,layer3_t,layer4_t),dim=1)


			out_t_c = self.transformer(layers,self.pos_enc)
			out_t_o = torch.flatten(self.avg7(out_t_c),start_dim=1)

			layer4_o = self.avg7(layer4)
			layer4_o = torch.flatten(layer4_o,start_dim=1)

		# fuse out_t_o before fc : 
		# 1) = reshape: [b*(k+1), d] -> [b, (k+1), d]
		# 2) = weighted sum: [b, (k+1), d] -> [b, d]
		if self.cfg.middle_fuse:
			out_t_o = out_t_o.reshape([batch_size, self.cfg.k + 1, -1])
			if self.cfg.attention_in_middle_fuse:
				out_t_o = self.first_middle_fuser(out_t_o[::, :1, ::], out_t_o[::, 1:, ::],  out_t_o[::, 1:, ::])
			else:
				out_t_o = self.first_middle_fuser(out_t_o)
		out_t_o = self.fc2(out_t_o)

		# fuse layer4_o before concat with out_t_o and fc:
		# 1) = reshape: [b*(k+1), d] -> [b, (k+1), d]
		# 2) = weighted sum: [b, (k+1), d] -> [b, d]
		if self.cfg.middle_fuse:
			layer4_o = layer4_o.reshape([batch_size, self.cfg.k + 1, -1])
			if self.cfg.attention_in_middle_fuse:
				layer4_o = self.second_middle_fuser(layer4_o[::, :1, ::], layer4_o[::, 1:, ::], layer4_o[::, 1:, ::])
			else:
				layer4_o = self.second_middle_fuser(layer4_o)

		# backbone output
		predictionQA = self.fc(torch.flatten(torch.cat((out_t_o,layer4_o),dim=1),start_dim=1))

		# fuse backbone's output with neighbours' labels
		
		if self.cfg.late_fuse:
			if self.cfg.weight_before_late_fuse:
				t = self.weighter(t)
			else:
				t = t.mean(dim=1).unsqueeze(1)
			labels = torch.cat([predictionQA, t], dim=1)
			predictionQA = self.final_fuser(labels)

		# =============================================================================
		# =============================================================================

		# double branch handler
		if self.cfg.double_branch:

			# main branch
			_,flayer1,flayer2,flayer3,flayer4_1 = self.model(torch.flip(x_1, [3])) 
			flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
			flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
			flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
			flayer4_t =            self.L2pooling_l4(F.normalize(flayer4_1,dim=1, p=2))
			flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
			fout_t_c_1 = self.transformer(flayers,self.pos_enc)

			# neighbours' branch
			_,flayer1,flayer2,flayer3,flayer4_2 = self.model_2(torch.flip(x_2, [3])) 
			flayer1_t = self.avg8_2( self.L2pooling_l1_2(F.normalize(flayer1,dim=1, p=2)))
			flayer2_t = self.avg4_2( self.L2pooling_l2_2(F.normalize(flayer2,dim=1, p=2)))
			flayer3_t = self.avg2_2( self.L2pooling_l3_2(F.normalize(flayer3,dim=1, p=2)))
			flayer4_t =            self.L2pooling_l4_2(F.normalize(flayer4_2,dim=1, p=2))
			flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
			fout_t_c_2 = self.transformer(flayers,self.pos_enc_2)

			# concat branches
			fout_t_c = torch.cat([fout_t_c_1, fout_t_c_2], dim=0)
			flayer4 = torch.cat([flayer4_1, flayer4_2], dim=0)


		else:
			_,flayer1,flayer2,flayer3,flayer4 = self.model(torch.flip(x, [3])) 
			flayer1_t = self.avg8( self.L2pooling_l1(F.normalize(flayer1,dim=1, p=2)))
			flayer2_t = self.avg4( self.L2pooling_l2(F.normalize(flayer2,dim=1, p=2)))
			flayer3_t = self.avg2( self.L2pooling_l3(F.normalize(flayer3,dim=1, p=2)))
			flayer4_t =            self.L2pooling_l4(F.normalize(flayer4,dim=1, p=2))
			flayers = torch.cat((flayer1_t,flayer2_t,flayer3_t,flayer4_t),dim=1)
			fout_t_c = self.transformer(flayers,self.pos_enc)
		
		# weighted sum before consistency loss
		if self.cfg.middle_fuse:
			out_t_c = self.consist1_fuser[0](out_t_c.reshape([batch_size, self.cfg.k + 1, *out_t_c.shape[1:]]))
			fout_t_c = self.consist1_fuser[1](fout_t_c.reshape([batch_size, self.cfg.k + 1, *fout_t_c.shape[1:]])).detach()
			layer4 = self.consist2_fuser[0](layer4.reshape([batch_size, self.cfg.k + 1, *layer4.shape[1:]]))
			flayer4 = self.consist2_fuser[1](flayer4.reshape([batch_size, self.cfg.k + 1, *flayer4.shape[1:]])).detach()

		consistloss1 = self.consistency(out_t_c,fout_t_c)
		consistloss2 = self.consistency(layer4,flayer4)
		consistloss = 1*(consistloss1+consistloss2)
				
		return predictionQA, consistloss


class  TReS(object):
	def __init__(self, config, device,  svPath, datapath, train_idx, test_idx,Net):
		super(TReS, self).__init__()
		
		self.device = device
		self.epochs = config.epochs
		self.test_patch_num = config.test_patch_num
		self.l1_loss = torch.nn.L1Loss()
		self.lr = config.lr
		self.lrratio = config.lrratio
		self.weight_decay = config.weight_decay
		self.net = Net(config,device).to(device) 
		self.droplr = config.droplr
		self.config = config

		# finetune logic
		# load checkpoint, change architecture and freeze internal params
		if not config.single_channel:
			if config.finetune:
				# load checkpoint,
				self.net.load_state_dict(torch.load(config.ckpt, map_location=device))
				# change architecture
				self.net.model.conv1 = nn.Conv2d(3*(config.k+1), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
				if config.multi_return:
					self.net.fc = nn.Linear(self.net.model.fc.in_features*2, config.k+1)
				# freeze internal params
				if not config.full_finetune:
					for parameter in self.net.parameters():
						parameter.requires_grad = False
					for parameter in self.net.model.conv1.parameters():
						parameter.requires_grad = True
					if config.multi_return:
						for parameter in self.net.fc.parameters():
							parameter.requires_grad = True
				self.net.to(device)
		else:
			if config.finetune:
				self.net.load_state_dict(torch.load(config.ckpt, map_location=device))
				self.net.initial_fuser = nn.Conv2d(3*(config.k+1), 3, 1, bias=config.conv_bias)
				if not config.full_finetune:
					for parameter in self.net.parameters():
						parameter.requires_grad = False
					for parameter in self.net.initial_fuser.parameters():
						parameter.requires_grad = True
				self.net.to(device)

		# Nirvana resume logic
		if config.resume:
			checkpoint = torch.load(config.stateSnapshot + '/state', map_location=device)
			self.net.load_state_dict(checkpoint['model_state_dict'])
			self.lr = checkpoint['lr']

			self.paras = [{'params': self.net.parameters(), 'lr': self.lr} ]
			if config.optimizer == "adam":
				self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
			elif config.optimizer == "radam":
				self.solver = torch.optim.RAdam(self.paras, weight_decay=self.weight_decay)
			elif config.optimizer == "sgd":
				self.solver = torch.optim.SGD(self.paras, weight_decay=self.weight_decay, momentum=config.momentum, nesterov=config.nesterov)

			if config.scheduler == "log":
				self.scheduler = torch.optim.lr_scheduler.StepLR(self.solver, step_size=self.droplr, gamma=self.lrratio)
			if config.scheduler == "cosine":
				self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=config.T_max, eta_min=checkpoint['eta_min'])

			self.solver.load_state_dict(checkpoint['optimizer_state_dict'])
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			if config.scheduler == "cosine":
				# to handle cosine dumping
				self.scheduler.base_lrs[0] = checkpoint['base_lrs']
				self.scheduler.eta_min = checkpoint['eta_min']

			self.loss = checkpoint['loss']
			self.start_epoch = checkpoint['epoch']
			self.best_srcc = checkpoint['best_srcc']
			self.best_plcc = checkpoint['best_plcc']

			# dump after loading state
			nirvana_dl.snapshot.dump_snapshot()

		else:
			self.start_epoch = 0
			self.best_plcc = 0.0
			self.best_srcc = 0.0
			self.paras = [{'params': self.net.parameters(), 'lr': self.lr} ]
			if config.optimizer == "adam":
				self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
			elif config.optimizer == "radam":
				self.solver = torch.optim.RAdam(self.paras, weight_decay=self.weight_decay)
			elif config.optimizer == "sgd":
				self.solver = torch.optim.SGD(self.paras, weight_decay=self.weight_decay, momentum=config.momentum, nesterov=config.nesterov)

			if config.scheduler == "log":
				self.scheduler = torch.optim.lr_scheduler.StepLR(self.solver, step_size=self.droplr, gamma=self.lrratio)
			if config.scheduler == "cosine":
				self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=config.T_max, eta_min=config.eta_min)

		# Initialize dataloaders
		train_loader = data_loader.DataLoader(config.dataset, datapath, 
											  train_idx, config.patch_size, 
											  config.train_patch_num,
											  seed=config.seed, k=config.k, 
											  batch_size=config.batch_size, istrain=True,
											  cross_root=config.cross_datapath, cross_dataset=config.cross_dataset,
											  delimeter=config.delimeter)
		
		test_loader = data_loader.DataLoader(config.dataset, datapath,
											 test_idx, config.patch_size,
											 config.test_patch_num,
											 seed=config.seed, k=config.k, istrain=False,
											 cross_root=config.cross_datapath, cross_dataset=config.cross_dataset,
											 delimeter=config.delimeter)
		
		self.train_data = train_loader.get_data()
		self.test_data = test_loader.get_data()


	def train(self,seed,svPath):
		# set best metrics to handle resume
		# set to 0 if no resume
		best_srcc = self.best_srcc
		best_plcc = self.best_plcc

		print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tLearning_Rate\tdroplr')

		# files for metrics logging
		steps = 0
		results = {}
		train_results = {}
		performPath = svPath +'/' + 'val_SRCC_PLCC_'+str(self.config.vesion)+'_'+str(seed)+'.json'
		trainPerformPath = svPath +'/' + 'train_LOSS_SRCC_'+str(self.config.vesion)+'_'+str(seed)+'.json'

		if not self.config.resume:
			with open(performPath, 'w') as json_file2:
				json.dump( {} , json_file2)
			with open(trainPerformPath, 'w') as json_file3:
				json.dump( {}, json_file3 )
		

		for epochnum in range(self.start_epoch, self.epochs):
			self.net.train()
			epoch_loss = []
			pred_scores = []
			gt_scores = []
			pbar = tqdm(self.train_data, leave=False)

			# setting lr manually for Nirvana restart
			# handling bug: https://github.com/Lightning-AI/lightning/issues/12812
			self.solver.param_groups[0]['lr'] = self.scheduler.get_last_lr()[0]

			for g, (img, label) in enumerate(pbar):
				img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
				label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

				steps+=1
				
				self.net.zero_grad()

				# if we use fuser that processes labels
				# labels must be transferred as the input as well
				# ! MUST NOT USE 0-th label because it's orginal label !
				# ! DON'T CONFUSE WITH NOT TAKING 0-th label at the preprocess stage ! 
				if self.config.unet or self.config.sin or self.config.late_fuse:
					pred,closs = self.net(img, label[::, 1:self.config.k_late+1])
					pred2,closs2 = self.net(torch.flip(img, [3]), label[::, 1:self.config.k_late+1])
				else:
					pred,closs = self.net(img)
					pred2,closs2 = self.net(torch.flip(img, [3]))   

				# multi return handler
				if self.config.multi_return:
					pred_scores = pred_scores + pred[:,0].flatten().cpu().tolist()
					gt_scores = gt_scores + label[:,0].flatten().cpu().tolist()
					loss_qa = self.l1_loss(pred, label.float().detach())
					loss_qa2 = self.l1_loss(pred2, label.float().detach())

					# =============================================================================
					# =============================================================================

					# simple multi return 
					if not self.config.multi_ranking:

						indexlabel = torch.argsort(label, dim=0)[:, 0].flatten() # small--> large
						anchor1 = torch.unsqueeze(pred.T[0, indexlabel[0],...].contiguous(),dim=0) # d_min
						positive1 = torch.unsqueeze(pred.T[0, indexlabel[1],...].contiguous(),dim=0) # d'_min+
						negative1_1 = torch.unsqueeze(pred.T[0, indexlabel[-1],...].contiguous(),dim=0) # d_max+

						
						anchor2 = torch.unsqueeze(pred.T[0, indexlabel[-1],...].contiguous(),dim=0)# d_max
						positive2 = torch.unsqueeze(pred.T[0, indexlabel[-2],...].contiguous(),dim=0)# d'_max+
						negative2_1 = torch.unsqueeze(pred.T[0, indexlabel[0],...].contiguous(),dim=0)# d_min+

						# =============================================================================
						# =============================================================================

						fanchor1 = torch.unsqueeze(pred2.T[0, indexlabel[0],...].contiguous(),dim=0)
						fpositive1 = torch.unsqueeze(pred2.T[0, indexlabel[1],...].contiguous(),dim=0)
						fnegative1_1 = torch.unsqueeze(pred2.T[0,indexlabel[-1],...].contiguous(),dim=0)

						fanchor2 = torch.unsqueeze(pred2.T[0, indexlabel[-1],...].contiguous(),dim=0)
						fpositive2 = torch.unsqueeze(pred2.T[0, indexlabel[-2],...].contiguous(),dim=0)
						fnegative2_1 = torch.unsqueeze(pred2.T[0, indexlabel[0],...].contiguous(),dim=0)

						assert (label.T[0, indexlabel[-1]]-label.T[0, indexlabel[1]])>=0
						assert (label.T[0, indexlabel[-2]]-label.T[0, indexlabel[0]])>=0
						triplet_loss1 = nn.TripletMarginLoss(margin=(label.T[0, indexlabel[-1]]-label.T[0, indexlabel[1]]), p=1) # d_min,d'_min,d_max
						triplet_loss2 = nn.TripletMarginLoss(margin=(label.T[0, indexlabel[-2]]-label.T[0, indexlabel[0]]), p=1)
						
						tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
						triplet_loss2(anchor2, positive2, negative2_1)
						ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
						triplet_loss2(fanchor2, fpositive2, fnegative2_1)

					# multi ranking multi return
					else:
						tripletlosses = torch.zeros([self.config.k])
						ftripletlosses = torch.zeros([self.config.k])

						for l in range(self.config.k):
							indexlabel = torch.argsort(label, dim=0)[:, l].flatten() # small--> large
							anchor1 = torch.unsqueeze(pred.T[l, indexlabel[0],...].contiguous(),dim=0) # d_min
							positive1 = torch.unsqueeze(pred.T[l, indexlabel[1],...].contiguous(),dim=0) # d'_min+
							negative1_1 = torch.unsqueeze(pred.T[l, indexlabel[-1],...].contiguous(),dim=0) # d_max+

							
							anchor2 = torch.unsqueeze(pred.T[l, indexlabel[-1],...].contiguous(),dim=0)# d_max
							positive2 = torch.unsqueeze(pred.T[l, indexlabel[-2],...].contiguous(),dim=0)# d'_max+
							negative2_1 = torch.unsqueeze(pred.T[l, indexlabel[0],...].contiguous(),dim=0)# d_min+

							# =============================================================================
							# =============================================================================

							fanchor1 = torch.unsqueeze(pred2.T[l, indexlabel[0],...].contiguous(),dim=0)
							fpositive1 = torch.unsqueeze(pred2.T[l, indexlabel[1],...].contiguous(),dim=0)
							fnegative1_1 = torch.unsqueeze(pred2.T[l,indexlabel[-1],...].contiguous(),dim=0)

							fanchor2 = torch.unsqueeze(pred2.T[l, indexlabel[-1],...].contiguous(),dim=0)
							fpositive2 = torch.unsqueeze(pred2.T[l, indexlabel[-2],...].contiguous(),dim=0)
							fnegative2_1 = torch.unsqueeze(pred2.T[l, indexlabel[0],...].contiguous(),dim=0)

							assert (label.T[l, indexlabel[-1]]-label.T[l, indexlabel[1]])>=0
							assert (label.T[l, indexlabel[-2]]-label.T[l, indexlabel[0]])>=0
							triplet_loss1 = nn.TripletMarginLoss(margin=(label.T[l, indexlabel[-1]]-label.T[l, indexlabel[1]]), p=1) # d_min,d'_min,d_max
							triplet_loss2 = nn.TripletMarginLoss(margin=(label.T[l, indexlabel[-2]]-label.T[l, indexlabel[0]]), p=1)

							tripletlosses[l] = triplet_loss1(anchor1, positive1, negative1_1) + \
							triplet_loss2(anchor2, positive2, negative2_1)

							ftripletlosses[l] = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
							triplet_loss2(fanchor2, fpositive2, fnegative2_1)
						tripletlosses = tripletlosses.mean()
						ftripletlosses = ftripletlosses.mean()

				# single return = standard approach.
				# usually use it
				else:
					label = label.T[0]
					pred_scores = pred_scores + pred.flatten().cpu().tolist()
					gt_scores = gt_scores + label.cpu().tolist()
					loss_qa = self.l1_loss(pred.flatten(), label.float().detach())
					loss_qa2 = self.l1_loss(pred2.flatten(), label.float().detach())

					# =============================================================================
					# =============================================================================

					indexlabel = torch.argsort(label) # small--> large
					anchor1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0) # d_min
					positive1 = torch.unsqueeze(pred[indexlabel[1],...].contiguous(),dim=0) # d'_min+
					negative1_1 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0) # d_max+

					anchor2 = torch.unsqueeze(pred[indexlabel[-1],...].contiguous(),dim=0)# d_max
					positive2 = torch.unsqueeze(pred[indexlabel[-2],...].contiguous(),dim=0)# d'_max+
					negative2_1 = torch.unsqueeze(pred[indexlabel[0],...].contiguous(),dim=0)# d_min+

					# =============================================================================
					# =============================================================================

					fanchor1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)
					fpositive1 = torch.unsqueeze(pred2[indexlabel[1],...].contiguous(),dim=0)
					fnegative1_1 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)

					fanchor2 = torch.unsqueeze(pred2[indexlabel[-1],...].contiguous(),dim=0)
					fpositive2 = torch.unsqueeze(pred2[indexlabel[-2],...].contiguous(),dim=0)
					fnegative2_1 = torch.unsqueeze(pred2[indexlabel[0],...].contiguous(),dim=0)

					assert (label[indexlabel[-1]]-label[indexlabel[1]])>=0
					assert (label[indexlabel[-2]]-label[indexlabel[0]])>=0
					triplet_loss1 = nn.TripletMarginLoss(margin=(label[indexlabel[-1]]-label[indexlabel[1]]), p=1) # d_min,d'_min,d_max
					triplet_loss2 = nn.TripletMarginLoss(margin=(label[indexlabel[-2]]-label[indexlabel[0]]), p=1)

					tripletlosses = triplet_loss1(anchor1, positive1, negative1_1) + \
					triplet_loss2(anchor2, positive2, negative2_1)
					ftripletlosses = triplet_loss1(fanchor1, fpositive1, fnegative1_1) + \
					triplet_loss2(fanchor2, fpositive2, fnegative2_1)

				# =============================================================================
				# =============================================================================

				# print GPU usage 
				if g == 0 and epochnum == 0:
					print("GPU usage")
					os.system("nvidia-smi")
				
				# mind blowing TReS loss
				loss = loss_qa + closs + loss_qa2 + closs2 + 0.5*( self.l1_loss(tripletlosses,ftripletlosses.detach())+ self.l1_loss(ftripletlosses,tripletlosses.detach()))+0.05*(tripletlosses+ftripletlosses)
				
				epoch_loss.append(loss.item())
				loss.backward()
				self.solver.step()
				

			# calculate train metrics: Spearman's rank correlation and loss
			train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
			train_loss = sum(epoch_loss) / len(epoch_loss)

			# log train metric
			train_results[epochnum] = (train_loss, train_srcc)
			with open(trainPerformPath, "a+") as file:
				file.write(f"{epochnum+1}: {train_loss}, {train_srcc}")

			# validation
			test_srcc, test_plcc = self.test(self.test_data,epochnum,svPath,seed)

			# log val metric
			results[epochnum]=(test_srcc, test_plcc)
			with open(performPath, "a+") as file:
				file.write(f"{epochnum+1}: {test_srcc}, {test_plcc}")

			# save best model's parameters according to the val's Spearman's correlation
			if test_srcc > best_srcc:
				modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))
				torch.save(self.net.state_dict(), modelPathbest)

				# update best metrics
				best_srcc = test_srcc
				best_plcc = test_plcc

			print('{}\t{:4.3f}\t\t{:4.4f}\t\t{:4.4f}\t\t{:4.3f}\t\t{}\t\t{:4.3f}'.format(epochnum + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, self.solver.param_groups[0]['lr'], self.droplr ))


			# scheduler step
			self.scheduler.step()

			# cosine scheduler dump
			if self.config.scheduler == "cosine" and self.config.dump_cosine > 0:
				# dump the lower bound of cosine annealing when reach it
				if (epochnum+1) % self.config.T_max == 0:
					self.scheduler.eta_min = self.scheduler.eta_min * self.config.dump_cosine
				# dump the upper bound of cosine annealing when reach it
				if (epochnum+1+self.config.T_max) % self.config.T_max == 0:
					self.scheduler.base_lrs[0] = self.scheduler.base_lrs[0] * self.config.dump_cosine

			# save all the resume-necessary info every epoch
			# in case of Nirvana restart
			fullModelPath = self.config.stateSnapshot + '/state'
			if self.config.scheduler == "cosine":
				torch.save({
					'epoch': epochnum+1,
					'model_state_dict': self.net.state_dict(),
					'optimizer_state_dict': self.solver.state_dict(),
					'scheduler_state_dict': self.scheduler.state_dict(),
					'loss': loss,
					'lr': self.scheduler.get_last_lr(),
					'base_lrs':	self.scheduler.base_lrs[0],
					'eta_min': self.scheduler.eta_min,
					'best_srcc': best_srcc,
					'best_plcc': best_plcc
				}, fullModelPath)
			else:
				torch.save({
					'epoch': epochnum+1,
					'model_state_dict': self.net.state_dict(),
					'optimizer_state_dict': self.solver.state_dict(),
					'scheduler_state_dict': self.scheduler.state_dict(),
					'loss': loss,
					'lr': self.scheduler.get_last_lr(),
					'best_srcc': best_srcc,
					'best_plcc': best_plcc
				}, fullModelPath)
			# copy logs to state dir to save them	
			copy_tree(self.config.svpath, self.config.stateSnapshot)
			# save everything in state path to Nirvana for restart
			nirvana_dl.snapshot.dump_snapshot()


		print('Best val SRCC %f, PLCC %f' % (best_srcc, best_plcc))

		return best_srcc, best_plcc


	def test(self, data, epochnum, svPath, seed, pretrained=0):
		# to handle test session (in opposite to val session)
		# set pretrained=1 if want test session
		if pretrained:
			self.net.load_state_dict(torch.load(svPath+'/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))))
			
		self.net.eval()
		pred_scores = []
		gt_scores = []
		
		pbartest = tqdm(data, leave=False)
		with torch.no_grad():
			steps2 = 0
		
			for img, label in pbartest:
				img = torch.as_tensor(img.to(self.device))
				label = torch.as_tensor(label.to(self.device))

				# if we use fuser that processes labels
				# labels must be transferred as the input as well
				# ! MUST NOT USE 0-th label because it's orginal label !
				# ! DON'T CONFUSE WITH NOT TAKING 0-th label at the preprocess stage !
				if self.config.unet or self.config.sin or self.config.late_fuse:
					pred,_ = self.net(img, label[::, 1:self.config.k_late + 1])
				else:
					pred,_ = self.net(img)

				if self.config.multi_return:
					pred_scores = pred_scores + pred[:,0].flatten().cpu().tolist()
					gt_scores = gt_scores + label[:,0].flatten().cpu().tolist()
				else:
					pred_scores = pred_scores + pred.flatten().cpu().tolist()
					gt_scores = gt_scores + label.T[0].cpu().tolist()
				
				
				steps2 += 1
				
		# average scores over pathes
		pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
		gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
		
		# if val session, save to val csv
		if not pretrained:
			dataPath = svPath + '/val_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion),str(seed),epochnum)
			with open(dataPath, 'w') as f:
				writer = csv.writer(f)
				writer.writerow(g for g in ['preds','gts'])
				writer.writerows(zip(pred_scores, gt_scores))
		# if test session, save to test csv
		else:
			dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion),str(seed),epochnum)
			with open(dataPath, 'w') as f:
				writer = csv.writer(f)
				writer.writerow(g for g in ['preds','gts'])
				writer.writerows(zip(pred_scores, gt_scores))
			
		# calculate test metrics: Spearman's and Pearson's correlations
		test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
		test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
		return test_srcc, test_plcc
	
if __name__=='__main__':
	import os
	import argparse
	import random
	import numpy as np
	from args import *
