import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import abc
from ..layers.BayesianLayer import BayesianLayer
from ...core.config import SMPL_MEAN_PARAMS
from ...core.config import AMASS_SAMPLE_PARAMS
from ...utils.geometry import rot6d_to_rotmat
from ..layers.GraphLayer import GraphConvolutionLayer



class ModelHead2(nn.Module):
    def __init__(self, num_input_features):
        super(ModelHead2, self).__init__()

        torch.autograd.set_detect_anomaly(True)
        self.npose = 24 * 6

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn.AvgPool2d(7, stride=1)
        self.fc1 = BayesianLayer(num_input_features + self.npose + 13, 1024, sigma = 0.0005)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = BayesianLayer(1024, 1024, sigma = 0.0005)
        self.drop2 = nn.Dropout(0.3)
        self.adj = self.get_adj_mat().clone() #shape [24*6, 24*6]
        self.gc1 = GraphConvolutionLayer(self.npose, self.npose, self.adj, False)
        self.gc2 = GraphConvolutionLayer(self.npose, self.npose, self.adj, False)
        self.drop3 = nn.Dropout(0.5)
        self.gc_decay = 0.7
        #self.activation = nn.ELU(inplace = True)

        '''
        self.decpose = nn.Linear(1024, self.npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        #idea: a = softmax(tanh(q^T * W * k)), then output = Sum_i a_i*v_i
        self.attn = nn.Linear(22*9, 22*9); #sample AMASS poses agrees with SMPL poses with the first 22 joints (except both hands)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        '''
        self.decpose = BayesianLayer(1024, self.npose, bias = False)
        self.decshape = BayesianLayer(1024, 10, bias = False)
        self.deccam = nn.Linear(1024, 3, bias = False)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)






    def get_adj_mat(self):
        adj = torch.zeros([24,24])
        adj[0,1] = adj[0,2] = adj[0,3] = 1
        adj[1,2] = adj[1,4] = adj[2,5] = 1
        adj[3,6] = adj[4,7] = adj[5,8] = 1
        adj[6,9] = adj[7,10] = adj[8,11] = 1
        adj[9,13] = adj[9,14] = 1
        adj[12,13] = adj[12,14] = adj[12,15] = 1
        adj[13,16] = adj[14,17] = adj[16,18] = 1
        adj[17,19] = adj[18,20] = adj[19,21] = 1
        adj[20,22] = adj[21,23] = 1


        for i in range(24):
            for j in range(24):
                if adj[i,j] == 1: #make symmetric
                    adj[j,i] = 1
                if i == j:
                    adj[i,j] = 1 #make loops


        adj_6D = torch.zeros([self.npose, self.npose])
        for i in range(24):
            for j in range(24):
                if adj[i,j] == 1:
                    for ii in range(i*6, i*6+3):  #here we assume that the rotation for x(y) is only dependent on rotation for x(y)
                        for jj in range(j*6, j*6+3):
                            adj_6D[ii,jj] = 1
                    for ii in range(i*6+3, i*6+6):
                        for jj in range(j*6+3, j*6+6):
                            adj_6D[ii,jj] = 1

                    '''
                    for ii in range(i*6, i*6+6): 
                        for jj in range(j*6, j*6+6):
                            adj_6D[ii,jj] = 1
                    '''

        return adj_6D


    def forward(self, features, init_pose=None, init_shape=None, init_cam=None, n_iter=3, gc_iter = 5):
        batch_size = features.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.avgpool(features)
        xf = xf.view(xf.size(0), -1)

        mediate_pose = init_pose.clone()
        pred_shape = init_shape.clone()
        pred_cam = init_cam.clone()

        kl_loss = torch.tensor(0.0).to(device)

        for i in range(n_iter):
            xc = torch.cat([xf, mediate_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            mediate_pose += self.decpose(xc)
            pred_shape += self.decshape(xc)
            pred_cam += self.deccam(xc)

            kl_loss += self.fc1.kl_divergence()
            kl_loss += self.fc2.kl_divergence()
            kl_loss += self.decpose.kl_divergence()
            kl_loss += self.decshape.kl_divergence()


        residual_pose = mediate_pose - init_pose
        #pred_pose = init_pose

        for i in range(gc_iter):
            xc = self.gc1(residual_pose)
            xc = self.drop3(xc)
            residual_pose += (self.gc_decay**i) * self.gc2(xc)

        pred_pose = residual_pose + init_pose
        kl_loss = kl_loss/(n_iter*4.0)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose,
            'loss_var': kl_loss,
        }
        return output

