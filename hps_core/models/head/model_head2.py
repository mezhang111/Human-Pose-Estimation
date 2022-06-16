import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import abc
from ..layers.BayesianLayer import BayesianLayer
from ...core.config import SMPL_MEAN_PARAMS
#from ...core.config import AMASS_SAMPLE_PARAMS
from ...utils.geometry import rot6d_to_rotmat
from ..layers.GraphLayer import GraphConvolutionLayer, BayesianGraphConvolutionLayer
from ..layers.softargmax import softargmax2d, get_heatmap_preds
from ..backbone.resnet import conv3x3, conv1x1, BasicBlock
from ...utils.positional_encoding import PositionalEncodingPermute2D

class ModelHead2(nn.Module):
    def __init__(self, num_input_features):
        super(ModelHead2, self).__init__()

        torch.autograd.set_detect_anomaly(True)
        self.npose = 24 * 6
        self.nroot = 6
        self.nbody = 12*6
        self.nlimbs = 12*6
        self.feature_dim = num_input_features
        self.deconv_dim = 512
        self.deconv_layer = self._make_deconv_layer([self.deconv_dim,self.deconv_dim,self.deconv_dim])
        self.deconv_layer2 = self._make_deconv_layer([self.deconv_dim,self.deconv_dim,self.deconv_dim])

        self.final_layer_cam = nn.Sequential(
            conv3x3(self.deconv_dim, self.deconv_dim),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm2d(self.deconv_dim),
            conv3x3(self.deconv_dim, self.deconv_dim),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm2d(self.deconv_dim),
            conv1x1(self.deconv_dim, self.deconv_dim, stride = 2)
        )  
        self.final_layer_2d = nn.Sequential(
                conv3x3(self.deconv_dim, self.deconv_dim),
                nn.LeakyReLU(inplace = True),
                nn.BatchNorm2d(self.deconv_dim),
                conv1x1(self.deconv_dim, self.deconv_dim, stride = 2)
        )

        self.final_layer_3d = nn.Sequential(
                conv3x3(self.deconv_dim, self.deconv_dim),
                nn.LeakyReLU(inplace = True),
                nn.BatchNorm2d(self.deconv_dim),
                conv1x1(self.deconv_dim, self.deconv_dim, stride = 2)
        )

        self.kp_layer = nn.Sequential(
            conv3x3(self.deconv_dim, 128),
            nn.AvgPool2d(3, padding = 1),
            conv1x1(128, 24),
        )

        self.joint_pairs_24 = [(0,1), (0,2), (0,3), (1,4), (2,5), 
            (3,6), (4,7), (5,8), (6,9), (7,10), (8,11), (9,12), (9,13), (9,14),
            (12,15), (13,16), (14,17), (16,18), (17,19), (18,20),
            (19,21), (20,22), (21,23)]

        self.final_layer = nn.Sequential(
            conv3x3(self.deconv_dim*2, self.deconv_dim*2),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm2d(self.deconv_dim*2),
            conv3x3(self.deconv_dim*2, self.deconv_dim*2),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm2d(self.deconv_dim*2),
        )

        self.final_layer2 = nn.Sequential(
            conv3x3(self.deconv_dim*2, self.deconv_dim),
            nn.LeakyReLU(inplace = True),
            nn.BatchNorm2d(self.deconv_dim),
            conv1x1(self.deconv_dim, self.feature_dim, stride = 2)
        )
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn.AvgPool2d(7, stride=1)
        '''
        self.fc0 = nn.Linear(self.deconv_dim+self.npose, 1024)
        self.drop0 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024,1024)
        self.drop3 = nn.Dropout(0.5)
        self.decpose0 = nn.Linear(1024, self.npose)

        self.fc4 = nn.Linear(self.deconv_dim+13,1024)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(1024, 1024)
        self.drop5 = nn.Dropout(0.5)
        self.decshape0 = nn.Linear(1024,10)
        self.deccam0 = nn.Linear(1024,3)
        '''

        self.poseMLP = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(0.5,inplace = True),
            nn.Linear(1024, self.npose, bias = False),
        )

        self.shapeMLP = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(0.5,inplace = True),
            nn.Linear(1024, 10, bias = False),
        )

        self.camMLP = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.Dropout(0.5,inplace = True),
            nn.Linear(512, 512),
            nn.Dropout(0.5,inplace = True),
            nn.Linear(512, 3),
        )

        '''
        nn.init.xavier_uniform_(self.poseMLP.weight, gain=0.01)
        nn.init.xavier_uniform_(self.shapeMLP.weight, gain=0.01)
        nn.init.xavier_uniform_(self.camMLP.weight, gain=0.01)
        '''
        

        self.fc1 = BayesianLayer(self.feature_dim + self.npose + 10 + 3, 1024, sigma = 0.0005)
        self.drop1 = nn.Dropout(0.3,inplace = True)
        self.fc2 = BayesianLayer(1024, 1024, sigma = 0.0005)
        self.drop2 = nn.Dropout(0.3,inplace = True)


        '''
        self.joint_pairs_24 = [(1, 2), (4, 5), (13, 14), (16, 17)]
        self.root = [0]
        self.body = [1,2,3,6,9,12,13,14,15,16,17]
        self.limbs = [4,5,7,8,10,11,18,19,20,21,22,23]
        '''

        
        '''

        self.adj = self.get_pair_mat().clone() #shape [24*6, 24*6]
        self.pairs = self.get_pair_mat().clone()
        self.gc = self.get_gcl()
        self.decay = 0.7
        #self.gc2 = BayesianGraphConvolutionLayer(self.npose, self.npose, self.pairs, False)
        #self.activation = nn.ELU(inplace = True)
        '''

        '''
        self.decpose = nn.Linear(1024, self.npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        #idea: a = softmax(tanh(q^T * W * k)), then output = Sum_i a_i*v_i
        #self.attn = nn.Linear(22*9, 22*9); #sample AMASS poses agrees with SMPL poses with the first 22 joints (except both hands)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        '''
        '''
        self.decpose = BayesianLayer(1024, self.npose, bias = False)
        self.decshape = BayesianLayer(1024, 10, bias = False)
        self.deccam = nn.Linear(1024, 3, bias = False)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        

        #amass_body_pose = np.load(AMASS_SAMPLE_PARAMS)['poses'][:, 3:66]
        #amass_body_pose = torch.from_numpy(amass_body_pose).type(torch.float).to('cuda')
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def get_gcl(self):
        gc1 = GraphConvolutionLayer(self.npose, self.npose, self.adj, False)
        bn1 = nn.LayerNorm(self.npose)
        dp1 = nn.Dropout(0.5)
        ac1 = nn.LeakyReLU()
        '''
        gc2 = GraphConvolutionLayer(self.npose, self.npose, self.adj, True)
        bn2 = nn.LayerNorm(self.npose)
        dp2 = nn.Dropout(0.5)
        ac2 = nn.LeakyReLU()
        '''
        gc3 = GraphConvolutionLayer(self.npose, self.npose, self.adj, False)

        #gc_layers = [gc1, ac1, dp1, bn1, gc2, ac2, dp2, bn2, gc3]
        gc_layers = [gc1, ac1, dp1, bn1, gc3]
        return nn.Sequential(*gc_layers)

    def get_pair_mat(self):
        adj = torch.zeros([self.npose, self.npose])
        for (i,j) in self.joint_pairs_24:
            for ii in range(i*6, i*6+6): 
                    for jj in range(j*6, j*6+6):
                        adj[ii,jj] = 1

            for jj in range(j*6, j*6+6): 
                    for ii in range(i*6, i*6+6):
                        adj[jj,ii] = 1
        return adj


    def get_adj_mat(self):

        adj = torch.zeros([24,24])
        adj[0,1] = adj[0,2] = adj[0,3] = 1
        adj[1,2] = adj[1,4] = adj[2,5] = 1
        adj[3,6] = adj[4,7] = adj[5,8] = 1
        adj[6,9] = 1
        adj[7,10] = adj[8,11] = 0.2
        adj[9,13] = adj[9,14] = 1
        adj[12,13] = adj[12,14] = adj[12,15] = 1
        adj[13,16] = adj[14,17] = adj[16,18] = 1
        adj[17,19] = 1
        adj[18,20] = adj[19,21] = 0.5
        adj[20,22] = adj[21,23] = 1

        for i in range(24):
            for j in range(24):
                if adj[i,j] > 0: #make symmetric
                    adj[j,i] = adj[i,j]

                if i == j:
                    adj[i,j] = 1 #make loops


        adj_6D = torch.zeros([self.npose, self.npose])

        for i in range(24):
            for j in range(24):
                if adj[i,j] > 0:
                    for ii in range(i*6, i*6+3):  #here we assume that the rotation for x(y) is only dependent on rotation for x(y)
                        for jj in range(j*6, j*6+3):
                            adj_6D[ii,jj] = adj[i,j]
                    for ii in range(i*6+3, i*6+6):
                        for jj in range(j*6+3, j*6+6):
                            adj_6D[ii,jj] = adj[i,j]


        for (i,j) in self.joint_pairs_24: #pairs adjacent
            for ii in range(i*6, i*6+6): 
                    for jj in range(j*6, j*6+6):
                        adj_6D[ii,jj] = 1

            for jj in range(j*6, j*6+6): 
                    for ii in range(i*6, i*6+6):
                        adj_6D[jj,ii] = 1

        return adj_6D

    def _make_deconv_layer(self, deconv_dim):
        deconv1 = nn.ConvTranspose2d(
            in_channels = self.feature_dim, 
            out_channels = deconv_dim[0],
            kernel_size = 4,
            stride = 2,
            padding = 1,
            bias = False
        )
        bn1 = nn.BatchNorm2d(deconv_dim[0])
        ac1 = nn.LeakyReLU(inplace = True)
        dp1 = nn.Dropout2d(0.25,inplace = True)
        deconv2 = nn.ConvTranspose2d(
            in_channels = deconv_dim[0], 
            out_channels = deconv_dim[1],
            kernel_size = 4,
            stride = 2,
            padding = 1,
            bias = False
        )
        bn2 = nn.BatchNorm2d(deconv_dim[1])
        ac2 = nn.LeakyReLU(inplace = True)
        dp2 = nn.Dropout2d(0.25,inplace = True)
        deconv3 = nn.ConvTranspose2d(
            in_channels = deconv_dim[1], 
            out_channels = deconv_dim[2],
            kernel_size = 4,
            stride = 2,
            padding = 1,
            bias = False
        )
        ac3 = nn.LeakyReLU(inplace = True)
        bn3 = nn.BatchNorm2d(deconv_dim[2])
        deconv_layers = [deconv1, ac1, bn1, dp1, deconv2, ac2, bn2, dp2, deconv3, ac3, bn3]
        return nn.Sequential(*deconv_layers)

    def get_diff_vec(self, kp, batch_size):
        diff = torch.zeros([batch_size, len(self.joint_pairs_24), 2])
        counter = 0
        for (i,j) in self.joint_pairs_24:
            diff[:, counter, :] = kp[:, i, :] - kp[:, j, :]
            counter += 1
        diff = diff.view(batch_size, -1)
        return diff

    def get_features(self, pfeatures):
        out = self.final_layer(pfeatures);
        out = out+pfeatures #residual cut
        out = self.final_layer2(out);
        return out

    def forward(self, features, init_pose=None, init_shape=None, init_cam=None, n_iter=3, g_iter = 1, images = None):
        batch_size = features.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        #keypoints, heatmaps = softargmax2d(features) #keypoints of shape [batch_size, feature_dim, 2] and heatmaps of shape [batch_size, feature_dim, 16]
        out2d = self.deconv_layer(features)
        pfeatures2d = self.final_layer_2d(out2d) #[batch_size * 512 * 16 * 16]
        kp = self.kp_layer(out2d)
        kp, _ = softargmax2d(kp)

        out3d = self.deconv_layer2(features)
        pfeatures3d = self.final_layer_3d(out3d) #[batch_size * 512 * 16 * 16]
        pfeatures = torch.cat([out3d, out2d], 1) #[batch_size * 1024 * 16 * 16]
        
        #p_enc = PositionalEncodingPermute2D(self.deconv_dim*2) #helpful?
        #pfeatures += p_enc(pfeatures)
        
        #preperation
        features2 = self.get_features(pfeatures) # [batch_size * 512 * 8 * 8]
        #print(features2.shape)
        features2 = self.avgpool(features2).view(batch_size,-1)


        pred_pose = init_pose.clone()
        pred_shape = init_shape.clone()
        pred_cam = init_cam.clone()

        features_cam = self.final_layer_cam(features)
        features_cam = self.avgpool(features_cam).view(batch_size,-1)
        pred_cam += self.camMLP(features_cam)

        kl_loss = torch.tensor(0.0).to(device)

        #residual cuts
        #features = self.avgpool(features).view(batch_size, -1)
        #features3 = features2 + features

        for i in range(n_iter):
            xc = self.fc1(torch.cat([features2, pred_pose, pred_shape, pred_cam], 1))
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose += self.poseMLP(xc)
            pred_shape += self.shapeMLP(xc)

        kl_loss += self.fc1.kl_divergence()
        kl_loss += self.fc2.kl_divergence()

        kl_loss = kl_loss/12.0;

        '''
        pred_pose = torch.zeros(batch_size, 24, 6).to(device)
        pred_pose[:,0,:] = pred_root.clone().to(device)
        pred_pose[:, self.body, :] = pred_body.view(batch_size, 11, -1).clone().to(device)
        pred_pose[:, self.limbs, :] = pred_limbs.view(batch_size, 12, -1).clone().to(device)
        pred_pose = pred_pose.view(batch_size, -1)
        '''
        '''
        for i in range(n_iter):
            xc = torch.cat([features, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose += self.decpose(xc)
            pred_shape += self.decshape(xc)
            pred_cam += self.deccam(xc)
        '''

        '''
        #pred_pose += self.poseMLP(torch.cat([features, pred_pose], 1))
        res_pose = pred_pose - init_pose
        for i in range(g_iter): #last step: add some symmetry
            xc = self.resposeMLP(torch.cat([features, res_pose, pred_shape, pred_cam], 1))
            res_pose = (self.decay**i)*self.gc(xc)
        
        pred_pose = pred_pose + res_pose
        kl_loss += self.resposeMLP[0].kl_divergence()
        kl_loss += self.resposeMLP[3].kl_divergence()
        kl_loss += self.resposeMLP[5].kl_divergence()
        
        kl_loss = kl_loss/3.0
        '''

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose,
            'pred_keypoints': kp,
            'loss_var': kl_loss,
        }
        return output

