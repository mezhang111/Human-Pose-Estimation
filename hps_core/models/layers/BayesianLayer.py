import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BayesianLayer(torch.nn.Module):
    '''
    Module implementing a single Bayesian feedforward layer.
    The module performs Bayes-by-backprop, that is, mean-field
    variational inference. It keeps prior and posterior weights
    (and biases) and uses the reparameterization trick for sampling.
    '''
    def __init__(self, input_dim, output_dim, mu = 0.0, sigma = 0.00075, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias
        self.prior_mu = 0
        self.prior_sigma = .01

        mu_init_distribution = torch.distributions.Normal(torch.ones(output_dim, input_dim) * torch.tensor(0), torch.ones(output_dim, input_dim) * torch.tensor(.01))
        self.weight_mu = nn.Parameter(mu*torch.ones(output_dim, input_dim, requires_grad=True))

        self.weight_logsigma = nn.Parameter(torch.log(sigma * torch.ones(output_dim, input_dim, requires_grad=True)))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.zeros(output_dim, requires_grad=True))
            self.bias_logsigma = nn.Parameter(torch.log(sigma * torch.ones(output_dim, requires_grad=True)))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logsigma', None)


    def forward(self, inputs):
        assert self.input_dim == inputs.shape[1]
        batch_size = inputs.shape[0]

        normal_distribution = torch.distributions.Normal(0, 1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        eps = normal_distribution.sample(self.weight_mu.size()).to(device)
        s_weight = self.weight_mu + torch.exp(self.weight_logsigma) * eps

        if self.use_bias:
            eps = normal_distribution.sample(self.bias_mu.size()).to(device)
            s_bias = self.bias_mu + torch.exp(self.bias_logsigma) * eps # sapled biases
        else:
            s_bias = None

        return F.linear(input=inputs, weight=s_weight, bias=s_bias)


    def kl_divergence(self):
        '''
        Computes the KL divergence between the priors and posteriors for this layer.
        '''
        kl_loss = self._kl_divergence(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            kl_loss += self._kl_divergence(self.bias_mu, self.bias_logsigma)
        return kl_loss



    def _kl_divergence(self, mu, logsigma):
        '''
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        '''
        d = logsigma.numel()
        kl = .5 * (torch.exp(2*logsigma).sum() / (self.prior_sigma**2)
                   + (torch.norm(mu) ** 2) / (self.prior_sigma**2)
                   - d
                   - 2*logsigma.sum().sum()
                   + d * torch.log(torch.tensor(float(self.prior_sigma**2))))
        return kl/d
