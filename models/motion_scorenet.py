import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, steps=100):
        super().__init__()
        self.dim = dim
        self.steps = steps

    def forward(self, x):
        x = self.steps * x
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, L, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(L, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(L, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features) * out + beta.view(-1, self.num_features)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features) * out
        
        return out


class ConditionalInstanceNorm1d(nn.Module):
    def __init__(self, num_features, L, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(L, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(L, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features) * h + beta.view(-1, self.num_features)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features) * h

        return out

class LatentSpaceTf(nn.Module):
    """
    Encode/decode features into a latent space or from a latent space

    Args:
        in_dim (int): dimensionality of the input
        out_dim (int): dimensionality of the output 
        hidden_layers (list): list of the number of neurons in the hidden layers (in order of the layers)
    """

    def __init__(self, in_dim, hidden_layers, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = hidden_layers

        self.layers.insert(0, self.in_dim)
        self.layers.append(self.out_dim)
        # print(self.layers)

        modules = []
        for i in range(len(self.layers) - 1):        
            modules.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i != (len(self.layers) - 2):
                modules.append(nn.ELU())

        self.encoder = nn.Sequential(*modules)



    def forward(self, x):
        torch.flatten(x)
        return self.encoder(x)


class FiLMEmbeddings(nn.Module):
    """
    Compute the FiLM Embedding from a conditioning input. https://arxiv.org/pdf/1709.07871.pdf

    Compute one set of scale, bias for each feature of the input
    
    Args:
        in_dim (int): dimensionality of the input
        cond_dim (int): dimensionality of the conditioning vector (cond)
        cond (tensor): conditioning tensor
        x (tensor): tensor to which conditioning is applied
    """
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.in_dim = in_dim
        self.film_encoder = nn.Sequential(
          nn.Mish(),
          nn.Linear(self.cond_dim, 2*in_dim),
          nn.Unflatten(-1, (-1, 2))
        )

    def forward(self, x, cond):
        film_encoding = self.film_encoder(cond)

        scale = film_encoding[:,:,0]
        bias = film_encoding[:,:,1]

        return scale*x + bias


class EnergyNet(nn.Module):
    """
    Encode an input and conditioning vector to extract features, apply FiLM conditioning to the input, and then decode the latent space features to get E()

    Args:
        in_dim (int): dimensionality of the input
        latent_space_dim (int): dimensionality of the latent space
        cond_dim (int): dimensionality of the conditioning vector (cond)
        encoder_hidden_layers (list): list of the number of neurons in the hidden layers of the encoder (in order of the layers)
        decoder_hidden_layers (list): list of the number of neurons in the hidden layers of the decoder (in order of the layers)
    """

    def __init__(self, config):
        super().__init__()
        in_dim = config.model.in_dim
        # cond_dim = config.model.cond_dim
        encoder_hidden_layers = config.model.encoder_hidden_layers
        latent_space_dim = config.model.latent_space_dim
        decoder_hidden_layers = config.model.decoder_hidden_layers
        # L = number of sigma levels
        self.L = config.model.L

        self.conditionalBN = ConditionalBatchNorm1d(num_features=in_dim, L=self.L)
        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)
        self.conditional_instance_norm = ConditionalInstanceNorm1d(num_features=latent_space_dim, L=self.L)
        self.decoder = nn.Sequential(*[LatentSpaceTf(latent_space_dim, decoder_hidden_layers, 1), nn.ELU()])


    def forward(self, x, cond):
        out = self.conditionalBN(x, cond)
        out = self.encoder(out)
        out = self.conditional_instance_norm(out, cond)
        energy = self.decoder(out)

        
        return energy




class SimpleNet(nn.Module):
    """
    Encode an input and conditioning vector to extract features, apply FiLM conditioning to the input, and then decode the latent space features to get E()

    Args:
        in_dim (int): dimensionality of the input
        latent_space_dim (int): dimensionality of the latent space
        cond_dim (int): dimensionality of the conditioning vector (cond)
        encoder_hidden_layers (list): list of the number of neurons in the hidden layers of the encoder (in order of the layers)
        decoder_hidden_layers (list): list of the number of neurons in the hidden layers of the decoder (in order of the layers)
    """

    def __init__(self, config):
        super().__init__()
        self.in_dim = in_dim = config.model.in_dim * config.model.numObsSteps
        # cond_dim = config.model.cond_dim
        encoder_hidden_layers = config.model.encoder_hidden_layers
        latent_space_dim = config.model.latent_space_dim
        decoder_hidden_layers = config.model.decoder_hidden_layers
        # L = number of sigma levels
        # self.L = config.model.L

        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)
        self.embed = SinusoidalPosEmb(latent_space_dim)
        self.decoder = nn.Sequential(*[LatentSpaceTf(latent_space_dim, decoder_hidden_layers, 1), nn.ELU()])


    def forward(self, x, cond):
        # out = self.conditionalBN(x, cond)
        out = self.encoder(x)
        # out = self.conditional_instance_norm(out, cond)
        out = out + self.embed(cond)
        energy = self.decoder(out)

        
        return energy




class DummyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        # self.norm = ConditionalInstanceNorm2d
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        # self.act = act = nn.ReLU(True)

        self.linear1 = nn.Linear(5, 5)


    def forward(self, x, y):

        output = self.linear1(x)

        return output