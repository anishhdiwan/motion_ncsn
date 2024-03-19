import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

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
                modules.append(nn.ReLU())

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
        cond_dim = config.model.cond_dim
        encoder_hidden_layers = config.model.encoder_hidden_layers
        latent_space_dim = config.model.latent_space_dim
        decoder_hidden_layers = config.model.decoder_hidden_layers
        self.batch_norm = config.model.batch_norm

        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)
        self.film_encoder = FiLMEmbeddings(latent_space_dim, cond_dim=latent_space_dim)
        self.batch_norm = nn.BatchNorm1d(latent_space_dim)
        self.decoder = nn.Sequential(*[LatentSpaceTf(latent_space_dim, decoder_hidden_layers, 1), nn.Sigmoid()])


    def forward(self, x, cond):
        # print(f"input shape {x.shape}")
        # print(f"cond shape {cond.shape}")
        encoded_input = self.encoder(x)
        encoded_cond = self.encoder(cond)
        # print(f"encoded input shape {encoded_input.shape}")
        # print(f"encoded cond shape {encoded_cond.shape}")
        # conditioned input
        cond_input = self.film_encoder(encoded_input, encoded_cond)
        # print(f"conditioned input shape {cond_input.shape}")
        if self.batch_norm:
            cond_input = self.batch_norm(cond_input)
            # print(f"B norm cond input shape {cond_input.shape}")
        
        energy = self.decoder(cond_input)
        # print(f"energy shape {energy.shape}")
        
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