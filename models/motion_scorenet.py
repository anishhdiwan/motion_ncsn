import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import math
from learning.motion_ncsn.models.ema import EMAHelper
from isaacgymenvs.tasks.amp.humanoid_amp_base import NUM_OBS, NUM_FEATURES, UPPER_BODY_MASK, LOWER_BODY_MASK
from isaacgymenvs.tasks.humanoid_amp import build_amp_observations
from omegaconf import OmegaConf
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
import os
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from copy import deepcopy

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
        self.layers = hidden_layers.copy()

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
        encoder_hidden_layers (list): list of the number of neurons in the hidden layers of the encoder (in order of the layers)
        decoder_hidden_layers (list): list of the number of neurons in the hidden layers of the decoder (in order of the layers)
    """

    def __init__(self, config, in_dim, feature_mask_type=None):
        super().__init__()
        # self.in_dim = in_dim = config.model.in_dim * config.model.numObsSteps
        self.in_dim = in_dim
        self.config = config

        # If not directly provided then look in cfg
        if feature_mask_type is None:
            self.feature_mask_type = self.config.model.get('feature_mask', None)
        else:
            self.feature_mask_type = feature_mask_type
        
        if self.feature_mask_type == "None":
            self.feature_mask_type = None

        self.get_mask(self.feature_mask_type)
        
        if self.config.model.get('ncsnv2', False):
            self.ncsn_version = 'ncsnv2'
        else:
            self.ncsn_version = 'ncsnv1'

        encoder_hidden_layers = config.model.encoder_hidden_layers
        latent_space_dim = config.model.latent_space_dim
        decoder_hidden_layers = config.model.decoder_hidden_layers

        self.encoder = LatentSpaceTf(in_dim, encoder_hidden_layers, latent_space_dim)
        self.embed = SinusoidalPosEmb(latent_space_dim)
        self.decoder = nn.Sequential(*[LatentSpaceTf(latent_space_dim, decoder_hidden_layers, 1), nn.ELU()])

        if config.optim.param_init == "xavier_uniform":
            self.encoder.apply(self.init_weights)
            self.embed.apply(self.init_weights)
            self.decoder.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, x, cond):
        if not self.feature_mask_type == None:
            x = x.masked_fill((~self.feature_mask).to(device=x.device), 0.0)

        if self.ncsn_version == 'ncsnv2':
            cond = cond['used_sigmas']
            out = self.encoder(x)
            # out = out + self.embed(cond)
            energy = self.decoder(out)
            energy = energy/cond
        elif self.ncsn_version == 'ncsnv1':
            cond = cond['labels']
            out = self.encoder(x)
            out = out + self.embed(cond)
            energy = self.decoder(out)
        
        return energy


    def get_mask(self, mask_type):

        pos_features_dict, vel_features_dict = deepcopy(NUM_FEATURES), deepcopy(NUM_FEATURES)
        pos_features_dict["num_features"] = [i*2 if i==3 else i for i in pos_features_dict["num_features"]]

        if mask_type is None:
            self.feature_mask = torch.ones(210, dtype=torch.bool)
        elif mask_type == "upper_body":
            root_mask = torch.ones(13, dtype=torch.bool)
            dof_pos_mask = self.create_mask(pos_features_dict, UPPER_BODY_MASK)
            dof_vel_mask = self.create_mask(vel_features_dict, UPPER_BODY_MASK)
            hand_body_mask = torch.ones(6, dtype=torch.bool)
            leg_body_mask = torch.zeros(6, dtype=torch.bool)
            feature_mask = torch.cat([root_mask, dof_pos_mask, dof_vel_mask, hand_body_mask, leg_body_mask])
            self.feature_mask = torch.cat([feature_mask, feature_mask])
        elif mask_type == "lower_body":
            root_mask = torch.ones(13, dtype=torch.bool)
            dof_pos_mask = self.create_mask(pos_features_dict, LOWER_BODY_MASK)
            dof_vel_mask = self.create_mask(vel_features_dict, LOWER_BODY_MASK)
            hand_body_mask = torch.zeros(6, dtype=torch.bool)
            leg_body_mask = torch.ones(6, dtype=torch.bool)
            feature_mask = torch.cat([root_mask, dof_pos_mask, dof_vel_mask, hand_body_mask, leg_body_mask])
            self.feature_mask = torch.cat([feature_mask, feature_mask])


    def create_mask(self, features_dict, masked_joints):
    
        # Initialize an empty mask with False values of size sum of num_features
        total_features = sum(features_dict['num_features'])
        mask = torch.zeros(total_features, dtype=torch.bool)
        
        # Initialize a starting index
        start_idx = 0
        
        # Iterate over the dof_names and num_features
        for i, joint in enumerate(features_dict['dof_names']):
            # Get the number of features for the current joint
            num_feat = features_dict['num_features'][i]
            
            # If the joint is in the upper body mask, set corresponding indices to True
            if joint in masked_joints:
                mask[start_idx:start_idx + num_feat] = True
            
            # Update the starting index for the next joint
            start_idx += num_feat

        return mask


class ComposedEnergyNet():
    def __init__(self, config, checkpoints, normalisation_checkpoints, device, in_dim_space, use_ema, ema_rate=None, scale_energies=False, env=None, keep_motion_libs=False):

        self.energy_networks = []
        self.norm_networks = []
        self.config = config
        self.energy_function_weights = [1/len(checkpoints)]*len(checkpoints)
        self.motion_styles = list(checkpoints.keys())
        feature_mask_types = self.config["inference"].get("composed_feature_mask", None)
        if (feature_mask_types == None) or (feature_mask_types == "None"):
            feature_mask_types = [None]*len(checkpoints)
        
        for norm_checkpoint in list(normalisation_checkpoints.values()):
            norm_net= RunningMeanStd(in_dim_space).to(device)
            energynet_input_norm_states = torch.load(norm_checkpoint, map_location=device)
            norm_net.load_state_dict(energynet_input_norm_states)
            norm_net.eval()
            self.norm_networks.append(norm_net)


        for idx, checkpoint in enumerate(list(checkpoints.values())):
            eb_model_states = torch.load(checkpoint, map_location=device)
            energynet = SimpleNet(self.config, in_dim=in_dim_space[0], feature_mask_type=feature_mask_types[idx]).to(device)
            energynet = torch.nn.DataParallel(energynet)
            energynet.load_state_dict(eb_model_states[0])

            if use_ema:
                print("Using EMA model weights")
                ema_helper = EMAHelper(mu=ema_rate)
                ema_helper.register(energynet)
                ema_helper.load_state_dict(eb_model_states[-1])
                ema_helper.ema(energynet)
            
            energynet.eval()
            self.energy_networks.append(energynet)


        if scale_energies:
            assert env is not None, "And env must be passed to compute scaling parameters"
            self.init_motion_sampling(env, device)
            self.energy_function_scaling = self.get_energy_function_scaling(device)
            if not keep_motion_libs:
                self.free_memory()
        else:
            self.energy_function_scaling = np.ones_like(self.energy_function_weights)

    def __call__(self, x, cond, energy_function_weights=None):
        """
        Automatically compose learnt energy functions. Assumes that both energy functions have the same network structure

        Args:
            x (torch.Tensor): Input samples
            cond: Noise level or the sigma value. Same as cond for SimpleNet.
            energy_function_weights: Relative weightage of energy functions when combining their rewards
        """
        if energy_function_weights == None:
            energy_function_weights = [torch.full((x.shape[0],1), weight, dtype=x.dtype, device=x.device) for weight in self.energy_function_weights]  #self.energy_function_weights
        assert (energy_function_weights[0].shape[0] == x.shape[0]) and (len(energy_function_weights)==len(self.energy_networks)), "Please pass the same number of weights as there are composed energy functions"
        output = None

        for idx, energynet in enumerate(self.energy_networks):
            if output is None:
                output = energy_function_weights[idx]*self.energy_function_scaling[idx]*energynet(self.norm_networks[idx](x), cond)
            else:
                output += energy_function_weights[idx]*self.energy_function_scaling[idx]*energynet(self.norm_networks[idx](x), cond)

        return output

    def get_energy_function_scaling(self, device):
        """Compute scaling factors to account for the scale difference between the learnt energy functions (energy functions are unnormalised)
        """
        num_samples = 4096
        c=0
        sigmas = self._get_ncsn_sigmas(device)
        avg_energies = []
        for motion_style, energynet in list(zip(self.motion_styles, self.energy_networks)):
            motion_samples = self.sample_paired_traj(num_samples, motion_style)
            labels = torch.ones(motion_samples.shape[0], device=device, dtype=torch.long) * c # c ranges from [0,L-1]
            used_sigmas = sigmas[labels].view(motion_samples.shape[0], *([1] * len(motion_samples.shape[1:])))
            perturbation_levels = {'labels':labels, 'used_sigmas':used_sigmas}
            energies = energynet(motion_samples, perturbation_levels)
            avg_energies.append(energies.squeeze().mean(dim=0))
        
        avg_energies = torch.tensor(avg_energies)
        scaling = avg_energies[0]/avg_energies

        return scaling
 

    def init_motion_sampling(self, env, device):
        """Initialise motion sampling to compute scaling factors

        Args:
            env (class): The environment class
        """
        motion_cfg = OmegaConf.create()
        motion_cfg.env = self.config.data.env_params
        motion_cfg.sim = self.config.data.sim_params
        self.motion_cfg = motion_cfg
        self._motion_libs = {}

        for motion_style in self.motion_styles:

            # First try to find motions in the main assets folder. Then try in the dataset directory
            motion_style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../assets/amp/motions', motion_style)
            if os.path.exists(motion_style_path):
                pass

            else:
                motion_style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../custom_envs/data/humanoid', motion_style)
                assert os.path.exists(motion_style_path), "Provided motion file can not be found in the assets/amp/motions or data/humanoid directories"


            self._motion_libs[motion_style] = MotionLib(motion_file=motion_style_path, 
                                     num_dofs=env.humanoid_num_dof,
                                     key_body_ids=env._key_body_ids.cpu().numpy(), 
                                     device=device)

    def free_memory(self):
        self._motion_libs = None


    def sample_paired_traj(self, num_samples, motion_style):
        num_obs_steps = self.motion_cfg["env"].get("numObsSteps", 2)
        dt = self.motion_cfg["sim"].get("dt", 0.0166)
        out_shape = (num_samples, num_obs_steps, NUM_OBS)
        num_obs = self.motion_cfg["env"].get("numObsSteps", 2) * NUM_OBS
        motion_ids = self._motion_libs[motion_style].sample_motions(num_samples)
        motion_times0 = self._motion_libs[motion_style].sample_time(motion_ids)

        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, num_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        

        time_steps = -dt * np.arange(0, num_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_libs[motion_style].get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self.motion_cfg["env"]["localRootObs"])
        obs_demo = obs_demo.view(out_shape)        
        obs_demo_flat = obs_demo.view(-1, num_obs)
        return obs_demo_flat
        

    def _get_ncsn_sigmas(self, device):
        """Return the noise scales depending on the ncsn version
        """
        if self.config["model"]["ncsnv2"]:
            # Geometric Schedule
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(self.config["model"]["sigma_begin"]), np.log(self.config["model"]["sigma_end"]),
                                self.config["model"]["L"]))).float().to(device)
        else:
            # Uniform Schedule
            sigmas = torch.tensor(
                    np.linspace(self.config["model"]["sigma_begin"], self.config["model"]["sigma_end"], self.config["model"]["L"])
                    ).float().to(device)

        return sigmas





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
