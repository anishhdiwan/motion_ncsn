# Assuming that motion_ncsn is a submodule in the algo directory
import sys
import os
MOTION_LIB_PATH = os.path.join(os.path.dirname(__file__),
                               '../../../custom_envs')
sys.path.append(MOTION_LIB_PATH)


from humanoid_motion_lib import HumanoidMotionLib
from gym_motion_lib import GymMotionLib
from models.motion_scorenet import SimpleNet
# Sample motion sets with each set having num_obs_steps motions. For example, if num_obs_steps = 2 then sample s,s' pairs. 
# In this case s' is the generated data while s is the conditioning vector

from omegaconf import OmegaConf
import numpy as np
import tqdm
from losses.dsm import *
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image


# Standardization using RunningMeanStd (compute running mean and stdevs during training and transform data with the latest values)
from rl_games.algos_torch.running_mean_std import RunningMeanStd


__all__ = ['AnnealRunner']

from sklearn import datasets
import matplotlib.pyplot as plt


class SwissRollDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=20000, state=0, norm=False):

        # Order the swiss roll
        swiss_roll = datasets.make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=state)
        swiss_roll = swiss_roll[0]
        
        if norm:
            # roll_data = swiss_roll_tensor.reshape(-1,data.shape[-1])
            stats = {
                'min': np.min(swiss_roll, axis=0),
                'max': np.max(swiss_roll, axis=0)
            }
            # nomalize to [0,1]
            ndata = (swiss_roll - stats['min']) / (stats['max'] - stats['min'])
            # normalize to [-1, 1]
            ndata = ndata * 2 - 1

            swiss_roll = ndata

        # Shifting the data to the particle env range (the aim is to learn an energy function in the same range that is seen in the real world)
        swiss_roll = (swiss_roll - swiss_roll.min(axis=0))*(512)/(swiss_roll.max(axis=0) - swiss_roll.min(axis=0))

        swiss_roll_tensor = torch.from_numpy(swiss_roll)

        self.swiss_roll_tensor = swiss_roll_tensor[:,[0,2]].to(torch.float)

    def __len__(self):
        # all possible segments of the dataset
        return len(self.swiss_roll_tensor)

    def __getitem__(self, idx):
        # get nomralized data using these indices
        nsample = self.swiss_roll_tensor[idx]

        return nsample


class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        # Added within a try block to also allow running anneal_runner from the main script within this package
        try:
            self.normalize = self.config.training.normalize_energynet_input
        except AttributeError:
            self.normalize = False

        if self.config.data.dataset != "humanoid":
            assert self.config.model.encode_temporal_feature == False, "Temporal feature encoding is not yet implemented for ray based envs"

        if self.config.model.encode_temporal_feature == True:
            self.in_dim = (self.config.model.in_dim * self.config.model.numObsSteps) + self.config.model.temporal_emb_dim*self.config.model.numObsSteps
        else:
            self.in_dim = self.config.model.in_dim * self.config.model.numObsSteps

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):

        if self.config.data.dataset == 'pushT':
            motion_lib = GymMotionLib(self.config.data.motion_file, self.config.model.numObsSteps, self.in_dim, episodic=False, normalize=False, test_split=True)
            dataloader, test_loader = motion_lib.get_traj_agnostic_dataloader(batch_size=self.config.training.batch_size, shuffle=True)

        if self.config.data.dataset == 'maze':
            motion_lib = GymMotionLib(self.config.data.motion_file, self.config.model.numObsSteps, self.in_dim, episodic=False, normalize=False, test_split=True, auto_ends=False)
            dataloader, test_loader = motion_lib.get_traj_agnostic_dataloader(batch_size=self.config.training.batch_size, shuffle=True)

        if self.config.data.dataset == 'humanoid':
            # Setting up a dict for motion lib processing
            humanoid_cfg = OmegaConf.create()
            humanoid_cfg.physics_engine = self.config.data.physics_engine
            humanoid_cfg.sim = self.config.data.sim_params
            humanoid_cfg.env = self.config.data.env_params
            humanoid_cfg.env.numEnvs = self.config.data.numEnvs
            humanoid_cfg.sim.use_gpu_pipeline = self.config.data.use_gpu_pipeline
            humanoid_cfg.sim.physx.num_threads = self.config.data.num_threads
            humanoid_cfg.sim.physx.solver_type = self.config.data.solver_type
            humanoid_cfg.sim.physx.num_subscenes = self.config.data.num_subscenes
            humanoid_cfg.sim.physx.use_gpu = self.config.data.use_gpu

            motion_lib = HumanoidMotionLib(self.config.data.env_params.motion_file, humanoid_cfg, self.config.device, encode_temporal_feature=self.config.model.encode_temporal_feature)
            dataloader = motion_lib.get_dataloader(batch_size=self.config.training.batch_size, buffer_size=self.config.training.buffer_size, shuffle=True)
            test_loader = dataloader

        if self.normalize:
            # Standardization
            self._running_mean_std = RunningMeanStd(torch.ones(self.in_dim).shape).to(self.config.device)

        if self.config.data.dataset == 'Swiss-Roll':

            if self.normalize:
                # Standardization
                self._running_mean_std = RunningMeanStd(torch.ones(self.config.model.in_dim).shape).to(self.config.device)
                self._running_mean_std.train()


            dataset = SwissRollDataset(n_samples=20000)
            test_dataset = SwissRollDataset(n_samples=4000, state=1)

            # create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.training.batch_size,
                num_workers=1,
                shuffle=True,
                # accelerate cpu-gpu transfer
                pin_memory=True,
                # don't kill worker process after each epoch
                persistent_workers=True
            )

            # create dataloader
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.config.training.batch_size,
                num_workers=1,
                shuffle=True,
                # accelerate cpu-gpu transfer
                pin_memory=True,
                # don't kill worker process after each epoch
                persistent_workers=True
            )
            


        test_iter = iter(test_loader)
        tb_path = os.path.join(self.args.run, 'summaries', self.args.doc)
        
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        # score = DummyNet(self.config).to(self.config.device)
        score = SimpleNet(self.config, in_dim=self.in_dim).to(self.config.device)
        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        # if self.args.resume_training:
        #     states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
        #     score.load_state_dict(states[0])
        #     optimizer.load_state_dict(states[1])

        step = 0
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.L))).float().to(self.config.device)



        for epoch in range(self.config.training.n_epochs):
            avg_loss = 0
            for i, X in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                if self.normalize:
                    X = self._running_mean_std(X)
          

                # if self.config.data.logit_transform:
                #     X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)

                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_loss(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                avg_loss += loss.item()
                # logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    # Quit
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X = next(test_iter)

                    test_X = test_X.to(self.config.device)

                    if self.normalize:
                        test_X = self._running_mean_std(test_X)

                    # if self.config.data.logit_transform:
                    #     test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)
                    # test_labels = test_y.to(test_X.device)

                    # with torch.no_grad():
                    # Instead of setting no_grad, explicitly compute scores as gradients without adding to the graph
                    test_dsm_loss, test_energies = anneal_dsm_loss(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power, grad=False)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)
                    for k, v in test_energies.items():
                        tb_logger.add_scalar(f'demo_data_energy/{k}', v, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                    
                    standardization_states = self._running_mean_std.state_dict()
                    torch.save(standardization_states, os.path.join(self.args.log, 'running_mean_std.pth'))

            print(f"Epoch {epoch} Avg Loss: {avg_loss/len(dataloader)}")
            # logging.info(f"Epoch {epoch} Avg Loss: {avg_loss/len(dataloader)}")



    def visualise_energy(self):
        if self.config.data.dataset == 'Swiss-Roll':
            self.visualise_sr_energy()
        elif self.config.data.dataset == 'maze':
            # self.visualise_2d_energy()
            self.plot_energy_landscape()
        elif self.config.data.dataset == 'humanoid':
            self.plot_energy_landscape()

    def plot_energy_landscape(self):
        """Plot a curve with the average energy of a set of samples on the y-axis and the distance of the samples from the demo dataset on the x-axis
        """

        if self.config.data.dataset == 'maze':
            motion_lib = GymMotionLib(self.config.data.motion_file, self.config.model.numObsSteps, self.in_dim, episodic=False, normalize=False, test_split=True, auto_ends=False)
            dataloader, test_loader = motion_lib.get_traj_agnostic_dataloader(batch_size=self.config.training.batch_size, shuffle=True)

        if self.config.data.dataset == 'humanoid':
            # Setting up a dict for motion lib processing
            humanoid_cfg = OmegaConf.create()
            humanoid_cfg.physics_engine = self.config.data.physics_engine
            humanoid_cfg.sim = self.config.data.sim_params
            humanoid_cfg.env = self.config.data.env_params
            humanoid_cfg.env.numEnvs = self.config.data.numEnvs
            humanoid_cfg.sim.use_gpu_pipeline = self.config.data.use_gpu_pipeline
            humanoid_cfg.sim.physx.num_threads = self.config.data.num_threads
            humanoid_cfg.sim.physx.solver_type = self.config.data.solver_type
            humanoid_cfg.sim.physx.num_subscenes = self.config.data.num_subscenes
            humanoid_cfg.sim.physx.use_gpu = self.config.data.use_gpu

            motion_lib = HumanoidMotionLib(self.config.data.env_params.motion_file, humanoid_cfg, self.config.device, encode_temporal_feature=self.config.model.encode_temporal_feature)
            dataloader = motion_lib.get_dataloader(batch_size=self.config.training.batch_size, buffer_size=self.config.training.buffer_size, shuffle=True)
            test_loader = dataloader


        if self.normalize:
            running_mean_std = RunningMeanStd(torch.ones(self.in_dim).shape).to(self.config.device)
            running_mean_std_states = torch.load(self.config.inference.running_mean_std_checkpoint, map_location=self.config.device)
            running_mean_std.load_state_dict(running_mean_std_states)
            running_mean_std.eval()
            
            print(f"EnergyNet was trained using normalised inputs. Data mean {running_mean_std.running_mean} Data var {running_mean_std.running_var}")


        device = self.config.device
        states = torch.load(self.config.inference.eb_model_checkpoint, map_location=self.config.device)
        network = SimpleNet(self.config, in_dim=self.in_dim).to(self.config.device)
        network = torch.nn.DataParallel(network)
        network.load_state_dict(states[0])
        network.eval()

        test_iter = iter(test_loader)
        test_X = next(test_iter)
        num_batches_to_sample = 15
        while test_X.shape[0] < num_batches_to_sample*self.config.training.batch_size:
            try:
                next_test_X = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                next_test_X = next(test_iter)
            
            test_X = torch.cat((test_X, next_test_X), 0)
            
        # test_X = next(test_iter)
        test_X = test_X.to(self.config.device)
        if self.normalize:
            test_X = running_mean_std(test_X)
        plot_energy_curve(network, test_X)


    def visualise_2d_energy(self):
        """Visualise the energy function for 2D maze environment

        An observation in the maze env is a 2D vector. The energy net is trained using s-s' pairs so the input is 4D. To visualise this in a 2D plane:
        - First a meshgrid of the same size as the env is created
        - For every point in the meshgrid, a window of the agent's reachable set is computed
        - s-s' pairs are the ncomputed by pairing every meshgrid point with another point in the window (reachable set)
        - The energy function is then computed in using this 4D input and the average energy is assigned to that meshgrid point
        """
        device = self.config.device
        states = torch.load(self.config.inference.eb_model_checkpoint, map_location=self.config.device)
        score = SimpleNet(self.config, in_dim=self.in_dim).to(self.config.device)
        score = torch.nn.DataParallel(score)
        score.load_state_dict(states[0])
        score.eval()

        colormask = self.config.visualise.colormask
        viz_min = 0
        viz_max = 512
        c = self.config.inference.sigma_level # c ranges from [0,L-1]
        kernel_size = 3 # must be odd
        grid_steps = 128
        window_idx_left = int((kernel_size - 1)/2)
        window_idx_right = int((kernel_size + 1)/2)

        if self.normalize:
            running_mean_std = RunningMeanStd(torch.ones(self.in_dim).shape).to(self.config.device)
            running_mean_std_states = torch.load(self.config.inference.running_mean_std_checkpoint, map_location=self.config.device)
            running_mean_std.load_state_dict(running_mean_std_states)
            running_mean_std.eval()
            
            print(f"EnergyNet was trained using normalised inputs. Data mean {running_mean_std.running_mean} Data var {running_mean_std.running_var}")


        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.L))
        print(f"Sigma levels {[(i,val.item()) for i,val in enumerate(sigmas)]}")

        xs = torch.linspace(viz_min, viz_max, steps=grid_steps)
        ys = torch.linspace(viz_min, viz_max, steps=grid_steps)
        x, y = torch.meshgrid(xs, ys, indexing='xy')

        grid_points = torch.cat((x.flatten().view(-1, 1),y.flatten().view(-1,1)), 1).to(device=self.config.device)
        grid_points = grid_points.reshape(grid_steps,grid_steps,2)
        energy_grid = torch.zeros(grid_steps,grid_steps,1)

        for i in range(grid_points.shape[0]):
            for j in range(grid_points.shape[1]):
                if i in [viz_min,viz_max] or j in [viz_min,viz_max]:
                    pass
                    
                else:
                    window = grid_points[i-window_idx_left:i+window_idx_right,j-window_idx_left:j+window_idx_right,:]
                    grid_pt_window = torch.zeros_like(window)
                    grid_pt_window[:,:,:] = grid_points[i,j]

                    obs_pairs = torch.cat((window, grid_pt_window), 2)

                    obs_pairs = obs_pairs.reshape(-1,4)
                    labels = torch.ones(obs_pairs.shape[0], device=grid_points.device) * c # c ranges from [0,L-1]
                    
                    obs_pairs = running_mean_std(obs_pairs)
                    energy = score(obs_pairs, labels)
                    mean_energy = torch.mean(energy).item()
                    energy_grid[i,j] = mean_energy

        energy_grid = energy_grid.reshape(-1,x.shape[0])

        if colormask:
            plt.figure(figsize=(8, 6))
            mesh = plt.pcolormesh(x.cpu().cpu().detach().numpy(), y.cpu().detach().numpy(), energy_grid.cpu().detach().numpy(), cmap ='gray')
            plt.gca().invert_yaxis()
            plt.xlabel("env - x")
            plt.ylabel("env - y")
            plt.title(f"Maze Env E(s,s' | c={c}) | Mean energy in agent's reachable set")
            plt.colorbar(mesh)


        plt.show()




    def visualise_sr_energy(self):
        """Visualise the energy function for the swiss-roll toy dataset
        """
        device = self.config.device
        states = torch.load(self.config.inference.eb_model_checkpoint, map_location=self.config.device)
        score = SimpleNet(self.config, in_dim=self.config.model.in_dim).to(self.config.device)
        score = torch.nn.DataParallel(score)
        score.load_state_dict(states[0])
        score.eval()

        plot3d = self.config.visualise.plot3d
        colormask = self.config.visualise.colormask
        plot_train_data = self.config.visualise.plot_train_data

        if self.normalize:
            running_mean_std = RunningMeanStd(torch.ones(self.config.model.in_dim).shape).to(self.config.device)
            running_mean_std_states = torch.load(self.config.inference.running_mean_std_checkpoint, map_location=self.config.device)
            running_mean_std.load_state_dict(running_mean_std_states)
            running_mean_std.eval()
            
            print(f"EnergyNet was trained using normalised inputs. Data mean {running_mean_std.running_mean} Data var {running_mean_std.running_var}")

            viz_min = [-5,-5]
            viz_max = [5,5]
            # viz_min = running_mean_std.running_mean - 5*torch.sqrt(running_mean_std.running_var)
            # viz_max = running_mean_std.running_mean + 5*torch.sqrt(running_mean_std.running_var)

            # viz_min = viz_min.cpu().detach().numpy()
            # viz_max = viz_max.cpu().detach().numpy()

        else:
            viz_min = [-1,-1]
            viz_max = [1,1]


        # if not os.path.exists(self.args.image_folder):
        #     os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.L))
        print(f"Sigma levels {[(i,val) for i,val in enumerate(sigmas)]}")


        if self.config.data.dataset == 'Swiss-Roll':

            
            xs = torch.linspace(viz_min[0], viz_max[0], steps=100)
            ys = torch.linspace(viz_min[1], viz_max[1], steps=100)
            x, y = torch.meshgrid(xs, ys, indexing='xy')

            c = self.config.inference.sigma_level # c ranges from [0,L-1]
            
            grid_points = torch.cat((x.flatten().view(-1, 1),y.flatten().view(-1,1)), 1).to(device=self.config.device)
            labels = torch.ones(grid_points.shape[0], device=grid_points.device) * c # c ranges from [0,L-1]


            energy = score(grid_points, labels)
            energy = energy.reshape(-1,x.shape[0])

            if plot3d:
                ax = plt.axes(projection='3d')
                ax.plot_surface(x.cpu().cpu().detach().numpy(), y.cpu().detach().numpy(), energy.cpu().detach().numpy())
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("energy")
                plt.show()


            if colormask:
                plt.figure(figsize=(8, 6))
                mesh = plt.pcolormesh(x.cpu().cpu().detach().numpy(), y.cpu().detach().numpy(), energy.cpu().detach().numpy(), cmap ='gray')
                plt.colorbar(mesh)

            if plot_train_data:

                # Only used for plots
                dataset = SwissRollDataset(n_samples=20000)
                sr_points = dataset[:]
                sr_points = sr_points.to(self.config.device)

                if self.normalize:
                    sr_points = running_mean_std(sr_points)
                    sr_points = sr_points.cpu().detach().numpy()

                plt.scatter(
                    sr_points[:, 0], sr_points[:, 1], s=5, alpha=0.1
                )

            plt.show()
