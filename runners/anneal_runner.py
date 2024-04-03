# Assuming that motion_ncsn is a submodule in the algo directory
import sys
import os
MOTION_LIB_PATH = os.path.join(os.path.dirname(__file__),
                               '../../../custom_envs')
sys.path.append(MOTION_LIB_PATH)


import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
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

from motion_lib import MotionLib
from models.motion_scorenet import SimpleNet
# Sample motion sets with each set having num_obs_steps motions. For example, if num_obs_steps = 2 then sample s,s' pairs. 
# In this case s' is the generated data while s is the conditioning vector


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
            motion_lib = MotionLib(self.config.data.motion_file, self.config.model.numObsSteps, self.config.model.in_dim, episodic=False, normalize=False, test_split=True)
            dataloader, test_loader = motion_lib.get_traj_agnostic_dataloader(batch_size=self.config.training.batch_size, shuffle=True)

            if self.normalize:
                # Standardization
                self._running_mean_std = RunningMeanStd(torch.ones(self.config.model.in_dim * self.config.model.numObsSteps).shape).to(self.config.device)


        if self.config.data.dataset == 'maze':
            motion_lib = MotionLib(self.config.data.motion_file, self.config.model.numObsSteps, self.config.model.in_dim, episodic=False, normalize=False, test_split=True, auto_ends=False)
            dataloader, test_loader = motion_lib.get_traj_agnostic_dataloader(batch_size=self.config.training.batch_size, shuffle=True)

            if self.normalize:
                # Standardization
                self._running_mean_std = RunningMeanStd(torch.ones(self.config.model.in_dim * self.config.model.numObsSteps).shape).to(self.config.device)




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
        score = SimpleNet(self.config).to(self.config.device)

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
                # labels = y.to(X.device)

                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
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
                    test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power, grad=False)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

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
        device = self.config.device
        states = torch.load(self.config.inference.eb_model_checkpoint, map_location=self.config.device)
        score = SimpleNet(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        score.load_state_dict(states[0])
        score.eval()

        plot3d = self.config.visualise.plot3d
        colormask = self.config.visualise.colormask
        plot_train_data = self.config.visualise.plot_train_data

        if self.config.training.normalize_energynet_input:
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

                if self.config.training.normalize_energynet_input:
                    sr_points = running_mean_std(sr_points)
                    sr_points = sr_points.cpu().detach().numpy()

                plt.scatter(
                    sr_points[:, 0], sr_points[:, 1], s=5, alpha=0.1
                )

            plt.show()


    def test(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        states = torch.load(os.path.join(self.args.log, 'checkpoint_200000.pth'), map_location=self.config.device)
        score = SimpleNet(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        plot3d = self.config.visualise.plot3d
        colormask = self.config.visualise.colormask
        plot_train_data = self.config.visualise.plot_train_data


        # if not os.path.exists(self.args.image_folder):
        #     os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.L))

        print(f"Sigma levels {[(i,val) for i,val in enumerate(sigmas)]}")

        score.eval()

        imgs = []
        if self.config.data.dataset == 'Swiss-Roll':

            
            xs = torch.linspace(-1, 1, steps=100)
            ys = torch.linspace(-1, 1, steps=100)
            x, y = torch.meshgrid(xs, ys, indexing='xy')

            c = self.config.visualise.sigma_level # c ranges from [0,L-1]
            
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

                plt.scatter(
                    sr_points[:, 0], sr_points[:, 1], s=5, alpha=0.1
                )

            plt.show()




    # def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
    #     images = []

    #     labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
    #     labels = labels.long()

    #     with torch.no_grad():
    #         for _ in range(n_steps):
    #             images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
    #             noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
    #             grad = scorenet(x_mod, labels)
    #             x_mod = x_mod + step_lr * grad + noise
    #             x_mod = x_mod
    #             print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

    #         return images

    # def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    #     images = []

    #     with torch.no_grad():
    #         for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
    #             labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
    #             labels = labels.long()
    #             step_size = step_lr * (sigma / sigmas[-1]) ** 2
    #             for s in range(n_steps_each):
    #                 images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
    #                 noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
    #                 grad = scorenet(x_mod, labels)
    #                 x_mod = x_mod + step_size * grad + noise
    #                 # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
    #                 #                                                          grad.abs().max()))

    #         return images