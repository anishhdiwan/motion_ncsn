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

# Assuming that motion_ncsn is a submodule in the algo directory
import sys
import os
MOTION_LIB_PATH = os.path.join(os.path.dirname(__file__),
                               '../../../custom_envs')
sys.path.append(MOTION_LIB_PATH)

from motion_lib import MotionLib
from models.motion_scorenet import SimpleNet
motion_file = MOTION_LIB_PATH + "/data/pusht/pusht_cchi_v7_replay.zarr"
# Sample motion sets with each set having num_obs_steps motions. For example, if num_obs_steps = 2 then sample s,s' pairs. 
# In this case s' is the generated data while s is the conditioning vector
num_obs_steps = 2
# Number of features per observation
num_obs_per_step = 5

__all__ = ['AnnealRunner']

from sklearn import datasets
import matplotlib.pyplot as plt

class SwissRollDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=20000, state=0, norm=True):

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
            motion_lib = MotionLib(motion_file, num_obs_steps, num_obs_per_step, episodic=False, normalize=True)
            dataloader = motion_lib.get_traj_agnostic_dataloader(batch_size=self.config.training.batch_size, shuffle=True)
            test_loader = dataloader


        if self.config.data.dataset == 'Swiss-Roll':

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
        self.config.input_dim = 2
        
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        # score = DummyNet(self.config).to(self.config.device)
        score = SimpleNet(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

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
                # X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

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
                    # test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

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

            logging.info(f"Epoch {epoch} Avg Loss: {avg_loss/len(dataloader)}")

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images


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


            # samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
            # all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            # for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
            #     sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
            #                          self.config.data.image_size)

            #     if self.config.data.logit_transform:
            #         sample = torch.sigmoid(sample)

            #     image_grid = make_grid(sample, nrow=grid_size)
            #     if i % 10 == 0:
            #         im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
            #         imgs.append(im)

            #     save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
            #     torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))



    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32 ,32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.L))
        score.eval()

        imgs = []
        if self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))


        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
