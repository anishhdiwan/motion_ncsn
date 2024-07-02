import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import pickle

def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_loss(network, samples, labels, sigmas, eval_labels, anneal_power=2., grad=True):
    """Implement either anneal DSM score matching or anneal DSM energy matching based on the defined presets

    Ultimately, both methods learn an energy based model. 
    Score matching matches the grad_x log q(x',x) with - grad_x EBM
    Energy matching matches DIST(x',x) with EBM

    Args:
        network (nn.Model): The energy based model
        samples (torch.Tensor): Input data points to train the EBM
        labels (torch.Tensor): Sigma labels based on which each sample is perturbed
        sigma (np.Array): An array of sigma values ranging from 0,L-1
        anneal_power (Float): Annealing regularisation power
        grad (Bool): Whether to backprop gradients
    """

    LOSS_TYPE = "score" # options are "score" or "energy"

    if LOSS_TYPE == "energy":
        return anneal_dsm_energy_estimation(network, samples, labels, sigmas, eval_labels, anneal_power, grad)
    elif LOSS_TYPE == "score":
        return anneal_dsm_score_estimation(network, samples, labels, sigmas, eval_labels, anneal_power, grad)




def anneal_dsm_score_estimation(network, samples, labels, sigmas, labels_to_evaluate, anneal_power=2., grad=True):

    REGULARISE_ENERGY = False 

    samples.requires_grad = True
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))    
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    perturbed_samples = perturbed_samples.to(torch.float)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)

    # Default NCSN
    # scores = network(perturbed_samples, labels)

    # Energy-NCSN
    perturbation_levels = {'labels':labels, 'used_sigmas':used_sigmas}
    energy = network(perturbed_samples, perturbation_levels)
    if grad:
        scores = autograd.grad(outputs=energy, inputs=perturbed_samples, grad_outputs=torch.ones_like(energy), retain_graph=True, create_graph=True)[0]
    else:
        scores = autograd.grad(outputs=energy, inputs=perturbed_samples, grad_outputs=torch.ones_like(energy), retain_graph=False, create_graph=False)[0]

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    if REGULARISE_ENERGY:
        # L = len(sigmas)
        # var_loss = 0
        # for i in range(L):
        #     mask_i = labels == i
        #     energy_i = energy[mask_i]
        #     var_i = energy_i.squeeze().var()
        #     if not torch.isnan(var_i):
        #         var_loss += var_i  

        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        loss = loss.mean(dim=0)
        loss = loss + 0.5*energy.squeeze().var()
    
    else:
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        loss = loss.mean(dim=0)

    if grad:
        return loss
    else:
        test_set_energies = compute_anneal_dsm_energies(network, samples, sigmas, labels_to_evaluate)
        return loss, test_set_energies



def anneal_dsm_energy_estimation(network, samples, labels, sigmas, labels_to_evaluate, anneal_power=2., grad=True):

    DIST_KERNEL = "gaussian" # options are "gaussian" or "uniform"

    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))    
    
    if DIST_KERNEL == "gaussian":
        # Default NCSN: gaussian perturbation
        perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    elif DIST_KERNEL == "uniform":
        # Perturb using uniform distribution
        perturbed_samples = samples + ((1 - 2*torch.rand(samples.shape, device=samples.device)) * used_sigmas)

    perturbed_samples = perturbed_samples.to(torch.float)

    if DIST_KERNEL == "gaussian":
        # Default NCSN: gaussian perturbation dist function
        target = ((perturbed_samples - samples)**2 )/ (2*(used_sigmas ** 2))
        target = torch.linalg.norm(target, dim=1).to(torch.float)
    elif DIST_KERNEL == "uniform":
        # Norm of perturbation distance scaled by the perturbation sigma level (here sigma is not covariance but some radius of perturbation in range [0,1])
        target = torch.linalg.norm((perturbed_samples - samples), dim=1).to(torch.float)/used_sigmas
    
    
    # Energy-NCSN
    perturbation_levels = {'labels':labels, 'used_sigmas':used_sigmas}
    if grad:
        energy = network(perturbed_samples, perturbation_levels)
    else:
        with torch.no_grad():
            energy = network(perturbed_samples, perturbation_levels)


    target = target.view(target.shape[0], -1)
    energy = energy.view(energy.shape[0], -1)

    loss = 1 / 2. * ((energy - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if grad:
        return loss.mean(dim=0)
    else:
        test_set_energies = compute_anneal_dsm_energies(network, samples, sigmas, labels_to_evaluate)
        return loss.mean(dim=0), test_set_energies


def compute_anneal_dsm_energies(network, samples, sigmas, labels_to_evaluate):
    """Compute the average energy of the samples for multiple label values
    """
    avg_energy = {}
    for noise_level in labels_to_evaluate:
        labels = torch.full((samples.shape[0],), noise_level, device=samples.device)
        used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        perturbation_levels = {'labels':labels, 'used_sigmas':used_sigmas}
        
        # perturbed_samples = perturbed_samples.to(torch.float)
        energy = network(samples.to(torch.float), perturbation_levels)
        avg_energy[f"sigma_level_{noise_level}"] = energy.squeeze().mean()

    return avg_energy

def plot_energy_curve(network, samples, sigmas, labels_to_evaluate, checkpoint_pth=None):
    """Plot a curve with the average energy of a set of samples on the y-axis and the distance of the samples from the demo dataset on the x-axis
    """
    # Absolute values of the range [-r, r] of a uniform distribution from which demo data is perturbed
    demo_sample_max_distances = np.linspace(0, 10, 100)

    plt.figure(1, figsize=(8, 6))
    plt.figure(2, figsize=(8, 6))

    energies = {}
    energy_stds = {}
    
    for noise_level in labels_to_evaluate:
        labels = torch.full((samples.shape[0],), noise_level, device=samples.device)
        used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        perturbation_levels = {'labels':labels, 'used_sigmas':used_sigmas}

        avg_energy = np.zeros_like(demo_sample_max_distances)
        std_energy = np.zeros_like(demo_sample_max_distances)

        for idx, max_dist in enumerate(demo_sample_max_distances):
            if max_dist == 0.0:
                perturbed_samples = copy.deepcopy(samples)
            else:
                perturbed_samples = copy.deepcopy(samples) + (max_dist -2*max_dist*torch.rand(samples.shape, device=samples.device))
            energy = network(perturbed_samples.to(torch.float), perturbation_levels)
            avg_energy[idx] = energy.squeeze().mean()
            std_energy[idx] = energy.squeeze().std()

        plt.figure(1)
        plt.plot(demo_sample_max_distances, avg_energy, label=f"sigma_level_{noise_level}")
        plt.figure(2)
        plt.plot(demo_sample_max_distances, std_energy, label=f"sigma_level_{noise_level}")

        energies[int(noise_level)] = np.flip(avg_energy, axis=0)
        energy_stds[int(noise_level)] = np.flip(std_energy, axis=0)
    
    plt.figure(1)
    plt.legend()
    plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
    plt.ylabel("avg energy E_theta(sample)")
    plt.title(f"Avg energy vs distance from demo data")

    plt.figure(2)
    plt.legend()
    plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
    plt.ylabel("std energy E_theta(sample)")
    plt.title(f"std energy vs distance from demo data")

    plt.show()

    
    combined_energy = np.zeros_like(demo_sample_max_distances)
    current_level_idx = 0
    current_min = 0
    thresh = 100.0 - labels_to_evaluate[current_level_idx]*10
    for i in range(len(demo_sample_max_distances)):  
        if current_level_idx == len(labels_to_evaluate) - 1:
            combined_energy[i] = current_min + energies[labels_to_evaluate[-1]][i]

        else:

            if energies[labels_to_evaluate[current_level_idx + 1]][i] < thresh:
                combined_energy[i] = current_min + energies[labels_to_evaluate[current_level_idx]][i]
            else:
                current_min += energies[labels_to_evaluate[current_level_idx]][i]
                current_level_idx += 1
                thresh = 100.0 - labels_to_evaluate[current_level_idx]*10
                combined_energy[i] = current_min + energies[labels_to_evaluate[current_level_idx]][i]
                


    plt.figure(figsize=(8, 6))
    combined_energy = np.flip(combined_energy, axis=0)
    plt.plot(demo_sample_max_distances, combined_energy, label=f"annealed energies")
    plt.legend()
    plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
    plt.ylabel("avg energy E_theta(sample)")
    plt.title(f"Avg energy vs distance from demo data")
    plt.show()


    # save data
    if checkpoint_pth != None:
        learnt_function_path = os.path.splitext(checkpoint_pth)[0] + '_learnt_fn.pkl'
        data = {'annealed_energy': combined_energy, 'nc_energies': energies, 'nc_energy_stdev':energy_stds, 'max_sample_perturbation': demo_sample_max_distances}

        with open(learnt_function_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("saved learnt function data")


