import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy

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


def anneal_dsm_loss(network, samples, labels, sigmas, anneal_power=2., grad=True):
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
        return anneal_dsm_energy_estimation(network, samples, labels, sigmas, anneal_power, grad)
    elif LOSS_TYPE == "score":
        return anneal_dsm_score_estimation(network, samples, labels, sigmas, anneal_power, grad)




def anneal_dsm_score_estimation(network, samples, labels, sigmas, anneal_power=2., grad=True):

    REGULARISE_ENERGY = True 

    samples.requires_grad = True
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))    
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    perturbed_samples = perturbed_samples.to(torch.float)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)

    # Default NCSN
    # scores = network(perturbed_samples, labels)

    # Energy-NCSN
    energy = network(perturbed_samples, labels)
    if grad:
        scores = autograd.grad(outputs=energy, inputs=perturbed_samples, grad_outputs=torch.ones_like(energy), retain_graph=True, create_graph=True)[0]
    else:
        scores = autograd.grad(outputs=energy, inputs=perturbed_samples, grad_outputs=torch.ones_like(energy), retain_graph=False, create_graph=False)[0]

    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)

    if REGULARISE_ENERGY:
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power + torch.linalg.norm(energy, dim=1, ord=1)
    
    else:
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if grad:
        return loss.mean(dim=0)
    else:
        test_set_energies = compute_anneal_dsm_energies(network, samples, sigmas, perturbation="gaussian")
        return loss.mean(dim=0), test_set_energies





def anneal_dsm_energy_estimation(network, samples, labels, sigmas, anneal_power=2., grad=True):

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
    if grad:
        energy = network(perturbed_samples, labels)
    else:
        with torch.no_grad():
            energy = network(perturbed_samples, labels)


    target = target.view(target.shape[0], -1)
    energy = energy.view(energy.shape[0], -1)

    loss = 1 / 2. * ((energy - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if grad:
        return loss.mean(dim=0)
    else:
        test_set_energies = compute_anneal_dsm_energies(network, samples, sigmas, perturbation=DIST_KERNEL)
        return loss.mean(dim=0), test_set_energies


def compute_anneal_dsm_energies(network, samples, sigmas, perturbation):
    """Compute the average energy of the samples for multiple label values
    """
    labels_to_evaluate = [0,3,5,7,9]
    avg_energy = {}
    for noise_level in labels_to_evaluate:
        labels = torch.full((samples.shape[0],), noise_level, device=samples.device)
        # used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))

        # if perturbation == "gaussian":
        #     perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
        # elif perturbation == "uniform":
        #     perturbed_samples = samples + ((1 - 2*torch.rand(samples.shape, device=samples.device)) * used_sigmas)
        
        # perturbed_samples = perturbed_samples.to(torch.float)
        energy = network(samples.to(torch.float), labels)
        avg_energy[f"sigma_level_{noise_level}"] = energy.squeeze().mean()

    return avg_energy

def plot_energy_curve(network, samples):
    """Plot a curve with the average energy of a set of samples on the y-axis and the distance of the samples from the demo dataset on the x-axis
    """
    # Absolute values of the range [-r, r] of a uniform distribution from which demo data is perturbed
    demo_sample_max_distances = np.linspace(0, 10, 100)
    labels_to_evaluate = [0,3,5,7,9]

    for noise_level in labels_to_evaluate:
        labels = torch.full((samples.shape[0],), noise_level, device=samples.device)
        avg_energy = np.zeros_like(demo_sample_max_distances)

        for idx, max_dist in enumerate(demo_sample_max_distances):
            if max_dist == 0.0:
                perturbed_samples = copy.deepcopy(samples)
            else:
                perturbed_samples = copy.deepcopy(samples) + (max_dist -2*max_dist*torch.rand(samples.shape, device=samples.device))
            energy = network(perturbed_samples.to(torch.float), labels)
            avg_energy[idx] = energy.squeeze().mean()

        plt.figure(figsize=(8, 6))
        plt.plot(demo_sample_max_distances, avg_energy)
        plt.xlabel("max perturbation r (where sample = sample + unif[-r,r])")
        plt.ylabel("avg energy E_theta(sample)")
        plt.title(f"Avg energy vs distance from demo data at sigma_level_{noise_level}")
        plt.show()
        




