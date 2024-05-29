import torch
import torch.autograd as autograd
import torch.nn.functional as F


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

    REGULARISE_ENERGY = False 

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
        # Regularise to get low energy values. Ensures consistency between the energy functions learnt for different noise levels
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power + torch.linalg.norm(energy, dim=1, ord=1)
    
    else:
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)



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

    return loss.mean(dim=0)