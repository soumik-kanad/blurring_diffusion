import torch
import numpy as np
import logging

def get_sampling_fn_blurring_diffusion(config, 
                                 
                                initial_noise,
                                denoise_fn,

                                 intermediate_sample_indices, device,
                                 share_noise=False):
    """ Returns our inverse heat process sampling function. 
    Arguments: 
    initial_noise : initial noisy states
    intermediate_sample_indices: list of indices to save (e.g., [0,1,2,3...] or [0,2,4,...])
    delta: Standard deviation of the sampling noise
    share_noise: Whether to use the same noises for all elements in the batch
    """
    T = config.model.T

    def sampler(model):

        if share_noise:
            noises = [torch.randn_like(initial_noise[0], dtype=torch.float)[None]
                      for i in range(T)]
        intermediate_samples_out = []

        with torch.no_grad():
            z_t = initial_noise.to(config.device).float()
            if intermediate_sample_indices != None and T in intermediate_sample_indices:
                intermediate_samples_out.append(z_t)
            for i in range(T, 0, -1):
                t = torch.ones(
                    initial_noise.shape[0], device=device, dtype=torch.long) * i
                # Predict less blurry-noisy image
                hat_eps = model(z_t, t)
                # Sampling step
                z_t =  denoise_fn(hat_eps, z_t, t, t-1)
                # Save trajectory
                if intermediate_sample_indices != None and i-1 in intermediate_sample_indices:
                    intermediate_samples_out.append(z_t)

            return z_t, config.model.T, intermediate_samples_out
    return sampler
