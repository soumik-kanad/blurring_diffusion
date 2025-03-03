import torch
import numpy as np
from model_code.torch_dct import dct_2d, idct_2d



class BlurringDiffusion:
    def __init__(self, T=100, res=64, 
                sigma_blur_max=20,
                min_scale=0.001,
                logsnr_min=-10, 
                logsnr_max=10,
                delta=1e-8,

                

                ):
        self.T = T
        self.res = res
        self.sigma_blur_max = sigma_blur_max
        self.min_scale = min_scale
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        self.delta = delta


    def get_frequency_scaling(self, t):
        """
        Compute the scaling for the frequencies at time t
        t : range [0,1]
        """
        assert torch.all(t >= 0) and torch.all(t <= 1), f"t should be between 0 and 1, got {t}"

        # Compute dissipation time
        sigma_blur = self.sigma_blur_max * (torch.sin(t * np.pi / 2) ** 2) #Soumik: adding this back into the code # Prateksha:Idk the need for this variable - present in the pseudo-code
        dissipation_time = (sigma_blur ** 2) / 2
        
        # Compute frequencies
        freqs = np.pi * torch.linspace(0, self.res - 1, self.res) / self.res
        labda = freqs[ :, None] ** 2 + freqs[None, :] ** 2 #changed the dimensions - made it similar to heat dissipation code
        labda = labda.to(t.device)
        
        # Compute scaling for frequencies
        scaling = torch.exp(-labda * dissipation_time[:, None, None]) * (1 - self.min_scale)
        scaling += self.min_scale
        
        return scaling



    def get_noise_scaling_cosine(self, t):

        assert torch.all(t >= 0) and torch.all(t <= 1), f"t should be between 0 and 1, got {t}"

        limit_max = torch.arctan(torch.exp(torch.tensor(-0.5 * self.logsnr_max)))
        limit_min = torch.arctan(torch.exp(torch.tensor(-0.5 * self.logsnr_min))) - limit_max

        # eps = 1e-5
        # tan_input = torch.clamp(limit_min * t + limit_max, eps, np.pi / 2 - eps) # Clipped the range of values passed to tan; not sure if this is okay to do?
        tan_input = limit_min * t + limit_max # code without clipping

        logsnr = -2 * torch.log(torch.tan(tan_input))
        # print(f"For t = {t}, logsnar = {logsnr}, {limit_min * t + limit_max}, {torch.tan(limit_min * t + limit_max)}, {torch.log(torch.tan(tan_input))}")
        
        # Transform logsnr to a, sigma
        a = torch.sqrt(1 / (1 + torch.exp(-logsnr)))  # sigmoid(logsnr)
        sigma = torch.sqrt(1 / (1 + torch.exp(logsnr)))  # sigmoid(-logsnr)
        
        return a, sigma

    def get_alpha_sigma(self, t):
        
        assert torch.all(t >= 0) and torch.all(t <= self.T), f"t should be between 0 and T, got {t}"

        freq_scaling = self.get_frequency_scaling(t/self.T)
        a, sigma = self.get_noise_scaling_cosine(t/self.T) # since scale for blurring and noise addition were different
        
        alpha = a[:, None, None] * freq_scaling  # Combine dissipation and scaling
        
        return alpha, sigma
    

    def diffuse(self, x, t):
        """
        Diffuse the input image x at time t
        t : range [0,T]
        """
        x_freq = dct_2d(x, norm='ortho')
        
        alpha, sigma = self.get_alpha_sigma(t)
        alpha = alpha[:,None,:,:].to(x.device)
        sigma = sigma[:,None, None, None].to(x.device)
        
        eps = torch.randn_like(x)
        
        u_recon = idct_2d(alpha * x_freq, norm='ortho')
        z_t = u_recon + sigma * eps


        return z_t, [u_recon, sigma, eps]
    

    def loss(self, model, x=None, z_t=None, eps=None):
        """
        Calculate loss
        if x is provided, calculate z_t and eps
        if z_t is provided, calculate error directly (eps should be provided as well)
        """
        batch_size = x.shape[0] if x is not None else z_t.shape[0]
        device = x.device if x is not None else z_t.device
        t = torch.rand(batch_size, device=device) * self.T
        
        if x is not None:
            z_t, [_,_,eps] = self.diffuse(x, t)
            # print(type(eps), type(z_t))
        else:
            assert z_t is not None and eps is not None, "z_t and eps should be provided"
        ## model expects t in [0,T]
        error = (eps - model(z_t, t))**2
        return error.mean(), torch.sum(error.reshape(error.shape[0], -1), dim=-1), t
    

    def denoise(self, hat_eps, z_t, t, s):
        """
        According to denoise() function in the blurring diffusion paper

        t,s : range [0,T]

        """
        assert torch.all(s<=t), "s should be less than or equal to t"
        
        alpha_s, sigma_s = self.get_alpha_sigma(s) 
        alpha_s, sigma_s = alpha_s[:,None,:,:], sigma_s[:,None, None, None]
        alpha_t, sigma_t = self.get_alpha_sigma(t) 
        alpha_t, sigma_t = alpha_t[:,None,:,:], sigma_t[:,None, None, None]
        
        # Compute helpful coefficients
        alpha_ts = alpha_t / alpha_s
        alpha_st = 1 / alpha_ts
        sigma2_ts = (sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2)

        # Denoising variance
        sigma2_denoise = 1 / torch.clamp(
            1 / torch.clamp(sigma_s ** 2, min=self.delta) +
            1 / torch.clamp(sigma_t ** 2 / alpha_ts ** 2 - sigma_s ** 2, min=self.delta),
            min=self.delta
        )

        # The coefficients for u_t and u_eps
        coeff_term1 = alpha_ts * sigma2_denoise / (sigma2_ts + self.delta)
        coeff_term2 = alpha_st * sigma2_denoise / torch.clamp(sigma_s ** 2, min=self.delta)

        # Compute terms
        u_t = dct_2d(z_t, norm='ortho')
        term1 = idct_2d(coeff_term1 * u_t, norm='ortho')
        term2 = idct_2d(coeff_term2 * (u_t - sigma_t * dct_2d(hat_eps, norm='ortho')), norm='ortho')
        mu_denoise = term1 + term2

        # Sample from the denoising distribution
        eps = torch.randn_like(mu_denoise)
        return mu_denoise + idct_2d(torch.sqrt(sigma2_denoise) * eps, norm='ortho')