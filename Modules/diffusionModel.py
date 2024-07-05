import torch
import torch.nn.functional as F

class DiffusionModel:

    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps=200):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps

        # linear scheduler
        #self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.betas = self.linear_schedule()

        # cosine scheduler
        #self.betas = self.cosine_schedule(s=0.008)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        #-------#
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        #----#

    def forward(self, x_0, t, device):
        """
        x_0: (B, C, H, W)
        t: (B,)
        """
        # sample noise from gaussian distribution
        # noise = torch.randn_like(x_0)
        noise = self.get_noise(x_0)
        # alpha cumulative product with right dimension
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        # sqrt( 1 - cumulative product )
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
        # mean
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        # variance
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        # noise image obtained
        # torch.clip(mean + variance, -1, 1)
        return mean + variance, noise.to(device)

    def backward_gnn(self, x, t, predicted_noise):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.

        mean = (1 / sqrt(alpha_t) * (x_t - (beta_t * model_error_prediction of image x_t)/ sqrt(1 - alpha_hat)))
        """

        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        # can take different values betas_t simplest one
        #posterior_variance_t = betas_t
        #------#
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        #----#


        # we passed through all the denoising step we are at time T0
        if t == 0:
            return mean
        else:
            # noise = torch.randn_like(x)
            noise = self.get_noise(x)
            variance = torch.sqrt(posterior_variance_t) * noise
            # clip the value between -1 and 1
            return torch.clip(mean + variance, -1, 1)



    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick the values 
        according to the indices stored in `t`
        """
        # take only the t values we are interesting in
        result = values.gather(-1, t.cpu())
        """
        if
        x_shape = (5, 3, 64, 64)
            -> len(x_shape) = 4
            -> len(x_shape) - 1 = 3

        and thus we reshape `out` to dims
        (batch_size, 1, 1, 1)

        """
        # reshape in a way we can use it
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def linear_schedule(self):
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        return betas

    def cosine_schedule(self, s=0.008):
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def get_noise(x_0):
        """
      Return noise sampled from a normal distribution with a mean of 0 and a variance of 1,
      followed by rescaling to the range of -1 to 1.
      """
        # Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard
        # deviation are given.
        # nor = torch.randn_like(x_0) * 0.1
        nor = torch.randn_like(x_0)
        # rescale to -1 to 1
        # nor = (((nor - nor.min()) / (nor.max() - nor.min())) * 2 - 1) * 0.1
        return nor
