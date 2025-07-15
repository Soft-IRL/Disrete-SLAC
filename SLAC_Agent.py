import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal, Independent, Bernoulli, kl_divergence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools

class StepType:
    FIRST = 0
    MID = 1
    LAST = 2

class MultivariateNormalDiag(nn.Module):
    """
    A neural network module for a multivariate normal distribution with diagonal covariance.
    Input: Observation
    Output: Learnt MultivariateNormal distribution with diagonal covariance matrix.
    """
    def __init__(self, input_dim, hidden_dim, latent_size):
        super().__init__()
        self.latent_size = latent_size
        #print("input_dim:", input_dim, "hidden_dim:", hidden_dim, "latent_size:", latent_size)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2 * latent_size)
    
    def forward(self, *inputs):
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        out = self.output_layer(x)

        loc = out[..., :self.latent_size]
        scale_diag = F.softplus(out[..., self.latent_size:]) + 1e-5  # ensure positivity

        base  = Normal(loc, scale_diag)
        return Independent(base, 1)               # diagonal Gaussian
    
class ConstantMultivariateNormalDiag(nn.Module):
    """
        Constant diagonal Gaussian broadcast to the batch shape of *any* dummy
    input tensor.

        Returns Independent(Normal(loc, scale), 1), where
          • loc   ≡ 0
          • scale ≡ σ  (same on every coordinate unless you pass a vector)
    """

    def __init__(self, latent_size: int, scale: float | torch.Tensor = 1.0):
        super().__init__()
        self.latent_size = latent_size

        self.register_buffer("loc_const",
                             torch.zeros(latent_size))
        if isinstance(scale, torch.Tensor):
            assert scale.shape == (latent_size,)
            self.register_buffer("scale_const", scale.clone())
        else:                                # scalar σ
            self.register_buffer("scale_const",
                                 torch.ones(latent_size) * float(scale))

    def forward(self, *inputs):
        # infer batch_shape from first dummy arg (or ())
        batch_shape = inputs[0].shape if inputs else ()

        loc   = self.loc_const  .expand(*batch_shape, self.latent_size)
        scale = self.scale_const.expand_as(loc)

        base = Normal(loc, scale)            # (..., D)
        return Independent(base, 1)          # event_dim = 1  ⇒ mv Normal

class Encoder(nn.Module):
    """Encodes observations to the latent space."""
    def __init__(self, base_depth, feature_size):
        super().__init__()
        self.feature_size = feature_size
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=base_depth, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(base_depth, 2 * base_depth, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2 * base_depth, 4 * base_depth, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4 * base_depth, 8 * base_depth, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(8 * base_depth, feature_size, kernel_size=4, stride=1, padding=0)  # VALID in TF = no padding

        self.activation = nn.LeakyReLU()

    def forward(self, image):
        """
        image: Tensor of shape (..., C, H, W), e.g., [B, T, C, H, W] or [B*T, C, H, W]
        Output: Tensor of shape (..., feature_size)
        """

        original_shape = image.shape[:-3]  # Save leading dims
        B = int(torch.prod(torch.tensor(original_shape)))  # Flatten batch

        x = image.view(B, *image.shape[-3:])  # reshape to (B*T, C, H, W)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.conv5(x))  # now shape (B*T, feature_size, 1, 1)

        x = x.view(*original_shape, self.feature_size)  # reshape to (..., feature_size)
        return x

class Decoder(nn.Module):
    """Probabilistic decoder for `p(x_t | z_t^1, z_t^2)`.

    Tensor Dimension Reference
    --------------------------
    B: Batch size (number of sequences)
    N: Number of parallel environments or agents
    T: Sequence length (time steps)
    C: Number of image channels (e.g., RGB = 3)
    H: Height of the image
    W: Width of the image
#
    Example Shapes:
    - Input image (PyTorch style):    (B, N, T, C, H, W)
    - Flattened for CNN:              (B*N*T, C, H, W)  # PyTorch
    - Decoder output (reconstruction): (B, N, T, H, W, C)
    """

    def __init__(self, base_depth, channels=3, scale=1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.leaky_relu = nn.LeakyReLU()
        self.base_depth = base_depth

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.base_depth, out_channels=8 * base_depth,
                                          kernel_size=4, stride=1, padding=0)  # VALID
        self.deconv2 = nn.ConvTranspose2d(8 * base_depth, 4 * base_depth, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(4 * base_depth, 2 * base_depth, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(2 * base_depth, base_depth, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(base_depth, channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, *inputs):
        if len(inputs) > 1:
            latent = torch.cat(inputs, dim=-1)
        else:
            latent = inputs[0]

        # Flatten leading dimensions and reshape to N x C x 1 x 1
        latent_shape = latent.shape
        leading_dims = latent_shape[:-1]
        flat_latent = latent.view(-1, latent_shape[-1])  # (B, latent_dim)
        x = flat_latent.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)

        # Lazy init for first deconv layer (in_channels depends on latent size)
        if not hasattr(self, "deconv1_initialized"):
            self.deconv1 = nn.ConvTranspose2d(latent_shape[-1], 8 * self.base_depth, kernel_size=4, stride=1, padding=0)
            self.deconv1_initialized = True
            self.deconv1.to(x.device)

        x = self.leaky_relu(self.deconv1(x))
        x = self.leaky_relu(self.deconv2(x))
        x = self.leaky_relu(self.deconv3(x))
        x = self.leaky_relu(self.deconv4(x))
        x = self.deconv5(x)  # no activation here (loc of Gaussian)

        # --- reshape back to (leading_dims, C, H, W) ----------------
        out_shape = (*leading_dims, *x.shape[1:]) # (..., C, H, W)
        x = x.view(out_shape) 

        # Return Independent Normal (per-pixel)
        return Independent(Normal(loc=x, scale=self.scale), reinterpreted_batch_ndims=3) #  indicates that the last 3 dimensions (H, W, C) should be considered part of a single event, i.e., an image


class ModelDistributionNetwork(nn.Module):
    def __init__(self, action_dim, args, model_reward=False, model_discount=False,
                 decoder_stddev=np.sqrt(0.1, dtype=np.float32), reward_stddev=None):
        
        super().__init__()
        self.base_depth = args.base_depth
        self.encoder_output_size = 8 * self.base_depth
        self.action_dim = action_dim
        self.device = args.device
        self.lr = args.m_learning_rate
        self.epsilon = args.start_e
        self.latent1_size = args.latent1_size
        self.latent2_size = args.latent2_size
        self.model_reward = model_reward
        self.model_discount = model_discount
        self.decoder_stddev = decoder_stddev
        self.reward_stddev = reward_stddev
        self.kl_analytic = args.kl_analytic
        
        # p(z_1^1)
        self.latent1_first_prior = ConstantMultivariateNormalDiag(self.latent1_size, scale=1.0).to(self.device)
        # p(z_1^2 | z_1^1)
        self.latent2_first_prior = MultivariateNormalDiag(self.latent1_size, 8 * self.base_depth, self.latent2_size).to(self.device)
        # p(z_{t+1}^1 | z_t^2, a_t)
        self.latent1_prior = MultivariateNormalDiag(self.latent2_size + self.action_dim, 8 * self.base_depth, self.latent1_size).to(self.device)
        # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_prior = MultivariateNormalDiag(self.latent1_size + self.latent2_size + self.action_dim, 8 * self.base_depth, self.latent2_size).to(self.device)

         # q(z_1^1 | x_1)
        self.latent1_first_posterior = MultivariateNormalDiag(self.encoder_output_size, 8 * self.base_depth, self.latent1_size).to(self.device)
        # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
        self.latent2_first_posterior = self.latent2_first_prior
        # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
        self.latent1_posterior = MultivariateNormalDiag(self.encoder_output_size + self.latent2_size + self.action_dim, 8 * self.base_depth, self.latent1_size).to(self.device)

        # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_posterior = self.latent2_prior

        # compresses x_t into a vector
        self.encoder = Encoder(self.base_depth, 8* self.base_depth).to(self.device)
        # p(x_t | z_t^1, z_t^2)
        self.decoder = Decoder(self.base_depth, scale=self.decoder_stddev).to(self.device)

        # ------------ optimizer ----------------------------------------------------
        # gather *all* parameters of sub-modules registered above
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    """def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()"""
        

    def compute_loss(self, images, actions, step_types, rewards=None, discounts=None, latent_posterior_samples_and_dists=None):
        
        #This gets the number of transitions, which is one less than the number of steps.
        sequence_length = step_types.shape[1] - 1
        
        #If not provided, sample the latent variables and distributions from the encoder (inference model) conditioned on the current sequence.
        if latent_posterior_samples_and_dists is None:
            latent_posterior_samples_and_dists = self.sample_posterior(images, actions, step_types) # q(z1_0 | x0)  , q(z2_0 | z1_0), q(z1_t | x_t, z2_{t-1}, a_{t-1}), q(z2_t | z1_t, z2_{t-1}, a_{t-1})
        
        #Latent variables and their corresponding distributions for both z1 and z2.
        (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists

        # ------------------------------------------------------------------ build PRIOR distributions (aligned)
        p_z1, p_z2 = self.get_prior(z1_post, z2_post, actions, step_types) # For every t=0…T−1: pψ(zt+1∣zt2,at) and pψ(zt+12∣zt+1,zt2,at)

        # ------------------------------------------------------------------ KL terms
        if self.kl_analytic:
            kl_z1 = kl_divergence(q_z1.dists, p_z1.dists).sum(-1)   
            kl_z2 = kl_divergence(q_z2.dists, p_z2.dists).sum(-1)
        else:
            # sample-based (still broadcasts)
            kl_z1 = q_z1.log_prob(z1_post) - p_z1.log_prob(z1_post)         # (B,T+1)
            kl_z2 = q_z2.log_prob(z2_post) - p_z2.log_prob(z2_post)
        
        # ------------------------------------------------------------------ recon term
        x_dist   = self.decoder(z1_post, z2_post)           # p(x|z)  Independent Normal
        log_px   = x_dist.log_prob(images).sum(1)           # (B,)

        # ------------------------------------------------------------------ ELBO
        elbo = log_px - kl_z1 - kl_z2

        # ------------------------------------------------ loss
        loss = -elbo.mean()

        return loss, (z1_post, z2_post)


    def sample_posterior(self, images, actions, step_types, features=None):
        """
        Sample latent1 and latent2 from the approximate posterior.
        Uses conditional_distribution and returns both samples and their stacked distributions.
        """

        # The sequence has T+1 timesteps (images.shape[1]), but actions only span T transitions. So we truncate actions to match the correct length.
        sequence_length = step_types.shape[1] - 1

        actions = actions[:, :sequence_length]

        if features is None:
            features = self.encoder(images)  # shape: (B, T+1, feat_dim)

        # Swap batch and time axes to get shape (T+1, B, ...)
        features = features.transpose(0, 1)       # (T+1, B, feature_dim)
        actions = actions.transpose(0, 1)         # (T, B, action_dim)
        step_types = step_types.transpose(0, 1)   # (T+1, B)

        latent1_dists, latent1_samples = [], []
        latent2_dists, latent2_samples = [], []

        for t in range(sequence_length + 1):
            if t == 0:
                # Initial step: no previous latents
                latent1_dist = self.latent1_first_posterior(features[t])           # q(z1_0 | x0)
                latent1_sample = latent1_dist.rsample()

                latent2_dist = self.latent2_first_posterior(latent1_sample)        # q(z2_0 | z1_0)
                latent2_sample = latent2_dist.rsample()
 
            else:
                latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1].unsqueeze(-1))  # q(z1_t | x_t, z2_{t-1}, a_{t-1})
                # Use conditional_distribution to conditionally select the correct posterior. Sample z1_t.
                latent1_sample = latent1_dist.rsample()
                latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1].unsqueeze(-1)) #  q(z2_t | z1_t, z2_{t-1}, a_{t-1})
                latent2_sample = latent2_dist.rsample()

            latent1_dists.append(latent1_dist)
            latent1_samples.append(latent1_sample)
            latent2_dists.append(latent2_dist)
            latent2_samples.append(latent2_sample)

        # Re-stack samples into shape (B, T+1, D)
        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        # Stack distributions into StackedNormal objects
        latent1_dists = stack_distributions(latent1_dists)
        latent2_dists = stack_distributions(latent2_dists)

        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)
    
    def get_prior(self, z1_post, z2_post, actions, step_types=None):
        
        sequence_length = step_types.shape[1] - 1
        actions = actions[:, :sequence_length]

        # t = 0  ---------
        p_z1_first = self.latent1_first_prior(step_types[:, :1])          # (B,1,d1) pψ​(z01​)
        p_z2_first = self.latent2_first_prior(z1_post[:, :1])             # (B,1,d2) pψ​(z02​∣z01​)

        # t = 1 … T  -----
        p_z1_auto  = self.latent1_prior(z2_post[:, :sequence_length], actions.unsqueeze(-1)) # For every t=0…T−1: pψ(zt+1∣zt2,at)
        p_z2_auto  = self.latent2_prior(z1_post[:, 1:], z2_post[:, :sequence_length], actions.unsqueeze(-1)) # For every t=0…T−1: pψ(zt+12∣zt+1,zt2,at)

        #------------------------ p_z1 -------------------------
        loc_first   = p_z1_first.base_dist.loc          # (B, 1, d1)
        scale_first = p_z1_first.base_dist.scale        # (B, 1, d1)
        loc_auto    = p_z1_auto .base_dist.loc          # (B, T, d1)
        scale_auto  = p_z1_auto .base_dist.scale        # (B, T, d1)
        locs_z1     = torch.cat([loc_first,   loc_auto],   dim=1)   # (B, T+1, d1)
        scales_z1   = torch.cat([scale_first, scale_auto], dim=1)   # (B, T+1, d1)
        p_z1 = StackedNormal(locs_z1, scales_z1)  

        #------------------------ p_z2 -------------------------
        loc_first2   = p_z2_first.base_dist.loc
        scale_first2 = p_z2_first.base_dist.scale
        loc_auto2    = p_z2_auto .base_dist.loc
        scale_auto2  = p_z2_auto .base_dist.scale
        locs_z2   = torch.cat([loc_first2,   loc_auto2],   dim=1)
        scales_z2 = torch.cat([scale_first2, scale_auto2], dim=1)
        p_z2 = StackedNormal(locs_z2, scales_z2)

        return p_z1, p_z2
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

def stack_distributions(dists):
    locs = torch.stack([d.base_dist.loc for d in dists], dim=1)
    scales = torch.stack([d.base_dist.scale for d in dists], dim=1)
    return StackedNormal(locs, scales)

class StackedNormal:
        """Utility to represent a sequence of Normal distributions as a single distribution."""
        def __init__(self, locs, scales):
            self.dists = Independent(Normal(locs, scales), 1)

        def log_prob(self, value):
            return self.dists.log_prob(value)

        def sample(self):
            return self.dists.rsample()

        @property
        def loc(self):
            return self.dists.base_dist.loc