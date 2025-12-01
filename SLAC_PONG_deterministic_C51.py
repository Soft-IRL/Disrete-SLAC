import os
import glob
import io
from PIL import Image

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.spaces import Box, Discrete
import ale_py
import numpy as np
#import matplotlib.pyplot as plt
import random
import time
from dataclasses import dataclass
import tyro
#import cv2
import copy
from tqdm import tqdm
from contextlib import contextmanager
import collections, math

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.utils as vutils
from torch.distributions import kl_divergence
from SequenceReplayBuffer import SequenceReplayBuffer
import matplotlib.pyplot as plt

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from SLAC_world_deterministic import ModelDistributionNetwork
from SLAC_Agent_DRLDoubleDQN import D3QNAgent



script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""

    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = False
    """if toggled, the trained model will be saved to disk"""
    from_scratch: bool = False
    """if toggled, the model will be trained from scratch"""
    ckpt_path = "..//checkpoints//ALE//Pong-v5__SLAC_PONG_deterministic__1__941_full_pretrain_checkpoint//model_pretrained_kl_teacher.pth"
    """If not from scratch, path to the pretrained model"""
    wandb_project_name: str = "SLAC_PONG_C51"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    q_learning_rate: float = 3e-4
    """the learning rate of the q_network optimizer"""
    m_learning_rate: float = 1e-4
    """the learning rate of the model_network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of the alpha optimizer"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    gamma: float = 0.99
    """the discount factor"""
    learning_starts: int = 1000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    target_network_frequency: int = 10000
    """the frequency of target network update"""
    tau: float = 0.005
    """the polyak averaging factor for target network update"""
    sequence_len : int = 8
    """the length of the sequence for training"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    kl_analytic: bool = True
    """if toggled, the KL divergence will be computed analytically"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    base_depth: int = 32
    """the base depth of the model network"""
    latent1_size: int = 32
    """the size of the first latent variable"""
    latent2_size: int = 256
    """the size of the second latent variable"""
    hidden_dims: tuple = (256, 256)
    """the hidden dimensions of the Q-network"""

    # =========== The following is used for C51
    # =========== Have to carefully tune Q_min, Q_max values 
    # =========== suggestion: get inspiration from deterministic version
    N_atoms: int = 51
    """the number of atoms in the C51 approach (51)"""
    Q_min: float = -1.
    """the minimum value of the support of the C51 approach """
    Q_max: float = 1.
    """the maximum value of the support of the C51 approach """


def make_env(env_id, seed, idx, capture_video=False, run_name=""):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1, full_action_space=False)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, frameskip=1, full_action_space=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = RallyDoneWrapper(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.GrayScaleObservation(env)


        env.action_space.seed(seed)
        return env

    return thunk

class RallyDoneWrapper(gym.Wrapper):
    """End episode after each rally (when a point is scored)."""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """Modify step function to end the episode after each rally."""
        obs, reward, done, truncated, info = self.env.step(action)

        # End the episode when a point is scored (reward ≠ 0)
        if reward != 0:
            done = True  # Force episode to end

        return obs, reward, done, truncated, info

def inspect_terminal_sequences(rb, num_sequences=20):
    """
    Sample `num_sequences` batches of size 1 from the buffer and
    print those that contain a done=True flag.
    """
    seq_len = rb.seq_len
    found = 0
    trial = 0

    while found < num_sequences and trial < 1000:
        trial += 1
        batch = rb.sample(1)                 # batch_size = 1
        r   = batch["reward"].cpu().numpy().squeeze()      # (S,)
        d   = batch["done"].cpu().numpy().squeeze()        # (S,)
        stp = batch["step_type"].cpu().numpy().squeeze()   # (S,)

        if d.any():                           # contains a terminal state?
            found += 1
            term_idx = np.where(d)[0][0]      # first occurrence
            print(f"\n=== Sequence {found}  "
                  f"(terminal at position {term_idx}) ===")
            print("idx | step_type | reward | done")
            print("--------------------------------")
            for i in range(seq_len):
                print(f"{i:3d} |     {stp[i]}     |  {r[i]:5.1f} |  {d[i]}")
            # simple assertions
            assert (d[:seq_len-1] == 0).all(),   "done in middle of slice!"
            if term_idx > 0:
                print("-- terminal reward is r[{}] = {:.1f}".format(
                      term_idx-1, r[term_idx-1]))
            else:
                print("-- episode ends exactly at first index (rare)")

    if found == 0:
        print("No terminal states found in 1000 samples – "
              "replay buffer may still be small.")

timings = collections.defaultdict(float)     # global dict

@contextmanager
def tic(name):
    t0 = time.perf_counter()
    yield
    timings[name] += time.perf_counter() - t0

def register_nan_hooks(module, name=""):
    for n, p in module.named_parameters():
        full_name = f"{name}.{n}" if name else n
        p.register_hook(
            lambda grad, n=full_name: (
                print(f"⚠️ NaN in grad of {n}") if torch.isnan(grad).any() else None
            )
        )

def visualize_rollout(images, title_prefix="Frame", cmap='gray'):
    """
    Visualize a sequence of grayscale images with shape (T, 1, H, W)
    
    :param images: Tensor or ndarray of shape (T, 1, H, W)
    :param title_prefix: Prefix for subplot titles
    :param cmap: Color map to use (default 'gray')
    """
    T = images.shape[0]
    plt.figure(figsize=(T * 2, 2))

    for t in range(T):
        img = images[t, 0].cpu().numpy()  # shape (H, W)
        plt.subplot(1, T, t + 1)
        plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"{title_prefix} {t}")

    plt.tight_layout()
    plt.show()

def log_rollout_grid(images, step, caption="Rollout"):
    """
    Log a rollout as a single grid image to Weights & Biases.
    
    :param images: Tensor of shape (T, 1, H, W)
    :param step: Global step or env step
    :param caption: Caption for the image
    """
    # Normalize and repeat channel to get (T, 3, H, W) for W&B
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)  # grayscale → RGB

    grid = vutils.make_grid(images, nrow=images.shape[0], pad_value=1)

    wandb.log({caption: wandb.Image(grid, caption=caption)}, step=step)

def save_world_model_ckpt(model, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"[✓] Saved world model checkpoint at {path}")

def log_checkpoint_to_wandb(path, step, run, *, aliases=("latest",), name="slac_world_model"):
    art = wandb.Artifact(name=name, type="model", metadata={"step": step})
    art.add_file(path, name=os.path.basename(path))             # optional 'name=' keeps a clean filename
    run.log_artifact(art, aliases=list(aliases))
    print(f"Uploaded {path} to W&B as artifact '{name}' with aliases {aliases}")

@torch.no_grad()
def visualize_terminal_debug(agent, model, z1, z2, actions, rewards, dones, global_step):
    """
    Visualizes the terminal state of a batch:
    1. Reconstructed Image (to see if the model 'sees' the ball)
    2. Target Distribution (what the Bellman update says value should be)
    3. Predicted Distribution (what the Agent believes the value is)
    Logs the resulting composite image to WandB.
    """
    
    # 1. Find a terminal sequence
    mask_term = dones[:, -1]
    if not mask_term.any():
        return  # No terminal state in this batch

    # Pick the first terminal index
    idx = torch.nonzero(mask_term, as_tuple=True)[0][0]

    # 2. Extract Data for the Terminal Step (t = T)
    # We use the final latent state (z at index -1)
    z1_last = z1[idx, -1].unsqueeze(0)  # (1, d1)
    z2_last = z2[idx, -1].unsqueeze(0)  # (1, d2)
    
    # Concatenate for Agent
    z_cat = torch.cat([z1_last, z2_last], dim=-1)
    z_cat = F.layer_norm(z_cat, z_cat.shape[-1:])

    #a_last = actions[idx, -1].long().unsqueeze(0)  # (1,)
    a_last_idx = actions[idx, -1].long().item()
    r_last = rewards[idx, -1].unsqueeze(0)         # (1,)
    d_last = torch.ones_like(r_last)               # (1,) - Always True for terminal

    # 3. Reconstruct Image (VAE Check)
    # This verifies if the latent z actually captured the ball's position
    recon_tensor = model.decoder(z1_last, z2_last) # (1, 1, 64, 64)
    recon_img = recon_tensor.squeeze().cpu().numpy()
    
    # 4. Compute Distributions
    
    # A) PREDICTED Distribution (Online Net)
    logits = agent._logits(z_cat) # (1, A, N)
    probs = logits.softmax(dim=-1)
    # Select the distribution for the specific action taken (the hit/miss)
    #pred_dist = probs[0, a_last.item()].cpu().numpy()
    
    # B) TARGET Distribution (Bellman Projection)
    # Online Selection (Greedy)
    q_online = (probs * agent.atoms).sum(-1)
    a_star = q_online.argmax(dim=1)
    
    # Target Net Evaluation (Dummy next state, masked by done=1 anyway)
    logits_target = agent.q_target_net(z_cat).view(1, agent.n_actions, agent.n_atoms)
    p_target = logits_target.softmax(-1)
    p_next_a = p_target[0, a_star]
    
    # Projection
    target_dist_tensor = agent._target_projection(p_next_a, r_last, d_last)
    target_dist = target_dist_tensor[0].cpu().numpy()

    # 5. Plotting
    n_actions = agent.n_actions
    #fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig = plt.figure(figsize=(6 * (n_actions + 1), 5))
    
    """# Subplot 1: Reconstructed Image
    axes[0].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Reconstructed State\nReward: {r_last.item():.1f}", fontsize=14)
    axes[0].axis('off')
    
    # Shared X-axis for distributions
    atoms_x = agent.atoms.cpu().numpy()
    bar_width = (atoms_x[-1] - atoms_x[0]) / len(atoms_x)
    
    # Subplot 2: Target Distribution
    axes[1].bar(atoms_x, target_dist, width=bar_width, color='green', alpha=0.7)
    axes[1].set_title("Target Distribution\n(Ground Truth)", fontsize=14)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Value")
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Predicted Distribution
    axes[2].bar(atoms_x, pred_dist, width=bar_width, color='blue', alpha=0.7)
    pred_mean = (pred_dist * atoms_x).sum()
    axes[2].set_title(f"Predicted Distribution\nMean: {pred_mean:.3f}", fontsize=14)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("Value")
    axes[2].grid(True, alpha=0.3)"""

    # Subplot 1: Image
    ax_img = fig.add_subplot(1, n_actions + 1, 1)
    ax_img.imshow(recon_img, cmap='gray', vmin=0, vmax=1)
    ax_img.set_title(f"State (Reward: {r_last.item():.0f})\nTerminated: True", fontsize=12)
    ax_img.axis('off')

    # Subplots for Actions
    atoms_x = agent.atoms.cpu().numpy()
    bar_width = (atoms_x[-1] - atoms_x[0]) / len(atoms_x)
    
    for a_i in range(n_actions):
        ax = fig.add_subplot(1, n_actions + 1, a_i + 2)
        
        # Get prediction for this specific action
        pred_dist = probs[0, a_i].cpu().numpy()
        pred_mean = (pred_dist * atoms_x).sum()
        
        # Colors
        is_taken = (a_i == a_last_idx)
        bar_color = 'blue' if not is_taken else 'purple'
        alpha = 0.6 if is_taken else 0.4
        
        # Plot Prediction
        ax.bar(atoms_x, pred_dist, width=bar_width, color=bar_color, alpha=alpha, label='Pred')
        
        # Plot Target (Only if taken)
        if is_taken:
            ax.bar(atoms_x, target_dist, width=bar_width, color='green', alpha=0.5, label='Target')
            ax.legend()
            title_text = f"Action {a_i} (TAKEN)\nQ: {pred_mean:.3f}"
            # visual border for the taken action
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        else:
            title_text = f"Action {a_i}\nQ: {pred_mean:.3f}"

        ax.set_title(title_text, fontsize=12, fontweight='bold' if is_taken else 'normal')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Return Value")
    
    plt.tight_layout()
    
    # 6. Save to buffer and Log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    
    wandb.log({
        "debug/terminal_analysis": wandb.Image(image, caption=f"Step {global_step}: Terminal Analysis"),
        "global_step": global_step
    })
    
    plt.close(fig)
    buf.close()

class FreezeParams:
    def __init__(self, params):
        self.params = list(params)
        self.prev = None
    def __enter__(self):
        self.prev = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad_(False)
    def __exit__(self, *exc):
        for p, r in zip(self.params, self.prev):
            p.requires_grad_(r)

def huber(x, delta=1.0):
    a = x.abs()
    return torch.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta))

def compute_loss(model, images, actions, step_types, step=None, rewards=None, discounts=None, latent_posterior_samples_and_dists=None, use_kl=False,  rollout_K=3):
    #If not provided, sample the latent variables and distributions from the encoder (inference model) conditioned on the current sequence.
    if latent_posterior_samples_and_dists is None:
        latent_posterior_samples_and_dists = model.sample_posterior(images, actions, step_types) # q(z1_0 | x0)  , q(z2_0 | z1_0), q(z1_t | x_t, z2_{t-1}, a_{t-1}), q(z2_t | z1_t, z2_{t-1}, a_{t-1})
        
    #Latent variables and their corresponding distributions for both z1 and z2.
    (z1_post, z2_post), (q_z1, q_z2) = latent_posterior_samples_and_dists
    preds_imgs   = model.decoder(z1_post, z2_post)
    mse = ((images - preds_imgs)**2).mean()
    output = {"mse": mse}

    if use_kl:
        p_z1, p_z2, p_z1_auto, p_z2_auto = model.get_prior(z1_post, z2_post, actions, step_types) # For every t=0…T−1: pψ(zt+1∣zt2,at) and pψ(zt+12∣zt+1,zt2,at)
        kl_z1 = kl_divergence(q_z1.dists, p_z1.dists).sum(-1)

        q1 = q_z1.dists.base_dist
        p1 = p_z1.dists.base_dist

        # KL balancing + free bits
        tau = 0.02; alpha = 0.8
        p1_det = torch.distributions.Normal(p1.loc.detach(), p1.scale.detach())
        q1_det = torch.distributions.Normal(q1.loc.detach(), q1.scale.detach())

        kl_q_raw = torch.distributions.kl_divergence(q1, p1_det)          # (B,T+1,D)
        kl_p_raw = torch.distributions.kl_divergence(q1_det, p1)

        kl_q = (kl_q_raw - tau).clamp_min(0).sum(-1).mean()               # scalar
        kl_p = (kl_p_raw - tau).clamp_min(0).sum(-1).mean()

        kl_bal = alpha * kl_q + (1 - alpha) * kl_p
        target = 0.2  # aim each auxiliary to be ~20% of recon
        kl_term   = (target * mse.detach() / (kl_bal.detach() + 1e-8)).clamp_(0, 1.0) * kl_bal
        #pred_term = (target * mse.detach() / (pred_loss.detach() + 1e-8)).clamp_(0, 1.0) * pred_loss


        # ----- 2) One-step prior consistency (GT inputs → predict t+1) ------------
        with torch.no_grad():
            z1_det, z2_det = z1_post.detach(), z2_post.detach()

        p_z1, p_z2, p_z1_auto, p_z2_auto = model.get_prior(z1_det, z2_det, actions, step_types)

        z1_next = z1_det[:, 1:]                 # (B,T,·)
        z2_next = z2_det[:, 1:]                 # (B,T,·)
        mu1 = p_z1_auto.base_dist.loc           # (B,T,·)
        mu2 = p_z2_auto.base_dist.loc           # (B,T,·)
        
        latent_tf_mse = ((mu1 - z1_next)**2).mean() + ((mu2 - z2_next)**2).mean()

        # Pixel one-step (decode predicted latents vs x_{t+1})
        x_pred_next = model.decoder(mu1, mu2)                      # (B,T,1,H,W)
        pix_tf_mse  = ((images[:, 1:] - x_pred_next) ** 2).mean()

        loss = mse + kl_term + 0.1*latent_tf_mse + pix_tf_mse

        output["kl_z1"] = kl_z1.mean()
        output["kl_q_raw"] = kl_q_raw.mean()
        output["kl_q"] = kl_q
        output["kl_term"] = kl_term
        output["latent_tf_mse"] = latent_tf_mse
        output["pix_tf_mse"] = pix_tf_mse
        #output["lat_roll_mse"] = lat_roll_mse
        #output["pix_roll_mse"] = pix_roll_mse

        return loss, output
    else:
        return mse, output
    
def evaluate_and_record(args, model, agent, step, run_name):
    """
    Runs a single evaluation episode/rally, records the video, 
    and logs it to Weights & Biases.
    """
    print(f"\n[Eval] Starting evaluation recording at step {step}...")
    
    # 1. Create a dedicated evaluation environment
    # We force num_envs=1 and capture_video=True
    video_run_name = f"{run_name}_eval_{step}"
    eval_env = SyncVectorEnv([make_env(
        args.env_id, 
        args.seed + 1000, # Different seed to avoid overfitting 
        0, 
        capture_video=True, 
        run_name=video_run_name
    )])
    
    device = args.device
    obs, _ = eval_env.reset()
    
    # 2. Initialize Latent State (Belief) - Same logic as training
    # SLAC requires us to 'burn in' the first state using the first_posterior
    prev_actions = torch.zeros(1, dtype=torch.long, device=device) # (1,)
    
    with torch.no_grad():
        imgs0 = torch.from_numpy(obs).unsqueeze(1).to(device).float() / 255.0
        feat0 = model.encoder(imgs0)
        z1_bel = model.latent1_first_posterior(feat0).rsample()
        z2_bel = model.latent2_first_posterior(z1_bel).rsample()

    done = False
    total_reward = 0
    
    # 3. Game Loop
    while not done:
        # --- A. Update Belief (Posterior) using current observation ---
        # This allows the agent to 'see' the current frame and update velocity estimates
        with torch.no_grad():
            imgs = torch.from_numpy(obs).unsqueeze(1).to(device).float() / 255.0
            feat = model.encoder(imgs)
            
            a_one = F.one_hot(prev_actions, num_classes=model.action_dim).float()
            
            # q(z^1_t | x_t, z^2_{t-1}, a_{t-1})
            q1 = model.latent1_posterior(feat, z2_bel, a_one)
            z1_t = q1.rsample()
            
            # q(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            q2 = model.latent2_posterior(z1_t, z2_bel, a_one)
            z2_t = q2.rsample()
            
            # Update belief for next step
            z1_bel, z2_bel = z1_t, z2_t

        # --- B. Select Action ---
        # Note: We act greedily (deterministic=True or epsilon=0) for evaluation
        with torch.no_grad():
            z_cat = torch.cat([z1_bel, z2_bel], dim=1)
            # D3QNAgent.act is usually epsilon-greedy. 
            # For eval, we usually want the purely greedy action (argmax).
            # If your agent.act() doesn't support a deterministic flag, 
            # you can manually call the network:
            
            # Assuming C51 Agent:
            z_cat = F.layer_norm(z_cat, z_cat.shape[-1:])
            logits = agent._logits(z_cat) 
            q_values = (logits.softmax(-1) * agent.atoms).sum(-1)
            action = q_values.argmax(dim=1)
            print(q_values)
            print(action)

        # --- C. Step Environment ---
        # Map back to real action space if using the mapping
        ACTION_MAPPING = {0: 0, 1: 2, 2: 3}
        real_action = np.array([ACTION_MAPPING[int(action.item())]])
        print(real_action)
        
        obs, reward, terminations, truncations, infos = eval_env.step(real_action)
        
        # SLAC Logic: Update previous action for next iteration
        prev_actions = action
        total_reward += reward[0]
        
        # Check done (SyncVectorEnv returns arrays)
        done = terminations[0] or truncations[0]

    # 4. Cleanup and Upload
    eval_env.close()
    
    # Find the video file. RecordVideo usually saves to videos/{run_name}/rl-video-step-0.mp4
    video_dir = f"videos/{video_run_name}"
    # Glob pattern to find the mp4 file
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if video_files:
        video_path = video_files[0]
        print(f"[Eval] Video recorded at: {video_path}. Uploading to WandB...")
        
        wandb.log({
            "eval/video": wandb.Video(video_path, fps=30, format="mp4", caption=f"Step {step} Reward {total_reward:.1f}"),
            "eval/reward": total_reward,
            "global_step": step
        })
    else:
        print("[Eval] Warning: No video file found.")


if __name__ == '__main__':
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.login()
        run = wandb.init(
            project=args.wandb_project_name,
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=False,
        )
        run.log_code(root=script_dir)
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device
    envs = SyncVectorEnv([make_env(args.env_id, args.seed + i, i,
              args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ACTION_MAPPING = {0: 0, 1: 2, 2: 3}
    n_actions = len(ACTION_MAPPING.keys())
    action_space = Discrete(n_actions)
    obs,_ = envs.reset(seed=args.seed)
    agent = D3QNAgent(action_space.n, args)   
    Model = ModelDistributionNetwork(action_space, args)
    use_kl = False

    if not args.from_scratch:
        latent_dim = args.latent1_size + args.latent2_size
        if Model.decoder.deconv1.weight.shape[0] != latent_dim:
            Model.decoder.deconv1 = nn.ConvTranspose2d(latent_dim, 8*args.base_depth, kernel_size=4, stride=1, padding=0).to(args.device)
            Model.decoder.deconv1_initialized = True
            use_kl=True
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        Model.load_state_dict(ckpt["model_state"])

    eval_counter = 0
    rb = SequenceReplayBuffer(
        capacity   = args.buffer_size,
        obs_shape  = (1,obs.shape[1],obs.shape[2]),
        act_shape  = (),
        seq_len    = args.sequence_len,
        device     = args.device,)
    
    obs, _ = envs.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
        
    ############################# THIS IS THE MODEL PRETRAINING ###############################
    if args.from_scratch:
        # 0. collect bootstrap data ------------------------------------------------
        print("Collecting bootstrap data for model pretraining...")
        while rb.ptr < 10_000:                 # or 10 episodes for DM-Control
            actions = np.array([action_space.sample() for _ in range(envs.num_envs)])
            real_actions = np.array([ACTION_MAPPING[a.item()] for a in actions])
            next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
            done = terminations | truncations 
            step_type = np.where(
                        done,               2,
                        np.where(episode_first, 0, 1)
                    ).astype(np.int64)      # shape (n_envs,)
                
            for k in range(args.num_envs):
                    rb.add(obs[k][None],
                    actions[k],
                    rewards[k],
                    done[k],
                    step_type[k])      
            obs = next_obs
            episode_first = done.copy() 
            
        # 1. model-only optimisation loop -----------------------------------------
        print("Pretraining the model...")
        for pretrain_step in tqdm(range(10_000)):
            batch = rb.sample(args.batch_size)
            images  = (batch["obs"].float() / 255.)
                #print(images.dtype, images.min().item(), images.max().item())
            actions = batch["action"]
            step_ty = batch["step_type"]
            model_loss, output = compute_loss(Model, images, actions, step_ty, step=pretrain_step, use_kl=use_kl)
            Model.optimizer.zero_grad()
            model_loss.backward()
            Model.optimizer.step()

            if args.track:
                if (pretrain_step) % 10_000 == 0:
                    sequence = rb.sample(1)
                    images  = (sequence["obs"].float() / 255.)
                    actions = sequence["action"]
                    step_ty = sequence["step_type"]
                    log_rollout_grid(images[0], step=0, caption="Sequence")
                    x_pred, aux = Model.one_step_prior_predict(images, actions, step_ty)
                    log_rollout_grid(x_pred[0], step=0, caption="One-step Prior Predict")
                    recon = Model.visual_diagnostics(images, actions, step_ty)
                    log_rollout_grid(recon, step=0, caption="Recon")
                    
                    preds_imgs, fd_pred, fd_true, mask = Model.build_motion_mask(images, actions, step_ty)
                    log_rollout_grid(preds_imgs.squeeze(0), step=0, caption="Predicted Images")

                if (pretrain_step) % 500 == 0:
                    writer.add_scalar("losses/model_loss", model_loss.item(), pretrain_step)
                    writer.add_scalar("losses/mse", output["mse"].item(), pretrain_step)
                    if use_kl:
                        writer.add_scalar("losses/kl_z1", output["kl_z1"].item(), pretrain_step)
                        writer.add_scalar("losses/kl_q_raw", output["kl_q_raw"].item(), pretrain_step)
                        writer.add_scalar("losses/kl_q", output["kl_q"].item(), pretrain_step)
                        writer.add_scalar("losses/kl_term", output["kl_term"].item(), pretrain_step)
                        writer.add_scalar("losses/latent_tf_mse", output["latent_tf_mse"].item(), pretrain_step)
                        writer.add_scalar("losses/pix_tf_mse", output["pix_tf_mse"].item(), pretrain_step)

        if args.save_model:
            path = f"checkpoints\\{run_name}\\model_pretrained_kl_teacher.pth"
            save_world_model_ckpt(Model, pretrain_step+1, path)
            log_checkpoint_to_wandb(path, pretrain_step+1, run, aliases=("pretrain", "latest"))

        ######################## END OF MODEL PRETRAINING ###############################
        print("Model pretraining completed.")


    obs, _ = envs.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)
    prev_actions = torch.zeros(args.num_envs, dtype=torch.long, device=args.device) # NOOP
    start_time = time.time()

    with torch.no_grad():
        imgs0  = torch.from_numpy(obs).unsqueeze(1).to(args.device).float() / 255.0
        feat0  = Model.encoder(imgs0)                           # (N, feat)
        z1_bel = Model.latent1_first_posterior(feat0).rsample() # (N, d1)
        z2_bel = Model.latent2_first_posterior(z1_bel).rsample()# (N, d2)

    for global_step in range(args.total_timesteps):
        agent.epsilon = agent.linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.total_timesteps), global_step)

     # -------- Bayes filter: PREDICT (use PRIORS) --------
        with torch.no_grad():
            a_one  = F.one_hot(prev_actions, num_classes=Model.action_dim).float()  # (N,A)
            # p(z^1_t | z^2_{t-1}, a_{t-1})
            p1     = Model.latent1_prior(z2_bel, a_one).base_dist
            z1_prd = p1.loc  # mean prediction (lower variance than sampling)
            # p(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            p2     = Model.latent2_prior(z1_prd, z2_bel, a_one).base_dist
            z2_prd = p2.loc

            # -------- Bayes filter: UPDATE (use POSTERIORS with current frame) --------
        with torch.no_grad():
            imgs = torch.from_numpy(obs).unsqueeze(1).to(device).float() / 255.0
            feat = Model.encoder(imgs)  # (N, feat)
            # q(z^1_t | x_t, z^2_{t-1}, a_{t-1})
            q1   = Model.latent1_posterior(feat, z2_bel, a_one)
            z1_t = q1.rsample()
            # q(z^2_t | z^1_t, z^2_{t-1}, a_{t-1})
            q2   = Model.latent2_posterior(z1_t, z2_bel, a_one)
            z2_t = q2.rsample()

            z1_bel, z2_bel = z1_t, z2_t
        
        if random.random() < agent.epsilon:     
            actions = torch.as_tensor([action_space.sample() for _ in range(args.num_envs)], device=device, dtype=torch.long)  # (N,)
        else:
            z_cat = torch.cat([z1_bel, z2_bel], dim=1)     # (N, d1+d2)
            actions = agent.act(z_cat).squeeze(1).to(device) # (N,)
            # map to env actions
        
        a_np  = actions.detach().cpu().numpy()
        real_actions = np.array([ACTION_MAPPING[int(a)] for a in a_np], dtype=np.int64)

        #execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
        done = terminations | truncations 

        # remember actions for next predict/update
        prev_actions = actions

        # -------- RE-INIT belief on resets (uses latent1_first_posterior again) --------
        if done.any():
            ids = np.nonzero(done)[0]
            ids_t = torch.from_numpy(ids).to(device)
            with torch.no_grad():
                imgs_r  = torch.from_numpy(next_obs[ids]).unsqueeze(1).to(device).float() / 255.0
                feat_r  = Model.encoder(imgs_r)
                z1_0    = Model.latent1_first_posterior(feat_r).rsample()
                z2_0    = Model.latent2_first_posterior(z1_0).rsample()
                z1_bel[ids_t] = z1_0
                z2_bel[ids_t] = z2_0
                prev_actions[ids_t] = 0  # NOOP
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        step_type = np.where(done, 2, np.where(episode_first, 0, 1)).astype(np.int64)  # shape (n_envs,)

        #with tic("store_buffer"):
        for k in range(args.num_envs):
            rb.add(obs[k][None],
            actions[k].item(),
            rewards[k],
            done[k],
            step_type[k])      

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        episode_first = done.copy() 

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:            
            #with tic("sample"):
            data = rb.sample(args.batch_size)
            images  = data["obs"].to(dtype=torch.float32).div_(255.)
            actions = data["action"]
            step_ty = data["step_type"]
            rewards = data["reward"]
            dones   = data["done"]

            # World model training
            """model_loss, output = compute_loss(Model, images, actions, step_ty, use_kl=True)
            Model.optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 20.0)
            Model.optimizer.step()"""

            # Agent training
            with torch.no_grad():
                (z1, z2), _ = Model.sample_posterior(images, actions, step_ty)
            
            q_loss, q_pred, target_q = agent.compute_loss(z1, z2, actions, rewards, dones)
            agent.update(q_loss)

            if global_step % 1000 == 0:
                actual_r, final_pred, gap = agent.check_terminal_accuracy(z1, z2, actions, rewards, dones)
                if actual_r is not None:
                    writer.add_scalar("charts/gap_final_state", gap, global_step)
                    if actual_r > 0:
                        writer.add_scalar("charts/positive_terminal_accuracy", final_pred, global_step)
                    if actual_r < 0:
                        writer.add_scalar("charts/negative_terminal_accuracy", final_pred, global_step)

            if global_step % 5000 == 0:
                visualize_terminal_debug(agent, Model, z1, z2, actions, rewards, dones, global_step)
                evaluate_and_record(args, Model, agent, global_step, run_name)

            if global_step % 500 == 0:
                #writer.add_scalar("losses/model_loss", model_loss.item(), global_step)
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/q_values", q_pred.mean().item(), global_step)
                writer.add_scalar("losses/target_q_min", target_q.min().item(), global_step)
                writer.add_scalar("losses/target_q_max", target_q.max().item(), global_step)
                writer.add_scalar("losses/target_q_mean", target_q.mean().item(), global_step)
                """writer.add_scalar("losses/mse", output["mse"].item(), global_step)
                writer.add_scalar("losses/kl_z1", output["kl_z1"].item(), global_step)
                writer.add_scalar("losses/kl_q_raw", output["kl_q_raw"].item(), global_step)
                writer.add_scalar("losses/kl_q", output["kl_q"].item(), global_step)
                writer.add_scalar("losses/kl_term", output["kl_term"].item(), global_step)
                writer.add_scalar("losses/latent_tf_mse", output["latent_tf_mse"].item(), global_step )
                writer.add_scalar("losses/pix_tf_mse", output["pix_tf_mse"].item(), global_step)"""
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/epsilon", agent.epsilon, global_step)

            if global_step % 5000 == 0:
                #model_grad_norm = Model.get_grad_norm()
                agent_grad_norm = agent.get_grad_norm()
                writer.add_scalar("charts/grad_norm", agent_grad_norm, global_step)
                #writer.add_scalar("charts/model_grad_norm", model_grad_norm, global_step)
                """td_error = agent.get_td_error(z1, z2, actions, rewards, dones)
                writer.add_scalar("charts/td_error_head_0", td_error[0], global_step)
                writer.add_scalar("charts/td_error_head_1", td_error[1], global_step)
                writer.add_scalar("charts/td_error_head_2", td_error[2], global_step)
                agent_head_grads = agent.get_head_grads(q_loss)
                writer.add_scalar("charts/agent_head_0_grads", agent_head_grads[0], global_step)
                writer.add_scalar("charts/agent_head_1_grads", agent_head_grads[1], global_step)
                writer.add_scalar("charts/agent_head_2_grads", agent_head_grads[2], global_step)"""
            
            # update target network
            if global_step % args.target_network_frequency == 0:
                agent.update_target_model()

            """if global_step % 100_000 == 0 and global_step > 0:
                sequence = rb.sample(1)
                images  = (sequence["obs"].float() / 255.)
                actions = sequence["action"]
                step_ty = sequence["step_type"]
                log_rollout_grid(images[0], step=0, caption="Sequence")
                x_pred, aux = Model.one_step_prior_predict(images, actions, step_ty)
                log_rollout_grid(x_pred[0].detach().cpu(), step=0, caption="One-step Prior Predict")
                preds_imgs, fd_pred, fd_true, mask = Model.build_motion_mask(images, actions, step_ty)
                log_rollout_grid(preds_imgs.squeeze(0).detach().cpu(), step=0, caption="Predicted Images")"""
            
            if global_step % 1_000_000 == 0 and global_step > 0:
                if args.save_model:
                    save_dir   = f"runs/{run_name}"
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = f"{save_dir}/{args.exp_name}.pt"

                    ckpt = {
                        "global_step"   : global_step,

                        # ---------- generative model ----------------------------------
                        "model_state"   : Model.state_dict(),
                        "model_opt"     : Model.optimizer.state_dict(),

                        # ---------- Q-agent ------------------------------------------
                        "q_state"       : agent.q_net.state_dict(),
                        "q_target"      : agent.q_target_net.state_dict(),
                        "q_opt"         : agent.q_opt.state_dict(),

                        # ---------- config -------------------------------------------
                        "args"          : vars(args),
                        }
                    
                    torch.save(ckpt, model_path)
                    print(f"Model saved to {model_path}")

                    # Upload the model to wandb
                    if args.track:
                        artifact = wandb.Artifact(f"model-{int(global_step/1_000_000)}M", type="model")
                        artifact.add_file(model_path)
                        run.log_artifact(artifact)
                        print(f"Model uploaded to wandb as artifact: model-{int(global_step/1_000_000)}M")
    
    envs.close()
    writer.close()


