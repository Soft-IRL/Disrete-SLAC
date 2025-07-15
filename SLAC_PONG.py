import os
import gymnasium as gym
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

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from SequenceReplayBuffer import SequenceReplayBuffer

from huggingface_hub import hf_hub_download

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from SLAC_Agent import ModelDistributionNetwork



script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """if toggled, the trained model will be saved to disk"""
    wandb_project_name: str = "SLAC_PONG"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    q_learning_rate: float = 3e-4
    """the learning rate of the q_network optimizer"""
    m_learning_rate: float = 1e-4
    """the learning rate of the model_network optimizer"""
    alpha_lr: float =3e-4
    """the learning rate of the alpha optimizer"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    gamma: float = 0.99
    """the discount factor"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    target_network_frequency: int = 1
    """the frequency of target network update"""
    tau: float = 0.005
    """the polyak averaging factor for target network update"""
    sequence_len : int = 8
    """the length of the sequence for training"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    kl_analytic: bool = True
    """if toggled, the KL divergence will be computed analytically"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    base_depth: int = 32
    """the base depth of the model network"""
    latent1_size: int = 32
    """the size of the first latent variable"""
    latent2_size: int = 256
    """the size of the second latent variable"""
    hidden_dims: tuple = (256, 256)
    """the hidden dimensions of the Q-network"""


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


class Qagent():
    """
    Discrete-action soft-Q / DQN agent for SLAC latents.
    - Input  : concat(z1, z2)  (size = latent1 + latent2)
    - Output : Q-values for all actions
    """

    def __init__(self, n_actions, args):
        self.state_size = args.latent1_size + args.latent2_size
        self.hidden_dims = args.hidden_dims
        self.n_actions = int(n_actions)
        self.target_entropy = -0.98 * np.log(self.n_actions)
        self.device = args.device
        self.lr = args.q_learning_rate
        self.alpha_lr = args.alpha_lr
        self.epsilon = args.start_e
        self.gamma = args.gamma

        # 1. Q-network & target network
        self.q_net        = self._build_mlp(self.state_size, self.n_actions, self.hidden_dims).to(self.device)
        self.q_target_net = copy.deepcopy(self.q_net).eval().requires_grad_(False)

        # ---------------- learnable log_alpha ------------
        init_alpha = 1.0
        self.log_alpha = torch.tensor(np.log(init_alpha),
                                      requires_grad=True,
                                      device=self.device)
        
        # ---------- optimizers -------------------------------------------
        self.q_opt     = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
    
    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dims):
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        return nn.Sequential(*layers)
    
    def act(self, z):
        """
        Select an action based on the current state z.
        :param z: latent representation (concatenation of z1 and z2)
        :return: selected action
        """
        q_values = self.q_net(z)
        alpha = self.log_alpha.exp()
        logits = q_values / alpha
        pi = torch.softmax(logits, dim=1)
        action = torch.multinomial(pi, num_samples=1)
        return action
    
    def compute_loss(self, z1, z2, actions, rewards, dones):
        z_t     = torch.cat([z1[:, :-1], z2[:, :-1]], dim=-1)      # (B,S-1, d1+d2)
        z_tp1   = torch.cat([z1[:, 1:],  z2[:, 1:]],  dim=-1)      # (B,S-1, d1+d2)
        a_t     = actions[:, :-1]                                  # (B,S-1)
        r_t     = rewards[:, :-1]                                  # (B,S-1)
        done_t  = dones[:, :-1].float()                            # (B,S-1)

        # Flatten transition dimension for network calls --------------------------
        z_t_f   = z_t.reshape(-1, z_t.shape[-1])                 # (B*(S-1), d)
        z_tp1_f = z_tp1.reshape(-1, z_tp1.shape[-1])
        a_t_f   = a_t.flatten()     

        # -------------------------------------------------------------------------
        # 1. TD target   y_t = r + γ α log Σ_a' exp(Q_target(z_{t+1},a')/α)
        # -------------------------------------------------------------------------
        with torch.no_grad():
            q_tp1_all = self.q_target_net(z_tp1_f)                # (B*(S-1), A)
            alpha     = self.log_alpha.exp()

            soft_max  = torch.logsumexp(q_tp1_all / alpha, dim=-1, keepdim=False)
            y = r_t.flatten() + self.gamma * (1.0 - done_t.flatten()) * (alpha * soft_max)
        
        # -------------------------------------------------------------------------
        # 2. Q-loss     ½ (Q(z_t,a_t) − y)^2
        # -------------------------------------------------------------------------
        q_pred = self.q_net(z_t_f).gather(1, a_t_f.unsqueeze(-1)).squeeze(-1)
        q_loss = 0.5 * F.mse_loss(q_pred, y, reduction="mean")

        # -------------------------------------------------------------------------
        # 3. α-loss     L_α = α · ( − log π − target_entropy )
        # -------------------------------------------------------------------------
        with torch.no_grad():
            # policy derived from current Q-values
            q_all      = self.q_net(z_t_f)                       # (B*(S-1), A)
            log_pi_all = F.log_softmax(q_all / alpha, dim=-1)
            entropy    = -(log_pi_all.exp() * log_pi_all).sum(-1) # (B*(S-1),)

        alpha_loss = (self.log_alpha.exp() * (entropy - self.target_entropy)).mean()

        return q_loss, alpha_loss
    
    """def update(self,q_loss, alpha_loss):
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()"""
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

# !!!!!!!!!!!!!!!!!!!! Ajouter batch size différent pour latent model !!!!!!!!!!!!!!!!!!!!!!

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

        artifact = wandb.Artifact("source_code", type="code")
        artifact.add_dir(script_dir) 
        run.log_artifact(artifact)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ACTION_MAPPING = {0: 0, 1: 2, 2: 3}
    n_actions = len(ACTION_MAPPING.keys())
    action_space = Discrete(n_actions)
    action_dim = int(np.prod(action_space.shape))

    obs,_ = envs.reset(seed=args.seed)

    Model = ModelDistributionNetwork(action_dim, args)
    agent = Qagent(action_space.n, args)

    eval_counter = 0

    rb = SequenceReplayBuffer(
        capacity   = args.buffer_size,
        obs_shape  = (1,obs.shape[1],obs.shape[2]),
        act_shape  = (),
        seq_len    = args.sequence_len,
        device     = args.device,)

    start_time = time.time()

    #start the game
    obs, _ = envs.reset(seed=args.seed)
    episode_first = np.ones(args.num_envs, dtype=bool)   # True right after reset
    
    ############################# THIS IS THE MODEL PRETRAINING ###############################
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
    for step in tqdm(range(50_000)):
        batch = rb.sample(args.batch_size)
        images  = (batch["obs"].float() / 255.).to(device)
        actions = batch["action"].to(device)
        step_ty = batch["step_type"].to(device)

        model_loss, _ = Model.compute_loss(images, actions, step_ty)
        Model.optimizer.zero_grad()
        model_loss.backward()
        Model.optimizer.step()
    
    ######################## END OF MODEL PRETRAINING ###############################
    print("Model pretraining completed.")
    for global_step in range(args.total_timesteps):
        agent.epsilon = agent.linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < agent.epsilon:      
            actions = np.array([action_space.sample() for _ in range(envs.num_envs)])
        else:
            flat_obs = Model.encoder(torch.FloatTensor(obs/255).unsqueeze(1).to(device))
            dist_z_1 = Model.latent1_first_posterior(flat_obs)
            z_1 = dist_z_1.rsample()
            dist_z_2 = Model.latent2_first_posterior(z_1)
            z_2 = dist_z_2.rsample()
            z = torch.cat((z_1, z_2), dim=1)
            actions = agent.act(z).cpu().numpy().flatten() 
            
        real_actions = np.array([ACTION_MAPPING[a.item()] for a in actions])

        #execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(real_actions)
        done = terminations | truncations 
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
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

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        episode_first = done.copy() 
        

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            images  = (data["obs"].float() / 255.).to(device) 
            actions = data["action"].to(device)
            step_ty = data["step_type"].to(device)
            rewards = data["reward"].to(device)
            dones   = data["done"].to(device)                  

            model_loss, (z1, z2) = Model.compute_loss(images,actions,step_ty)
            #Model.update(model_loss)

            #(z1, z2), _ = Model.sample_posterior(images, actions[:, :-1], step_ty)
            q_loss, alpha_loss = agent.compute_loss(z1, z2, actions, rewards, dones)
            #agent.update(q_loss, alpha_loss)

            total_loss = model_loss + q_loss + alpha_loss

            Model.optimizer.zero_grad()
            agent.q_opt.zero_grad()
            agent.alpha_opt.zero_grad()

            total_loss.backward()

            Model.optimizer.step()     
            agent.q_opt.step()              
            agent.alpha_opt.step()

            if global_step % 100 == 0:
                writer.add_scalar("losses/model_loss", model_loss.item(), global_step)
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/epsilon", agent.epsilon, global_step)

            if global_step % 5000 == 0:
                model_grad_norm = Model.get_grad_norm()
                agent_grad_norm = agent.get_grad_norm()
                writer.add_scalar("charts/grad_norm", agent_grad_norm, global_step)
                writer.add_scalar("charts/model_grad_norm", model_grad_norm, global_step)
            
             # update target network
            if global_step % args.target_network_frequency == 0:
                # -------------------------------------------------------------------------
                # 4. Target-net update (polyak)
                # -------------------------------------------------------------------------
                with torch.no_grad():
                    for p, p_targ in zip(agent.q_net.parameters(), agent.q_target_net.parameters()):
                        p_targ.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
            
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

                        # temperature parameter α
                        "log_alpha"     : agent.log_alpha.detach().cpu(),
                        "alpha_opt"     : agent.alpha_opt.state_dict(),

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



