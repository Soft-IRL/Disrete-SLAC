from envs.light_dark_navigation_env import make_env
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env, Wrapper
from gymnasium.spaces import Discrete

from stable_baselines3.common.buffers import ReplayBuffer

from tqdm import tqdm
#import imageio.v2 as imageio
#import imageio.v3 as iio
import cv2


class DiscreteActions(Wrapper):
    """Map a 2D Box action env to a 9-action discrete grid: (dx,dy) in {-1,0,1}*max_speed."""
    def __init__(self, env):
        super().__init__(env)
        ms = env.unwrapped.cfg.max_speed
        grid = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                grid.append([dx * ms, dy * ms])
        self._grid = np.asarray(grid, dtype=np.float32)
        self.action_space = Discrete(len(self._grid))
        # observation_space is the same as the base env
        self.observation_space = env.observation_space

    def step(self, a_idx):
        a = self._grid[int(a_idx)]
        #print(f"Discrete action {a_idx} -> {a}")  
        return self.env.step(a)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)




class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(obs_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)

def evaluate_policy(env, q, episodes=5, seed=0, render=False, filename="dqn_eval.mp4", fps=15):
    #env = DiscreteActions(env)

    rng = np.random.default_rng(seed)
    device = next(q.parameters()).device
    world_radius = env.unwrapped.cfg.world_radius

    returns, steps_list, successes = [], [], 0

    obs, info = env.reset(seed=rng.integers(0, 1_000_000))
    if render:
        frame = env.render()
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    for ep in range(episodes):
        if ep > 0:
            obs, _ = env.reset(seed=rng.integers(0, 1_000_000))
            if render:
                frame = env.render()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        ep_ret, steps = 0.0, 0
        while True:
            with torch.no_grad():
                o = torch.as_tensor(obs/world_radius, dtype=torch.float32, device=device).unsqueeze(0)
                a = q(o).argmax(dim=1).detach().cpu().numpy()[0]

            obs, r, terminated, truncated, info = env.step(a)

            ep_ret += r; steps += 1
            if render: 
                frame = env.render()
                if frame is not None: writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                successes += int(terminated)
                break
        returns.append(np.round(ep_ret))
        steps_list.append(steps)


    env.close()
    # duration is seconds per frame
    if render:
        writer.release()
    return returns, steps_list, successes


""" def train_dqn_zero_uncertainty(env, 
                               episodes: int = 500,
                                max_steps: int = 200,
                                gamma: float = 0.99,
                                lr: float = 5e-4,
                                batch_size: int = 128,
                                start_steps: int = 1_000,
                                train_after: int = 5_000,
                                train_every: int = 1,
                                target_update: int = 500,
                                eps_start: float = 1.0,
                                eps_end: float = 0.05,
                                eps_decay_steps: int = 20_000,
                                seed: int = 0,):


    # Enable randomized goal/start at reset
    env.unwrapped.cfg.max_steps = max_steps

    # Wrap discrete actions
    env = DiscreteActions(env)
    world_radius = env.unwrapped.cfg.world_radius

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buf = ReplayBuffer(
        buffer_size=100_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNet(obs_dim, n_actions).to(device)
    qt = QNet(obs_dim, n_actions).to(device)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)

    def epsilon(t):
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    total_steps = 0
    returns = []

    for ep in tqdm(range(episodes)):
        
        obs, info = env.reset(seed=rng.integers(0, 1_000_000))
        #print(obs)
        obs = obs/world_radius   # normalize observations to [-1,1]
        ep_ret = 0.0
        for t in range(max_steps):
            total_steps += 1
            if total_steps < start_steps or rng.random() < epsilon(total_steps):
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    o   = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                    qv  = q(o)
                    a   = torch.argmax(qv, dim=1).detach().cpu().numpy()[0]

            next_obs, r, term, trunc, info = env.step(a)
            next_obs = next_obs/world_radius   # normalize observations to [-1,1]
            if r > 0: tqdm.write(f"Success in ep {ep+1} at step {t+1}!")

            d = float(term or trunc)
            buf.add(obs=obs,
                    next_obs=next_obs,
                    action=a,
                    reward=r,
                    done=d,
                    infos=[info])
            
            obs = next_obs
            ep_ret += r
                
            # Train
            if total_steps >= train_after and total_steps % train_every == 0 and buf.size() >= batch_size:
                batch = buf.sample(batch_size)
                o  = batch.observations                  # (B, obs_dim), torch.float32
                no = batch.next_observations             # (B, obs_dim)
                a_b  = batch.actions.long().squeeze(-1)    # (B,)
                r_b  = batch.rewards.squeeze(-1)           # (B,)
                d_b  = batch.dones.squeeze(-1).float()     # (B,)

                q_pred = q(o).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = qt(no).max(1)[0]
                    target = r_b + gamma * (1.0 - d_b) * next_q

                
                loss = nn.functional.smooth_l1_loss(q_pred, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()


            if total_steps % target_update == 0:
                qt.load_state_dict(q.state_dict())


            if term or trunc:
                break

        returns.append(ep_ret)
        # Simple print every 10 eps
        if (ep + 1) % 10 == 0:
            tqdm.write(f"Ep {ep+1:4d} | Return {ep_ret:7.3f} | eps {epsilon(total_steps):.3f} | total steps {total_steps}")
        
        if (ep + 1) % 10 == 0:
            if (ep + 1) % 100 == 0:
                rets, steps, succ = evaluate_policy(env, q, episodes=5, render=True)
            else:
                rets, steps, succ = evaluate_policy(env, q, episodes=5, render=False)
            tqdm.write(f"Evaluation | Return {rets} | Steps {steps} | Success {succ}")


    env.close()
    return returns """

def train_dqn_zero_uncertainty(
    env,
    total_training_steps: int = 300_000,
    max_episode_steps: int = 200,
    gamma: float = 0.99,
    lr: float = 5e-4,
    batch_size: int = 128,
    start_steps: int = 1_000,
    train_after: int = 5_000,
    train_every: int = 1,
    target_update: int = 2_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 100_000,
    seed: int = 0,
    eval_every_steps: int = 10_000,
    do_render_eval_every: int = 25_000,   # render eval each this many steps
):
    # keep your env; just set per-episode step cap
    env.unwrapped.cfg.max_steps = max_episode_steps

    # Discretize actions
    env = DiscreteActions(env)
    world_radius = env.unwrapped.cfg.world_radius

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SB3 ReplayBuffer (as in your code)
    buf = ReplayBuffer(
        buffer_size=100_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q = QNet(obs_dim, n_actions).to(device)
    qt = QNet(obs_dim, n_actions).to(device)
    qt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=lr)

    def epsilon(t):
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + frac * (eps_end - eps_start)

    total_steps = 0
    next_eval_at = eval_every_steps
    next_render_eval_at = do_render_eval_every
    returns = []

    # start first episode
    obs, info = env.reset(seed=rng.integers(0, 1_000_000))
    obs = obs / world_radius
    ep_ret = 0.0
    ep_len = 0
    ep_idx = 0

    from tqdm import tqdm
    pbar = tqdm(total=total_training_steps, desc="DQN training (steps)")

    while total_steps < total_training_steps:
        total_steps += 1
        ep_len += 1

        # act
        if total_steps < start_steps or rng.random() < epsilon(total_steps):
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                qv = q(o)
                a   = torch.argmax(qv, dim=1).detach().cpu().numpy()[0]

        next_obs, r, term, trunc, info = env.step(a)
        next_obs = next_obs / world_radius

        d = float(term or trunc)
        buf.add(
            obs, next_obs, a, r, d,
            infos=[info],
        )

        ep_ret += r
        obs = next_obs

        # learn
        if total_steps >= train_after and total_steps % train_every == 0 and buf.size() >= batch_size:
            batch = buf.sample(batch_size)
            o  = batch.observations
            no = batch.next_observations
            a_b = batch.actions.long().squeeze(-1)
            r_b = batch.rewards.squeeze(-1)
            d_b = batch.dones.squeeze(-1).float()

            q_pred = q(o).gather(1, a_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = qt(no).max(1)[0]
                target = r_b + gamma * (1.0 - d_b) * next_q

            loss = nn.functional.smooth_l1_loss(q_pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if total_steps % target_update == 0:
            qt.load_state_dict(q.state_dict())

        # evaluation by steps
        if total_steps >= next_eval_at:
            render_eval = total_steps >= next_render_eval_at
            rets, steps, succ = evaluate_policy(
                env, q,
                episodes=5,
                seed=rng.integers(0, 1_000_000),
                render=render_eval,
            )
            tqdm.write(f"[{total_steps}] Eval | Return {rets} | Steps {steps} | Success {succ}")
            next_eval_at += eval_every_steps
            if render_eval:
                next_render_eval_at += do_render_eval_every

        pbar.update(1)

        # episode end or max length
        if term or trunc or ep_len >= max_episode_steps:
            returns.append(ep_ret)
            ep_idx += 1
            obs, info = env.reset(seed=rng.integers(0, 1_000_000))
            obs = obs / world_radius
            ep_ret = 0.0
            ep_len = 0

    pbar.close()
    env.close()
    return returns


if __name__ == "__main__":
    env = make_env(
        render_mode="rgb_array",
        world_radius=10.0,
        max_speed=0.5,
        goal_radius=1.0,
        band_center=(-9.0, 0.0),
        band_angle_deg=90.0,
        band_width=2.0,
        sigma_dark=0.0,     # zero uncertainty baseline
        sigma_light=0.0,
        include_goal_in_obs=True,
        randomize_start=True,
        randomize_goal=True,
        min_start_goal_dist=6.0,
        require_opposite_band_side=False,
    )

    rets = train_dqn_zero_uncertainty(env)
    
    """plt.plot(rets)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()"""