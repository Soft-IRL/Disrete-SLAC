import numpy as np
import torch

class SequenceReplayBuffer:
    """Ring buffer that stores individual time-steps
       and can return contiguous windows of fixed length L."""
    
    def __init__(self, capacity, obs_shape, act_shape,
             seq_len=4, device="cpu"):
        
        self.capacity = capacity          # max timesteps, not sequences
        self.seq_len  = seq_len
        self.device   = device
        
        self.ptr   = 0        # write index
        self.full  = False    # becomes True once we wrapped around

        self.obs     = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        #self.nextobs = np.zeros_like(self.obs)
        self.action  = np.zeros((capacity, *act_shape), dtype=np.int64)
        self.reward  = np.zeros((capacity,), dtype=np.float32)
        self.done    = np.zeros((capacity,), dtype=np.bool_)
        self.step_type = np.zeros((capacity,), dtype=np.int64)  # 0: FIRST, 1: MID, 2: LAST

    # ---------- add one transition ----------------------------------
    def add(self, obs, action, reward, done, step_type):
        self.obs[self.ptr]     = obs
        self.action[self.ptr]  = action
        self.reward[self.ptr]  = reward
        #self.nextobs[self.ptr] = next_obs
        self.done[self.ptr]    = done
        self.step_type[self.ptr] = step_type

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True  # After capacity calls, full=True and ptr starts overwriting from index 0.
    
    # ---------- internal: valid start indices -----------------------
    # Return all indices i that can start a complete length-seq_len window.
    # Result: idx is a NumPy array of valid start indices.
    def _valid_starts(self):
        max_i = (self.capacity if self.full else self.ptr) - self.seq_len
        idx = np.arange(max_i)
        # exclude windows crossing the write head when full
        if self.full:
            bad = (idx + self.seq_len > self.capacity) & (idx < self.ptr)
            idx = idx[~bad]
        # exclude windows containing a done=True *inside* (allow at last pos)
        good = []
        for i in idx:
            if not self.done[i : i + self.seq_len - 1].any():
                good.append(i)
        return np.array(good)
    
     # ---------- sample B sequences ----------------------------------
    def sample(self, batch):
        # Return a Python dict whose entries already have shape (batch, seq_len, …)
        starts = self._valid_starts()
        if len(starts) == 0:
            raise ValueError("Not enough data for one full sequence.")
        idx = np.random.choice(starts, size=batch, replace=True) # Randomly choose batch start indices (with replacement)
        rang = np.arange(self.seq_len)
        # idx[:,None] + rang produces a (batch, seq_len) matrix of absolute indices.

        obs      = torch.as_tensor(self.obs[idx[:,None] + rang], device=self.device)
        #next_obs = torch.as_tensor(self.nextobs[idx[:,None] + rang], device=self.device)
        action   = torch.as_tensor(self.action[idx[:,None] + rang], device=self.device)
        reward   = torch.as_tensor(self.reward[idx[:,None] + rang], device=self.device)
        done     = torch.as_tensor(self.done[idx[:,None]   + rang], device=self.device)
        step_type = torch.as_tensor(self.step_type[idx[:,None] + rang], device=self.device)

        return dict(obs=obs, #next_obs=next_obs,
                    action=action, reward=reward, done=done, step_type=step_type)