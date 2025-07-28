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

        """self.obs     = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        #self.nextobs = np.zeros_like(self.obs)
        self.action  = np.zeros((capacity, *act_shape), dtype=np.int64)
        self.reward  = np.zeros((capacity,), dtype=np.float32)
        self.done    = np.zeros((capacity,), dtype=np.bool_)
        self.step_type = np.zeros((capacity,), dtype=np.int64)  # 0: FIRST, 1: MID, 2: LAST"""

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8,
                               device="cpu", pin_memory=(self.device.type == "cpu"))
        self.action    = torch.empty((capacity, *act_shape), dtype=torch.int64 ,
                             device="cpu", pin_memory=(self.device.type == "cpu"))
        self.reward    = torch.empty((capacity,),        dtype=torch.float32,
                             device="cpu", pin_memory=(self.device.type == "cpu"))
        self.done      = torch.empty((capacity,),        dtype=torch.bool   ,
                             device="cpu", pin_memory=(self.device.type == "cpu"))
        self.step_type = torch.empty((capacity,),        dtype=torch.int64 ,
                             device="cpu", pin_memory=(self.device.type == "cpu"))
        self.valid = torch.zeros(capacity, dtype=torch.bool)  # start flags
        self.full  = False

    # ---------- add one transition ----------------------------------
    def add(self, obs, action, reward, done, step_type):
        idx = self.ptr
        ...
        # write transition -------------------------------------------
        self.obs     [idx].copy_(torch.as_tensor(obs,    dtype=torch.uint8))
        self.action  [idx] = action
        self.reward  [idx] = reward
        self.done    [idx].copy_(torch.as_tensor(done,    dtype=torch.bool))
        self.step_type[idx] = step_type
        # -------------------------------------------------------------
        self._update_valid(idx)       # O(L) instead of O(capacity)
        # advance pointer ---------------------------------------------
        self.ptr = (idx + 1) % self.capacity
        if self.ptr == 0:
            self.full = True
    
    def _update_valid(self, idx):
        """
        Recompute validity of starts in [idx-L+1, idx] inclusive.
        Only those windows are affected by writing at idx.
        """
        L = self.seq_len
        for s in range(idx - L + 1, idx + 1):
            if s < 0:                 # before buffer start
                continue
            if not self.full and s + L > self.ptr:   # window not full yet
                self.valid[s] = False
                continue
            # window [s, s+L-1]
            mid_done = self.done[s : s + L - 1].any()
            self.valid[s] = not mid_done
    
    # ---------- internal: valid start indices -----------------------
    # Return all indices i that can start a complete length-seq_len window.
    # Result: idx is a NumPy array of valid start indices.
    """def _valid_starts(self):
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
        return np.array(good)"""
    
    def _valid_starts(self):
        max_i = (self.capacity if self.full else self.ptr) - self.seq_len
        return torch.nonzero(self.valid[:max_i], as_tuple=False).squeeze(-1)
    
     # ---------- sample B sequences ----------------------------------
    """def sample(self, batch):
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
                    action=action, reward=reward, done=done, step_type=step_type)"""

    import torch, numpy as np

class SequenceReplayBuffer:
    """
    Ring-buffer that stores single time-steps on CPU and returns
    fixed-length sequences on request.
    """

    def __init__(self, capacity, obs_shape, act_shape,
                 seq_len=8, device="cuda"):
        self.capacity = capacity          # max *time-steps*
        self.seq_len  = seq_len           # L
        self.device   = torch.device(device)

        # ------------------------------------------------------------------ #
        #  storage – **CPU**, pinned so GPU transfer is DMA & non-blocking   #
        # ------------------------------------------------------------------ #
        pin = True                        # always pin; harmless on CPU
        self.obs       = torch.empty((capacity, *obs_shape),  dtype=torch.uint8 , pin_memory=pin)
        self.action    = torch.empty((capacity, *act_shape),  dtype=torch.int64 , pin_memory=pin)
        self.reward    = torch.empty((capacity,),             dtype=torch.float32, pin_memory=pin)
        self.done      = torch.empty((capacity,),             dtype=torch.bool  , pin_memory=pin)
        self.step_type = torch.empty((capacity,),             dtype=torch.int64 , pin_memory=pin)

        # start-flags: valid[i] == True  ⇒  slice [i … i+L-1] is usable
        self.valid = torch.zeros(capacity, dtype=torch.bool)  # CPU

        self.ptr  = 0              # next index to write
        self.full = False

    # ---------------------------------------------------------------------- #
    # helper – recompute validity of windows that *touch* idx                #
    # ---------------------------------------------------------------------- #
    def _update_valid(self, idx: int):
        """
        Recompute the validity flags for the only windows that can have
        changed after writing the transition at index `idx`.

        A start index s is *valid* iff
            - the window [s, s+L-1] exists (no hole in an unfinished buffer),
            - and  done[k] is False  for all k in [s, s+L-2].
            (the last position is allowed to be done=True)
        """
        L = self.seq_len                     # window length
        C = self.capacity

        # ------------------------------------------------------------
        # 1. candidate starts possibly affected by this write
        #    They are the L indices preceding `idx`  (inclusive).
        # ------------------------------------------------------------
        s0 = max(0, idx - L + 1)             # clamp to 0 for an un-filled buffer
        s1 = idx + 1                         #   idx  is included, idx+1  is not
        starts = torch.arange(s0, s1, device="cpu")    # ≤ L elements

        # ------------------------------------------------------------
        # 2. windows that would overrun the *current write pointer*
        #    Only relevant while the buffer is not full yet.
        # ------------------------------------------------------------
        if not self.full:
            overflow = starts + L > self.ptr         # bool mask
            self.valid[starts[overflow]] = False

        # ------------------------------------------------------------
        # 3. Build a *contiguous* view of the circular buffer
        #    so that windows that cross the physical end are handled
        #    without special-casing.
        # ------------------------------------------------------------
        # done == (C,)  →  done2 == (2C,)   [head | tail]
        done2 = torch.cat([self.done, self.done], dim=0)   # CPU tensor

        # For every candidate s  we want the slice  done2[s+1 : s+L]
        # (length = L-1).  unfold gives us all these slices at once:
        #   windows.shape == (len(starts), L-1)
        windows = done2.unfold(0, L - 1, 1)[starts]

        # ------------------------------------------------------------
        # 4. Any `done=True` inside the *middle* of the window?
        # ------------------------------------------------------------
        mid_done = windows.any(dim=1)        # bool tensor, len == len(starts)

        # valid ⇐ no done in the middle  ∧  (passes overflow test already set)
        self.valid[starts] = ~mid_done

    # ---------------------------------------------------------------------- #
    # public – add one transition                                            #
    # ---------------------------------------------------------------------- #
    def add(self, obs, action, reward, done, step_type):
        idx = self.ptr

        # write ----------------------------------------------------------------
        self.obs      [idx].copy_(torch.as_tensor(obs,  dtype=torch.uint8))
        self.action   [idx] = action
        self.reward   [idx] = reward
        self.done     [idx].copy_(torch.as_tensor(done,  dtype=torch.bool))
        self.step_type[idx] = step_type

        # update flags BEFORE moving ptr ---------------------------------------
        self._update_valid(idx)

        # advance pointer -------------------------------------------------------
        self.ptr = (idx + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    # ---------------------------------------------------------------------- #
    # public – sample a batch of sequences                                   #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, batch_size: int):
        starts = self.valid.nonzero(as_tuple=False).squeeze(-1)  # CPU
        if starts.numel() == 0:
            raise RuntimeError("replay buffer: no valid sequences yet")

        # (B,)   choose random start indices on CPU
        idx = starts[ torch.randint(0, starts.numel(), (batch_size,)) ]

        # absolute indices (B,L) ––– wrap with %capacity  <<<————─★
        seq_idx = (idx.unsqueeze(1) +
                torch.arange(self.seq_len)).remainder(self.capacity)

        # gather once per field (still CPU)
        obs   = self.obs     [seq_idx]          # uint8  (B,L,1,H,W)
        act   = self.action  [seq_idx]
        rew   = self.reward  [seq_idx]
        done  = self.done    [seq_idx]
        stype = self.step_type[seq_idx]

        # single asynchronous copy to GPU
        obs   = obs  .to(self.device, non_blocking=True)
        act   = act  .to(self.device, non_blocking=True)
        rew   = rew  .to(self.device, non_blocking=True)
        done  = done .to(self.device, non_blocking=True)
        stype = stype.to(self.device, non_blocking=True)

        return dict(obs=obs, action=act, reward=rew,
                    done=done, step_type=stype)
