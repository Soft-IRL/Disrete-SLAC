import copy
import torch
import numpy as np
from torch import optim
from torch import nn
import torch.nn.functional as F

# ========================================================================== #
#                Distributional Double Deep Q-Network based on C51           #
# ========================================================================== #

class D3QNAgent():
    """
    Discrete-action soft-Q / DQN agent for SLAC latents.
    - Input  : concat(z1, z2)  (size = latent1 + latent2)
    - Output : For each action a, a categorical distribution over supp_dims (51) atoms.
               Acts greedily w.r.t. the *expected* value of those distributions.
               For now, no entropy on information is considered.
    """

    def __init__(self, n_actions, args):
        self.state_size = args.latent1_size + args.latent2_size
        self.hidden_dims = args.hidden_dims
        self.n_actions = int(n_actions)
        self.device = args.device
        self.lr = args.q_learning_rate
        self.alpha_lr = args.alpha_lr
        self.epsilon = args.start_e
        self.gamma = args.gamma
        
        # specific for C51 approach
        self.n_atoms = int(getattr(args, "N_atoms", getattr(args, "n_atoms", 51)))
        self.Qmin    = float(getattr(args, "Q_min", getattr(args, "Q_min", -1.0)))
        self.Qmax    = float(getattr(args, "Q_max", getattr(args, "Q_max",  1.0)))
        assert self.Qmax > self.Qmin and self.n_atoms >= 2
        self.dz = (self.Qmax - self.Qmin) / (self.n_atoms - 1)

        # fixed support
        z = torch.linspace(self.Qmin, self.Qmax, self.n_atoms)
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        self.register_buffer("atoms", z.to(self.device))  # (N,)

        # networks and optimizer
        out_dim = self.n_actions * self.n_atoms
        self.q_net        = self._build_mlp(self.state_size, out_dim, self.hidden_dims).to(self.device)
        self.q_target_net = copy.deepcopy(self.q_net).eval().requires_grad_(False)
        self.q_opt = optim.Adam(self.q_net.parameters(), lr=self.lr)
    
    @torch.no_grad()
    def act(self, z: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        """
        ε-greedy over mean Q implied by the categorical distribution.
        Returns action indices of shape (B, 1)
        """
        z = F.layer_norm(z, z.shape[-1:])
        logits = self.q_net(z).view(z.size(0), self.n_actions, self.n_atoms)   # row output (B, A, N)
        probs  = logits.softmax(dim=-1)                                        # softmax probability (B, A, N)
        q      = (probs * self.atoms).sum(-1)                                  # expected Q value (B, A)
        greedy_action = q.argmax(dim=1, keepdim=True)                                 # (B, 1)
        return greedy_action

    
    # compute loss given batched sequence pairs
    def compute_loss(self, z1, z2, actions, rewards, dones):
        """
        Sequence loss exactly like deterministic DQN:
        uses transitions t = 0..S-2, with reward r_{t+1} and done_{t+1}.
        """
        B, S, d1 = z1.shape
        d = d1 + z2.size(-1)

        z_all    = torch.cat([z1, z2], dim=-1)          # (B, S, d)
        z_all = F.layer_norm(z_all, z_all.shape[-1:])
        a_all    = actions.long()                       # (B, S)
        r_all    = rewards                              # (B, S)
        done_all = dones.float()                        # (B, S)

        # regular transitions
        z_t    = z_all[:, :-1]                          # (B, S-1, d)
        z_tp1  = z_all[:,  1:]                          # (B, S-1, d)
        a_t    = a_all[:, :-1]                          # (B, S-1)
        r_tp1  = r_all[:,  1:]                          # (B, S-1)
        d_tp1  = done_all[:, 1:]                        # (B, S-1)

        BT = B * (S - 1)
        z_t_f    = z_t.reshape(BT, d)
        z_tp1_f  = z_tp1.reshape(BT, d)
        a_t_f    = a_t.reshape(BT)
        r_tp1_f  = r_tp1.reshape(BT)
        d_tp1_f  = d_tp1.reshape(BT)

        # ------------------- Double DQN action selection ------------------- #
        with torch.no_grad():
            # online net picks a* at s'
            logits_online_tp1 = self._logits(z_tp1_f)                          # (BT, A, N)
            probs_online_tp1  = logits_online_tp1.softmax(-1)                  # (BT, A, N)
            q_online_tp1      = (probs_online_tp1 * self.atoms).sum(-1)        # (BT, A)
            a_star            = q_online_tp1.argmax(dim=1)                     # (BT,)

            # target net evaluates Z(s', a*)
            logits_target_tp1 = self.q_target_net(z_tp1_f).view(BT, self.n_actions, self.n_atoms)
            p_target_tp1      = logits_target_tp1.softmax(-1)                  # (BT, A, N)
            p_next_a          = p_target_tp1[torch.arange(BT, device=z_t_f.device), a_star]  # (BT, N)
            #print(p_next_a.shape)
            #bla

            # project Bellman-updated distribution onto fixed support
            target_proj = self._target_projection(p_next_a, r_tp1_f, d_tp1_f)  # (BT, N)

            target_mean = (target_proj * self.atoms).sum(-1)    # (BT,)

        # ------------------- Online log-probs for taken actions ------------ #
        logits_t = self._logits(z_t_f)                                         # (BT, A, N)
        # gather logits for the *taken* actions at time t
        logits_taken = logits_t[torch.arange(BT, device=z_t_f.device), a_t_f]  # (BT, N)
        log_prob     = F.log_softmax(logits_taken, dim=-1)                     # (BT, N)

        # cross-entropy loss between target_proj (stop-grad) and current logits
        loss = -(target_proj * log_prob).sum(dim=-1).mean()

        # for logging: expected Q(s_t, a_t)
        with torch.no_grad():
            probs_t      = logits_taken.softmax(-1)
            q_taken_mean = (probs_t * self.atoms).sum(-1)                      # (BT,)

        # sanity checks
        assert a_t_f.min().item() >= 0 and a_t_f.max().item() < self.n_actions, \
            f"Bad action indices in buffer: [{a_t_f.min().item()}, {a_t_f.max().item()}]"

        return loss, q_taken_mean, target_mean
    
    # ==================================== helper functions ==================================== #
    
    # Create MLP model
    @staticmethod
    def _build_mlp(in_dim, out_dim, hidden_dims):
        layers, last = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        return nn.Sequential(*layers)
    
    # map model output to logits in correct shape (batchsize, n_actions, n_atoms)
    def _logits(self, z_flat):
        """Return logits shaped (BT, A, N)"""
        return self.q_net(z_flat).view(z_flat.size(0), self.n_actions, self.n_atoms)
    
    # projection the target distribution Y = r + gamma(1 - done)Z(s,a) to the current support.
    @torch.no_grad()
    def _target_projection(self, p_next_a: torch.Tensor, r: torch.Tensor, done: torch.Tensor):
        BT, N = p_next_a.size()
        atoms = self.atoms.unsqueeze(0)  # (1, N)

        # Bellman update of support
        Tz = r.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * self.gamma * atoms  # (BT, N)
        Tz = Tz.clamp(self.Qmin, self.Qmax)

        # Map to [0, N-1] grid
        b = (Tz - self.Qmin) / self.dz              # (BT, N)
        l = b.floor()
        u = b.ceil()

        l_long = l.long().clamp(0, self.n_atoms - 1)
        u_long = u.long().clamp(0, self.n_atoms - 1)

        offset = (torch.arange(BT, device=self.device) * N).unsqueeze(1)  # (BT,1)
        l_idx = (l_long + offset).view(-1)
        u_idx = (u_long + offset).view(-1)

        m = torch.zeros(BT * N, device=self.device)

        # 1) Standard fractional mass
        lower_mass = (p_next_a * (u - b)).view(-1)
        upper_mass = (p_next_a * (b - l)).view(-1)
        m.index_add_(0, l_idx, lower_mass)
        m.index_add_(0, u_idx, upper_mass)

        # 2) Fix exact-integer case: l == u
        eq = (u_long == l_long)                 # (BT, N) bool
        if eq.any():
            # indices where l == u
            eq_idx = (l_long + offset)[eq]      # (num_eq,)
            # add full probability mass for those positions
            m.index_add_(0, eq_idx.view(-1), p_next_a[eq].view(-1))

        m = m.view(BT, N)
        # normalise
        m_sum = m.sum(dim=1, keepdim=True).clamp_(min=1e-8)
        m /= m_sum

        return m

    
    # online model update
    def update(self, loss):
        self.q_opt.zero_grad()
        loss.backward()
        # optional: inspect last layer head grads per action
        last = [m for m in self.q_net.modules() if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            row_grad = last.weight.grad.norm(dim=1).detach().cpu().numpy()
            # print("head grad norms:", row_grad)
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_opt.step()
        
    # target model update
    def update_target_model(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    # epsilon scheduler
    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)
    
    def get_grad_norm(self):
        total_norm = 0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    @torch.no_grad()
    def inspect_terminal_distribution(self, z1, z2, actions, rewards, dones):
        """
        Debug tool: Finds a terminal sequence in the batch and prints
        the Predicted vs Target distributions for the final step.
        """
        B, S, d1 = z1.shape
        
        # 1. Identify sequences that ended (done=True at the last step)
        mask_term = dones[:, -1]
        
        if not mask_term.any():
            print("\n[DEBUG] No terminal states found in this batch.")
            return

        print(f"\n[DEBUG] Found {mask_term.sum()} terminal states. Inspecting the first one...")

        # 2. Extract the data for the first terminal sequence found
        idx = torch.nonzero(mask_term, as_tuple=True)[0][0]
        
        # We look at the very last transition: s_{L-1} -> Terminal
        z_last = torch.cat([z1[idx, -1], z2[idx, -1]], dim=-1).unsqueeze(0) # (1, d)
        z_last = F.layer_norm(z_last, z_last.shape[-1:])
        a_last = actions[idx, -1].long().unsqueeze(0)                       # (1,)
        r_last = rewards[idx, -1].unsqueeze(0)                              # (1,)
        d_last = torch.ones_like(r_last)                                    # (1,)

        print(f"Action: {a_last.item()} | Reward: {r_last.item()} | Done: {d_last.item()}")

        # 3. Calculate PREDICTED Distribution (Online Net)
        logits = self._logits(z_last)                                       # (1, A, N)
        preds_all = logits.softmax(-1)
        # Select the distribution for the action actually taken
        preds_dist = preds_all[0, a_last.item()].cpu().numpy()

        # 4. Calculate TARGET Distribution (Projection)
        # For terminal state, next_state doesn't matter, but we run the logic anyway
        # (Double DQN selection on next state, but masked by done=1)
        
        # a) Online net selection (irrelevant due to done=1 but required for code path)
        q_online = (logits.softmax(-1) * self.atoms).sum(-1)
        a_star   = q_online.argmax(dim=1)

        # b) Target net evaluation
        logits_target = self.q_target_net(z_last).view(1, self.n_actions, self.n_atoms)
        p_target = logits_target.softmax(-1)
        p_next_a = p_target[0, a_star] 

        # c) Projection
        # This is where the magic happens. If r=-1, target should be a spike at bin 0.
        target_dist_tensor = self._target_projection(p_next_a, r_last, d_last)
        target_dist = target_dist_tensor[0].cpu().numpy()

        # 5. Visualization (ASCII Bar Chart)
        print("\nDistribution Comparison (Left: Bin Value | Middle: Target | Right: Pred)")
        print("-" * 65)
        atoms_np = self.atoms.cpu().numpy()
        
        # Accumulate mass for simple stats
        target_mass = target_dist.sum()
        pred_mass = preds_dist.sum()
        target_mean = (target_dist * atoms_np).sum()
        pred_mean   = (preds_dist * atoms_np).sum()

        # Print bins where there is significant mass (skip empty ones for clarity)
        for i in range(self.n_atoms):
            # Only print if either distribution has > 1% mass
            if target_dist[i] > 0.01 or preds_dist[i] > 0.01:
                bar_t = "#" * int(target_dist[i] * 50)
                bar_p = "|" * int(preds_dist[i] * 50)
                print(f"Atom {atoms_np[i]:5.2f} : {target_dist[i]:.4f} [{bar_t:<20}] vs {preds_dist[i]:.4f} [{bar_p:<20}]")

        print("-" * 65)
        print(f"Sum Mass:    Target={target_mass:.5f}      Pred={pred_mass:.5f}")
        print(f"Exp Value:   Target={target_mean:.5f}      Pred={pred_mean:.5f}")
        print("-" * 65)
        
        if abs(target_mass - 1.0) > 0.1:
             print("⚠️ WARNING: Target mass is NOT 1.0. Projection logic is deleting mass!")
        elif target_mean < -0.9 and r_last.item() == -1:
             print("✅ SUCCESS: Target correctly identifies a loss.")
        elif target_mean > 0.9 and r_last.item() == 1:
             print("✅ SUCCESS: Target correctly identifies a win.")
        else:
             print("❓ UNCERTAIN: Target is not sharp.")
    

    @torch.no_grad()
    def check_terminal_accuracy(self, z1, z2, actions, rewards, dones):
        """
        Scans batch for a terminal state and prints the Reward vs Predicted Mean.
        Useful for tracking if the agent is beginning to 'get it'.
        """
        # 1. Find indices of sequences that end with done=True
        mask_term = dones[:, -1]
        
        if not mask_term.any():
            return None, None, None

        # 2. Pick the first one found
        idx = torch.nonzero(mask_term, as_tuple=True)[0][0]

        # 3. Extract State, Action, Reward
        # Reconstruct the latent state z at the final step
        z_last = torch.cat([z1[idx, -1], z2[idx, -1]], dim=-1).unsqueeze(0) 
        z_last = F.layer_norm(z_last, z_last.shape[-1:])
        
        a_last = actions[idx, -1].long().item()
        r_actual = rewards[idx, -1].item()

        # 4. Get Network Prediction
        logits = self._logits(z_last)           # (1, A, N)
        probs  = logits.softmax(dim=-1)         # (1, A, N)
        dist_a = probs[0, a_last]               # (N,) - distribution for taken action
        
        # 5. Calculate Expected Value (Mean)
        pred_mean = (dist_a * self.atoms).sum().item()
        gap = r_actual - pred_mean

        # 6. Log it
        #print(f"[Terminal Check] Reward: {r_actual: .4f} | Pred Mean: {pred_mean: .4f} | Gap: {r_actual - pred_mean:.4f}")
        return r_actual, pred_mean, gap
    
    import matplotlib.pyplot as plt


