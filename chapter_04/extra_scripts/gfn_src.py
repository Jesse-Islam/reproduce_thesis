import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from IPython.display import clear_output
from torch.distributions import Categorical

class SimpleExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        """
        Args:
            input_dim: Dimensionality of the input (state)
            hidden_dim: Number of hidden units in each expert
            output_dim: Dimensionality of the expert output (e.g. logits for actions)
            num_experts: Number of expert networks
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        # Create a list of experts.
        self.experts = nn.ModuleList(
            [SimpleExpert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )
        
        # Gating network: a small MLP that outputs a weight distribution over experts.
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x):
        # x has shape (batch_size, input_dim) - if using single state, add batch dim if needed.
        # Compute gating scores and normalize them.
        gate_logits = self.gate(x)              # shape: (batch_size, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # shape: (batch_size, num_experts)
        
        # Compute each expert's output.
        expert_outputs = [expert(x) for expert in self.experts]  # list of (batch_size, output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)        # shape: (batch_size, num_experts, output_dim)
        
        # Weighted sum of expert outputs.
        # Expand weights to match the experts shape.
        gate_weights = gate_weights.unsqueeze(-1)                # (batch_size, num_experts, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=1)   # shape: (batch_size, output_dim)
        return output


class TBModelMoE(nn.Module):
    def __init__(self, action_space, hidden_dim=256, num_experts=2, lr=3e-4):
        super(TBModelMoE, self).__init__()
        # include the stop bit
        self.action_space = action_space + 1

        # Mixture-of-Experts with same parameterization as TBModel
        self.moe = MixtureOfExperts(
            input_dim=self.action_space,
            hidden_dim=hidden_dim,
            output_dim=self.action_space * 2,
            num_experts=num_experts
        )

        # Normalizing constant
        self.logZ = nn.Parameter(torch.ones(1))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Get logits from MoE
        logits = self.moe(x)  # shape: (batch_size, action_space*2)
        A = self.action_space

        # Forward policy: only where x == 0
        P_F = logits[..., :A] * (1 - x) + x * -100
        # Backward policy: only where x == 1
        P_B = logits[..., A:] * x + (1 - x) * -100

        # Remove batch dim if input was single state
        return P_F.squeeze(0), P_B.squeeze(0)



class TBModel(nn.Module):
    def __init__(self,action_space, hidden_dim=256, lr=3e-4):
        super().__init__()
        # include the stop bit
        self.action_space = action_space + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.action_space, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, (self.action_space * 2))
        )
        self.optimizer=torch.optim.Adam(self.parameters(),  lr)
        self.logZ = nn.Parameter(torch.ones(1))
    def forward(self, x):
        logits = self.mlp(x)
        A = self.action_space
        # first A entries for P_F
        P_F = logits[..., :A] * (1 - x) + x * -100
        # next A entries for P_B
        P_B = logits[..., A:] * x + (1 - x) * -100
        # final 1 for flow
        #flow = logits[..., 2*A:]
        return P_F, P_B#, flow







class gflownet:
    def __init__(
        self,
        model,
        reward_fn,
        device='cpu',
        action_space=500,
        max_steps=7,
        min_steps=2,
        initial_temp=10.0,
        final_temp=1.0,
        total_episodes=20_000,
        batch_size=100,
        decay_fraction=0.1,
        verbose=False
    ):
        """
        model        : your torch.nn.Module (must have .logZ attribute)
        optimizer    : optimizer for the model
        reward_fn    : function(batch_states) -> rewards
        device       : 'cpu' or torch.device
        action_space : int
        max_steps    : int
        min_steps    : int
        initial_temp : float, starting temperature
        final_temp   : float, ending temperature
        total_episodes: int, total number of rollouts
        batch_size   : int
        decay_fraction: float, fraction of total batches over which to decay temp
        """
        self.model          = model.to(device)
        self.optimizer      = model.optimizer
        self.reward_fn      = reward_fn
        self.device         = device
        self.action_space   = action_space
        self.max_steps      = max_steps + 1
        self.min_steps      = min_steps
        self.initial_temp   = initial_temp
        self.final_temp     = final_temp
        self.total_episodes = total_episodes
        self.batch_size     = batch_size
        self.verbose        = verbose
        # how many batches to decay over
        self.decay_batches  = (self.total_episodes / self.batch_size) * decay_fraction

    def cosine_similarity_matrix(self,final_states):
        """
        Computes a matrix of pairwise cosine similarities for a batch of state vectors.
        values are fixed between 0 and 1 because of my inputs being fixed as 0 or 1
        
        Args:
            final_states: Tensor of shape (batch_size, state_dim)
        
        Returns:
            A tensor of shape (batch_size, batch_size) containing pairwise cosine similarities.
        """
        # Normalize the states along the feature dimension
        normalized_states = F.normalize(final_states, dim=1)
        # Compute cosine similarity as the dot product of normalized vectors
        sim_matrix = torch.matmul(normalized_states, normalized_states.transpose(0, 1))
        return sim_matrix
    
    def diversity_loss(self,final_states, weight=1.0):
        """
        Computes a diversity loss that penalizes pairwise similarity among final states.
        
        Args:
            final_states: Tensor of shape (batch_size, state_dim)
            weight: A scaling factor for the loss
        
        Returns:
            A scalar diversity loss.
        """
        sim_matrix = self.cosine_similarity_matrix(final_states)
        batch_size = sim_matrix.shape[0]
        # Exclude the diagonal (each state compared with itself)
        mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=sim_matrix.device)
        pairwise_sim = sim_matrix * mask
        # Sum up the similarities; higher sum means less diversity.
        loss = weight * pairwise_sim.sum() / (batch_size * (batch_size - 1))
        return loss
    def compute_temperature(self, batch_idx):
        frac = batch_idx / self.decay_batches
        return max(
            self.final_temp,
            self.initial_temp - (self.initial_temp - self.final_temp) * frac
        )

    
    def forward_rollout_batch(self,temp):
        """
        Same functionality; fused masks, single idx tensor, inline backward,
        clones preserved for autograd.
        """
        A            = self.model.action_space
        stop_idx     = A - 1
        action_value = 1.0
    
        # Initialize
        state       = torch.zeros(self.batch_size, A, device=device)
        log_P_F     = torch.zeros(self.batch_size, device=device)
        log_P_B     = torch.zeros(self.batch_size, device=device)
        finished    = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        reward      = torch.zeros(self.batch_size, device=device)
        steps       = torch.zeros(self.batch_size, dtype=torch.int32, device=device)
    
        idx = torch.arange(A, device=device)
    
        for t in range(self.max_steps):
            alive = torch.nonzero(~finished, as_tuple=True)[0]
            if alive.numel() == 0:
                break
    
            s = state[alive]                     # [B_alive, A]
            logits, _ = self.model(s)                 # [B_alive, A]
    
            # build & apply fused mask
            m = s.eq(action_value)
            fill = float('-1e8')
            if t < self.min_steps:
                m |= idx.eq(stop_idx)
            elif t == self.max_steps - 1:
                m |= idx.ne(stop_idx)
                fill = float('-1e9')
    
            logits = logits.masked_fill(m, fill) / temp
    
            # sample forward
            cat = Categorical(logits=logits)
            acts = cat.sample()                  # [B_alive]
            log_P_F[alive]     += cat.log_prob(acts)
    
            # update state (clone)
            new_s = s.clone()
            new_s[torch.arange(len(acts), device=self.device), acts] = action_value
            state[alive] = new_s
    
            # inline backward log-prob
            log_P_B[alive] += Categorical(logits=self.model(new_s)[1]).log_prob(acts)
    
            # record stops
            stops = acts == stop_idx
            if stops.any():
                idxs = alive[stops]
                reward[idxs]   = reward_fn(new_s[stops, :-1]).clamp(min=0).float()
                steps[idxs]    = t + 1
                finished[idxs] = True
    
        # finalize any trajectories that never stopped
        never = ~finished
        if never.any():
            steps[never]  = max_steps
            reward[never] = self.reward_fn(state[never, :-1]).clamp(min=0).float()
        return state, log_P_B, log_P_F, reward
        

    
    def rollout_and_compute_loss(self, temp):
        """
        Performs one rollout and returns:
          - final states tensor
          - average reward (float)
          - combined loss tensor (TB + entropy + optional diversity)
        """
        state, log_P_B, log_P_F, rewards= self.forward_rollout_batch(temp=temp)

        # === 2) TB loss ===
        # TB term: (logZ + forward_logp - log(reward) - backward_logp)^2
        tb_term = self.model.logZ + log_P_F - torch.log(rewards.clamp_min(1e-9)) - log_P_B
        tb_loss = 0.5*tb_term.pow(2).mean()


        return state, rewards.mean().item(), tb_loss

    def train(self):
        n_batches = int(self.total_episodes / self.batch_size)
        print("Starting training…")
        for batch_idx in range(n_batches):
            # 1) temperature
            temp = self.compute_temperature(batch_idx)

            # 2) rollout + basic loss
            final_states, mean_reward, loss = self.rollout_and_compute_loss(temp)

            # 3) diversity penalty (skip on first batch)
            if batch_idx > 0:
                div_pen = self.diversity_loss(final_states[:, :-1], weight=1.0)
                loss = loss + div_pen
            else:
                div_pen = 0.0

            # 4) backward & step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5) logging
            if (batch_idx % 10 == 0) and (batch_idx > 0) and self.verbose:
                clear_output(wait=True)
                print(
                    f"[Batch {batch_idx:4d}/{n_batches:4d}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Diversity Penalty: {div_pen:.4f}"
                )

        print("Training complete.")

    def generate(self,batch_size):
        """
        Same functionality; fused masks, single idx tensor, inline backward,
        clones preserved for autograd.
        """
        self.model.eval()
        A            = self.model.action_space
        stop_idx     = A - 1
        action_value = 1.0
        temp=1.0
        # Initialize
        state       = torch.zeros(batch_size, A, device=device)
        finished    = torch.zeros(batch_size, dtype=torch.bool, device=device)
        reward      = torch.zeros(batch_size, device=device)
        steps       = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
        idx = torch.arange(A, device=device)
    
        for t in range(self.max_steps):
            alive = torch.nonzero(~finished, as_tuple=True)[0]
            if alive.numel() == 0:
                break
    
            s = state[alive]                     # [B_alive, A]
            logits, _ = self.model(s)                 # [B_alive, A]
    
            # build & apply fused mask
            m = s.eq(action_value)
            fill = float('-1e8')
            if t < self.min_steps:
                m |= idx.eq(stop_idx)
            elif t == self.max_steps - 1:
                m |= idx.ne(stop_idx)
                fill = float('-1e9')
    
            logits = logits.masked_fill(m, fill) / temp
    
            # sample forward
            cat = Categorical(logits=logits)
            acts = cat.sample()                  # [B_alive]
    
            # update state (clone)
            new_s = s.clone()
            new_s[torch.arange(len(acts), device=self.device), acts] = action_value
            state[alive] = new_s
    
    
            # record stops
            stops = acts == stop_idx
            if stops.any():
                idxs = alive[stops]
                reward[idxs]   = reward_fn(new_s[stops, :-1]).clamp(min=0).float()
                steps[idxs]    = t + 1
                finished[idxs] = True
    
        # finalize any trajectories that never stopped
        never = ~finished
        if never.any():
            steps[never]  = max_steps
            reward[never] = self.reward_fn(state[never, :-1]).clamp(min=0).float()
        self.model.train()
        return state, reward
    

    def estimate_bit_frequencies(self,samples):
        """
        Estimate the frequency of each bit being 1 across a set of samples.
        
        Args:
            samples (Tensor): A binary tensor of shape (num_samples, n), where each row is a candidate vector.
        
        Returns:
            Tensor: A tensor of shape (n,) where each element is the fraction of samples with that bit set to 1.
        """
        samples_float = samples.float()
        frequencies = samples_float.mean(dim=0)
        return frequencies[:-1]   





def plot_bit_frequencies(grouped_data, title):
    """
    Creates a plot of bit frequency means with 95% confidence intervals.
    
    Args:
        grouped_data (dict): Dictionary mapping noise values to tensors of shape (replications, num_bits).
        title (str): Title of the figure.
    """
    plt.figure(figsize=(10, 6))
    
    # For each bit position, compute the mean and confidence interval across noise levels.
    for bit in range(num_bits):
        means = []
        cis = []
        for noise in noise_levels:
            # Get the data for a given noise level.
            data = grouped_data[noise].numpy()  # shape: (replications, num_bits)
            bit_data = data[:, bit]             # values for this bit across replications
            mean = np.mean(bit_data)
            std  = np.std(bit_data, ddof=1)       # use ddof=1 for an unbiased estimate
            sem  = std / np.sqrt(replications)    # standard error of the mean
            ci   = t_val * sem                    # confidence interval
            
            means.append(mean)
            cis.append(ci)
        # Plot error bars for this bit position.
        plt.errorbar(noise_levels, means, yerr=cis, marker='o', capsize=3, label=f'Bit {bit}')
    
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Bit Frequency')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
