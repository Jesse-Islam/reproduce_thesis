import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from IPython.display import clear_output
from functools import partial
#####################################################################################################################################
### Models
#####################################################################################################################################
class TBModel(nn.Module):
    def __init__(self,action_space, hidden_dim=256, lr=3e-4):
        super().__init__()
        # include the stop bit
        self.action_space = action_space + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.action_space, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, (self.action_space * 2))# + 1)#this is a bit if i need to track flow.
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



import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        A simple feed-forward expert network.

        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Number of hidden units.
            output_dim: Dimensionality of the output.
        """
        super(SimpleExpert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_experts: int):
        """
        Mixture of Experts (MoE) module that computes a weighted combination of expert outputs.

        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Number of hidden units in each expert.
            output_dim: Dimensionality of the output from each expert.
            num_experts: Number of expert networks.
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        # Initialize the list of experts.
        self.experts = nn.ModuleList(
            [SimpleExpert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )
        
        # Gating network: outputs unnormalized weights for each expert.
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Aggregated output tensor of shape (batch_size, output_dim).
        """
        # Compute the gating logits and convert them to a probability distribution.
        gate_logits = self.gate(x)                    # Shape: (batch_size, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # Shape: (batch_size, num_experts)

        # Get outputs from each expert.
        expert_outputs = [expert(x) for expert in self.experts]  # Each: (batch_size, output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)        # Shape: (batch_size, num_experts, output_dim)

        # Combine expert outputs using the gating weights.
        gate_weights = gate_weights.unsqueeze(-1)                # Shape: (batch_size, num_experts, 1)
        output = torch.sum(gate_weights * expert_outputs, dim=1)   # Shape: (batch_size, output_dim)
        return output


class TBModelMoE(nn.Module):
    def __init__(self, action_space: int, hidden_dim: int, num_experts: int = 2, lr: float = 3e-4):
        """
        TBModelMoE uses a Mixture of Experts to generate separate forward and backward policy logits.

        The model adjusts the action space by adding a stop bit, and the MoE outputs two sets of
        logits (one for forward actions and one for backward actions).

        Args:
            action_space: Number of actions (the model internally adds one more for the stop bit).
            hidden_dim: Number of hidden units in experts and gating network.
            num_experts: Number of experts in the mixture.
            lr: Learning rate for the optimizer.
        """
        super(TBModelMoE, self).__init__()
        # Adjust action space to include a stop bit.
        self.action_space = action_space + 1

        # The MoE produces twice the number of action logits (forward and backward).
        self.moe = MixtureOfExperts(
            input_dim=self.action_space,
            hidden_dim=hidden_dim,
            output_dim=self.action_space * 2,
            num_experts=num_experts
        )

        # Learnable log-partition parameter.
        self.logZ = nn.Parameter(torch.ones(1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for TBModelMoE.

        This function processes the input state to compute two sets of logits:
          - Forward policy logits (P_F)
          - Backward policy logits (P_B)
        
        Args:
            x: Input tensor of shape (action_space,) or (1, action_space).

        Returns:
            A tuple (P_F, P_B) where each is a tensor of shape (action_space,).
        """
        # Add batch dimension if missing.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Compute the combined logits from the MoE.
        logits = self.moe(x)  # Shape: (batch_size, self.action_space * 2)
        A = self.action_space

        # Process logits into forward and backward parts.
        # P_F: forward policy logits; favors actions not taken as indicated by x.
        # P_B: backward policy logits; favors actions taken as indicated by x.
        P_F = logits[:, :A] * (1 - x) + x * -1e-7
        P_B = logits[:, A:] * x + (1 - x) * -1e-7

        return P_F.squeeze(0), P_B.squeeze(0)

#####################################################################################################################################
### gflownet class
#####################################################################################################################################

class GFlowNetTrainer:   
    def __init__(self,
                 model, 
                 reward_fn,
                 action_space=50,
                 update_freq=20,
                 max_steps=10,
                 min_steps=3,
                 reward_min=0.0,
                 reward_max=None,              # Defaults to action_space + 1
                 initial_temp=10.0,
                 final_temp=1.0,
                 decay_episodes=2000,
                 beta_entropy=0.01,
                 total_episodes=20000,
                 clamp_min = 1e-7,
                 k_rewards = 1,
                 k_chunk_size=10,
                 device=None,
                verbose=False):
        self.model=model
        self.reward_fn=reward_fn
        #self.reward_fn_batched = torch.vmap(self.reward_fn, in_dims=(0,),randomness='different',chunk_size=min(k_rewards,k_chunk_size))  # maps over first dim
        self.action_space = action_space
        self.update_freq = update_freq
        self.batch_size = update_freq
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.reward_min = reward_min
        self.reward_max = reward_max if reward_max is not None else action_space + 1
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_episodes = decay_episodes
        self.beta_entropy = beta_entropy
        self.total_episodes = total_episodes
        self.clamp_min=clamp_min
        self.k_rewards = k_rewards
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose=verbose
        torch.autograd.set_detect_anomaly(False)
        

        
        self.model.to(self.device)
        self.logZs = []  # To track the log normalization constant



    '''
    def forward_rollout_batched(self, temp):
        
        device = self.device
        action_value = 1.0
        action_space = self.model.action_space
        stop_action = action_space - 1
    
        # Initialize states for the batch (each row corresponds to one trajectory)
        states = torch.zeros(self.batch_size, action_space, device=device).float()
    
        # Accumulators per trajectory
        total_entropy = torch.zeros(self.batch_size, device=device)
        total_P_F = torch.zeros(self.batch_size, device=device)
        total_P_B = torch.zeros(self.batch_size, device=device)
        rewards = torch.zeros(self.batch_size, device=device)
    
        # A mask for trajectories that have terminated (via selecting the stop action)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        t = 0
        while (t < self.max_steps) and (not done.all()):
            active_idx = torch.nonzero(~done, as_tuple=True)[0]
            if active_idx.numel() > 0:
                active_states = states[active_idx, :]
                if active_states.dim() == 1:
                    active_states = active_states.unsqueeze(0)
                P_F_s, _ = self.model(active_states)  # get the forward probabilities for each state
                #P_F_s, _ = self.model(torch.zeros(active_states.shape, device=device))
                if P_F_s.dim() == 1:
                    P_F_s = P_F_s.unsqueeze(0)
                mask = (active_states == action_value)  # boolean tensor, True if action is disallowed
                
                # Clone the logits so we don't modify them in-place
                masked_logits = P_F_s.clone()
                masked_logits[mask] = self.clamp_min  # set to a large negative value
    
                if t < self.min_steps:
                    masked_logits[:, stop_action] = self.clamp_min
    
                # Scale logits by dividing by the current temperature.
                masked_logits = masked_logits / temp
                cat = Categorical(logits=masked_logits)
                actions = cat.sample()
                if actions.dim() == 0:
                    actions = actions.unsqueeze(0)
                entropies = cat.entropy()
                total_entropy[active_idx] = total_entropy[active_idx] + entropies  # accumulate entropy
                
                new_states = active_states.clone()
                # Make sure indices are created on the correct device.
                indices = torch.arange(new_states.size(0), device=device)
                new_states[indices, actions] = action_value
                states[active_idx,:] = new_states
    
                total_P_F[active_idx] = total_P_F[active_idx] + cat.log_prob(actions)
    
                # Check if any active trajectory has selected the stop action.
                stop_mask = (actions == stop_action)
                if stop_mask.any():
                    finished_idx = active_idx[stop_mask]
                    for i in finished_idx:
                        idx = i.item()
                        rewards[idx] = self.scale_rewards(
                            self.reward_resample(states[idx, :-1]).float()
                        )
                        #rewards[idx] = self.reward_resample(states[idx, :-1]).float() 
                    # Mark these trajectories as finished (using an out-of-place update)
                    done = done.clone()
                    done[finished_idx] = True
                _, P_B_s = self.model(new_states)
                if P_B_s.dim() == 1:
                    P_B_s = P_B_s.unsqueeze(0)
                total_P_B[active_idx] = total_P_B[active_idx] + Categorical(logits=P_B_s).log_prob(actions)
                t = t + 1
        not_finished = (states[:, stop_action] != action_value)
        if not_finished.any():
            nf_idx = torch.nonzero(not_finished, as_tuple=True)[0]
            nf_states = states[nf_idx, :]
            if nf_states.dim() == 1:
                nf_states = nf_states.unsqueeze(0)
            # Create a mask that disallows all actions except the stop action.
            mask_final = torch.ones_like(nf_states, dtype=torch.bool)
            mask_final[:, stop_action] = False
            # Clone the logits so we don't modify them in-place
            P_F_s, _ = self.model(nf_states)
            if P_F_s.dim() == 1:
                P_F_s = P_F_s.unsqueeze(0)
            masked_logits = P_F_s.clone()
            masked_logits[mask_final] = self.clamp_min  # set to a large negative value
            cat = Categorical(logits=masked_logits)
            actions = cat.sample()
            if actions.dim() == 0:
                actions = actions.unsqueeze(0)
            entropies = cat.entropy()
            total_entropy[nf_idx] = total_entropy[nf_idx] + entropies  # accumulate entropy
            
            nf_states = nf_states.clone()
            indices = torch.arange(nf_states.size(0), device=device)
            nf_states[indices, actions] = action_value
            states[nf_idx,:] = nf_states
            stop_mask = (actions == stop_action)
            if stop_mask.any():
                finished_idx = nf_idx
                for i in finished_idx:
                    idx = i.item()
                    rewards[idx] = self.scale_rewards(self.reward_resample(states[idx, :-1]).float() )
                    #rewards[idx] = self.reward_resample(states[idx, :-1]).float() 
                # Mark these trajectories as finished (using an out-of-place update)
                done = done.clone()
                done[finished_idx] = True
            _, P_B_s = self.model(nf_states)
            if P_B_s.dim() == 1:
                P_B_s = P_B_s.unsqueeze(0)
            total_P_B[nf_idx] = total_P_B[nf_idx] + Categorical(logits=P_B_s).log_prob(actions)
        total_entropy = total_entropy / (states.detach().sum(axis=1))
        return states.detach().clone(), total_P_B, total_P_F, rewards, total_entropy
    '''
    def forward_rollout_batched(self, temp):
        # Initialize everything
        states, total_entropy, total_P_F, total_P_B, rewards, done = self._init_rollout()
        t = 0
        # Main rollout loop
        while (t < self.max_steps) and (not done.all()):
            (states, total_entropy, total_P_F,
             total_P_B, rewards, done) = self._rollout_step(
                states, total_entropy, total_P_F,
                total_P_B, rewards, done, temp, t
            )
            t += 1

        # Final forced-stop step
        (states, total_entropy, total_P_F,
         total_P_B, rewards, done) = self._final_rollout(
            states, total_entropy, total_P_F,
            total_P_B, rewards, done, temp
        )

        # Return batch of results
        return states.detach().clone(), total_P_B, total_P_F, rewards, total_entropy

    def _init_rollout(self):
        device = self.device
        A = self.model.action_space
        # Initial empty states and accumulators
        states = torch.zeros(self.batch_size, A, device=device).float()
        total_entropy = torch.zeros(self.batch_size, device=device)
        total_P_F = torch.zeros(self.batch_size, device=device)
        total_P_B = torch.zeros(self.batch_size, device=device)
        rewards = torch.zeros(self.batch_size, device=device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=device)
        return states, total_entropy, total_P_F, total_P_B, rewards, done

    def _mask_forward_logits(self, logits, states, t, temp):
        # Disable already-taken actions and early stopping
        mask = states == 1.0
        masked = logits.clone()
        masked[mask] = self.clamp_min
        if t < self.min_steps:
            masked[:, self.model.action_space - 1] = self.clamp_min
        return masked / temp

    def _rollout_step(
        self, states, total_entropy,
        total_P_F, total_P_B, rewards, done,
        temp, t
    ):
        device = self.device
        stop_action = self.model.action_space - 1
        active_idx = torch.nonzero(~done, as_tuple=True)[0]
        if active_idx.numel() == 0:
            return states, total_entropy, total_P_F, total_P_B, rewards, done

        # Gather active states
        active_states = states[active_idx]
        if active_states.dim() == 1:
            active_states = active_states.unsqueeze(0)

        # Forward policy logits
        P_F_s, _ = self.model(active_states)
        if P_F_s.dim() == 1:
            P_F_s = P_F_s.unsqueeze(0)
        masked_logits = self._mask_forward_logits(P_F_s, active_states, t, temp)

        # Sample actions
        cat = Categorical(logits=masked_logits)
        actions = cat.sample()
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)

        # Accumulate entropy and log-prob
        total_entropy[active_idx] += cat.entropy()
        total_P_F[active_idx] += cat.log_prob(actions)

        # Update states with sampled actions
        new_states = active_states.clone()
        idx = torch.arange(new_states.size(0), device=device)
        new_states[idx, actions] = 1.0
        states[active_idx] = new_states

        # Handle any stops
        stop_mask = (actions == stop_action)
        if stop_mask.any():
            finished_idx = active_idx[stop_mask]
            for i in finished_idx:
                idx_item = i.item()
                rewards[idx_item] =self.reward_resample(states[idx_item, :-1]).float()
            done = done.clone()
            done[finished_idx] = True

        # Backward policy log-prob
        _, P_B_s = self.model(new_states)
        if P_B_s.dim() == 1:
            P_B_s = P_B_s.unsqueeze(0)
        total_P_B[active_idx] += Categorical(logits=P_B_s).log_prob(actions)

        return states, total_entropy, total_P_F, total_P_B, rewards, done

    def _final_rollout(
        self, states, total_entropy,
        total_P_F, total_P_B, rewards, done,
        temp
    ):
        device = self.device
        stop_action = self.model.action_space - 1

        not_finished = (states[:, stop_action] != 1.0)
        if not_finished.any():
            nf_idx = torch.nonzero(not_finished, as_tuple=True)[0]
            nf_states = states[nf_idx]
            if nf_states.dim() == 1:
                nf_states = nf_states.unsqueeze(0)

            # Force only stop action
            P_F_s, _ = self.model(nf_states)
            if P_F_s.dim() == 1:
                P_F_s = P_F_s.unsqueeze(0)
            mask_final = torch.ones_like(nf_states, dtype=torch.bool)
            mask_final[:, stop_action] = False
            masked_logits = P_F_s.clone()
            masked_logits[mask_final] = self.clamp_min
            cat = Categorical(logits=masked_logits)
            actions = cat.sample()
            if actions.dim() == 0:
                actions = actions.unsqueeze(0)

            # Accumulate entropy
            total_entropy[nf_idx] += cat.entropy()

            # Update nf_states
            new_nfs = nf_states.clone()
            idx = torch.arange(new_nfs.size(0), device=device)
            new_nfs[idx, actions] = 1.0
            states[nf_idx] = new_nfs

            # Original behavior: reward all not-finished if any chose stop
            stop_mask = (actions == stop_action)
            if stop_mask.any():
                finished_idx = nf_idx
                for i in finished_idx:
                    idx_item = i.item()
                    rewards[idx_item] = self.reward_resample(states[idx_item, :-1]).float()
                done = done.clone()
                done[finished_idx] = True

            # Accumulate backward log-prob
            _, P_B_s = self.model(new_nfs)
            if P_B_s.dim() == 1:
                P_B_s = P_B_s.unsqueeze(0)
            total_P_B[nf_idx] += Categorical(logits=P_B_s).log_prob(actions)

        # Normalize entropy by Hamming weight (unchanged behavior)
        total_entropy = total_entropy / (states.detach().sum(axis=1))
        return states, total_entropy, total_P_F, total_P_B, rewards, done

    def forward_rollout(self,temp):
        
        total_entropy = 0  # Accumulate entropy across steps.
        state = torch.zeros(self.model.action_space).float().to(self.device)
        action_value=1.0
        total_P_F = 0
        total_P_B = 0
        t=0
        while t < self.max_steps:
            P_F_s, _ = self.model(state)
            mask = (state == action_value)  # boolean tensor, True if action is disallowed
            # clone the logits so we don't modify them in-place
            masked_logits = P_F_s.clone()
            masked_logits[mask] = self.clamp_min  # set to a large negative value
            if t < self.min_steps:
                masked_logits[self.model.action_space - 1] = self.clamp_min
            # Scale logits by dividing by the current temperature.
            masked_logits = masked_logits / temp
            cat = Categorical(logits=masked_logits)
            action = cat.sample()
            entropy = cat.entropy()
            total_entropy += entropy  # accumulate entropy
            new_state = state.clone()
            new_state[action] = action_value
            
            total_P_F = total_P_F + cat.log_prob(action)
            if (action==(self.model.action_space-1)):
                reward =  R=self.reward_resample(new_state[:-1]).float()
                t=self.max_steps
            _, P_B_s  = self.model(new_state)
            total_P_B = total_P_B + Categorical(logits=P_B_s).log_prob(action)
            t=t+1
            state = new_state
        if state[self.model.action_space-1]!=action_value:
            mask = (state == state)  # boolean tensor, True if action is disallowed
            mask[self.model.action_space-1]=False
            # clone the logits so we don't modify them in-place
            masked_logits = P_F_s.clone()
            masked_logits[mask] = self.clamp_min  # set to a large negative value
            cat = Categorical(logits=masked_logits)
            action = cat.sample()
            entropy = cat.entropy()
            total_entropy += entropy  # accumulate entropy
            new_state = state.clone()
            new_state[action] = action_value
            reward =  rewards[idx] = self.reward_resample(states[idx, :-1]).float() 
            _, P_B_s  = self.model(new_state)
            total_P_B = total_P_B + Categorical(logits=P_B_s).log_prob(action)
            state = new_state
        total_entropy=total_entropy/t
        return state,total_P_B,total_P_F,reward,total_entropy


    
    def cosine_similarity_matrix(self,final_states):
        normalized_states = F.normalize(final_states, dim=1)
        sim_matrix = torch.matmul(normalized_states, normalized_states.transpose(0, 1))
        return sim_matrix
    
    def diversity_loss(self,final_states, weight=1.0):
        sim_matrix = self.cosine_similarity_matrix(final_states)
        batch_size = sim_matrix.shape[0]
        mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=sim_matrix.device)
        pairwise_sim = sim_matrix * mask
        loss = weight * pairwise_sim.sum() / (batch_size * (batch_size - 1))
        return loss

    def reward_resample(self, state):
        """
        Estimate the expected reward by averaging k noisy evaluations,
        now using the vectorized reward_fn_batch.
        """
        self.k_rewards
        
        # 1) Ensure state has shape (1, D)
        if state.dim() == 1:
            state = state.unsqueeze(0)          # (1, D)
        
        # 2) Repeat to form a batch of size k: (1, D) -> (k, D)
        state_batch = state.repeat(self.k_rewards, 1)        # (k, D)
        
        # 3) Compute all k rewards in one call
        rewards = self.reward_fn(state_batch) # (k,)
        
        # 4) Return the mean reward
        return rewards.mean(0)                  # scalar tensor


    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    def train(self):
        minibatch_loss = 0
        mean_rewards = 0.0
        batch_states = []
        episode=0
        
        while episode < (self.total_episodes+1):
            # Decay the temperature linearly from initial_temp to final_temp.
            temp = max(self.final_temp,
                       self.initial_temp - (self.initial_temp - self.final_temp) * (episode / self.decay_episodes))
            
            # Generate a rollout using the placeholder forward_rollout function.
            batch_states, total_P_Bs, total_P_Fs, rewards, total_entropies = self.forward_rollout_batched(temp=temp)
            episode=episode+self.batch_size
            # Compute the TB loss:
            #entropy_bonus = self.beta_entropy * total_entropies
            entropy_bonus=0
            minibatch_loss = ((self.model.logZ + total_P_Fs - torch.log(torch.clamp(rewards, min=1e-8)).clamp(-20) - total_P_Bs).pow(2) - entropy_bonus).mean()
            mean_rewards = rewards.mean()
            # Perform an update every 'update_freq' episodes.
            if (episode % self.update_freq) == 0 and episode > 0:
                diversity_penalty = self.diversity_loss(final_states=batch_states[:-1], weight=1.0)
                minibatch_loss = minibatch_loss + diversity_penalty
                batch_states = []
                minibatch_loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                
                avg_loss = minibatch_loss.item()
                minibatch_loss = 0
                self.logZs.append(self.model.logZ.item())
                if self.verbose:
                    clear_output(wait=True)
                    print(f"[Episode {episode}] TB Loss: {avg_loss:.4g} | reward: {mean_rewards:.2g} | current_penalty: {diversity_penalty:.2g}")
                mean_rewards = 0

    def generate(self,num_samples,temp=1.0):
        self.batch_size=num_samples
        self.model.eval()
        with torch.no_grad():
            states, _, _, rewards, _=self.forward_rollout_batched(temp=temp)
        self.model.train()
        self.batch_size=self.update_freq
        return states[:,:-1],rewards



def estimate_bit_frequencies(samples):
    samples_float = samples.float()
    frequencies = samples_float.mean(dim=0)
    return frequencies


def plot_bit_frequencies(bit_frequencies, red_indices=None):
    n = len(bit_frequencies)
    if red_indices is None:
        red_indices = []
    
    # Create a color list: "red" if the index is in red_indices, "skyblue" otherwise.
    colors = ["red" if i in red_indices else "skyblue" for i in range(n)]
    
    plt.figure(figsize=(24, 6))
    plt.bar(range(n), bit_frequencies.cpu().numpy(), color=colors)
    plt.xlabel("Bit Position")
    plt.ylabel("Frequency of 1")
    plt.title("Frequency of 1s per Bit Position")
    plt.xticks(range(n))
    plt.show()
