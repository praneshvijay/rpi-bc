# Repository Overview
This repository implements **Attention-Based Credit (ABC)** assignment for Reinforcement Learning from Human Feedback (RLHF). Normally in RLHF, the reward model provides a single scalar reward at the very end of a generated response, which is sparse and makes step-by-step credit assignment difficult. This repository modifies the PPO training loop to redistribute this final scalar reward across the individual tokens of the generated response, using attention maps to determine which tokens deserve the most credit.

The core PPO logic is inside [abcrl/rl/ppo.py](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/rl/ppo.py), which features [PPOTrainerABC](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/rl/ppo.py#30-460)—a subclass of `trl.PPOTrainer`. It adapts the [step](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/rl/ppo.py#35-325) and [compute_rewards](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/rl/ppo.py#326-359) methods to handle per-token dense rewards instead of just a single sequence-level reward.

The experiment scripts (e.g., [experiments/scripts/rlhf_imdb.py](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/experiments/scripts/rlhf_imdb.py)) oversee generating responses, gathering baseline sequence-level rewards (e.g., from a sentiment classifier acting as the reward model), computing the attention distribution, scattering the sequential reward to specific tokens, and finally stepping the [PPOTrainerABC](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/rl/ppo.py#30-460).

# Existing Attention Distribution Mechanisms

The attention redistribution functions are defined in [abcrl/attention/redistribution.py](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/attention/redistribution.py). Below is the global list of implementations used across the experiment scripts:

### Baselines (No custom redistribution)
- **`rlhf`**: The standard PPO baseline. The full scalar reward is given solely and sparsely to the last token.
- **`uniform`**: A dense reward baseline where the final reward is uniformly distributed equally across all generated tokens.

### Generator Attention Mechanisms
1. **[get_generator_attention_distribution](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/attention/redistribution.py#44-93) (Methods: `abcde`, `abcde2`)**:
   - Computes the attention map using the **Generator Policy Model's** attention (the model actually generating the text).
   - `last_only=True` (`abcde2`): Considers only the attention map of the very last generated token.
   - `last_only=False` (`abcde`): Computes a weighted average of the attention maps across all generated tokens throughout the generation process.

### Reward Model Attention Mechanisms (Shallow)
2. **[get_attention_distribution](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/attention/redistribution.py#7-41) (Method: `abc`)**:
   - Computes the attention map using the **Reward Model's** attention layers.
   - Extracts the last row from the attention matrix (representing how the final classification token attends to all previous tokens), aligns it, and normalizes it.

### Reward Model Attention Mechanisms (Deep / Diffusion-Based)
3. **[get_rpi_attention_distribution_batch](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/attention/redistribution.py#146-216) (Method: `rpibc`)**:
   - Uses **Integrated Gradients** combined with a diffusion process (adding noise over time steps) on a single attention block of the Reward Model.
   - Diffuses the attention map, measures the gradient of the reward output with respect to the attention weights, and yields robust token-level attribution scores (last-row based).

4. **[get_rpi_deep_attention_distribution_batch](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/attention/redistribution.py#297-387) (Method: `rpibc_deep`)**:
   - A multi-layer adaptation of RPIBC. It patches multiple (or all) attention blocks in the Reward Model.
   - Diffuses attention probabilities across all layers simultaneously and computes GradCAM-style gradient weightings for each layer.
   - Uses **column-sum** (how much each token was attended to by ALL subsequent tokens) instead of last-row, distributing credit more fairly across the entire generation span.
   - *(Note: Includes specialized variants like `..._gptj` and `..._llama` to handle model-specific attention hooking).*

# How to Implement and Add Your Own Method

To add a brand new attention distribution method (e.g., `"my_custom_method"`), you will need to follow three steps:

### Step 1: Define your distribution function
Add your custom logic in [abcrl/attention/redistribution.py](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/abcrl/attention/redistribution.py):
```python
def get_custom_attention_distribution(
    response: torch.Tensor, query: torch.Tensor, attention: torch.Tensor, **kwargs
) -> torch.Tensor:
    # 1. Process the `attention` tensor (from either the reward model or generator)
    # 2. Map the attention weights to the `response` tokens
    out = torch.zeros_like(response, dtype=float)
    
    # ... your custom logic to populate `out` with attention weights ...
    
    # 3. Return the normalized distribution
    return (out / out.sum()).detach()
```

### Step 2: Update the Experiment Script(s)
Modify the relevant experiment scripts that you wish to run (e.g., [experiments/scripts/rlhf_imdb.py](file:///home/prnes/Desktop/8th_Sem/BTP/attention-based-credit/experiments/scripts/rlhf_imdb.py)).

**1. Add your method to the assertions/arguments:**
Around line 69, add your method name to the allowed list:
```python
assert method in ["rlhf", "abc", "abcde", "abcde2", "uniform", "my_custom_method"]
```

**2. Extract the correct attention tensor:**
Around line 160, ensure your method receives the right attention data (from the Reward Model or from the Generator). For example, if your method uses the reward model's attention:
```python
if method in ["abc", "my_custom_method"]:  
    attention = out.attentions[-1].mean(1)  # last layer averaged over heads
elif method in ["abcde", "abcde2"]: 
    attention = response_attentions
```

**3. Apply your custom redistribution:**
Inside the token-level reward assignment loop (around line 176), add an `elif` block for your method:
```python
elif method == "my_custom_method":
    # Optional: Reserve some portion (1 - beta) of the reward for the final token
    reward[-1] = (1 - beta) * total
    
    # Calculate and add redistributed reward to all tokens based on your function
    distribution = get_custom_attention_distribution(response, query, attention)
    redist_reward = torch.tensor(distribution, device=reward.device) * total * beta
    reward += redist_reward
```

### Step 3: Run the Experiment
You can now run the experiment script passing your new method:
```bash
python experiments/scripts/rlhf_imdb.py --method my_custom_method --beta 0.5
```
