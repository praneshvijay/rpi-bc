import torch
import torch.nn as nn
import numpy as np
import contextlib


def get_attention_distribution(
    response: torch.Tensor, query: torch.Tensor, attention: torch.Tensor
) -> torch.Tensor:
    """
    Compute the ABC attention map for a given response and query.

    Args:
        response (torch.Tensor): The response tensor.
        query (torch.Tensor): The query tensor.
        attention (torch.Tensor): The attention tensor.

    Returns:
        torch.Tensor: The attention distribution.

    """
    attention = attention.squeeze().cpu().detach().numpy()
    attention_matrix = attention[
        len(query) : len(query) + len(response),
        len(query) : len(query) + len(response),
    ]
    attention_map = attention_matrix[-1, :]  # / attention_matrix[-1, :].sum()

    out = torch.zeros_like(response, dtype=float)
    if len(attention_map) == len(out):
        out += torch.tensor(attention_map, device=out.device)
    elif len(attention_map) < len(out):
        out[len(out) - len(attention_map) :] += torch.tensor(
            attention_map, device=out.device
        )
    else:
        out += torch.tensor(
            attention_map[len(attention_map) - len(out) :], device=out.device
        )

    return (out / out.sum()).detach()


def get_generator_attention_distribution(
    response: torch.Tensor,
    query: torch.Tensor,
    attention: torch.Tensor,
    last_only: bool = False,
) -> torch.Tensor:
    """
    Compute the ABC-D attention distribution for a generator given a response and query.

    Args:
        response (torch.Tensor): The response tensor.
        query (torch.Tensor): The query tensor.
        attention (torch.Tensor): The attention tensor.
        last_only (bool, optional): Whether to consider only the last tokens attention
        map, otherwise weighted average over generation. Defaults to False.

    Returns:
        torch.Tensor: The attention distribution.

    """
    out = torch.zeros_like(response, dtype=float)

    if last_only:
        attention_map = attention[-1][-1].squeeze().mean(0)[len(query) :]
        attention_map = torch.cat(
            (torch.zeros(1, device=attention_map.device), attention_map), 0
        )

    else:
        attention_matrix = torch.zeros((len(response), len(response)))

        for i, token_att in enumerate(attention[1:]):
            att_map = token_att[-1].squeeze().mean(0)[len(query) :]
            attention_matrix[i + 1, 1 : len(att_map) + 1] = att_map

        weight = torch.nan_to_num(1 / (attention_matrix != 0).sum(axis=0), posinf=0)
        attention_map = (attention_matrix * weight).sum(axis=0)

    if len(attention_map) == len(out):
        out += torch.tensor(attention_map, device=out.device)
    elif len(attention_map) < len(out):
        out[len(out) - len(attention_map) :] += torch.tensor(
            attention_map, device=out.device
        )
    else:
        out += torch.tensor(
            attention_map[len(attention_map) - len(out) :], device=out.device
        )

    return (out / out.sum()).detach()

@contextlib.contextmanager
def patch_gpt2_attention(model, rpi_attn_prob):
    # Find last block
    last_block = model.transformer.h[-1]
    original_attn = last_block.attn._attn
    
    def patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = rpi_attn_prob
        
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    last_block.attn._attn = patched_attn.__get__(last_block.attn, type(last_block.attn))
    try:
        yield
    finally:
        last_block.attn._attn = original_attn

def _compute_alphas_from_time_steps(device, num_time_steps):
    beta1 = 1e-4
    beta2 = 0.02
    beta_t = (beta2 - beta1) * torch.linspace(0, 1, num_time_steps + 1, device=device) + beta1
    alpha_t = 1 - beta_t
    alpha_cumulative = torch.cumsum(alpha_t.log(), dim=0).exp()
    alpha_cumulative[0] = 1
    return alpha_cumulative

def _compute_attention_gradients(model, inputs, attention_probabilities):
    model.zero_grad()
    attention_probabilities = attention_probabilities.detach().requires_grad_(True)
    attention_probabilities.retain_grad()
    
    with patch_gpt2_attention(model, attention_probabilities):
        out = model(**inputs)
        
    logits = out.logits
    total = (logits[:, 1] - logits[:, 0]).sum()
    total.backward()
    
    gradients = attention_probabilities.grad.detach()
    attention_probabilities.requires_grad = False
    model.zero_grad()
    torch.cuda.empty_cache()
    return gradients

def get_rpi_attention_distribution_batch(
    model, 
    inputs, 
    responses: list, 
    queries: list, 
    attention_probabilities: torch.Tensor,
    num_time_steps: int = 10,
    num_interpolation: int = 5,
    num_samples: int = 1
) -> list:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    alpha_cumulative = _compute_alphas_from_time_steps(model.device, num_time_steps)
    
    all_attribution_scores = []
    
    for _ in range(num_samples):
        noise_tensor = torch.randn_like(attention_probabilities)
        baseline_attention = (
            alpha_cumulative[num_time_steps].sqrt() * attention_probabilities + 
            (1 - alpha_cumulative[num_time_steps]) * noise_tensor
        )
        
        delta = attention_probabilities - baseline_attention
        scales = torch.linspace(0, 1, num_interpolation + 1, device=model.device).view(-1, 1, 1, 1, 1)
        interpolated_attention = baseline_attention.unsqueeze(0) + scales * delta.unsqueeze(0)
        
        gradients_list = []
        for i in range(num_interpolation + 1):
            att_step = interpolated_attention[i]
            grad = _compute_attention_gradients(model, inputs, att_step)
            gradients_list.append(grad)
            
        gradients = torch.stack(gradients_list)
        mean_gradient = torch.mean(gradients, dim=0)
        attribution_scores = mean_gradient * attention_probabilities
        all_attribution_scores.append(attribution_scores.detach())
        
    avg_attribution = torch.stack(all_attribution_scores).mean(dim=0)
    avg_attribution = avg_attribution.mean(dim=1).cpu().numpy()
    
    distributions = []
    for b in range(len(responses)):
        response = responses[b]
        query = queries[b]
        att = avg_attribution[b]
        
        attention_matrix = att[
            len(query) : len(query) + len(response),
            len(query) : len(query) + len(response),
        ]
        attention_map = np.abs(attention_matrix[-1, :])
        
        out = torch.zeros_like(response, dtype=float)
        if len(attention_map) == len(out):
            out += torch.tensor(attention_map, device=out.device)
        elif len(attention_map) < len(out):
            out[len(out) - len(attention_map) :] += torch.tensor(
                attention_map, device=out.device
            )
        else:
            out += torch.tensor(
                attention_map[len(attention_map) - len(out) :], device=out.device
            )

        out_sum = out.sum()
        if out_sum == 0:
            distributions.append((torch.ones_like(out) / len(out)).detach())
        else:
            distributions.append((out / out_sum).detach())
            
    return distributions


# ─── Deep Multi-Layer RPIBC ─────────────────────────────────────────────────
# Patches ALL (or the last N) GPT-2 transformer blocks simultaneously and
# aggregates gradient-weighted attributions across layers (GradCAM-style).
# Uses column-sum instead of last-row to give credit to consistently
# attended tokens across the whole generation.

@contextlib.contextmanager
def patch_gpt2_attention_multilayer(model, rpi_attn_probs_list, num_layers=None):
    """
    Patches the last `num_layers` GPT-2 blocks (default: all blocks).
    rpi_attn_probs_list: list of per-layer attention tensors (len == num_layers).
    """
    blocks = model.transformer.h
    if num_layers is None:
        num_layers = len(blocks)
    target_blocks = blocks[-num_layers:]

    original_attns = []
    for i, (block, attn_prob) in enumerate(zip(target_blocks, rpi_attn_probs_list)):
        original_attns.append(block.attn._attn)

        def make_patched(prob):
            def patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
                aw = prob.type(value.dtype)
                aw = self.attn_dropout(aw)
                if head_mask is not None:
                    aw = aw * head_mask
                return torch.matmul(aw, value), aw
            return patched_attn

        block.attn._attn = make_patched(attn_prob).__get__(block.attn, type(block.attn))

    try:
        yield
    finally:
        for block, orig in zip(target_blocks, original_attns):
            block.attn._attn = orig


def _compute_attention_gradients_multilayer(model, inputs, attn_probs_list):
    """
    Compute the gradient of the reward output w.r.t. each layer's attention.
    Returns a GradCAM-weighted average: layers with larger gradient magnitude
    contribute more to the final attribution map.
    """
    model.zero_grad()
    # Make each layer's attention require grad
    req_grads = []
    for ap in attn_probs_list:
        ap = ap.detach().requires_grad_(True)
        ap.retain_grad()
        req_grads.append(ap)

    with patch_gpt2_attention_multilayer(model, req_grads, num_layers=len(req_grads)):
        out = model(**inputs)

    logits = out.logits
    total = (logits[:, 1] - logits[:, 0]).sum()
    total.backward()

    # GradCAM-style weighting: weight each layer by the L1 norm of its gradient
    grads = []
    weights = []
    for ap in req_grads:
        g = ap.grad.detach()
        grads.append(g)
        weights.append(g.abs().mean().item())
        ap.requires_grad = False

    model.zero_grad()
    torch.cuda.empty_cache()

    # Weighted average across layers
    total_w = sum(weights) + 1e-8
    agg = sum(w / total_w * g for w, g in zip(weights, grads))
    return agg


def get_rpi_deep_attention_distribution_batch(
    model,
    inputs,
    responses: list,
    queries: list,
    attention_probs_per_layer: list,   # list of [B, H, S, S] tensors, one per layer
    num_time_steps: int = 5,
    num_interpolation: int = 10,
    num_samples: int = 1,
    num_layers: int = None,            # None = use all available layers
) -> list:
    """
    Deep RPIBC: compute IG-based attribution across ALL (or last N) GPT-2 blocks
    then aggregate with GradCAM weighting. Uses column-sum over the full attention
    matrix for each token's credit (vs last-row only in shallow RPIBC).
    """
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if num_layers is not None:
        attn_layers = attention_probs_per_layer[-num_layers:]
    else:
        attn_layers = attention_probs_per_layer

    alpha_cumulative = _compute_alphas_from_time_steps(model.device, num_time_steps)

    all_attribution_scores = []

    for _ in range(num_samples):
        # Per-layer diffusion baselines
        baselines = []
        deltas = []
        for ap in attn_layers:
            noise = torch.randn_like(ap)
            baseline = (
                alpha_cumulative[num_time_steps].sqrt() * ap
                + (1 - alpha_cumulative[num_time_steps]) * noise
            )
            baselines.append(baseline)
            deltas.append(ap - baseline)

        scales = torch.linspace(0, 1, num_interpolation + 1, device=model.device)

        gradients_list = []
        for scale in scales:
            interp = [
                (b + scale * d).to(model.device)
                for b, d in zip(baselines, deltas)
            ]
            grad = _compute_attention_gradients_multilayer(model, inputs, interp)
            gradients_list.append(grad)

        gradients = torch.stack(gradients_list)          # [K+1, B, H, S, S]
        mean_gradient = torch.mean(gradients, dim=0)     # [B, H, S, S]

        # Use the first (reference) layer's attention for element-wise product
        ref_attn = attn_layers[-1]  # last layer — closest to the output
        attribution = mean_gradient * ref_attn
        all_attribution_scores.append(attribution.detach())

    avg_attribution = torch.stack(all_attribution_scores).mean(dim=0)  # [B, H, S, S]
    # Average over heads → [B, S, S]
    avg_attribution = avg_attribution.mean(dim=1).cpu().numpy()

    distributions = []
    for b in range(len(responses)):
        response = responses[b]
        query = queries[b]
        att = avg_attribution[b]  # [S, S]

        ql, rl = len(query), len(response)
        attention_matrix = att[ql: ql + rl, ql: ql + rl]   # [R, R]

        # Column-sum: how much each token was attended to by ALL subsequent tokens
        # vs last-row only (which only uses the final token's attention)
        attention_map = np.abs(attention_matrix).sum(axis=0)  # [R]

        out = torch.zeros(rl, dtype=torch.float64)
        if len(attention_map) == rl:
            out += torch.tensor(attention_map)
        elif len(attention_map) < rl:
            out[rl - len(attention_map):] += torch.tensor(attention_map)
        else:
            out += torch.tensor(attention_map[len(attention_map) - rl:])

        out_sum = out.sum()
        if out_sum == 0:
            distributions.append((torch.ones(rl) / rl).detach())
        else:
            distributions.append((out / out_sum).detach())

    return distributions


# ─── GPT-J Multi-Layer RPIBC ─────────────────────────────────────────────────
# GPT-J uses the same _attn hook signature as GPT-2, so the patch is
# structurally identical. We expose separate names for clarity.

@contextlib.contextmanager
def patch_gptj_attention_multilayer(model, rpi_attn_probs_list, num_layers=None):
    """
    Patches the last `num_layers` GPT-J blocks (default: all blocks).
    GPT-J exposes model.transformer.h[i].attn._attn identical to GPT-2.
    rpi_attn_probs_list: list of [B, H, S, S] tensors, one per patched layer.
    """
    blocks = model.transformer.h
    if num_layers is None:
        num_layers = len(blocks)
    target_blocks = blocks[-num_layers:]

    original_attns = []
    for block, attn_prob in zip(target_blocks, rpi_attn_probs_list):
        original_attns.append(block.attn._attn)

        def make_patched(prob):
            def patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
                aw = prob.type(value.dtype)
                aw = self.attn_dropout(aw)
                if head_mask is not None:
                    aw = aw * head_mask
                return torch.matmul(aw, value), aw
            return patched_attn

        block.attn._attn = make_patched(attn_prob).__get__(block.attn, type(block.attn))

    try:
        yield
    finally:
        for block, orig in zip(target_blocks, original_attns):
            block.attn._attn = orig


def _compute_attention_gradients_multilayer_gptj(model, inputs, attn_probs_list):
    """
    Same as the GPT-2 version but uses patch_gptj_attention_multilayer.
    The TL;DR reward model (Holarissun/trl_rm_tldr_gptj) outputs logits [B, 1],
    so we sum directly rather than taking the difference between two classes.
    """
    model.zero_grad()
    req_grads = []
    for ap in attn_probs_list:
        ap = ap.detach().requires_grad_(True)
        ap.retain_grad()
        req_grads.append(ap)

    with patch_gptj_attention_multilayer(model, req_grads, num_layers=len(req_grads)):
        out = model(**inputs)

    logits = out.logits
    if logits.shape[-1] == 1:
        total = logits.squeeze(-1).sum()
    else:
        total = (logits[:, 1] - logits[:, 0]).sum()
    total.backward()

    grads, weights = [], []
    for ap in req_grads:
        g = ap.grad.detach()
        grads.append(g)
        weights.append(g.abs().mean().item())
        ap.requires_grad = False

    model.zero_grad()
    torch.cuda.empty_cache()

    total_w = sum(weights) + 1e-8
    agg = sum(w / total_w * g for w, g in zip(weights, grads))
    return agg


def get_rpi_deep_attention_distribution_batch_gptj(
    model,
    inputs,
    responses: list,
    queries: list,
    attention_probs_per_layer: list,   # list of [B, H, S, S] tensors, one per layer
    num_time_steps: int = 5,
    num_interpolation: int = 10,
    num_samples: int = 1,
    num_layers: int = None,
) -> list:
    """
    Deep RPIBC for GPT-J reward models (e.g. trl_rm_tldr_gptj).
    Identical logic to the GPT-2 version but uses GPT-J patching.
    """
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    attn_layers = attention_probs_per_layer if num_layers is None \
        else attention_probs_per_layer[-num_layers:]

    alpha_cumulative = _compute_alphas_from_time_steps(model.device, num_time_steps)
    all_attribution_scores = []

    for _ in range(num_samples):
        baselines, deltas = [], []
        for ap in attn_layers:
            noise = torch.randn_like(ap)
            baseline = (
                alpha_cumulative[num_time_steps].sqrt() * ap
                + (1 - alpha_cumulative[num_time_steps]) * noise
            )
            baselines.append(baseline)
            deltas.append(ap - baseline)

        scales = torch.linspace(0, 1, num_interpolation + 1, device=model.device)
        gradients_list = []
        for scale in scales:
            interp = [(b + scale * d).to(model.device) for b, d in zip(baselines, deltas)]
            grad = _compute_attention_gradients_multilayer_gptj(model, inputs, interp)
            gradients_list.append(grad)

        gradients = torch.stack(gradients_list)
        mean_gradient = torch.mean(gradients, dim=0)
        ref_attn = attn_layers[-1]
        attribution = mean_gradient * ref_attn
        all_attribution_scores.append(attribution.detach())

    avg_attribution = torch.stack(all_attribution_scores).mean(dim=0)
    avg_attribution = avg_attribution.mean(dim=1).cpu().numpy()   # [B, S, S]

    distributions = []
    for b in range(len(responses)):
        response = responses[b]
        query = queries[b]
        att = avg_attribution[b]
        ql, rl = len(query), len(response)
        attention_matrix = att[ql: ql + rl, ql: ql + rl]
        attention_map = np.abs(attention_matrix).sum(axis=0)

        out = torch.zeros(rl, dtype=torch.float64)
        if len(attention_map) == rl:
            out += torch.tensor(attention_map)
        elif len(attention_map) < rl:
            out[rl - len(attention_map):] += torch.tensor(attention_map)
        else:
            out += torch.tensor(attention_map[len(attention_map) - rl:])

        out_sum = out.sum()
        if out_sum == 0:
            distributions.append((torch.ones(rl) / rl).detach())
        else:
            distributions.append((out / out_sum).detach())

    return distributions

# ─── LLaMA Multi-Layer RPIBC ─────────────────────────────────────────────────
# LLaMA doesn't expose an _attn hook, so we must monkey-patch the forward pass
# of LlamaAttention directly. 

import types

@contextlib.contextmanager
def patch_llama_attention_multilayer(model, rpi_attn_probs_list, num_layers=None):
    """
    Patches the last `num_layers` Llama blocks (default: all blocks).
    We dynamically override `block.self_attn.forward` to use the provided attribution probabilities.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    import torch.nn.functional as F
    import math

    blocks = model.model.layers
    if num_layers is None:
        num_layers = len(blocks)
    target_blocks = blocks[-num_layers:]

    original_forwards = []

    for block, attn_prob in zip(target_blocks, rpi_attn_probs_list):
        original_forwards.append(block.self_attn.forward)

        def make_patched_forward(prob):
            def patched_forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                if self.config.pretraining_tp > 1:
                    key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                    query_slices = self.q_proj.weight.split(
                        (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                    )
                    key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                    value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                    query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                    query_states = torch.cat(query_states, dim=-1)

                    key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                    key_states = torch.cat(key_states, dim=-1)

                    value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                    value_states = torch.cat(value_states, dim=-1)

                else:
                    query_states = self.q_proj(hidden_states)
                    key_states = self.k_proj(hidden_states)
                    value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                past_key_value = getattr(self, "past_key_value", past_key_value)
                cos, sin = self.rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; position_ids needed for the static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                # --- RPIBC PATCH HERE: INJECT ATTN PROBS ---
                attn_weights = prob.type(value_states.dtype)
                
                # if mask logic is needed we would do it here, but rpi provides actual probabilities directly
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)
                # -------------------------------------------

                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous()

                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None

                return attn_output, attn_weights, past_key_value

            return patched_forward

        block.self_attn.forward = types.MethodType(make_patched_forward(attn_prob), block.self_attn)

    try:
        yield
    finally:
        for block, orig in zip(target_blocks, original_forwards):
            block.self_attn.forward = orig

def _compute_attention_gradients_multilayer_llama(model, inputs, attn_probs_list):
    model.zero_grad()
    req_grads = []
    for ap in attn_probs_list:
        ap = ap.detach().requires_grad_(True)
        ap.retain_grad()
        req_grads.append(ap)

    with patch_llama_attention_multilayer(model, req_grads, num_layers=len(req_grads)):
        out = model(**inputs)

    logits = out.logits
    if logits.shape[-1] == 1:
        total = logits.squeeze(-1).sum()
    else:
        total = (logits[:, 1] - logits[:, 0]).sum()
    total.backward()

    grads = []
    weights = []
    for ap in req_grads:
        g = ap.grad.detach()
        grads.append(g)
        weights.append(g.abs().mean().item())
        ap.requires_grad = False

    model.zero_grad()
    torch.cuda.empty_cache()

    total_w = sum(weights) + 1e-8
    agg = sum(w / total_w * g for w, g in zip(weights, grads))
    return agg

def get_rpi_deep_attention_distribution_batch_llama(
    model,
    inputs,
    responses: list,
    queries: list,
    num_time_steps: int = 20,
    num_interpolation: int = 10,
    num_samples: int = 1,
) -> list:
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    alpha_cumulative = _compute_alphas_from_time_steps(model.device, num_time_steps)

    with torch.no_grad():
        baseline_out = model(**inputs)
    
    attn_layers = baseline_out.attentions
    num_layers = len(attn_layers)
    all_attribution_scores = []
    
    for _ in range(num_samples):
        noise_tensors = [torch.randn_like(ap) for ap in attn_layers]
        
        baselines = []
        for i in range(num_layers):
            ap = attn_layers[i]
            n = noise_tensors[i]
            base = (
                alpha_cumulative[num_time_steps].sqrt() * ap
                + (1 - alpha_cumulative[num_time_steps]) * n
            )
            baselines.append(base)

        deltas = [ap - b for ap, b in zip(attn_layers, baselines)]
        scales = torch.linspace(0, 1, num_interpolation + 1, device=model.device)
        
        gradients_list = []
        for scale in scales:
            interp = [(b + scale * d).to(model.device) for b, d in zip(baselines, deltas)]
            grad = _compute_attention_gradients_multilayer_llama(model, inputs, interp)
            gradients_list.append(grad)

        gradients = torch.stack(gradients_list)
        mean_gradient = torch.mean(gradients, dim=0)
        ref_attn = attn_layers[-1]
        attribution = mean_gradient * ref_attn
        all_attribution_scores.append(attribution.detach())

    avg_attribution = torch.stack(all_attribution_scores).mean(dim=0)
    avg_attribution = avg_attribution.mean(dim=1).cpu().numpy()

    distributions = []
    for b in range(len(responses)):
        response = responses[b]
        query = queries[b]
        att = avg_attribution[b]
        ql, rl = len(query), len(response)
        attention_matrix = att[ql: ql + rl, ql: ql + rl]
        attention_map = np.abs(attention_matrix).sum(axis=0)

        out = torch.zeros(rl, dtype=torch.float64)
        if len(attention_map) == rl:
            out += torch.tensor(attention_map)
        elif len(attention_map) < rl:
            out[rl - len(attention_map):] += torch.tensor(attention_map)
        else:
            out += torch.tensor(attention_map[len(attention_map) - rl:])

        out_sum = out.sum()
        if out_sum == 0:
            distributions.append((torch.ones(rl) / rl).detach())
        else:
            distributions.append((out / out_sum).detach())

    return distributions

