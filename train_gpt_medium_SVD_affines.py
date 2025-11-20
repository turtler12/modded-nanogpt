import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint as cp # medhaven: import checkpoint module for patching
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
torch._dynamo.config.compiled_autograd = False # medhaven: disable compiled autograd due to bugs


# medhaven: added this to disable activation checkpointing globally
def _no_checkpoint(function, *args, **kwargs):
    # Drop any checkpoint-specific kwargs (like use_reentrant, debug)
    kwargs.pop("use_reentrant", None)
    kwargs.pop("debug", None)
    return function(*args, **kwargs)

# Patch the module-level function so anything that calls
# torch.utils.checkpoint.checkpoint(...) will just run the function
cp.checkpoint = _no_checkpoint

# Local alias for this file if needed
checkpoint = _no_checkpoint

# We still import checkpoint_sequential just in case it’s used somewhere,
# but we are NOT using the original checkpoint at all anymore.
from torch.utils.checkpoint import checkpoint_sequential

# medhaven:  patch checkpoint_sequential to disable checkpointing
# ---- absolutely disable compile/inductor at runtime, belt-and-suspenders ----
os.environ["TORCH_COMPILE_DISABLE"] = "1"
try:
    torch._dynamo.reset()
except Exception:
    pass

# Stable attention backend: avoid flex/flash in DDP+BF16
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Triton path
    torch.backends.cuda.enable_math_sdp(True)           # fallback ok
except Exception:
    pass

# fixing all types
torch.set_float32_matmul_precision("high")  # or "medium" if you want
# If you’re training in bf16, keep autocast; otherwise comment:
amp_dtype = torch.bfloat16
# You can also switch to float16 if you aren’t using bf16 everywhere:
# amp_dtype = torch.float16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# SVD mapping utilities (for affine intercept sweep)
def _svd_map_batched(X: torch.Tensor, map_fn):
    """
    Apply a singular-value mapping to a batch of matrices.
    X: (..., m, n)
    map_fn: function f(singular_vals) -> mapped_singular_vals, elementwise on last dim
    """
    orig_dtype = X.dtype
    X32 = X.to(torch.float32)
    # econ SVD: U (..., m, k), S (..., k), Vh (..., k, n)
    U, S, Vh = torch.linalg.svd(X32, full_matrices=False)
    S_mapped = map_fn(S)
    US = U * S_mapped.unsqueeze(-2)
    Y = US @ Vh
    return Y.to(orig_dtype)

def make_svd_map_affine(intercept: float, normalize: bool = True):
    """
    Affine family that interpolates between:
      - intercept = 0:   normalized identity      (s_norm)
      - intercept = 1:   normalized step mapping ((s_norm > 0) -> 1, zeros stay 0)

    With normalize=True this behaves like the SVD+step map used elsewhere:
      • we first normalize by s_max so singulars are in [0, 1]
      • then apply a convex combination between s_norm and the step(0) map
      • we DO NOT rescale back by s_max, so op-norm <= 1 and intercept=1 matches the
        SVD+step behavior.
    """
    i = float(intercept)

    def _map(s: torch.Tensor) -> torch.Tensor:
        s = s.to(torch.float32)
        if normalize:
            s_max = s.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            s_norm = s / s_max
        else:
            s_norm = s
        step = (s_norm > 0).to(s.dtype)
        s_new = (1.0 - i) * s_norm + i * step
        return s_new

    return _map


def make_svd_map_affine_no_norm(intercept: float):
    """
    Affine mapping without spectral normalization. Interpolates singular values
    between the original s and s_max (per-matrix largest singular value).

    For intercept=0: returns s (identity).
    For intercept=1: returns s_max * ones_like(s) (i.e., U (s_max * I) V^T).
    """
    i = float(intercept)

    def _map(s: torch.Tensor) -> torch.Tensor:
        s = s.to(torch.float32)
        s_max = s.amax(dim=-1, keepdim=True).clamp_min(0.0)
        step = (s > 0).to(s.dtype)
        # convex combination between original s and s_max (for positive singulars)
        s_new = (1.0 - i) * s + i * (s_max * step)
        return s_new

    return _map

# medhaven: edited to avoid uint32 shifts on CUDA (bunch of errors otherwise)
@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)
    # medhaven: modified below to avoid uint32 shifts on CUDA
    # acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    # Do bit-twiddling in int32 because uint32 << is not implemented on CUDA
    acc_m_i32 = (acc_bf16_view_u16.to(torch.int32) << 16) | mantissa.to(torch.int32)
    acc_m_u32 = acc_m_i32.to(torch.uint32)

    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)

    # medhaven: modified below to avoid uint32 shifts on CUDA
    # acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    acc_m_i32 = acc_m_u32.to(torch.int32)
    upper16 = (acc_m_i32 >> 16).to(torch.uint16)  # high 16 bits
    lower16 = acc_m_i32.to(torch.uint16)    

    # medhaven
    # mantissa.copy_(acc_m_u32.to(torch.uint16))
    acc_bf16_view_u16.copy_(upper16)
    mantissa.copy_(lower16)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1, use_svd_mapping: bool = False, svd_map_fn=None):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])
        self.use_svd_mapping = use_svd_mapping
        self.svd_map_fn = svd_map_fn

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank >= len(params):
                    # still need to enqueue a gather for padding alignment
                    futures.append(
                        dist.all_gather(params_pad[base_i:base_i + self.world_size],
                                        params_pad[base_i + self.rank],
                                        async_op=True).get_future()
                    )
                    continue

                p = params[base_i + self.rank]
                state = self.state[p]
                if len(state) == 0:
                    state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)   # used by NS path
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)

                mbuf: torch.Tensor = state["momentum_buffer"]
                g32 = p.grad.float()

                # momentum buffer update (same in both paths)
                mbuf.copy_(momentum * mbuf + (1 - momentum) * g32)

                eff_lr = torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)
                eff_wd = torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0))

                if self.use_svd_mapping and p.ndim >= 2 and self.svd_map_fn is not None:
                    # ------------ MAPPING-ONLY PATH (NO NS) ------------
                    # same pre-update as earlier experiments
                    upd32 = momentum * mbuf + (1 - momentum) * g32

                    # SVD → map (Yes-SN or No-SN) → recompose
                    v = _svd_map_batched(upd32, self.svd_map_fn).to(p.dtype)

                    # weight decay then SGD step
                    p.mul_(1 - eff_wd)
                    p.add_(v, alpha=-eff_lr.item())

                else:
                    # ------------ NS-ONLY PATH ------------
                    update(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=eff_lr,
                        eff_weight_decay=eff_wd,
                    )

                # schedule the all_gather for this stride-chunk
                futures.append(
                    dist.all_gather(params_pad[base_i:base_i + self.world_size],
                                    params_pad[base_i + self.rank],
                                    async_op=True).get_future()
                )

        torch.futures.collect_all(futures).wait()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

@torch.no_grad()
def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkvo_w = nn.Parameter(init_linear(torch.empty(4, hdim, dim)).bfloat16())
        self.qkvo_w.detach()[3].zero_() # out zero init suggested by @Grad62304977
        self.rotary = Rotary(head_dim, max_seq_len)
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask, lambdas: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w[3])
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.proj_w)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask, lambdas: Tensor, sa_lambdas: Tensor):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, block_mask, sa_lambdas)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head_w = nn.Parameter(torch.zeros(next_multiple_of_n(vocab_size, n=128), model_dim))
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
        ]))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)
    
    # medhaven: FIX THIS SECTION IF THERE IS AN ERROR
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        skip_connections = []
        skip_map = {
            9: 6,
            10: 4,
            11: 2,
        }
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        for i in range(len(self.blocks)):
            if i in skip_map:
                x = x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]
            x = self.blocks[i](x, ve[i], x0, block_masks[i], lambdas[i], sa_lambdas[i])
            skip_connections.append(x)

        x = norm(x)
        if self.training:
            logits: Tensor = F.linear(x.flatten(end_dim=1), self.lm_head_w.bfloat16()).float()
            loss = F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq)
            return loss

        loss = 0
        for i in range(4):
            logits: Tensor = F.linear(x.flatten(end_dim=1).chunk(4)[i], self.lm_head_w.bfloat16()).float()
            loss += F.cross_entropy(15 * logits * torch.rsqrt(logits.square() + 225), target_seq.chunk(4)[i]) / 4
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 4096 # before: 64*1024 # FlexAttention sequence length
    val_seq_len = 4096 # before: 4*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 5960 # number of iterations to run
    cooldown_frac = 0.7 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
args = Hyperparameters()

run_id = int(os.environ.get("RUN_ID", 0))
# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# medhaven: removed 
# assert world_size == 8 # this code is designed for 8xH100
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
if master_process:
    run_id_full = f"{run_id:03d}_{uuid.uuid4()}"
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id_full}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)
from torch._logging._internal import trace_structured # noqa: E402
import torch._inductor.codecache # noqa: E402
import torch._inductor.graph # noqa: E402
def _patched_trace_structured(name, metadata_fn, **kwargs):
    if name == "inductor_output_code":
        print0(f"inductor_output_code: {metadata_fn().get('filename', 'Unknown')}")
    trace_structured(name, metadata_fn, **kwargs)
torch._inductor.codecache.trace_structured = _patched_trace_structured
torch._inductor.graph.trace_structured = _patched_trace_structured

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)

# --- Mapping controls (replicate Yes-SN / No-SN from previous model) ---
MAP_KIND       = os.environ.get("MAP_KIND", "identity").lower()   # "affine" to activate mapping
INTERCEPT      = float(os.environ.get("INTERCEPT", "1.0"))        # 0..1
MAP_NORMALIZE  = int(os.environ.get("MAP_NORMALIZE", "1"))        # 1 => Yes-SN, 0 => No-SN
APPLY_INIT_MAP = int(os.environ.get("APPLY_INIT_MAP", "0"))       # default 0 to avoid confound

svd_map_fn_selected = None
if MAP_KIND == "affine":
    # Yes-SN = normalize=True, No-SN = normalize=False
    svd_map_fn_selected = make_svd_map_affine(INTERCEPT, normalize=bool(MAP_NORMALIZE))

# --- Optional: one-shot SVD map on initial hidden weights (disabled by default) ---
if svd_map_fn_selected is not None and APPLY_INIT_MAP:
    print0(f"[INIT MAP] intercept={INTERCEPT}, normalize={MAP_NORMALIZE}")
    with torch.no_grad():
        for p in hidden_matrix_params:
            if p.ndim >= 2:
                p.copy_(_svd_map_batched(p, svd_map_fn_selected))




embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
scalar_params = [model.scalars]
head_params: list[nn.Parameter] = [model.lm_head_w]
# sanity check
params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
optimized_parameters_set = {p for params in params_collections for p in params}
assert optimized_parameters_set == {*model.parameters()}
assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)



# init the optimizer(s)
adam_param_groups = [dict(params=head_params, lr=1/320), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)

# Optional SVD affine mapping selection via environment. Set MAP_KIND="affine" and
# INTERCEPT in [0.0..1.0] to apply the mapping to all hidden 2D parameters before training.
# MAP_KIND = os.environ.get("MAP_KIND", "identity").lower()
# INTERCEPT = float(os.environ.get("INTERCEPT", "1.0"))
# if MAP_KIND == "affine":
#     svd_map_fn = make_svd_map_affine_no_norm(INTERCEPT)
#     print0(f"Applying SVD affine mapping (no-norm): intercept={INTERCEPT}")
#     for p in hidden_matrix_params:
#         if p.ndim >= 2:
#             # compute mapped weight and copy in-place
#             with torch.no_grad():
#                 mapped = _svd_map_batched(p.data, svd_map_fn)
#                 p.data.copy_(mapped)



# medhaven: modified to pass svd_map_fn_selected to Muon optimizer
# optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
optimizer2 = Muon(
    hidden_matrix_params,
    lr=0.025, momentum=0.95, rank=rank, world_size=world_size,
    use_svd_mapping=(svd_map_fn_selected is not None),
    svd_map_fn=svd_map_fn_selected,
)

optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
def opt_params(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for group in opt.param_groups for p in group["params"]]
opt2params = {opt: opt_params(opt) for opt in optimizers}
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        return (1 - x) / args.cooldown_frac

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x
    window_size = next_multiple_of_n(3456 * factor, n=128)
    return get_window_size_blocks_helper(window_size)

# medhaven: replaced the following line with the maybe_compile function below for debugging stuff
# model: nn.Module = torch.compile(model, dynamic=False)

# Safe, opt-in compile (defaults OFF). Turn on only after things are stable.
def maybe_compile(m: nn.Module):
    if os.environ.get("TORCH_COMPILE_DISABLE", "1") != "1":
        try:
            return torch.compile(m, dynamic=False)
        except Exception:
            pass
    return m

model = maybe_compile(model)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 0 # medhaven: set to 0 to skip warmup (was originally 10)
initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#        Training and validation       #
########################################

torch.cuda.reset_peak_memory_stats()
train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
dist.barrier()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        dist.barrier()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id_full}", exist_ok=True)
            torch.save(log, f"logs/{run_id_full}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    opt2futures = {
        opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
        for opt, params in opt2params.items()
    }
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        torch.futures.collect_all(opt2futures[opt]).wait()
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()