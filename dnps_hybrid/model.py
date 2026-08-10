import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from dnps_hybrid import const


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class PaddedCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = PaddedCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class OODSatisficingLoss(nn.Module):
    def __init__(self, max_conf_threshold=0.1):
        super().__init__()
        self.threshold = max_conf_threshold

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1) # Shape: [Batch, Seq, Vocab]
        max_probs, _ = torch.max(probs, dim=-1) # Shape: [Batch, Seq]
        loss = F.relu(max_probs - self.threshold)
        return loss.mean()

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size + 1, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        self.ood_criterion = OODSatisficingLoss()
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        padding_mask = (idx == const.VOCAB.index('-'))
        x[padding_mask] = 0
        non_padding_mask = ~padding_mask
        mask_k = non_padding_mask.view(b, 1, 1, t)
        mask_q = non_padding_mask.view(b, 1, t, 1)

        causal_mask = torch.tril(torch.ones((t, t), dtype=torch.bool, device=device)).view(1, 1, t, t)
        attn_mask = causal_mask & mask_k & mask_q
        attn_mask = attn_mask.expand(b, self.config.n_head, t, t)
        for block in self.transformer.h:
            x = block(x, attn_mask)
        x = self.transformer.ln_f(x)

        if targets is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x) # note: using list [-1] to preserve the time dim
            mem_loss = None
            rand_loss = None
        else:
            logits = self.lm_head(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            ood_mask = (targets_flat == -1)
            ood_logits = logits_flat[ood_mask]

            if ood_logits.shape[0] > 0:
                rand_loss = self.ood_criterion(ood_logits)
            else:
                rand_loss = torch.tensor(0.0, device=logits.device)

            padding_idx = const.VOCAB.index('-')
            mem_mask = (~ood_mask) & (targets_flat != padding_idx)

            mem_logits = logits_flat[mem_mask]
            mem_targets = targets_flat[mem_mask]

            if mem_logits.shape[0] > 0:
                mem_loss = F.cross_entropy(mem_logits, mem_targets.long())
            else:
                mem_loss = torch.tensor(0.0, device=logits.device)

        return logits, mem_loss, rand_loss

    # Cache of per-tokenizer-identity translation tensors so we only build them
    # once per process. Keyed by id(tokenizer) because depthcharge's
    # PeptideTokenizer is unhashable but stable through the run.
    _casanovo_translation_cache: dict = {}

    def _build_casanovo_translation(self, tokenizer):
        """Build a token-id -> pepLM-VOCAB-id table from a live tokenizer.

        Handles the LUAD fine-tune (Casanovo v5 + TMT6plex residues), where
        the bundled ``const.CASANOVO_TRANSLATION`` (frozen against the v5
        default residue dict) is too short and would OOB-index on
        ``K[TMT6plex]`` / ``[TMT6plex]-`` etc.
        """
        n = max(tokenizer.index.values()) + 1
        table = torch.zeros(n, dtype=torch.int64, device=const.DEVICE)
        for aa, idx in tokenizer.index.items():
            head = aa[0]
            if head == "[":
                head = "."
            elif head == "I":
                head = "L"
            try:
                table[idx] = const.VOCAB.index(head)
            except ValueError as e:
                raise RuntimeError(
                    f"Casanovo token {aa!r} (id={idx}) has no pepLM VOCAB "
                    f"mapping (head={head!r}); extend dnps_hybrid.const.VOCAB "
                    "or special-case it here."
                ) from e
        return table

    def input_from_casanovo_vocab(self, tokens_casanovo, tokenizer=None):
        if tokenizer is not None:
            cache = self._casanovo_translation_cache
            key = id(tokenizer)
            table = cache.get(key)
            if table is None:
                table = self._build_casanovo_translation(tokenizer)
                cache[key] = table
        else:
            table = const.CASANOVO_TRANSLATION
        out = torch.zeros(tokens_casanovo.size(0), const.PLM_BLOCK_SIZE, dtype=torch.int64, device=const.DEVICE)
        # tokens_casanovo lives on the model device (which may differ from
        # const.DEVICE if e.g. CUDA_VISIBLE_DEVICES restricts the device).
        # Move the translation table once per (tokenizer, device) so the
        # gather kernel sees matching devices.
        if table.device != tokens_casanovo.device:
            table = table.to(tokens_casanovo.device)
            if tokenizer is not None:
                self._casanovo_translation_cache[id(tokenizer)] = table
        out[:, 1:tokens_casanovo.size(1) + 1] = table[tokens_casanovo][:, :out.size(1) - 1]
        out[:, 0] = const.START_TOKEN
        return out

    def input_from_contranovo_vocab(self, tokens_contranovo, translation):
        out = torch.zeros(tokens_contranovo.size(0), const.PLM_BLOCK_SIZE, dtype=torch.int64, device=const.DEVICE)
        out[:, 1:tokens_contranovo.size(1) + 1] = translation[tokens_contranovo][:, :out.size(1) - 1]
        out[:, 0] = const.START_TOKEN
        return out

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # get next token
            idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
            # append next token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == const.VOCAB.index('$'):
                break

        return idx


class FusionModel(nn.Module):
    """Fusion head: (casanovo_logits [| pepLM_logits]) -> casanovo-vocab logits.

    Asymmetric per-stream norm: BatchNorm1d on the casanovo stream (fixed
    upstream model, so running stats stay valid and per-amino-acid calibration
    is preserved) and LayerNorm on the pepLM stream (swappable upstream model,
    so we avoid coupling running stats to a specific pepLM). For the null
    model (no pepLM stream), ``casanovo_vocab_size == input_size`` and only
    the casanovo block is normed.
    """

    def __init__(self, input_size, hidden_size, output_size, casanovo_vocab_size):
        super().__init__()
        self.casanovo_vocab_size = casanovo_vocab_size
        plm_size = input_size - casanovo_vocab_size
        self.bn_cas = nn.BatchNorm1d(casanovo_vocab_size)
        self.ln_plm = nn.LayerNorm(plm_size) if plm_size > 0 else None
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        cas = self.bn_cas(x[:, :self.casanovo_vocab_size])
        if self.ln_plm is not None:
            plm = self.ln_plm(x[:, self.casanovo_vocab_size:])
            x = torch.cat([cas, plm], dim=1)
        else:
            x = cas
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def remove_prefix(state_dict):
    for k,_v in list(state_dict.items()):
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            state_dict[new_key] = state_dict.pop(k)
    return state_dict

def _maybe_compile(model):
    import os
    if os.environ.get("DNPS_DISABLE_TORCH_COMPILE", "0") in ("1", "true", "True"):
        return model
    return torch.compile(model)


def load_plm_model():
    import os
    if not os.path.exists(const.PLM_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"PLM checkpoint not found at {const.PLM_CHECKPOINT_PATH!r}. "
            "Set DNPS_PLM_CKPT_PATH to the .pt file inside the container, "
            "or bind-mount it at the default path."
        )
    checkpoint = torch.load(const.PLM_CHECKPOINT_PATH, map_location=const.DEVICE)
    model_args = checkpoint['model_args']
    state_dict = checkpoint['model']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = remove_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(const.DEVICE)
    model = _maybe_compile(model)
    return model

def load_fusion_model(null_model, vocab_size, path=None, output_size=None):
    """Load a FusionModel checkpoint.

    Parameters
    ----------
    null_model : bool
        Load the null (Casanovo-only) baseline instead of the full fusion.
    vocab_size : int
        Casanovo vocabulary size (input block size for the Casanovo stream).
    path : str, optional
        Override checkpoint path.
    output_size : int, optional
        Output dimensionality of the FusionModel.  Defaults to ``vocab_size``.
    """
    import os
    if path is None:
        fusion_path = const.NULL_MODEL_PATH if null_model else const.FUSION_MODEL_PATH
    else:
        fusion_path = path
    if not os.path.exists(fusion_path):
        raise FileNotFoundError(
            f"Fusion model weights not found at {fusion_path!r}. "
            "Set DNPS_FUSION_MODEL_PATH (or DNPS_NULL_MODEL_PATH) accordingly."
        )
    if output_size is None:
        output_size = vocab_size
    input_size = vocab_size
    if not null_model:
        input_size += len(const.VOCAB)
    fusion_model = FusionModel(
        input_size=input_size, hidden_size=128, output_size=output_size,
        casanovo_vocab_size=vocab_size,
    )
    fusion_state_dict = torch.load(fusion_path, map_location=const.DEVICE)
    fusion_state_dict = remove_prefix(fusion_state_dict)
    fusion_model.load_state_dict(fusion_state_dict)
    fusion_model.eval()
    fusion_model.to(const.DEVICE)
    fusion_model = _maybe_compile(fusion_model)
    return fusion_model