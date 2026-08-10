import os
import time
import math
from contextlib import nullcontext
import wandb
import torch
import pickle
from dnps_hybrid.model import GPTConfig, GPT
from dnps_hybrid import const
import torch._functorch.config

tokens_per_iter = const.PLM_BATCH_SIZE * const.PLM_BLOCK_SIZE
print(f"tokens per iteration will be: {tokens_per_iter:,}")

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.current_device())

os.makedirs(const.RUN_PATH, exist_ok=True)
os.makedirs(os.path.dirname(const.PLM_CHECKPOINT_PATH), exist_ok=True)
torch.manual_seed(const.SEED)
torch._functorch.config.donated_buffer = False
torch.cuda.manual_seed(const.SEED)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# note: float16 data type will automatically use a GradScaler
ctx = nullcontext() if const.DEVICE == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=const.DTYPE)

print("Loading X")
ALL_X = torch.load(const.PLM_SEQ_X_PATH, map_location=const.DEVICE)
print("Loading Y")
ALL_Y = torch.load(const.PLM_SEQ_Y_PATH, map_location=const.DEVICE)
print("Done loading data")

def add_rand_suffix(x, y, pep_lens):
    B, T = x.shape
    device = x.device
    if const.PLM_RAND_SUFFIX_FULL_LEN:
        # Per-sample uniform kept-prefix length in [0, pep_lens). rand_N=0 →
        # the entire peptide is OOD; rand_N=pep_lens-1 → only the last token
        # is OOD. The post-peptide +2 tail is still always replaced.
        start_indices = (torch.rand(B, device=device) * pep_lens.float()).long()
    else:
        start_indices = torch.clamp(pep_lens - 5, min=6)
    end_indices = torch.clamp(pep_lens + 2, max=const.PLM_BLOCK_SIZE - 1)
    cols = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    mask_x = (cols >= start_indices.unsqueeze(1)) & (cols < end_indices.unsqueeze(1))
    mask_y = (cols >= (start_indices.unsqueeze(1) + 1)) & (cols < end_indices.unsqueeze(1))
    rand_tokens = torch.randint(3, len(const.VOCAB) - 1, (B, T), device=device)
    x[mask_x] = rand_tokens[mask_x]
    y[mask_y] = -1

def add_mutations(x, pep_lens, start_idx, n_mutated, mask_ratio):
    end_idx = start_idx + n_mutated
    mut_pos = (torch.rand(n_mutated, device=x.device) * (pep_lens - 1)).long()
    rows = torch.arange(n_mutated, device=x.device)
    rand_tokens = torch.randint(3, len(const.VOCAB)-1, (n_mutated,), device=x.device)
    x[start_idx:end_idx][rows, mut_pos] = rand_tokens
    is_mask = torch.rand(n_mutated, device=x.device) < mask_ratio
    x[start_idx:end_idx][rows[is_mask], mut_pos[is_mask]] = MASK_TOKEN_INDEX

def get_batch(iter_num):
    third = const.PLM_BATCH_SIZE // 3
    ix = torch.randint(len(ALL_X), (third,), device=const.DEVICE)
    x_chunk = ALL_X[ix]
    y_chunk = ALL_Y[ix]
    pep_lens = (x_chunk == 0).float().argmax(dim=1)
    pep_lens = torch.where(pep_lens == 0, 
                           torch.tensor(const.PLM_BLOCK_SIZE - 1, device=x_chunk.device), 
                           pep_lens)
    x = x_chunk.repeat(3, 1)
    y = y_chunk.repeat(3, 1)
    add_rand_suffix(x[third:2*third], y[third:2*third], pep_lens)

    progress = iter_num / const.PLM_MAX_ITERS
    mask_ratio = max(0.0, 0.8 * (1.0 - progress))
    add_mutations(x, pep_lens, 2*third, third, mask_ratio)
    return x, y

# PATCH: The vocabulary size increased by 1 (due to <MASK>), so START_TOKEN index shifted.
# Old data has START_TOKEN at the index that is now occupied by the MASK token.
MASK_TOKEN_INDEX = const.VOCAB.index('#')
if (ALL_X == MASK_TOKEN_INDEX).any():
    print(f"Patching old data: Shifting START_TOKEN from {MASK_TOKEN_INDEX} to {const.START_TOKEN}")
    ALL_X[ALL_X == MASK_TOKEN_INDEX] = const.START_TOKEN

iter_num = 0

if const.PLM_INIT_FROM_CHECKPOINT:
    checkpoint = torch.load(const.PLM_CHECKPOINT_PATH, map_location=const.DEVICE)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(const.DEVICE)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
else:
    # model init
    model_args = dict(n_layer=const.PLM_N_LAYER, n_head=const.PLM_N_HEAD, n_embd=const.PLM_N_EMBD, block_size=const.PLM_BLOCK_SIZE,
                    bias=False, dropout=0, vocab_size=len(const.VOCAB))
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(const.DEVICE)

    # WARM START: Initialize <MASK> embedding to average of vocab
    # This prevents a massive loss spike when the model first sees the new token
    print("Initializing <MASK> token embedding to average of vocab...")
    with torch.no_grad():
        # Average all tokens except START_TOKEN and the new MASK
        avg_emb = model.transformer.wte.weight[:MASK_TOKEN_INDEX].mean(dim=0)
        model.transformer.wte.weight[MASK_TOKEN_INDEX] = avg_emb

scaler = torch.amp.GradScaler(enabled=True)

# optimizer
optimizer = model.configure_optimizers(const.PLM_WEIGHT_DECAY, const.PLM_LEARNING_RATE, (const.PLM_BETA1, const.PLM_BETA2), const.DEVICE)

# compile the model
print("compiling the model... (takes a ~minute)")
model = torch.compile(model)
print("done compiling")


@torch.no_grad()
def estimate_loss(iter_num):
    model.eval()
    mem_losses, rand_losses = torch.zeros(const.PLM_EVAL_ITERS), torch.zeros(const.PLM_EVAL_ITERS)
    for k in range(const.PLM_EVAL_ITERS):
        X, Y = get_batch(iter_num)
        with ctx:
            logits, mem_loss, rand_loss = model(X, Y)
        mem_losses[k] = mem_loss.item()
        rand_losses[k] = rand_loss.item()
    model.train()
    return mem_losses.mean(), rand_losses.mean()

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < const.PLM_WARMUP_ITERS:
        return const.PLM_LEARNING_RATE * (it + 1) / (const.PLM_WARMUP_ITERS + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > const.PLM_MAX_ITERS:
        return const.PLM_MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - const.PLM_WARMUP_ITERS) / (const.PLM_MAX_ITERS - const.PLM_WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return const.PLM_MIN_LR + coeff * (const.PLM_LEARNING_RATE - const.PLM_MIN_LR)

# logging
# Gather all constants from const that start with 'PLM_'
plm_config = {k: getattr(const, k) for k in dir(const) if k.startswith("PLM_")}
wandb.init(
    project=const.WANDB_PROJECT,
    name=f"{const.RUN_NAME}_plm",
    dir=const.RUN_PATH,
    config=plm_config
)

# training loop
X, Y = get_batch(iter_num) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % const.PLM_EVAL_INTERVAL == 0:
        mem_loss, rand_loss = estimate_loss(iter_num)
        print(f"step {iter_num}: mem_loss {mem_loss:.4f}, rand_loss {rand_loss:.4f}")
        wandb.log({
            "iter": iter_num,
            "mem_loss": mem_loss,
            "rand_loss": rand_loss,
            "lr": lr,
            "mfu": running_mfu*100, # convert to percentage
        })
        if iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
            }
            print(f"saving checkpoint to {const.PLM_CHECKPOINT_PATH}")
            torch.save(checkpoint, const.PLM_CHECKPOINT_PATH)
    if iter_num == 0 and const.PLM_EVAL_ONLY:
        break

    with ctx:
        logits, mem_loss, rand_loss = model(X, Y)
        total_loss = mem_loss + (rand_loss * const.RAND_LOSS_WEIGHT)

        if iter_num % const.PLM_EVAL_INTERVAL == 0:
            try:
                grads_mem = torch.autograd.grad(mem_loss, model.parameters(), retain_graph=True, allow_unused=True)
                grads_rand = torch.autograd.grad(rand_loss, model.parameters(), retain_graph=True, allow_unused=True)
                grads_mem_norm = torch.sqrt(sum(grad.norm()**2 for grad in grads_mem if grad is not None))
                grads_rand_norm = torch.sqrt(sum(grad.norm()**2 for grad in grads_rand if grad is not None))
                wandb.log({
                    "iter": iter_num,
                    "grads_mem_norm": grads_mem_norm,
                    "grads_rand_norm": grads_rand_norm * const.RAND_LOSS_WEIGHT,
                })
            except:
                print("Error calculating gradients")
                pass
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch(iter_num)
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(total_loss).backward()
    # clip the gradient
    if const.PLM_GRAD_CLIP != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), const.PLM_GRAD_CLIP)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % const.PLM_LOG_INTERVAL == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(const.PLM_BATCH_SIZE, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: mem_loss {mem_loss.item():.4f}, rand_loss {rand_loss.item():.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > const.PLM_MAX_ITERS:
        break