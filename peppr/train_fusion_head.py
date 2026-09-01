"""Train the (cas-backbone + pepLM) fusion model.

Loads cas-backbone teacher scores + pepLM teacher scores and trains a fusion
head plus a null (cas-only) baseline.  Backbone is selected via the
``PEPPR_BACKBONE`` env var: ``casanovo`` (default) saves to
``const.FUSION_MODEL_PATH``; ``contranovo`` saves to
``const.CONTRANOVO_FUSION_MODEL_PATH``
(both require ``PEPPR_CONTRANOVO_FUSION_PATH`` /
``PEPPR_CONTRANOVO_FUSION_PATH`` to be set to an experiment-specific path,
to avoid overwriting the shared Apr-27 checkpoints).
Backbone selection only swaps file paths; the fusion head architecture
(asymmetric BN-cas + LN-plm) and loss are identical across backbones.

The fusion and null heads are trained exclusively on the canonical
MassIVE-KB PSM corpus configured in :mod:`peppr.const`.

Optional augmentations
----------------------
``PEPPR_FUSION_PRIOR_TOP2_SWAP_FRAC`` (float, default 0.5):
    Legacy top-2 swap: swap the values of the top-1 and top-2 pepLM logits
    for a Bernoulli(frac) sample of rows. Only creates training signal for
    the case where the correct token is the runner-up.

"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import wandb
from peppr import const
from peppr.model import FusionModel

# Backbone selection: ``casanovo`` (default) or ``contranovo``. The fusion
# head architecture and loss are identical; only the input-side teacher
# artifacts and output checkpoint paths differ.
_BACKBONE = os.environ.get('PEPPR_BACKBONE', 'casanovo').lower()
if _BACKBONE not in ('casanovo', 'contranovo'):
    raise RuntimeError(
        f"PEPPR_BACKBONE={_BACKBONE!r}; expected 'casanovo' or 'contranovo'."
    )

config = {
    'model_type': 'fusion',
    'hidden_size': 128,
    'vocab_size': len(const.VOCAB),
    'batch_size': 8192,
    'learning_rate': 0.002,
    'num_epochs': int(os.environ.get('PEPPR_FUSION_EPOCHS', 8)),
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'lr_decay_factor': 0.7,
    'seed': const.SEED,
    'prior_top2_swap_frac': float(
        os.environ.get('PEPPR_FUSION_PRIOR_TOP2_SWAP_FRAC', '0.5')
    ),
}


def _apply_plm_top2_swap(
    batch_X_fusion: torch.Tensor,
    casanovo_vocab_size: int,
    frac: float,
) -> int:
    """In-place: swap the top-1 and top-2 pepLM logit values for a
    Bernoulli(``frac``) sample of rows in ``batch_X_fusion``.

    ``batch_X_fusion`` has shape ``[B, V_cas + V_plm]`` (one token per
    row); the pepLM block lives in columns ``[casanovo_vocab_size:]``.
    Casanovo columns and the label tensor are untouched.

    The swap is a value swap (not an index permutation) -- the runner-up
    logit value is written to the original argmax slot and vice versa --
    so the post-augmentation argmax of the pepLM block is what was
    previously the second-best token. Returns the number of rows mutated
    (handy for logging).
    """
    if frac <= 0.0:
        return 0
    plm = batch_X_fusion[:, casanovo_vocab_size:]
    if plm.size(1) < 2:
        return 0
    mask = torch.rand(plm.size(0), device=plm.device) < frac
    n_swap = int(mask.sum().item())
    if n_swap == 0:
        return 0
    sub = plm[mask]
    top_vals, top_idx = sub.topk(2, dim=1)
    rows = torch.arange(sub.size(0), device=sub.device)
    new_sub = sub.clone()
    new_sub[rows, top_idx[:, 0]] = top_vals[:, 1]
    new_sub[rows, top_idx[:, 1]] = top_vals[:, 0]
    plm[mask] = new_sub
    return n_swap

def load_data(Y_path, scores_plm_path, scores_casanovo_path):
    Y = torch.load(Y_path)
    scores_casanovo = torch.load(scores_casanovo_path, map_location=const.DEVICE)
    scores_prior = torch.load(scores_plm_path, map_location=const.DEVICE)

    casanovo_vocab_size = scores_casanovo.size(2)
    prior_vocab_size = scores_prior.size(2)

    Y = Y.reshape(-1)
    # IL-vocab fusion_y uses 0 (Casanovo padding) as the padding sentinel,
    # same as the standard fusion_y.  The standard padding check looks for
    # VOCAB.index('-') == 0, which also equals Casanovo padding index 0.
    padding_mask = Y == 0
    Y = Y[~padding_mask]
    scores_casanovo = scores_casanovo.reshape(-1, casanovo_vocab_size)[~padding_mask]
    scores_prior = scores_prior.reshape(-1, prior_vocab_size)[~padding_mask]
    X = torch.cat([scores_casanovo, scores_prior], dim=1)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return loader, casanovo_vocab_size, prior_vocab_size


def load_model(input_size, output_size, casanovo_vocab_size=None):
    model = FusionModel(
        input_size, config['hidden_size'], output_size,
        casanovo_vocab_size=casanovo_vocab_size if casanovo_vocab_size is not None else input_size,
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr_decay_factor'])
    model.to(const.DEVICE)
    model = torch.compile(model)
    return model, optimizer, scheduler


def get_val_loss(fusion_model, fusion_criterion, val_loader):
    fusion_model.eval()
    total_fusion = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(const.DEVICE)
            batch_Y = batch_Y.to(const.DEVICE).long()
            total_fusion += fusion_criterion(fusion_model(batch_X), batch_Y).item()
            n_batches += 1
    return total_fusion / n_batches


def main():
    torch.manual_seed(const.SEED)
    torch.cuda.manual_seed(const.SEED)
    torch.set_float32_matmul_precision('high')

    if _BACKBONE == 'casanovo':
        y_train_path = const.FUSION_Y_TRAIN_PATH
        y_test_path  = const.FUSION_Y_TEST_PATH
        cas_train_path = const.CASANOVO_TEACHER_SCORES_TRAIN_PATH
        cas_test_path  = const.CASANOVO_TEACHER_SCORES_TEST_PATH
        prior_train_path = const.PRIOR_PSM_TEACHER_SCORES_TRAIN_PATH
        prior_test_path  = const.PRIOR_PSM_TEACHER_SCORES_TEST_PATH
        fusion_out_path = const.FUSION_MODEL_PATH
    else:  # contranovo
        y_train_path  = const.CONTRANOVO_FUSION_Y_TRAIN_PATH
        y_test_path   = const.CONTRANOVO_FUSION_Y_TEST_PATH
        cas_train_path = const.CONTRANOVO_TEACHER_SCORES_TRAIN_PATH
        cas_test_path  = const.CONTRANOVO_TEACHER_SCORES_TEST_PATH
        prior_train_path = const.CONTRANOVO_PRIOR_PSM_TEACHER_SCORES_TRAIN_PATH
        prior_test_path  = const.CONTRANOVO_PRIOR_PSM_TEACHER_SCORES_TEST_PATH
        fusion_out_path = const.CONTRANOVO_FUSION_MODEL_PATH
        # Guard against clobbering the shared ContraNovo checkpoint: require an
        # experiment-specific override for new training runs.
        _shared = os.path.join(
            const.SHARED_MODEL_RUN_PATH, "contranovo_fusion_model.pth"
        )
        if fusion_out_path == _shared:
            raise RuntimeError(
                "Refusing to overwrite the shared ContraNovo checkpoint. Set "
                "PEPPR_CONTRANOVO_FUSION_PATH to an experiment-specific path."
            )
    print(f"[backbone] {_BACKBONE}  fusion_out={fusion_out_path}")
    os.makedirs(os.path.dirname(fusion_out_path), exist_ok=True)

    train_loader, casanovo_vocab_size, prior_vocab_size = load_data(
        y_train_path,
        prior_train_path,
        cas_train_path,
    )
    val_loader, _, _ = load_data(
        y_test_path,
        prior_test_path,
        cas_test_path,
    )
    print(f"casanovo_vocab_size={casanovo_vocab_size}  prior_vocab_size={prior_vocab_size}")

    fusion_output_size = casanovo_vocab_size

    fusion_model, fusion_opt, fusion_sched = load_model(
        casanovo_vocab_size + prior_vocab_size, fusion_output_size,
        casanovo_vocab_size=casanovo_vocab_size,
    )
    print(f"Fusion params: {sum(p.numel() for p in fusion_model.parameters()):,}")

    fusion_criterion = nn.CrossEntropyLoss()

    prior_top2_swap_frac = config['prior_top2_swap_frac']
    if prior_top2_swap_frac > 0.0:
        print(
            f"[aug] pepLM top-2 logit swap on {prior_top2_swap_frac*100:.2f}% "
            f"of training tokens per batch (val unaffected)."
        )
    use_wandb = os.environ.get('PEPPR_WANDB', '0') in ('1', 'true', 'True')
    if use_wandb:
        wandb.init(
            project=const.WANDB_PROJECT,
            name=f"{const.RUN_NAME}_{_BACKBONE}_fusion",
            dir=const.RUN_PATH,
            config=config,
        )

    val_loss_fusion = get_val_loss(fusion_model, fusion_criterion, val_loader)
    print(f"[epoch 0] val/loss_fusion={val_loss_fusion:.4f}")
    if use_wandb:
        wandb.log({'epoch': 0, 'val/loss_fusion': val_loss_fusion})

    for epoch in range(config['num_epochs']):
        fusion_model.train()
        total_fusion = 0.0
        total_swapped = 0
        total_rows = 0
        n_batches = 0
        for batch_X, batch_Y in train_loader:
            batch_X_fusion = batch_X.to(const.DEVICE)
            batch_Y = batch_Y.to(const.DEVICE).long()

            n_swapped = _apply_plm_top2_swap(
                batch_X_fusion, casanovo_vocab_size, prior_top2_swap_frac,
            )
            total_swapped += n_swapped
            total_rows += batch_X_fusion.size(0)

            fusion_opt.zero_grad()
            loss_fusion = fusion_criterion(fusion_model(batch_X_fusion), batch_Y)
            loss_fusion.backward()
            fusion_opt.step()

            total_fusion += loss_fusion.item()
            n_batches += 1

        train_loss_fusion = total_fusion / n_batches
        val_loss_fusion = get_val_loss(fusion_model, fusion_criterion, val_loader)
        fusion_sched.step()

        swap_rate = total_swapped / total_rows if total_rows else 0.0
        any_swap = prior_top2_swap_frac > 0.0
        swap_str = f" prior_swap_rate={swap_rate*100:.2f}%" if any_swap else ""
        print(
            f"[epoch {epoch + 1}] "
            f"train/loss_fusion={train_loss_fusion:.4f} "
            f"val/loss_fusion={val_loss_fusion:.4f} "
            f"lr={fusion_opt.param_groups[0]['lr']:.5f}"
            f"{swap_str}"
        )
        if use_wandb:
            log = {
                'epoch': epoch + 1,
                'train/loss_fusion': train_loss_fusion,
                'val/loss_fusion': val_loss_fusion,
                'learning_rate': fusion_opt.param_groups[0]['lr'],
            }
            if prior_top2_swap_frac > 0.0:
                log['train/prior_top2_swap_rate'] = swap_rate
            wandb.log(log)

    os.makedirs(os.path.dirname(fusion_out_path), exist_ok=True)
    torch.save(fusion_model.state_dict(), fusion_out_path)
    print(f"Saved fusion -> {fusion_out_path}")
    if use_wandb:
        wandb.log({'fusion_model_path': fusion_out_path})
        wandb.finish()


if __name__ == "__main__":
    main()
