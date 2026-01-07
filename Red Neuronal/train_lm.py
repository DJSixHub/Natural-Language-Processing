from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from red_neuronal.data.jsonl_dataset import JsonlSFTDataset, collate_pad
from red_neuronal.modeling.model import DecoderOnlyTransformerLM, ModelConfig

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def split_params_for_weight_decay(model: nn.Module):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("bias") or "norm" in n.lower() or "weight" in n.lower() and p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            # shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def save_checkpoint(out_dir: Path, name: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    path = out_dir / "checkpoints" / f"{name}.pt"
    torch.save(ckpt, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[0] / "model"))

    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1, help="Accumulate gradients for N micro-batches")
    ap.add_argument("--eval-batch-size", type=int, default=0, help="If 0, reuse --batch-size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)

    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--d-ff", type=int, default=1408)

    ap.add_argument("--checkpoint-every", type=int, default=2, help="epochs")
    ap.add_argument("--validate-every", type=int, default=2, help="epochs")
    ap.add_argument("--max-steps", type=int, default=0, help="If >0, stop after N optimizer steps")
    ap.add_argument("--resume", default="", help="Path to a checkpoint .pt to resume from")

    args = ap.parse_args()

    if args.grad_accum < 1:
        raise SystemExit("--grad-accum debe ser >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()

    cfg = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
    )

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    train_ds = JsonlSFTDataset(Path(args.train_jsonl), tok, max_seq_len=cfg.max_seq_len)
    val_ds = JsonlSFTDataset(Path(args.val_jsonl), tok, max_seq_len=cfg.max_seq_len)
    print(f"\nDataset: train={len(train_ds)} ejemplos, val={len(val_ds)} ejemplos")

    pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        raise RuntimeError("Tokenizer missing <pad>")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, pad_id=pad_id),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=(args.eval_batch_size or args.batch_size),
        shuffle=False,
        collate_fn=lambda b: collate_pad(b, pad_id=pad_id),
        num_workers=0,
    )

    model = DecoderOnlyTransformerLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo: {n_params:,} parámetros totales")
    print(f"Config: {cfg.n_layers}L x {cfg.n_heads}H x {cfg.d_model}D (d_ff={cfg.d_ff})\n")

    decay, no_decay = split_params_for_weight_decay(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup)

    def lr_for(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    global_step = 0
    model.train()

    # Optional resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        resumed_epoch = int(ckpt.get("epoch", 0))
        print(f"Reanudando desde {args.resume} (epoch={resumed_epoch})")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True) if tqdm else train_loader
        batch_idx = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            # gradient accumulation: scale loss by accum steps
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            running += loss.item() * args.grad_accum

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            lr = lr_for(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            global_step += 1
            batch_idx += 1
            
            # Update progress bar
            if tqdm and hasattr(loader, 'set_postfix'):
                avg_loss = running / max(1, batch_idx)
                loader.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})  # type: ignore

            if args.max_steps and global_step >= args.max_steps:
                break

        avg_train = running / max(1, len(train_loader))
        dt = time.time() - t0
        print(f"epoch {epoch} train_loss={avg_train:.4f} time={dt:.1f}s")

        if epoch % args.validate_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"epoch {epoch} val_loss={val_loss:.4f}")

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(out_dir, f"epoch_{epoch:04d}", model, optimizer, epoch)
            print(f"checkpoint guardado: epoch_{epoch:04d}.pt")

        if args.max_steps and global_step >= args.max_steps:
            break

    # final
    final_path = out_dir / "final.pt"
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, final_path)
    print("OK: modelo final guardado en", final_path)


if __name__ == "__main__":
    main()
