from dataclasses import dataclass
from typing import Any, Literal, cast
import math
import argparse
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
# Since we don't have a spiking_llama yet, we use the standard model but support spiking attention if registered
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transforms.types import Potential, set_spike_time_noise
from tqdm import tqdm

_TB_LOG_BATCHES = 10

# Register spiking SDPA attention implementation
AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)

DATASET_PRESETS = {
    "wikitext2": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "dataset_split": "test",
        "text_column": "text",
        "model_id": "meta-llama/Llama-3.1-8B",
    },
}

@dataclass
class Arguments:
    experiment_name: str
    model_backend: Literal["hf", "spiking"]
    task: Literal["wikitext2"]
    model_id: str
    dataset_name: str | None
    dataset_config_name: str | None
    dataset_split: str
    max_length: int
    batch_size: int
    device: Literal["cuda", "cpu"]
    max_eval_batches: int
    spiking_attention: bool
    theta: float
    tau_s: float
    spike_time_noise_std: float
    spike_time_noise_kind: Literal["gaussian", "uniform"]
    spike_time_noise_eval: bool

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Llama-3 on WikiText-2.")
    parser.add_argument("--experiment_name", type=str, default="llama3_eval",
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_backend", type=str, choices=["hf", "spiking"], default="hf",
                        help="Model backend to use (hf: vanilla HF Llama, spiking: currently only supports spiking attention).")
    parser.add_argument("--task", type=str, choices=["wikitext2"], default="wikitext2",
                        help="Preset task to evaluate. Sets dataset, split, and default model.")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Optional Hugging Face model ID. If omitted, task preset default is used.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Optional dataset name override. If omitted, task preset is used.")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Optional dataset config override. If omitted, task preset is used.")
    parser.add_argument("--dataset_split", type=str, default=None,
                        help="Optional dataset split override. If omitted, task preset is used.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum token length for tokenizer padding/truncation.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation. Default reduced for Llama-3 8B.")
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="If > 0, stop after this many evaluation batches for smoke testing.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--spiking-attention", action=argparse.BooleanOptionalAction, default=True,
                        help="Use spiking SDPA attention when --model_backend spiking is selected.")
    parser.add_argument("--theta", type=float, default=100.0,
                        help="Domain bound theta used by spiking attention.")
    parser.add_argument("--tau-s", type=float, default=1.0,
                        help="Spike-time constant tau_s.")
    parser.add_argument("--spike_time_noise_std", type=float, default=0.0,
                        help="Relative std for spike-time jitter against time-domain range.")
    parser.add_argument("--spike_time_noise_kind", type=str, choices=["gaussian", "uniform"], default="gaussian",
                        help="Noise distribution for spike-time jitter.")
    parser.add_argument("--spike_time_noise_eval", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply spike-time jitter in eval mode as well.")

    args = parser.parse_args()
    preset = DATASET_PRESETS[args.task]
    model_id = cast(str, args.model_id or preset["model_id"])
    dataset_name = cast(str | None, args.dataset_name or preset["dataset_name"])
    dataset_config_name = cast(str | None, args.dataset_config_name if args.dataset_config_name is not None else preset["dataset_config_name"])
    dataset_split = cast(str, args.dataset_split or preset["dataset_split"])

    return Arguments(
        experiment_name=args.experiment_name,
        model_backend=args.model_backend,
        task=args.task,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        dataset_split=dataset_split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        max_eval_batches=args.max_eval_batches,
        spiking_attention=args.spiking_attention,
        theta=args.theta,
        tau_s=args.tau_s,
        spike_time_noise_std=args.spike_time_noise_std,
        spike_time_noise_kind=args.spike_time_noise_kind,
        spike_time_noise_eval=args.spike_time_noise_eval,
    )

def infer_text_column(column_names: list[str], preferred: str | None = None) -> str:
    if preferred is not None and preferred in column_names:
        return preferred
    for candidate in ("text", "content", "sentence"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"No supported text column found in dataset columns: {column_names}")

def evaluate_llama3_model(args: Arguments):
    model_backend = args.model_backend
    model_id = cast(str, args.model_id)
    dataset_name = cast(str | None, args.dataset_name)
    dataset_config_name = cast(str | None, args.dataset_config_name)
    dataset_split = cast(str, args.dataset_split)
    max_length = args.max_length
    batch_size = args.batch_size
    max_eval_batches = args.max_eval_batches
    device_str = args.device

    torch_device = torch.device(device_str)

    cfg = vars(args)
    effective_attn_impl = "eager"
    if model_backend == "spiking" and torch_device.type != "cpu" and args.spiking_attention:
        effective_attn_impl = "spiking_sdpa"
    cfg["attn_impl"] = effective_attn_impl
    
    wandb.init(entity="CIDA", project="llama3-evaluation", config=cfg, name=args.experiment_name)
    print(f"Using device: {torch_device}")
    print(f"Model backend: {model_backend}")
    if model_backend == "spiking":
        print(
            "Spiking config - "
            f"attn:{args.spiking_attention}, "
            f"theta:{args.theta}, tau_s:{args.tau_s}, "
            f"time_noise_std:{args.spike_time_noise_std}, "
            f"time_noise_kind:{args.spike_time_noise_kind}, "
            f"time_noise_eval:{args.spike_time_noise_eval}"
        )
        # Register global spike-time noise
        set_spike_time_noise(
            std=args.spike_time_noise_std,
            kind=args.spike_time_noise_kind,
            eval_mode=args.spike_time_noise_eval,
        )

    print(f"Loading dataset: {dataset_name}/{dataset_config_name} ({dataset_split})...")
    assert dataset_name is not None
    if dataset_config_name is None:
        dataset = load_dataset(dataset_name, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split)

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    preferred_text_column = DATASET_PRESETS.get(args.task, {}).get("text_column")
    text_column = infer_text_column(dataset.column_names, preferred=preferred_text_column)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        labels = []
        for i in range(len(tokenized["input_ids"])):
            label = [
                -100 if mask == 0 else token
                for mask, token in zip(tokenized["attention_mask"][i], tokenized["input_ids"][i])
            ]
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    processed_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(cast(Any, processed_dataset), batch_size=batch_size, shuffle=False)

    print(f"Loading model: {model_id}...")
    # Use bfloat16 for Llama-3.1
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        attn_implementation=effective_attn_impl,
        trust_remote_code=True
    )

    model = model.to(torch_device)
    model.eval()
    print(type(model))
    tb_writer = SummaryWriter(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []

    def make_norm_hook(tag):
        def hook_fn(_module, inp, out):
            if log_step[0] < _TB_LOG_BATCHES:
                inp_val = inp[0].value if isinstance(inp[0], Potential) else inp[0]
                out_val = out.value if isinstance(out, Potential) else out
                # Convert to float for histogram logging
                tb_writer.add_histogram(f"{tag}/input", inp_val.detach().cpu().float(), log_step[0])
                tb_writer.add_histogram(f"{tag}/output", out_val.detach().cpu().float(), log_step[0])
        return hook_fn

    for name, module in model.named_modules():
        # Llama uses RMSNorm. We look for modules containing 'norm' or specific RMSNorm types.
        if "norm" in name.lower():
            hooks.append(module.register_forward_hook(make_norm_hook(name)))

    print("Starting evaluation...")

    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        labels = batch["labels"].to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        if not torch.isnan(loss):
            total_loss += loss.item()
            total_steps += 1
            wandb.log({"Batch Loss": loss.item(), "Batch Perplexity": math.exp(min(loss.item(), 20.0))})

        log_step[0] += 1
        if max_eval_batches > 0 and log_step[0] >= max_eval_batches:
            break

    for h in hooks:
        h.remove()
    tb_writer.close()

    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < float("inf") else float("inf")

    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    wandb.log({"Final Average Loss": avg_loss, "Final Perplexity": perplexity})
    print("-" * 30)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_llama3_model(args)
