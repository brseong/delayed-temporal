from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal, cast
import math
import argparse

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from utils.transformers.models.spiking_gpt2.modeling_spiking_gpt2 import GPT2LMHeadModel, SpikingConv1D
from utils.transformers.models.spiking_gpt2.configuration_gpt2 import GPT2Config
from utils.transformers.models.spiking_ops import SpikingLayerNorm, SpikingLinear
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transforms.types import Potential
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
from utils.transforms import types
from tqdm import tqdm

_TB_LOG_BATCHES = 10
_QUANTILE_DIR = _REPO_ROOT / "artifacts" / "quantiles"

AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)

DATASET_PRESETS = {
    "wikitext2": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "dataset_split": "test",
        "text_column": "text",
        "model_id": "neulab/gpt2-finetuned-wikitext103",
    },
    "wikitext103": {
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-103-raw-v1",
        "dataset_split": "test",
        "text_column": "text",
        "model_id": "neulab/gpt2-finetuned-wikitext103",
    },
}

@dataclass
class Arguments:
    """Command-line configuration consumed by the GPT-2 evaluator.

    Direct Gaussian spike-time error is the sole dynamic event-noise interface.
    Evaluation converts its dimensionless standard-deviation fraction with the
    base identity window ``2 * theta`` and uses the absolute mean and seed for one
    evaluation-wide seeded noise state.
    """

    # Dataset, backend, and model-conversion controls remain independent from the
    # stochastic timing experiment selected below.
    experiment_name: str
    model_backend: Literal["hf", "spiking"]
    task: Literal["wikitext2", "wikitext103"]
    model_id: str
    dataset_name: str | None
    dataset_config_name: str | None
    dataset_split: str
    max_length: int
    batch_size: int
    device: Literal["cuda", "cpu"]
    max_eval_batches: int
    spiking_layernorm: bool
    spiking_attention: bool
    spiking_ln_mul: bool
    spiking_ln_log: bool
    spiking_ln_expdiff: bool
    spiking_mlp: bool
    activation: str
    theta: float
    tau_s: float

    # These four fields match ViT, BERT, and RoBERTa exactly; distribution choice
    # and a separate evaluation-mode switch are intentionally absent.
    gaussian_time_noise: bool
    time_noise_std_frac: float
    time_noise_mean: float
    time_noise_seed: int

    # Quantile collection is calibration instrumentation, not dynamic noise state.
    collect_quantiles: bool

def parse_arguments() -> Arguments:
    """Parse GPT-2 evaluation and direct Gaussian timing options.

    This function resolves WikiText presets and preserves the relative timing scale
    exactly as entered. Absolute-sigma conversion and generator installation remain
    responsibilities of :func:`evaluate_gpt2_model`.

    Returns:
        A dataset-resolved :class:`Arguments` instance.
    """
    # Keep language-model, dataset, and spiking-ablation controls independent from
    # the shared Gaussian replica parameters.
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on WikiText-2/103.")
    parser.add_argument("--experiment_name", type=str, default="gpt2_eval",
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_backend", type=str, choices=["hf", "spiking"], default="hf",
                        help="Model backend to use (hf: vanilla HF GPT-2, spiking: spiking_gpt2 class).")
    parser.add_argument("--task", type=str, choices=["wikitext2", "wikitext103"], default="wikitext2",
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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="If > 0, stop after this many evaluation batches for smoke testing.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the evaluation on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--spiking-layernorm", action=argparse.BooleanOptionalAction, default=True,
                        help="Use SpikingLayerNorm when --model_backend spiking is selected.")
    parser.add_argument("--spiking-attention", action=argparse.BooleanOptionalAction, default=True,
                        help="Use spiking SDPA attention when --model_backend spiking is selected.")
    parser.add_argument("--spiking-ln-mul", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 1: use ψ_M for variance.")
    parser.add_argument("--spiking-ln-log", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 2: use φ_NL for spike encoding.")
    parser.add_argument("--spiking-ln-expdiff", action=argparse.BooleanOptionalAction, default=True,
                        help="[SpikingLayerNorm] Stage 3: use ψ_ED for normalisation.")
    parser.add_argument("--spiking-mlp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use SpikingConv1D in MLP layers when --model_backend spiking is selected.")
    parser.add_argument("--activation", type=str, default="gelu_new",
                        help="Activation function for the spiking backend (default: gelu_new).")
    parser.add_argument("--theta", type=float, default=100.0,
                        help="Domain bound theta used by spiking backend modules.")
    parser.add_argument("--tau-s", type=float, default=1.0,
                        help="Spike-time constant tau_s used by SpikingLayerNorm.")

    # These options match the other model evaluators so one experiment convention
    # can configure every supported architecture.
    parser.add_argument(
        "--gaussian-time-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply direct Gaussian error to every event-aware spike time.",
    )
    parser.add_argument(
        "--time-noise-std-frac",
        type=float,
        default=0.0,
        help="Gaussian time std as a fraction of the identity window 2*theta.",
    )
    parser.add_argument(
        "--time-noise-mean",
        type=float,
        default=0.0,
        help="Gaussian timing mean in absolute time units (default: 0.0).",
    )
    parser.add_argument(
        "--time-noise-seed",
        type=int,
        default=0,
        help="Seed for the evaluation-wide timing-noise generator.",
    )
    parser.add_argument("--collect-quantiles", action="store_true",
                        help="Collect and print 99.9%% quantiles of absolute activations.")

    # Resolve dataset defaults first and copy all Gaussian values without changing
    # units; evaluation will perform the single 2*theta conversion.
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
        spiking_layernorm=args.spiking_layernorm,
        spiking_attention=args.spiking_attention,
        spiking_ln_mul=args.spiking_ln_mul,
        spiking_ln_log=args.spiking_ln_log,
        spiking_ln_expdiff=args.spiking_ln_expdiff,
        spiking_mlp=args.spiking_mlp,
        activation=args.activation,
        theta=args.theta,
        tau_s=args.tau_s,
        gaussian_time_noise=args.gaussian_time_noise,
        time_noise_std_frac=args.time_noise_std_frac,
        time_noise_mean=args.time_noise_mean,
        time_noise_seed=args.time_noise_seed,
        collect_quantiles=args.collect_quantiles,
    )

def infer_text_column(column_names: list[str], preferred: str | None = None) -> str:
    if preferred is not None and preferred in column_names:
        return preferred

    for candidate in ("text", "content", "sentence"):
        if candidate in column_names:
            return candidate

    raise ValueError(f"No supported text column found in dataset columns: {column_names}")

def evaluate_gpt2_model(args: Arguments) -> None:
    """Evaluate one GPT-2 backend with optional direct Gaussian event timing.

    The evaluator converts ``time_noise_std_frac`` to absolute time using the base
    identity-code window ``2 * theta`` and installs one seeded process-wide noise
    state. Causal-language-model loss and perplexity aggregation remain unchanged;
    Gaussian event and saturation diagnostics are emitted after the task loop.

    Args:
        args: Parsed GPT-2 dataset, conversion, and timing-noise settings.

    Raises:
        RuntimeError: If a Gaussian-enabled model is wrapped in ``DataParallel``.
        ValueError: If the shared Gaussian configuration rejects its parameters.
    """
    # Resolve model and dataset identity once so timing logs and task results refer
    # to the same effective evaluation configuration.
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

    # Convert the common user-facing fraction exactly once. tau_s does not rescale
    # this value; every encoder receives the same absolute sigma based on 2*theta.
    identity_time_window = 2.0 * float(args.theta)
    time_noise_std = float(args.time_noise_std_frac) * identity_time_window
    gaussian_enabled = bool(
        model_backend == "spiking" and args.gaussian_time_noise
    )

    # Install one evaluation-wide seeded generator and clear previous counters.
    # Explicitly disable it for the dense HF backend in reused Python processes.
    set_gaussian_time_noise(
        enabled=gaussian_enabled,
        time_std=time_noise_std,
        time_mean=args.time_noise_mean,
        seed=args.time_noise_seed,
        device=torch_device,
    )

    # Log both relative and absolute timing scales so theta sweeps do not obscure
    # the physical perturbation applied by the shared encoder boundary.
    cfg = {
        **vars(args),
        "gaussian_time_noise_effective": gaussian_enabled,
        "identity_time_window": identity_time_window,
        "time_noise_std": time_noise_std,
    }
    effective_attn_impl = "eager"
    if model_backend == "spiking" and torch_device.type != "cpu" and args.spiking_attention:
        effective_attn_impl = "spiking_sdpa"
    cfg["attn_impl"] = effective_attn_impl
    wandb.init(entity="CIDA", project="gpt2-evaluation", config=cfg, name=args.experiment_name)
    print(f"Using device: {torch_device}")
    print(f"Model backend: {model_backend}")
    print(
        "Gaussian time noise — "
        f"enabled: {gaussian_enabled}, "
        f"std_frac: {args.time_noise_std_frac}, "
        f"identity_window: {identity_time_window}, "
        f"std_abs: {time_noise_std}, "
        f"mean_abs: {args.time_noise_mean}, "
        f"seed: {args.time_noise_seed}"
    )
    if model_backend == "spiking":
        print(
            "Spiking config - "
            f"ln:{args.spiking_layernorm}, attn:{args.spiking_attention}, "
            f"mul:{args.spiking_ln_mul}, log:{args.spiking_ln_log}, "
            f"expdiff:{args.spiking_ln_expdiff}, mlp:{args.spiking_mlp}, "
            f"act:{args.activation}, theta:{args.theta}, tau_s:{args.tau_s}"
        )

    print(f"Loading dataset: {dataset_name}/{dataset_config_name} ({dataset_split})...")
    assert dataset_name is not None
    if dataset_config_name is None:
        dataset = load_dataset(dataset_name, split=dataset_split, cache_dir="/data/nas/datasets/")
    else:
        dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split, cache_dir="/data/nas/datasets/")

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
    model: nn.Module
    if model_backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(model_id)
    else:
        config = GPT2Config.from_pretrained(model_id)
        config.use_spiking_layernorm = args.spiking_layernorm
        config.spiking_ln_mul = args.spiking_ln_mul
        config.spiking_ln_log = args.spiking_ln_log
        config.spiking_ln_expdiff = args.spiking_ln_expdiff
        config.use_spiking_mlp = args.spiking_mlp
        config.activation_function = args.activation
        config.theta = args.theta
        config.tau_s = args.tau_s
        model = GPT2LMHeadModel.from_pretrained(model_id, config=config, attn_implementation=effective_attn_impl)

    # GPT-2 currently constructs no DataParallel wrapper itself. Retain an explicit
    # guard so future or externally inserted wrapping cannot share one global RNG.
    if gaussian_enabled and isinstance(model, nn.DataParallel):
        raise RuntimeError(
            "Gaussian spike-time noise does not support DataParallel; "
            "run one evaluation process per GPU"
        )

    if torch_device.type == "cuda":
        model = nn.Module.cuda(model)
    else:
        model = nn.Module.cpu(model)
    model.eval()

    tb_writer = SummaryWriter(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []

    def make_ln_hook(tag):
        def hook_fn(_module, inp, out):
            if log_step[0] < _TB_LOG_BATCHES:
                inp_val = inp[0].value if isinstance(inp[0], Potential) else inp[0]
                out_val = out.value if isinstance(out, Potential) else out
                tb_writer.add_histogram(f"{tag}/input", inp_val.detach().cpu().float(), log_step[0])
                tb_writer.add_histogram(f"{tag}/output", out_val.detach().cpu().float(), log_step[0])
        return hook_fn

    def make_clamp_hook(name):
        def pre_hook(_module, _inp):
            types.set_current_module_name(name)
        def post_hook(_module, _inp, _out):
            types.set_current_module_name(None)
        return pre_hook, post_hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            hooks.append(module.register_forward_hook(make_ln_hook(name)))
        
        if isinstance(module, (SpikingLayerNorm, SpikingLinear, SpikingConv1D)):
            pre_h, post_h = make_clamp_hook(name)
            hooks.append(module.register_forward_pre_hook(pre_h))
            hooks.append(module.register_forward_hook(post_h))

    quantiles = []
    def make_quantile_hook():
        def hook_fn(module, inp, out):
            val = out.value if isinstance(out, Potential) else out
            if isinstance(val, torch.Tensor):
                val_flat = val.detach().abs().float().view(-1)
                if val_flat.numel() > 16000000:
                    step = val_flat.numel() // 16000000 + 1
                    val_flat = val_flat[::step]
                q = torch.quantile(val_flat, 0.999).item()
                quantiles.append(q)
        return hook_fn

    if args.collect_quantiles:
        from transformers.pytorch_utils import Conv1D
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding, SpikingLayerNorm, SpikingLinear, SpikingConv1D, Conv1D)):
                hooks.append(module.register_forward_hook(make_quantile_hook()))

    if model_backend == "spiking":
        types.set_clamp_log_enabled(True)

    print("Starting evaluation...")

    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        labels = batch["labels"].to(torch_device)

        types.clear_clamp_stats()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Log clamp stats
        clamp_stats = types.get_clamp_stats()
        for (module_name, clamp_name), stats in clamp_stats.items():
            total = stats["total"]
            if total > 0:
                underflow_ratio = stats["underflow"] / total
                overflow_ratio = stats["overflow"] / total
                tb_writer.add_scalar(f"clamp/{module_name}/{clamp_name}/underflow", underflow_ratio, log_step[0])
                tb_writer.add_scalar(f"clamp/{module_name}/{clamp_name}/overflow", overflow_ratio, log_step[0])
                tb_writer.add_scalar(f"clamp/{module_name}/{clamp_name}/total_clamped", underflow_ratio + overflow_ratio, log_step[0])

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
    types.set_clamp_log_enabled(False)

    if args.collect_quantiles and quantiles:
        max_q = max(quantiles)
        print(f"RESULT_QUANTILE: {max_q}")
        _QUANTILE_DIR.mkdir(parents=True, exist_ok=True)
        with (_QUANTILE_DIR / f"quantile_gpt2_{args.task}.txt").open("w") as f:
            f.write(str(max_q))

    avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < float("inf") else float("inf")

    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    wandb.log({"Final Average Loss": avg_loss, "Final Perplexity": perplexity})

    # Convert only the maintained raw counters into report rates, preserving event
    # and output denominators as separate physical populations.
    if gaussian_enabled:
        for site, counts in sorted(get_gaussian_noise_stats().items()):
            events = counts["events"]
            outputs = counts["outputs"]
            miss_rate = counts["misses"] / events if events else 0.0
            underflow_rate = (
                counts["output_underflows"] / outputs if outputs else 0.0
            )
            overflow_rate = (
                counts["output_overflows"] / outputs if outputs else 0.0
            )
            print(
                f"Gaussian[{site}] events={events}, misses={counts['misses']} "
                f"(rate={miss_rate:.6g}), outputs={outputs}, "
                f"underflows={counts['output_underflows']} "
                f"(rate={underflow_rate:.6g}), "
                f"overflows={counts['output_overflows']} "
                f"(rate={overflow_rate:.6g})"
            )
            wandb.log({
                f"Gaussian/{site}/events": events,
                f"Gaussian/{site}/misses": counts["misses"],
                f"Gaussian/{site}/miss_rate": miss_rate,
                f"Gaussian/{site}/outputs": outputs,
                f"Gaussian/{site}/output_underflows": counts["output_underflows"],
                f"Gaussian/{site}/output_underflow_rate": underflow_rate,
                f"Gaussian/{site}/output_overflows": counts["output_overflows"],
                f"Gaussian/{site}/output_overflow_rate": overflow_rate,
            })

    print("-" * 30)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_gpt2_model(args)
