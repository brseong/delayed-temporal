from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.transformers.optional_tensorboard import create_summary_writer
from datasets import load_dataset
from transformers import AttentionInterface, AutoModelForSequenceClassification, AutoTokenizer
from utils.transformers.models.spiking_roberta.configuration_roberta import RobertaConfig
from utils.transformers.models.spiking_roberta.modeling_spiking_roberta import (
    RobertaEncoder,
    RobertaForSequenceClassification,
    RobertaSelfAttention,
    SpikingLayerNorm,
)
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transforms.types import Potential
from utils.transforms import types
from utils.transformers.models.spiking_ops import SpikingLinear
from utils.transforms.noise import get_gaussian_noise_stats, set_gaussian_time_noise
import evaluate
from tqdm import tqdm

_TB_LOG_BATCHES = 10  # 처음 N 배치에서만 히스토그램 로그
_QUANTILE_DIR = _REPO_ROOT / "artifacts" / "quantiles"
AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)

DATASET_PRESETS = {
    "sst2": {
        "dataset_name": "glue",
        "dataset_config_name": "sst2",
        "dataset_split": "validation",
        "text_column": "sentence",
        "model_id": "textattack/roberta-base-SST-2",
    },
    "agnews": {
        "dataset_name": "ag_news",
        "dataset_config_name": None,
        "dataset_split": "test",
        "text_column": "text",
        "model_id": "textattack/roberta-base-ag-news",
    },
    "imdb": {
        "dataset_name": "imdb",
        "dataset_config_name": None,
        "dataset_split": "test",
        "text_column": "text",
        "model_id": "textattack/roberta-base-imdb",
    },
}

@dataclass
class Arguments:
    """Command-line configuration consumed by the RoBERTa evaluator.

    Direct Gaussian spike-time error is the only dynamic event-noise model. Its
    standard-deviation fraction remains dimensionless until evaluation converts it
    with ``2 * theta``; the absolute mean and seed identify one evaluation-wide
    seeded noise state.
    """

    # Dataset, backend, and conversion controls specify the deterministic path
    # independently from stochastic event timing.
    experiment_name: str
    model_backend: Literal["hf", "spiking"]
    task: Literal["sst2", "agnews", "imdb"]
    model_id: str
    dataset_name: str | None
    dataset_config_name: str | None
    dataset_split: str
    cache_dir: str
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
    activation: Literal["relu", "gelu"]
    theta: float

    # Keep the same four Gaussian fields across every evaluator so scripts can
    # configure replicas without model-specific timing parameters.
    gaussian_time_noise: bool
    time_noise_std_frac: float
    time_noise_mean: float
    time_noise_seed: int

    # Quantile collection remains a deterministic calibration diagnostic.
    collect_quantiles: bool
    report_clamp_stats: bool

def parse_arguments() -> Arguments:
    """Parse RoBERTa evaluation and direct Gaussian timing options.

    Dataset presets are resolved after argparse, while Gaussian scale conversion is
    deliberately deferred to :func:`evaluate_roberta_model` so logging can retain
    both the input fraction and its eventual absolute value.

    Returns:
        A dataset-resolved :class:`Arguments` instance.
    """
    # Preserve the existing task and spiking-ablation interface while replacing
    # only the obsolete dynamic event-noise controls.
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face RoBERTa on SST-2, AG News, or IMDB.")
    parser.add_argument("--experiment_name", type=str, default="roberta_eval",
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_backend", type=str, choices=["hf", "spiking"], default="hf",
                        help="Model backend to use (hf: vanilla HF RoBERTa, spiking: spiking_roberta class).")
    parser.add_argument("--task", type=str, choices=["sst2", "agnews", "imdb"], default="sst2",
                        help="Preset task to evaluate. Sets dataset, split, and default model.")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Optional Hugging Face model ID. If omitted, task preset default is used.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Optional dataset name override. If omitted, task preset is used.")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Optional dataset config override. If omitted, task preset is used.")
    parser.add_argument("--dataset_split", type=str, default=None,
                        help="Optional dataset split override. If omitted, task preset is used.")
    parser.add_argument("--cache-dir", type=str, default="/data/nas/datasets/",
                        help="Hugging Face dataset cache directory.")
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
                        help="Use spiking MLP when --model_backend spiking is selected.")
    parser.add_argument("--activation", type=str, choices=["relu", "gelu"], default="gelu",
                        help="Activation function used by the spiking backend config.")
    parser.add_argument("--theta", type=float, default=100.0,
                        help="Domain bound theta used by spiking backend modules.")

    # Use the common Gaussian-only CLI. BooleanOptionalAction also supplies the
    # explicit --no-gaussian-time-noise form without a compatibility alias.
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
    parser.add_argument("--report-clamp-stats", action="store_true",
                        help="Aggregate and print per-site fixed-domain clamp counts.")

    # Resolve task defaults once, then transfer Gaussian values unchanged into the
    # dataclass; absolute-sigma calculation belongs to the later evaluation step.
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
        cache_dir=args.cache_dir,
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
        gaussian_time_noise=args.gaussian_time_noise,
        time_noise_std_frac=args.time_noise_std_frac,
        time_noise_mean=args.time_noise_mean,
        time_noise_seed=args.time_noise_seed,
        collect_quantiles=args.collect_quantiles,
        report_clamp_stats=args.report_clamp_stats,
    )


def infer_text_column(column_names: list[str], preferred: str | None = None) -> str:
    if preferred is not None and preferred in column_names:
        return preferred

    for candidate in ("sentence", "text", "content", "review"):
        if candidate in column_names:
            return candidate

    raise ValueError(f"No supported text column found in dataset columns: {column_names}")

def evaluate_roberta_model(args: Arguments) -> None:
    """Evaluate one RoBERTa backend with optional Gaussian event timing.

    The dimensionless noise fraction is converted once with the identity-code
    window ``2 * theta`` and installed with one seeded process-wide generator.
    Gaussian execution rejects this evaluator's multi-GPU ``DataParallel`` path,
    while deterministic and Hugging Face evaluations retain existing behavior.

    Args:
        args: Parsed RoBERTa dataset, conversion, and timing-noise settings.

    Raises:
        RuntimeError: If Gaussian timing would execute through ``DataParallel``.
        ValueError: If the shared Gaussian configuration rejects its parameters.
    """
    # Resolve the effective dataset and backend identity once for model setup,
    # logging, and calibration metadata.
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

    # Convert the shared relative CLI scale exactly once; all decorated encoders in
    # this evaluation use the resulting absolute timing standard deviation.
    identity_time_window = 2.0 * float(args.theta)
    time_noise_std = float(args.time_noise_std_frac) * identity_time_window
    gaussian_enabled = bool(
        model_backend == "spiking" and args.gaussian_time_noise
    )

    # RoBERTa normally wraps multiple visible GPUs in DataParallel. One mutable
    # process-wide generator cannot provide valid per-device replica streams.
    use_data_parallel = (
        torch_device.type == "cuda" and torch.cuda.device_count() > 1
    )
    if gaussian_enabled and use_data_parallel:
        raise RuntimeError(
            "Gaussian spike-time noise does not support DataParallel; "
            "run one evaluation process per GPU"
        )

    # Seed one evaluation-wide noise state and reset its counters before any model
    # or dataset work. HF execution explicitly installs the disabled state.
    set_gaussian_time_noise(
        enabled=gaussian_enabled,
        time_std=time_noise_std,
        time_mean=args.time_noise_mean,
        seed=args.time_noise_seed,
        device=torch_device,
    )

    # Preserve both relative and absolute timing scales in W&B configuration so
    # theta sweeps can be interpreted directly.
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
    wandb.init(entity="CIDA", project="roberta-evaluation", config=cfg, name=args.experiment_name)
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
            f"act:{args.activation}, theta:{args.theta}"
        )

    print(f"Loading dataset: {dataset_name}/{dataset_config_name} ({dataset_split})...")
    assert dataset_name is not None
    if dataset_config_name is None:
        dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split, cache_dir=args.cache_dir)
    preferred_text_column = DATASET_PRESETS.get(args.task, {}).get("text_column")
    text_column = infer_text_column(dataset.column_names, preferred=preferred_text_column)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    metric_tot = evaluate.load("accuracy")

    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    processed_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(cast(Any, processed_dataset), batch_size=batch_size, shuffle=False)

    print(f"Loading model: {model_id}...")
    model: nn.Module
    if model_backend == "hf":
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        config = RobertaConfig.from_pretrained(model_id)
        config.use_spiking_layernorm = args.spiking_layernorm
        config.spiking_ln_mul = args.spiking_ln_mul
        config.spiking_ln_log = args.spiking_ln_log
        config.spiking_ln_expdiff = args.spiking_ln_expdiff
        config.use_spiking_mlp = args.spiking_mlp
        config.hidden_act = args.activation
        config.theta = args.theta
        model = RobertaForSequenceClassification.from_pretrained(model_id, config=config, attn_implementation=effective_attn_impl)
    if torch_device.type == "cuda":
        model = nn.Module.cuda(model)
    else:
        model = nn.Module.cpu(model)
    
    # GPU 병렬화 (DataParallel) 설정
    if use_data_parallel:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)    
    
    model.eval()

    tb_writer = create_summary_writer(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []
    clamp_totals: dict[tuple[str, str], dict[str, int]] = {}

    def make_clamp_hook(name: str):
        previous_names: list[str | None] = []

        def pre_hook(_module, _inp):
            previous_names.append(types.get_current_module_name())
            types.set_current_module_name(name)

        def post_hook(_module, _inp, _out):
            previous = previous_names.pop() if previous_names else None
            types.set_current_module_name(previous)

        return pre_hook, post_hook

    def make_ln_hook(tag):
        def hook_fn(module, inp, out):
            if log_step[0] < _TB_LOG_BATCHES:
                inp_val = inp[0].value if isinstance(inp[0], Potential) else inp[0]
                out_val = out.value if isinstance(out, Potential) else out
                tb_writer.add_histogram(f"{tag}/input", inp_val.detach().cpu().float(), log_step[0])
                tb_writer.add_histogram(f"{tag}/output", out_val.detach().cpu().float(), log_step[0])
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, SpikingLayerNorm)):
            hooks.append(module.register_forward_hook(make_ln_hook(name)))

    if model_backend == "spiking" and args.report_clamp_stats:
        for name, module in model.named_modules():
            if isinstance(module, (RobertaEncoder, RobertaSelfAttention, SpikingLayerNorm, SpikingLinear)):
                pre_hook, post_hook = make_clamp_hook(name)
                hooks.append(module.register_forward_pre_hook(pre_hook))
                hooks.append(module.register_forward_hook(post_hook))
        types.clear_clamp_stats()
        types.set_current_module_name(None)
        types.set_clamp_log_enabled(True)

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
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding, SpikingLayerNorm)):
                hooks.append(module.register_forward_hook(make_quantile_hook()))

    print("Starting evaluation...")

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        labels = batch["labels"].to(torch_device)

        if model_backend == "spiking" and args.report_clamp_stats:
            types.clear_clamp_stats()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        if model_backend == "spiking" and args.report_clamp_stats:
            for tag, stats in types.get_clamp_stats().items():
                aggregate = clamp_totals.setdefault(
                    tag, {"underflow": 0, "overflow": 0, "total": 0}
                )
                for field in ("underflow", "overflow", "total"):
                    aggregate[field] += stats[field]
            types.set_current_module_name(None)

        log_step[0] += 1

        predictions = torch.argmax(outputs.logits, dim=-1)

        metric_tot.add_batch(predictions=predictions, references=labels)
        wandb.log({"Batch Accuracy": (predictions == labels).float().mean().item()})

        if max_eval_batches > 0 and log_step[0] >= max_eval_batches:
            break

    for h in hooks:
        h.remove()
    tb_writer.close()
    types.set_current_module_name(None)
    if model_backend == "spiking" and args.report_clamp_stats:
        types.set_clamp_log_enabled(False)

    if args.collect_quantiles and quantiles:
        max_q = max(quantiles)
        print(f"RESULT_QUANTILE: {max_q}")
        _QUANTILE_DIR.mkdir(parents=True, exist_ok=True)
        model_name = args.model_id.replace("/", "_")
        with (_QUANTILE_DIR / f"quantile_roberta_{args.task}_{model_name}.txt").open("w") as f:
            f.write(str(max_q))

    final_score = cast(dict[str, float], metric_tot.compute())
    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Accuracy: {final_score['accuracy']:.4f}")
    wandb.log({"Final Accuracy": final_score["accuracy"]})

    for (module_name, clamp_name), stats in sorted(clamp_totals.items()):
        total = stats["total"]
        underflow_rate = stats["underflow"] / total if total else 0.0
        overflow_rate = stats["overflow"] / total if total else 0.0
        site = f"{module_name}/{clamp_name}"
        print(
            f"Clamp[{site}] values={total}, underflows={stats['underflow']} "
            f"(rate={underflow_rate:.6g}), overflows={stats['overflow']} "
            f"(rate={overflow_rate:.6g})"
        )

    # Keep event and output denominators distinct when deriving report-only rates
    # from the maintained five-field Gaussian statistics records.
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
    evaluate_roberta_model(args)
