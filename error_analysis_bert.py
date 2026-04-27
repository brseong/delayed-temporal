from dataclasses import dataclass
from typing import Literal

import argparse
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import AttentionInterface, AutoModelForSequenceClassification, AutoTokenizer
from utils.transformers.models.spiking_bert.configuration_bert import BertConfig
from utils.transformers.models.spiking_bert.modeling_spiking_bert import BertForSequenceClassification, SpikingLayerNorm
from utils.transformers.integrations.spiking_sdpa_attention import spiking_sdpa_attention_forward
from utils.transforms.types import Potential
import evaluate
from tqdm import tqdm

_TB_LOG_BATCHES = 10  # 처음 N 배치에서만 히스토그램 로그
AttentionInterface.register("spiking_sdpa", spiking_sdpa_attention_forward)

@dataclass
class Arguments:
    experiment_name: str
    model_backend: Literal["hf", "spiking"]
    model_id: str
    dataset_name: str
    dataset_config_name: str
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
    activation: Literal["relu", "gelu"]
    theta: float

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Hugging Face BERT on SST-2.")
    parser.add_argument("--experiment_name", type=str, default="bert_eval",
                        help="Name of the experiment for logging purposes.")
    parser.add_argument("--model_backend", type=str, choices=["hf", "spiking"], default="hf",
                        help="Model backend to use (hf: vanilla HF BERT, spiking: spiking_bert class).")
    parser.add_argument("--model_id", type=str, default="textattack/bert-base-uncased-SST-2",
                        help="Pretrained BERT model ID from Hugging Face.")
    parser.add_argument("--dataset_name", type=str, default="glue",
                        help="Dataset name from Hugging Face datasets library.")
    parser.add_argument("--dataset_config_name", type=str, default="sst2",
                        help="Dataset config name for GLUE SST-2.")
    parser.add_argument("--dataset_split", type=str, default="validation",
                        help="Dataset split to evaluate.")
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

    args = parser.parse_args()
    return Arguments(
        experiment_name=args.experiment_name,
        model_backend=args.model_backend,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
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
    )

def evaluate_bert_model(args:Arguments):
    model_backend = args.model_backend
    model_id = args.model_id
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    dataset_split = args.dataset_split
    max_length = args.max_length
    batch_size = args.batch_size
    max_eval_batches = args.max_eval_batches
    device_str = args.device
    
    device = torch.device(device_str)
    
    cfg = vars(args)
    effective_attn_impl = "eager"
    if model_backend == "spiking" and device.type != "cpu" and args.spiking_attention:
        effective_attn_impl = "spiking_sdpa"
    cfg["attn_impl"] = effective_attn_impl
    wandb.init(entity="CIDA", project="bert-evaluation", config=cfg, name=args.experiment_name)
    print(f"Using device: {device}")
    print(f"Model backend: {model_backend}")
    if model_backend == "spiking":
        print(
            "Spiking config - "
            f"ln:{args.spiking_layernorm}, attn:{args.spiking_attention}, "
            f"mul:{args.spiking_ln_mul}, log:{args.spiking_ln_log}, "
            f"expdiff:{args.spiking_ln_expdiff}, mlp:{args.spiking_mlp}, "
            f"act:{args.activation}, theta:{args.theta}"
        )
    print(f"Loading dataset: {dataset_name}/{dataset_config_name} ({dataset_split})...")
    dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    metric_tot = evaluate.load("accuracy")

    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    processed_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loading model: {model_id}...")
    if model_backend == "hf":
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        config = BertConfig.from_pretrained(model_id)
        config.use_spiking_layernorm = args.spiking_layernorm
        config.spiking_ln_mul = args.spiking_ln_mul
        config.spiking_ln_log = args.spiking_ln_log
        config.spiking_ln_expdiff = args.spiking_ln_expdiff
        config.use_spiking_mlp = args.spiking_mlp
        config.hidden_act = args.activation
        config.theta = args.theta
        model = BertForSequenceClassification.from_pretrained(model_id, config=config, attn_implementation=effective_attn_impl)
    model.to(device)
    model.eval()

    tb_writer = SummaryWriter(log_dir=f"runs/{args.experiment_name}")
    log_step = [0]
    hooks = []

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

    print("Starting evaluation...")

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        log_step[0] += 1

        predictions = torch.argmax(outputs.logits, dim=-1)

        metric_tot.add_batch(predictions=predictions, references=labels)
        wandb.log({"Batch Accuracy": (predictions == labels).float().mean().item()})

        if max_eval_batches > 0 and log_step[0] >= max_eval_batches:
            break

    for h in hooks:
        h.remove()
    tb_writer.close()

    final_score = metric_tot.compute()
    print("-" * 30)
    print(f"Evaluation Results for {model_id}:")
    print(f"Accuracy: {final_score['accuracy']:.4f}")
    wandb.log({"Final Accuracy": final_score["accuracy"]})
    print("-" * 30)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_bert_model(args)
