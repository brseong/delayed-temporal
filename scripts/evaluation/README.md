# Evaluation entry points

The four `error_analysis_*` programs are the stable model-family command-line interface for dense and TTFS evaluation.

| Entry point | Model family | Primary metric |
|---|---|---|
| `error_analysis_vit.py` | ViT image classification | accuracy |
| `error_analysis_bert.py` | BERT sequence classification | accuracy |
| `error_analysis_roberta.py` | RoBERTa sequence classification | accuracy |
| `error_analysis_gpt2.py` | GPT-2 causal language modeling | loss and perplexity |

An evaluator owns model construction, dataset order, preprocessing, calibration attachment, per-batch progress, final metrics, and diagnostic counters. Options that change any of those belong here even if only one campaign currently uses them.

`text_calibration_runtime.py` is shared evaluator support for the three text-model entry points. It is not an independent experiment.

Campaign-specific grids, host allocation, retry policy, artifact publication, and paper-table selection do not belong in this directory.
