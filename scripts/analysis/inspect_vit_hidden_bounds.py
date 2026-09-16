"""Capture declared ViT hidden bounds from an explicitly frozen experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime import identity


# @lat: [[evaluation#Evaluation and Verification#Hidden Activation Bounds Inspection]]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--experiment-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    experiment = json.loads(args.experiment_json.read_text())
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != experiment['source_commit']:
        raise ValueError('Source commit mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Frozen source has tracked modifications')
    if identity.sha256_file(source / experiment['evaluator_path']) != experiment['evaluator_sha256']:
        raise ValueError('Evaluator hash mismatch')
    if experiment['evaluator_args'] != ['--gelu-cubic-implementation', 'phi_nl_psi_ed', '--gelu-cubic-floor', '1e-5']:
        raise ValueError('Unsupported evaluator configuration')

    os.environ.update(CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled', HF_HUB_OFFLINE='1')
    sys.path[:0] = [str(source), str(source / 'src/transformers/src'), str(source / 'src/spikingjelly')]
    import torch
    from scripts.analysis.gelu_cubic_phi_nl_vit import install_phi_nl_psi_ed_cube
    from scripts.evaluation.error_analysis_vit import ViTConfig, ViTForImageClassification, ViTImageProcessor, image_processor_pixel_bounds
    from utils.transforms.noise import set_gaussian_time_noise
    from utils.transforms.types import Potential

    torch.set_num_threads(2)
    checkpoint = Path(experiment['checkpoint_path'])
    checkpoint_identity = identity.artifact_identity(checkpoint)
    if checkpoint_identity['aggregate_sha256'] != experiment['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    config = ViTConfig.from_pretrained(
        checkpoint, use_spiking_layernorm=True, spiking_ln_mul=True,
        spiking_ln_log=True, spiking_ln_expdiff=True, use_spiking_mlp=True,
        spiking_mlp_exact_gelu=False, hidden_act='gelu', theta=experiment['selected_theta'],
        local_files_only=True,
    )
    processor = ViTImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    pixel_domain = image_processor_pixel_bounds(processor, num_channels=config.num_channels)
    config.pixel_value_min, config.pixel_value_max = pixel_domain.min, pixel_domain.max
    install_phi_nl_psi_ed_cube(magnitude_floor=1e-5)
    model = ViTForImageClassification.from_pretrained(
        checkpoint, config=config, attn_implementation='spiking_sdpa',
        torch_dtype=torch.float64, local_files_only=True,
    ).eval()
    assert model.config._attn_implementation == 'spiking_sdpa'
    assert all(p.device.type == 'cpu' and p.dtype == torch.float64 for p in model.parameters())
    records = {}

    def capture(name, phase, value):
        if isinstance(value, Potential):
            records[(name, phase)] = (float(value.domain.min), float(value.domain.max))
        elif isinstance(value, tuple) and value and isinstance(value[0], Potential):
            capture(name, phase, value[0])

    for name, module in model.named_modules():
        if name:
            module.register_forward_pre_hook(lambda m, values, n=name: capture(n, 'input', values))
            module.register_forward_hook(lambda m, values, out, n=name: capture(n, 'output', out))

    reference = None
    checks = []
    for label, fill, noisy in [('clean_zero', 0.0, False), ('clean_one', 1.0, False), ('gaussian_zero', 0.0, True)]:
        records.clear()
        set_gaussian_time_noise(enabled=noisy, time_std=80e-5, time_mean=0.0,
                                deadline_margin=4 * 80e-5, seed=0, device=torch.device('cpu'))
        with torch.inference_mode():
            output = model(pixel_values=torch.full((1, 3, 224, 224), fill, dtype=torch.float64))
        assert bool(torch.isfinite(output.logits).all())
        assert len(records) > 200
        if reference is None:
            reference = dict(records)
        else:
            assert reference == records, f'Declared bounds changed: {label}'
        checks.append(label)
        print(f'{label}: {len(records)} declared bounds verified', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'module_bounds.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['module', 'phase', 'lower', 'upper'])
        for (name, phase), bounds in sorted(reference.items()):
            writer.writerow([name, phase, *bounds])
    stages = {'input': ('', 'input'), 'layernorm_before': ('.layernorm_before', 'output'),
              'attention_projection': ('.attention', 'output'), 'attention_residual': ('.layernorm_after', 'input'),
              'layernorm_after': ('.layernorm_after', 'output'), 'linear_intermediate': ('.intermediate.dense', 'output'),
              'gelu': ('.intermediate', 'output'), 'linear_output': ('.output.dense', 'output'),
              'output': ('', 'output')}
    for layer in range(config.num_hidden_layers):
        prefix = f'vit.encoder.layer.{layer}'
        input_bounds = reference[(prefix, 'input')]
        attention_bounds = reference[(prefix + '.attention', 'output')]
        residual_bounds = reference[(prefix + '.layernorm_after', 'input')]
        assert residual_bounds == tuple(a + b for a, b in zip(input_bounds, attention_bounds))
        linear_bounds = reference[(prefix + '.output.dense', 'output')]
        assert reference[(prefix, 'output')] == tuple(a + b for a, b in zip(residual_bounds, linear_bounds))
    with (args.output_dir / 'layer_bounds.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['layer', 'stage', 'lower', 'upper'])
        for layer in range(config.num_hidden_layers):
            for stage, (suffix, phase) in stages.items():
                bounds = reference[(f'vit.encoder.layer.{layer}{suffix}', phase)]
                writer.writerow([layer + 1, stage, *bounds])
    metadata = dict(source_commit=head, evaluator_sha256=experiment['evaluator_sha256'],
                    experiment_sha256=sha256(args.experiment_json), checkpoint_identity=checkpoint_identity,
                    diagnostic_sha256=sha256(Path(__file__)), theta=config.theta,
                    dtype='float64', calibration_mode='none', attention='spiking_sdpa',
                    device='cpu', sequence_length=197, samples_from_dataset=0,
                    checks=checks, module_bound_count=len(reference),
                    interpretation='Declared fixed bounds, not observed activation extrema.')
    (args.output_dir / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Outputs:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
