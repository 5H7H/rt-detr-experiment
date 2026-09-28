"""Read-only EMA gate diagnostics on the first N validation images."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/sgf_r50_72ep.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    torch.set_num_threads(2)
    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    cfg.yaml_cfg['val_dataloader'].update(num_workers=0, batch_size=1, shuffle=False)
    model = cfg.model.eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['ema']['module'])
    del state
    if model.encoder.sgf is None:
        parser.error('The config does not enable SGF')
    cache, rows = {}, []
    def gate_hook(module, inputs, output):
        cache['gate'] = output.sigmoid()
    def fusion_hook(module, inputs, output):
        f3 = inputs[1]
        gate = cache['gate']
        rows.append(dict(gate_mean=gate.mean().item(),
                         gate_p05=gate.quantile(.05).item(), gate_p95=gate.quantile(.95).item(),
                         gate_lt01=(gate < .1).float().mean().item(),
                         gate_gt09=(gate > .9).float().mean().item(),
                         injection_norm_ratio=((output-f3).norm()/f3.norm()).item()))
    handles = [model.encoder.sgf.gate.register_forward_hook(gate_hook),
               model.encoder.sgf.register_forward_hook(fusion_hook)]
    with torch.inference_mode():
        for images, targets in itertools.islice(cfg.val_dataloader, args.limit):
            model.encoder(model.backbone(images))
    for handle in handles:
        handle.remove()
    if not rows:
        raise ValueError('Empty validation loader')
    print(json.dumps({'images': len(rows), 'device': 'cpu', 'weights': 'ema',
                      'lambda': model.encoder.sgf.residual_scale.item(),
                      'mean_of_per_image_statistics': {
                          key: sum(row[key] for row in rows)/len(rows) for key in rows[0]}}, indent=2))


if __name__ == '__main__':
    main()
