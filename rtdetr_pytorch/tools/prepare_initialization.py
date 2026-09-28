"""Save paired baseline/SGF initialization with identical shared tensors."""
import argparse
import copy
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import YAMLConfig
from src.misc.dist import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; choose a new directory')
    torch.set_num_threads(2)
    set_seed(args.seed)
    cfg = YAMLConfig('configs/experiments/r50_baseline_72ep.yml')
    cfg.yaml_cfg = copy.deepcopy(cfg.yaml_cfg)
    if args.baseline_checkpoint:
        cfg.yaml_cfg['PResNet']['pretrained'] = False
    baseline = cfg.model
    if args.baseline_checkpoint:
        checkpoint = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=False)
        baseline.load_state_dict(checkpoint['model'], strict=True)
    state = baseline.state_dict()
    set_seed(args.seed)
    cfg = YAMLConfig('configs/experiments/sgf_r50_72ep.yml')
    cfg.yaml_cfg['PResNet']['pretrained'] = False
    sgf = cfg.model
    missing, unexpected = sgf.load_state_dict(state, strict=False)
    assert missing and all(k.startswith('encoder.sgf.') for k in missing) and not unexpected
    assert all(torch.equal(value, sgf.state_dict()[key]) for key, value in state.items())
    assert sgf.encoder.sgf.residual_scale.item() == 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({'model': state}, args.output_dir / 'baseline.pth')
    torch.save({'model': sgf.state_dict()}, args.output_dir / 'sgf.pth')
    print('Saved paired initialization; common tensors are identical; SGF lambda=0.')


if __name__ == '__main__':
    main()
