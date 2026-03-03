import argparse
import torch


def extract_state(checkpoint: dict, source: str):
    if source == 'model':
        return checkpoint.get('model', checkpoint)
    if source == 'ema1':
        return checkpoint.get('model_ema1')
    if source == 'ema2':
        return checkpoint.get('model_ema2')
    raise ValueError(f'Unknown source: {source}')


def main():
    parser = argparse.ArgumentParser(description='Print gate-related parameters from a JiT checkpoint')
    parser.add_argument('--checkpoint', default="/NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/caJiT_CP/round4/noLoss_noHintsDropout_dot_concat_sam2_b8_decouple_CA/checkpoint-last.pth", help='Path to checkpoint .pth')
    parser.add_argument('--source', choices=['model', 'ema1', 'ema2'], default='model',
                        help='Which state dict in checkpoint to inspect')
    parser.add_argument('--max-items', type=int, default=20,
                        help='Maximum tensor entries to print per parameter')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = extract_state(ckpt, args.source)
    if state is None:
        raise KeyError(f"State dict '{args.source}' not found in checkpoint.")

    gate_keys = [
        k for k in state.keys()
        if ('ca_gate' in k) or ('norm_ca_gate' in k) or ('ca_scale' in k)
    ]

    if not gate_keys:
        print('No gate-related parameters found.')
        return

    print(f'Found {len(gate_keys)} gate-related parameters in {args.source}:')
    for k in sorted(gate_keys):
        v = state[k]
        if not torch.is_tensor(v):
            print(f'- {k}: <non-tensor type={type(v)}>')
            continue
        flat = v.detach().reshape(-1)
        sample = flat[:args.max_items].tolist()
        print(f'- {k}: shape={tuple(v.shape)}, dtype={v.dtype}')
        print(f'  values(sample {len(sample)}): {sample}')


if __name__ == '__main__':
    main()