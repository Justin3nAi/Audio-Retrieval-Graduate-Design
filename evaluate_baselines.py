"""
evaluate_baselines.py
用途：在 Clotho 测试集上评估多个 checkpoint，输出 Chapter 6 所需的全部指标。
用法：
    python evaluate_baselines.py \
        --passt_ckpt path/to/passt_only.ckpt \
        --clap_ckpt path/to/clap_only.ckpt \
        --simple_fusion_ckpt path/to/simple_fusion.ckpt \
        --final_ckpt path/to/final_system.ckpt \
        --data_root path/to/clotho
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from d25_t6.retrieval_module import AudioRetrievalModel


def evaluate_checkpoint(ckpt_path: str, dataloader, device='cuda') -> dict:
    """
    在给定 dataloader 上评估单个 checkpoint。
    返回包含 mAP@10, R@1, R@5, R@10 的字典。
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_path}")
    print(f"{'='*60}")

    model = AudioRetrievalModel.load_from_checkpoint(
        ckpt_path, map_location='cpu', strict=False
    )
    model.eval()
    model = model.to(device)

    all_audio_embs = []
    all_text_embs  = []
    all_paths      = []
    all_captions   = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            audio_emb, text_emb = model.forward(batch)
            all_audio_embs.append(audio_emb.detach().cpu())
            all_text_embs.append(text_emb.detach().cpu())
            all_paths.extend(batch['fname'])
            all_captions.extend([c[0] for c in batch['captions']])

    paths    = np.array(all_paths)
    audio_embs = torch.cat(all_audio_embs)   # (N, D)
    text_embs  = torch.cat(all_text_embs)    # (N, D)

    # --- deduplicate audio (Clotho: 5 captions per clip) ---
    target, select = [], []
    first_occurrence = {}
    for i, p in enumerate(paths):
        idx = first_occurrence.get(p)
        if idx is None:
            idx = len(first_occurrence)
            first_occurrence[p] = idx
            select.append(i)
        target.append(idx)

    audio_embs_unique = audio_embs[select]   # unique clips
    target = np.array(target)

    # --- similarity matrix ---
    C = torch.matmul(text_embs, audio_embs_unique.T)   # (N_text, N_audio)
    top10 = C.topk(10, dim=1)[1].detach().cpu().numpy()

    # --- metrics ---
    r1  = (top10[:, :1]  == target[:, None]).any(axis=1).mean()
    r5  = (top10[:, :5]  == target[:, None]).any(axis=1).mean()
    r10 = (top10         == target[:, None]).any(axis=1).mean()

    hit = (top10 == target[:, None])
    AP  = np.where(hit.any(axis=1), 1.0 / (hit.argmax(axis=1) + 1), 0.0)
    mAP = AP.mean()

    results = {
        'mAP@10': round(float(mAP), 4),
        'R@1':    round(float(r1),  4),
        'R@5':    round(float(r5),  4),
        'R@10':   round(float(r10), 4),
    }
    print(f"  mAP@10 = {results['mAP@10']:.4f}")
    print(f"  R@1    = {results['R@1']:.4f}")
    print(f"  R@5    = {results['R@5']:.4f}")
    print(f"  R@10   = {results['R@10']:.4f}")
    return results


def compute_improvements(results: dict, baseline_key='PaSST-only') -> None:
    """
    输出 Final system 相对于各个基线的百分比改进。
    """
    final = results.get('Final system')
    if final is None:
        return
    print(f"\n{'='*60}")
    print("Improvement of Final system over baselines:")
    print(f"{'='*60}")
    for name, metrics in results.items():
        if name == 'Final system':
            continue
        for metric in ['mAP@10', 'R@1', 'R@5', 'R@10']:
            base_val = metrics[metric]
            final_val = final[metric]
            if base_val > 0:
                delta = (final_val - base_val) / base_val * 100
                print(f"  {metric} vs {name}: {delta:+.1f}%")


def generate_latex_table(results: dict) -> str:
    """
    自动生成 Chapter 6 Table 2 的 LaTeX 代码。
    """
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Performance comparison on Clotho evaluation set}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{mAP@10} & \textbf{R@1} & \textbf{R@5} & \textbf{R@10} \\\\",
        r"\midrule",
    ]

    final = results.get('Final system', {})
    for name, metrics in results.items():
        bold = name == 'Final system'
        def fmt(v):
            s = f"{v:.3f}"
            return f"\\textbf{{{s}}}" if bold else s
        lines.append(
            f"{name} & {fmt(metrics['mAP@10'])} & {fmt(metrics['R@1'])} & "
            f"{fmt(metrics['R@5'])} & {fmt(metrics['R@10'])} \\\\"
        )

    lines.append(r"\midrule")
    # Improvement rows
    for name, metrics in results.items():
        if name == 'Final system':
            continue
        base_map = metrics['mAP@10']
        if base_map > 0:
            delta = (final['mAP@10'] - base_map) / base_map * 100
            lines.append(
                f"$\\Delta$ vs {name} & {delta:+.1f}\\% & - & - & - \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Chapter 6 baselines.")
    parser.add_argument('--passt_ckpt',         type=str, default=None, help='PaSST-only checkpoint path')
    parser.add_argument('--clap_ckpt',           type=str, default=None, help='CLAP-only checkpoint path')
    parser.add_argument('--simple_fusion_ckpt',  type=str, default=None, help='Simple fusion checkpoint path')
    parser.add_argument('--final_ckpt',          type=str, required=True, help='Final system checkpoint path')
    parser.add_argument('--data_root',           type=str, required=True, help='Path to Clotho dataset root')
    parser.add_argument('--batch_size',          type=int, default=32)
    parser.add_argument('--output_json',         type=str, default='chapter6_results.json')
    parser.add_argument('--device',              type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # --- Build dataloader for Clotho evaluation split ---
    # 你需要根据你的数据集类调整这里
    # 示例：假设你有一个 ClothoDataset 类
    try:
        from d25_t6.datasets.clotho_dataset import ClothoDataset
        eval_dataset = ClothoDataset(
            data_root=args.data_root,
            split='evaluation',
            sr=32000
        )
    except ImportError:
        # 如果没有 ClothoDataset，使用 train.py 中的数据加载方式
        print("Cannot import ClothoDataset, please adjust the dataloader section.")
        raise

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_dataset.collate_fn if hasattr(eval_dataset, 'collate_fn') else None
    )

    # --- Evaluate each checkpoint ---
    results = {}

    ckpt_map = {
        'PaSST-only':     args.passt_ckpt,
        'CLAP-only':      args.clap_ckpt,
        'Simple fusion':  args.simple_fusion_ckpt,
        'Final system':   args.final_ckpt,
    }

    for name, ckpt_path in ckpt_map.items():
        if ckpt_path is None:
            print(f"Skipping {name} (no checkpoint provided)")
            continue
        results[name] = evaluate_checkpoint(ckpt_path, eval_loader, device=args.device)

    # --- Print improvements ---
    compute_improvements(results)

    # --- Generate LaTeX table ---
    print("\n" + "="*60)
    print("LaTeX Table 2 (copy into chapter06.tex):")
    print("="*60)
    latex = generate_latex_table(results)
    print(latex)

    # --- Save to JSON ---
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_json}")
