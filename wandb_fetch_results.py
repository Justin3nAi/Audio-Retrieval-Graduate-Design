"""
wandb_fetch_results.py
用途：从 W&B 自动抓取所有训练 run，找出最佳模型，生成 Chapter 6 所需的数据。
安装依赖：pip install wandb pandas
用法：python wandb_fetch_results.py --entity YOUR_WANDB_USERNAME --project YOUR_PROJECT_NAME
"""
import argparse
import json
import pandas as pd
from datetime import datetime

try:
    import wandb
except ImportError:
    print("请先安装 wandb: pip install wandb")
    exit(1)


METRICS = ['val/mAP@10', 'val/R@1', 'val/R@5', 'val/R@10']
KEY_CONFIGS = [
    'use_multi_encoder', 'use_attentive_aggregation', 'use_attention_pooling',
    'use_cross_attention', 'use_moe', 'use_improved_projection',
    'fusion_type', 'roberta_base', 'max_lr', 'batch_size'
]


def fetch_all_runs(entity: str, project: str) -> pd.DataFrame:
    """抓取所有 run 的最终指标和关键配置"""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    records = []
    print(f"Found {len(runs)} runs in {entity}/{project}")
    print("Fetching metrics...")

    for run in runs:
        record = {
            'run_id':    run.id,
            'run_name':  run.name,
            'state':     run.state,
            'created':   run.created_at,
        }
        # 获取 summary（最终指标）
        for metric in METRICS:
            record[metric] = run.summary.get(metric, None)
        # 获取关键配置
        for cfg in KEY_CONFIGS:
            record[f'cfg_{cfg}'] = run.config.get(cfg, None)
        records.append(record)

    df = pd.DataFrame(records)
    # 只保留 finished 的 run
    df = df[df['state'] == 'finished'].copy()
    # 按 mAP@10 降序排列
    df = df.sort_values('val/mAP@10', ascending=False).reset_index(drop=True)
    return df


def identify_configurations(df: pd.DataFrame) -> dict:
    """
    根据配置特征识别4个关键配置：
    - PaSST-only: use_multi_encoder=False, use_attentive_aggregation=False 或 True
    - CLAP-only:  use_multi_encoder=True, 只用 CLAP
    - Simple fusion: use_multi_encoder=True, no attention
    - Final system: 最高 mAP@10 的 run
    """
    results = {}

    # Final system: 最高 mAP@10
    best = df.iloc[0]
    results['Final system'] = {
        'run_id':   best['run_id'],
        'run_name': best['run_name'],
        'mAP@10':   round(float(best['val/mAP@10']), 4) if best['val/mAP@10'] else None,
        'R@1':      round(float(best['val/R@1']),    4) if best['val/R@1']    else None,
        'R@5':      round(float(best['val/R@5']),    4) if best['val/R@5']    else None,
        'R@10':     round(float(best['val/R@10']),   4) if best['val/R@10']   else None,
    }

    # PaSST-only: use_multi_encoder = False 中最好的
    passt_runs = df[df['cfg_use_multi_encoder'] == False]
    if not passt_runs.empty:
        r = passt_runs.iloc[0]
        results['PaSST-only'] = {
            'run_id':   r['run_id'],
            'run_name': r['run_name'],
            'mAP@10':   round(float(r['val/mAP@10']), 4) if r['val/mAP@10'] else None,
            'R@1':      round(float(r['val/R@1']),    4) if r['val/R@1']    else None,
            'R@5':      round(float(r['val/R@5']),    4) if r['val/R@5']    else None,
            'R@10':     round(float(r['val/R@10']),   4) if r['val/R@10']   else None,
        }

    # CLAP-only: use_multi_encoder=True 且 fusion_type 暗示只用 CLAP
    clap_runs = df[
        (df['cfg_use_multi_encoder'] == True) &
        (df['cfg_fusion_type'].astype(str).str.contains('clap', case=False, na=False))
    ]
    if not clap_runs.empty:
        r = clap_runs.iloc[0]
        results['CLAP-only'] = {
            'run_id':   r['run_id'],
            'run_name': r['run_name'],
            'mAP@10':   round(float(r['val/mAP@10']), 4) if r['val/mAP@10'] else None,
            'R@1':      round(float(r['val/R@1']),    4) if r['val/R@1']    else None,
            'R@5':      round(float(r['val/R@5']),    4) if r['val/R@5']    else None,
            'R@10':     round(float(r['val/R@10']),   4) if r['val/R@10']   else None,
        }

    # Simple fusion: use_multi_encoder=True, use_cross_attention=False
    simple_runs = df[
        (df['cfg_use_multi_encoder'] == True) &
        (df['cfg_use_cross_attention'] == False)
    ]
    if not simple_runs.empty:
        r = simple_runs.iloc[0]
        results['Simple fusion'] = {
            'run_id':   r['run_id'],
            'run_name': r['run_name'],
            'mAP@10':   round(float(r['val/mAP@10']), 4) if r['val/mAP@10'] else None,
            'R@1':      round(float(r['val/R@1']),    4) if r['val/R@1']    else None,
            'R@5':      round(float(r['val/R@5']),    4) if r['val/R@5']    else None,
            'R@10':     round(float(r['val/R@10']),   4) if r['val/R@10']   else None,
        }

    return results


def generate_latex_table2(results: dict) -> str:
    """生成 Chapter 6 Table 2 的 LaTeX 代码"""
    order = ['PaSST-only', 'CLAP-only', 'Simple fusion', 'Final system']
    final = results.get('Final system', {})

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
    for name in order:
        if name not in results:
            continue
        m = results[name]
        bold = (name == 'Final system')
        def f(v, b=bold):
            if v is None: return 'N/A'
            s = f"{v:.3f}"
            return f"\\textbf{{{s}}}" if b else s
        lines.append(f"{name} & {f(m['mAP@10'])} & {f(m['R@1'])} & {f(m['R@5'])} & {f(m['R@10'])} \\\\")

    lines.append(r"\midrule")
    for name in order:
        if name == 'Final system' or name not in results:
            continue
        base = results[name]['mAP@10']
        final_val = final.get('mAP@10')
        if base and final_val:
            delta = (final_val - base) / base * 100
            lines.append(f"$\\Delta$ vs {name} & {delta:+.1f}\\% & - & - & - \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_top_runs(df: pd.DataFrame, n=20):
    """打印前 N 个最佳 run，帮助手动识别配置"""
    print(f"\n{'='*80}")
    print(f"Top {n} runs by val/mAP@10:")
    print(f"{'='*80}")
    cols = ['run_name', 'val/mAP@10', 'val/R@1', 'val/R@5', 'val/R@10',
            'cfg_use_multi_encoder', 'cfg_use_cross_attention', 'cfg_use_moe',
            'cfg_use_attentive_aggregation', 'cfg_fusion_type']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(n).to_string(index=True))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity',  type=str, required=True,  help='W&B username or team name')
    parser.add_argument('--project', type=str, required=True,  help='W&B project name')
    parser.add_argument('--top_n',   type=int, default=20,      help='Print top N runs')
    parser.add_argument('--output',  type=str, default='chapter6_wandb_results.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 1. 抓取所有 run
    df = fetch_all_runs(args.entity, args.project)
    print(f"\nTotal finished runs: {len(df)}")

    # 2. 保存完整数据到 CSV
    csv_path = args.output.replace('.json', '_all_runs.csv')
    df.to_csv(csv_path, index=False)
    print(f"All runs saved to: {csv_path}")

    # 3. 打印最佳 run
    print_top_runs(df, n=args.top_n)

    # 4. 自动识别关键配置
    results = identify_configurations(df)

    print(f"\n{'='*80}")
    print("Identified configurations:")
    print(f"{'='*80}")
    for name, data in results.items():
        print(f"\n[{name}]")
        print(f"  Run: {data['run_name']} ({data['run_id']})")
        print(f"  mAP@10 = {data['mAP@10']}")
        print(f"  R@1    = {data['R@1']}")
        print(f"  R@5    = {data['R@5']}")
        print(f"  R@10   = {data['R@10']}")

    # 5. 生成 LaTeX 表格
    print(f"\n{'='*80}")
    print("LaTeX Table 2 for Chapter 6:")
    print(f"{'='*80}")
    print(generate_latex_table2(results))

    # 6. 保存 JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nKey results saved to: {args.output}")
