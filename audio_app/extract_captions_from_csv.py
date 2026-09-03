"""
从CSV文件中提取所有caption（无需下载音频文件）
"""

import os
import json
import pandas as pd
from tqdm import tqdm


def extract_captions_from_csv(data_path: str, output_path: str = "candidate_captions.json"):
    """
    从CSV文件中提取所有唯一的caption
    
    Args:
        data_path: 数据集根目录（包含CSV文件）
        output_path: 输出文件路径
    """
    print("=" * 60)
    print("从CSV文件中提取caption...")
    print("=" * 60)
    
    all_captions = set()  # 使用set去重
    
    # 1. 从Clotho CSV提取
    print("\n处理Clotho数据集...")
    clotho_path = os.path.join(data_path, "CLOTHO")
    
    clotho_files = {
        "development": "clotho_captions_development.csv",
        "validation": "clotho_captions_validation.csv",
        "evaluation": "clotho_captions_evaluation.csv"
    }
    
    for subset, filename in clotho_files.items():
        csv_path = os.path.join(clotho_path, filename)
        
        if os.path.exists(csv_path):
            print(f"   - 读取 {filename}...")
            try:
                df = pd.read_csv(csv_path)
                
                # Clotho的caption列名通常是 caption_1, caption_2, ..., caption_5
                caption_cols = [col for col in df.columns if col.startswith('caption_')]
                
                for col in caption_cols:
                    captions = df[col].dropna().astype(str)
                    for cap in captions:
                        if cap.strip():
                            all_captions.add(cap.strip().lower())
                
                print(f"   [OK] {subset}: 累计 {len(all_captions)} 个唯一caption")
            
            except Exception as e:
                print(f"   [WARN] 读取失败: {e}")
        else:
            print(f"   [WARN] 文件不存在: {csv_path}")
    
    # 2. 从AudioCaps CSV提取
    print("\n处理AudioCaps数据集...")
    audiocaps_path = os.path.join(data_path, "AUDIOCAPS")
    
    audiocaps_files = ["train.csv", "val.csv", "test.csv"]
    
    for filename in audiocaps_files:
        csv_path = os.path.join(audiocaps_path, filename)
        
        if os.path.exists(csv_path):
            print(f"   - 读取 {filename}...")
            try:
                df = pd.read_csv(csv_path)
                
                # AudioCaps的caption列名通常是 'caption'
                if 'caption' in df.columns:
                    captions = df['caption'].dropna().astype(str)
                    for cap in captions:
                        if cap.strip():
                            all_captions.add(cap.strip().lower())
                
                print(f"   [OK] {filename}: 累计 {len(all_captions)} 个唯一caption")
            
            except Exception as e:
                print(f"   [WARN] 读取失败: {e}")
        else:
            print(f"   [WARN] 文件不存在: {csv_path}")
    
    # 3. 转换为列表并排序
    caption_list = sorted(list(all_captions))
    
    print("\n" + "=" * 60)
    print(f"提取完成！")
    print(f"   - 总计唯一caption: {len(caption_list)} 个")
    print(f"   - 输出文件: {output_path}")
    print("=" * 60)
    
    # 4. 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(caption_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n已保存到: {output_path}")
    
    # 5. 显示一些示例
    print("\n示例caption (前20个):")
    for i, cap in enumerate(caption_list[:20], 1):
        print(f"   {i}. {cap}")
    
    if len(caption_list) > 20:
        print(f"   ... 还有 {len(caption_list) - 20} 个")
    
    # 6. 统计信息
    print("\n统计信息:")
    print(f"   - 平均长度: {sum(len(cap) for cap in caption_list) / len(caption_list):.1f} 字符")
    print(f"   - 最短: {min(len(cap) for cap in caption_list)} 字符")
    print(f"   - 最长: {max(len(cap) for cap in caption_list)} 字符")
    
    return caption_list


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从CSV文件中提取caption")
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data',
        help='数据集根目录路径（包含CLOTHO和AUDIOCAPS文件夹）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='candidate_captions.json',
        help='输出文件名'
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    data_path = os.path.abspath(args.data_path)
    
    if not os.path.exists(data_path):
        print(f"[ERROR] 数据集路径不存在: {data_path}")
        print(f"提示: 请创建以下目录结构:")
        print(f"   {data_path}/")
        print(f"   +-- CLOTHO/")
        print(f"   |   +-- clotho_captions_development.csv")
        print(f"   |   +-- clotho_captions_validation.csv")
        print(f"   |   +-- clotho_captions_evaluation.csv")
        print(f"   +-- AUDIOCAPS/")
        print(f"       +-- train.csv")
        print(f"       +-- val.csv")
        print(f"       +-- test.csv")
        exit(1)
    
    print(f"数据集路径: {data_path}")
    
    # 检查必要的文件
    print("\n检查CSV文件...")
    clotho_path = os.path.join(data_path, "CLOTHO")
    audiocaps_path = os.path.join(data_path, "AUDIOCAPS")
    
    required_files = [
        os.path.join(clotho_path, "clotho_captions_development.csv"),
        os.path.join(clotho_path, "clotho_captions_validation.csv"),
        os.path.join(clotho_path, "clotho_captions_evaluation.csv"),
        os.path.join(audiocaps_path, "train.csv"),
        os.path.join(audiocaps_path, "val.csv"),
        os.path.join(audiocaps_path, "test.csv"),
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"[WARN] 缺少以下文件:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"提示: 至少需要一个CSV文件才能提取caption")
    
    captions = extract_captions_from_csv(data_path, args.output)
    
    print("\n[SUCCESS] 完成！现在可以重启应用使用新的caption库了。")
