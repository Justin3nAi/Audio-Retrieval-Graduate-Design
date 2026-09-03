"""
从训练数据集中提取所有caption作为候选库
"""

import os
import sys
import json
import pickle
from tqdm import tqdm

# 添加父目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from aac_datasets import Clotho, AudioCaps
from d25_t6.datasets.audio_loading import custom_loading


def extract_all_captions(data_path: str, output_path: str = "candidate_captions.json"):
    """
    从Clotho和AudioCaps数据集中提取所有唯一的caption
    
    Args:
        data_path: 数据集根目录
        output_path: 输出文件路径
    """
    print("=" * 60)
    print("📝 从训练数据集中提取caption...")
    print("=" * 60)
    
    all_captions = set()  # 使用set去重
    
    # 1. 从Clotho提取
    print("\n📦 加载Clotho数据集...")
    try:
        clotho_subsets = ["dev", "val", "eval"]
        for subset in clotho_subsets:
            print(f"   - 加载Clotho {subset}...")
            ds = Clotho(subset=subset, root=data_path, flat_captions=True)
            
            for i in tqdm(range(len(ds)), desc=f"   提取{subset}"):
                sample = ds[i]
                captions = sample.get('captions', [])
                
                # Clotho每个音频有5个caption
                if isinstance(captions, list):
                    for cap in captions:
                        if isinstance(cap, str) and cap.strip():
                            all_captions.add(cap.strip().lower())
                elif isinstance(captions, str) and captions.strip():
                    all_captions.add(captions.strip().lower())
            
            print(f"   ✅ Clotho {subset}: 累计 {len(all_captions)} 个唯一caption")
    
    except Exception as e:
        print(f"   ⚠️ Clotho加载失败: {e}")
    
    # 2. 从AudioCaps提取
    print("\n📦 加载AudioCaps数据集...")
    try:
        audiocaps_subsets = ["train", "val", "test"]
        for subset in audiocaps_subsets:
            print(f"   - 加载AudioCaps {subset}...")
            try:
                ds = AudioCaps(
                    subset=subset, 
                    root=data_path, 
                    download=False,
                    download_audio=False,
                    audio_format='mp3'
                )
                
                for i in tqdm(range(len(ds)), desc=f"   提取{subset}"):
                    sample = ds[i]
                    captions = sample.get('captions', [])
                    
                    if isinstance(captions, list):
                        for cap in captions:
                            if isinstance(cap, str) and cap.strip():
                                all_captions.add(cap.strip().lower())
                    elif isinstance(captions, str) and captions.strip():
                        all_captions.add(captions.strip().lower())
                
                print(f"   ✅ AudioCaps {subset}: 累计 {len(all_captions)} 个唯一caption")
            
            except Exception as e:
                print(f"   ⚠️ AudioCaps {subset}加载失败: {e}")
    
    except Exception as e:
        print(f"   ⚠️ AudioCaps加载失败: {e}")
    
    # 3. 转换为列表并排序
    caption_list = sorted(list(all_captions))
    
    print("\n" + "=" * 60)
    print(f"✅ 提取完成！")
    print(f"   - 总计唯一caption: {len(caption_list)} 个")
    print(f"   - 输出文件: {output_path}")
    print("=" * 60)
    
    # 4. 保存为JSON
    output_file = os.path.join(script_dir, output_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(caption_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 已保存到: {output_file}")
    
    # 5. 显示一些示例
    print("\n📄 示例caption (前20个):")
    for i, cap in enumerate(caption_list[:20], 1):
        print(f"   {i}. {cap}")
    
    if len(caption_list) > 20:
        print(f"   ... 还有 {len(caption_list) - 20} 个")
    
    return caption_list


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从数据集中提取caption")
    parser.add_argument(
        '--data_path',
        type=str,
        default='../data',
        help='数据集根目录路径'
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
        print(f"❌ 错误: 数据集路径不存在: {data_path}")
        exit(1)
    
    print(f"📁 数据集路径: {data_path}")
    
    captions = extract_all_captions(data_path, args.output)
    
    print("\n✅ 完成！现在可以重启应用使用新的caption库了。")
