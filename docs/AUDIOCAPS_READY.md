# ✅ AudioCaps数据集配置完成检查清单

## 📁 你的文件结构（正确✅）

```
data/
└── AUDIOCAPS/
    ├── test.csv
    ├── train.csv
    ├── val.csv
    └── audio_32000Hz/
        ├── test.csv
        ├── train.csv
        ├── val.csv
        ├── test/
        │   └── *.mp3 (音频文件)
        ├── train/
        │   └── *.mp3 (音频文件)
        └── val/
            └── *.mp3 (音频文件)
```

**这个结构是正确的！** `aac_datasets`库会自动识别这个结构。

---

## ✅ 已完成的代码修改

### 修改1：禁用自动下载
```python
# 已注释掉自动下载代码
# download_clotho(args["data_path"])
# download_audiocaps(args["data_path"])
```

### 修改2：添加数据集验证
```python
# 启动时会检查AudioCaps是否存在
if args['audiocaps']:
    audiocaps_path = os.path.join(args["data_path"], "AUDIOCAPS")
    if not os.path.exists(audiocaps_path):
        raise FileNotFoundError(f"❌ AudioCaps数据集未找到！")
    else:
        print(f"✅ 找到AudioCaps数据集")
```

### 修改3：禁用AudioCaps自动下载
```python
ac = custom_loading(
    AudioCaps(
        subset="train", 
        root=args["data_path"], 
        download=False,  # ✅ 禁用自动下载
        download_audio=False, 
        audio_format='mp3'
    )
)
```

### 修改4：添加详细日志
```python
# 会显示每个数据集的加载情况
✅ Clotho dev: 3800 样本
✅ AudioCaps train: 49838 样本
✅ 合并后总训练样本: 53638
```

---

## 🚀 开始训练

### 命令1：基础训练（推荐）
```bash
python train.py \
  --audiocaps \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --max_epochs 60 \
  --warmup_epochs 5 \
  --max_lr 2e-5 \
  --min_lr 1e-7 \
  --weight_decay 0.01 \
  --initial_tau 0.07 \
  --tau_trainable \
  --use_improved_projection \
  --no-use_cross_attention \
  --use_multi_layer_text \
  --use_ema \
  --ema_decay 0.999 \
  --use_layerwise_lr \
  --use_improved_schedule \
  --loss_type improved_infonce \
  --hard_negative_weight 0.15
```

### 命令2：快速测试（验证数据集是否正确）
```bash
python train.py \
  --audiocaps \
  --batch_size 8 \
  --max_epochs 2 \
  --no-logging
```

---

## 📊 预期输出

### 启动时应该看到：
```
Data path: data
Contents of data path: ['AUDIOCAPS', 'CLOTHO', ...]
✅ 找到AudioCaps数据集: data/AUDIOCAPS
==================================================
📊 加载训练数据集...
==================================================
✅ Clotho dev: 3800 样本
📥 加载AudioCaps数据集...
✅ AudioCaps train: 49838 样本
✅ 合并后总训练样本: 53638
✅ Clotho val: 1045 样本
==================================================
```

### 训练开始后应该看到：
```
Epoch 1/60
train/loss: 4.2
train/tau: 0.07
train/lr: 1.5e-6
...
```

---

## ⚠️ 可能的问题和解决方案

### 问题1：找不到AudioCaps数据集
**错误信息：**
```
❌ AudioCaps数据集未找到！请确保已解压到: data/AUDIOCAPS
```

**解决方案：**
```bash
# 检查目录是否存在
ls -la data/AUDIOCAPS

# 如果不存在，检查是否在其他位置
find data -name "AUDIOCAPS" -type d

# 确保目录名是大写的AUDIOCAPS
```

### 问题2：AudioCaps加载失败
**错误信息：**
```
❌ AudioCaps加载失败: [某个错误]
```

**可能原因和解决：**

**原因A：CSV文件缺失**
```bash
# 检查CSV文件
ls data/AUDIOCAPS/*.csv
ls data/AUDIOCAPS/audio_32000Hz/*.csv

# 应该看到：
# data/AUDIOCAPS/train.csv
# data/AUDIOCAPS/val.csv
# data/AUDIOCAPS/test.csv
# data/AUDIOCAPS/audio_32000Hz/train.csv
# data/AUDIOCAPS/audio_32000Hz/val.csv
# data/AUDIOCAPS/audio_32000Hz/test.csv
```

**原因B：音频文件缺失**
```bash
# 检查音频文件数量
ls data/AUDIOCAPS/audio_32000Hz/train/ | wc -l
# 应该显示接近 49838

ls data/AUDIOCAPS/audio_32000Hz/val/ | wc -l
# 应该显示接近 494

ls data/AUDIOCAPS/audio_32000Hz/test/ | wc -l
# 应该显示接近 957
```

**原因C：文件权限问题**
```bash
# 修改权限
chmod -R 755 data/AUDIOCAPS/
```

### 问题3：样本数量不对
**如果显示的样本数量明显少于预期：**

```bash
# 检查是否有损坏的音频文件
cd data/AUDIOCAPS/audio_32000Hz/train/
for f in *.mp3; do
    ffmpeg -v error -i "$f" -f null - 2>&1 | grep -q error && echo "损坏: $f"
done
```

### 问题4：显存不足
**错误信息：**
```
CUDA out of memory
```

**解决方案：**
```bash
# 减小batch size
python train.py \
  --audiocaps \
  --batch_size 16 \
  --accumulate_grad_batches 4  # 保持有效batch=64
```

---

## 🔍 验证数据集是否正确加载

### 方法1：运行快速测试
```bash
python train.py \
  --audiocaps \
  --batch_size 8 \
  --max_epochs 1 \
  --no-logging
```

如果能正常运行1个epoch，说明数据集配置正确！

### 方法2：Python脚本测试
```python
# test_audiocaps.py
from aac_datasets import AudioCaps
from d25_t6.datasets.audio_loading import custom_loading

try:
    ac = custom_loading(
        AudioCaps(
            subset="train", 
            root="data", 
            download=False, 
            audio_format='mp3'
        )
    )
    print(f"✅ AudioCaps加载成功！")
    print(f"   训练样本数: {len(ac)}")
    print(f"   第一个样本: {ac[0]}")
except Exception as e:
    print(f"❌ 加载失败: {e}")
```

运行：
```bash
python test_audiocaps.py
```

---

## 📈 预期训练效果

### 只用Clotho（之前）
- 训练样本：3,800
- Epoch 30 mAP@10：26-30%

### Clotho + AudioCaps（现在）
- 训练样本：53,638 ✨
- Epoch 30 mAP@10：**33-37%** ✨
- **预期提升：+5-7个百分点**

### 训练时间对比
| 配置 | 每个Epoch | 总时间(60 epochs) |
|------|----------|------------------|
| 只用Clotho | 2-3分钟 | 2-3小时 |
| Clotho + AudioCaps | 15-20分钟 | **15-20小时** |

---

## 💡 训练建议

### 1. 先快速验证（5分钟）
```bash
python train.py --audiocaps --batch_size 8 --max_epochs 2 --no-logging
```

### 2. 如果验证成功，开始完整训练
```bash
python train.py \
  --audiocaps \
  --batch_size 32 \
  --max_epochs 60 \
  --max_lr 2e-5 \
  --use_improved_projection \
  --use_ema \
  --loss_type improved_infonce \
  --hard_negative_weight 0.15
```

### 3. 使用nohup后台运行（推荐）
```bash
nohup python train.py \
  --audiocaps \
  --batch_size 32 \
  --max_epochs 60 \
  --max_lr 2e-5 \
  > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 4. 使用screen保持会话
```bash
# 创建新会话
screen -S training

# 运行训练
python train.py --audiocaps ...

# 断开会话（Ctrl+A, D）
# 重新连接
screen -r training
```

---

## 🎯 成功标志

如果看到以下输出，说明一切正常：

```
✅ 找到AudioCaps数据集: data/AUDIOCAPS
✅ Clotho dev: 3800 样本
✅ AudioCaps train: 49838 样本
✅ 合并后总训练样本: 53638
✅ Clotho val: 1045 样本

Epoch 1/60: 100%|████████| 1676/1676 [15:23<00:00]
train/loss: 4.123
val/mAP@10: 0.182
```

恭喜！你已经成功配置AudioCaps数据集！🎉

现在可以开始训练，预期在Epoch 30-40时达到35%+ mAP@10！

