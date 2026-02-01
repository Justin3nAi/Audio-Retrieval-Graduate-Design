# 📥 AudioCaps数据集导入完整指南

## 方法1：自动下载（推荐）✅

我已经帮你启用了自动下载功能。现在只需运行：

```bash
python train.py \
  --audiocaps \
  --data_path data \
  --batch_size 32 \
  --max_epochs 60
```

**会自动执行：**
1. 下载AudioCaps数据集（~2GB）
2. 自动解压到 `data/AUDIOCAPS/` 目录
3. 与Clotho数据集合并训练

**下载时间：**
- 网速好：10-20分钟
- 网速一般：30-60分钟

---

## 方法2：手动下载（备用）

如果自动下载失败，可以手动下载：

### 步骤1：下载数据集

```bash
# 在服务器上执行
cd data
wget https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip
```

或者使用curl：
```bash
curl -L -o AUDIOCAPS.zip https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip
```

### 步骤2：解压

```bash
# 需要安装7z
apt-get install p7zip-full  # Ubuntu/Debian
# 或
yum install p7zip  # CentOS

# 解压
7z x AUDIOCAPS.zip -oAUDIOCAPS
```

### 步骤3：验证

```bash
ls -lh data/AUDIOCAPS/
# 应该看到：
# - train/  (训练集音频)
# - val/    (验证集音频)
# - test/   (测试集音频)
# - *.csv   (标注文件)
```

---

## 方法3：从其他来源下载

如果上面的链接不可用，可以从官方源下载：

### AudioCaps官方链接
- GitHub: https://github.com/cdjkim/audiocaps
- 需要从YouTube下载音频（较慢）

### 使用aac-datasets库自动下载
```python
from aac_datasets import AudioCaps

# 会自动下载和处理
dataset = AudioCaps(
    root="data",
    subset="train",
    download=True,
    download_audio=True,  # 自动下载音频
    audio_format='mp3'
)
```

---

## 📊 数据集信息

### AudioCaps统计
- **训练集**: 49,838个音频-文本对
- **验证集**: 494个音频-文本对
- **测试集**: 957个音频-文本对
- **音频格式**: MP3
- **总大小**: ~2GB

### 目录结构
```
data/
├── AUDIOCAPS/
│   ├── train/
│   │   ├── Y0001.mp3
│   │   ├── Y0002.mp3
│   │   └── ...
│   ├── val/
│   ├── test/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── CLOTHO/
    └── ...
```

---

## 🚀 训练命令

### 基础训练（Clotho + AudioCaps）
```bash
python train.py \
  --audiocaps \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --max_epochs 60 \
  --warmup_epochs 5 \
  --max_lr 2e-5 \
  --weight_decay 0.01 \
  --tau_trainable \
  --use_improved_projection \
  --no-use_cross_attention \
  --use_multi_layer_text \
  --use_ema \
  --loss_type improved_infonce \
  --hard_negative_weight 0.15
```

### 优化训练（更高学习率）
```bash
python train.py \
  --audiocaps \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --max_epochs 70 \
  --warmup_epochs 8 \
  --max_lr 3e-5 \
  --weight_decay 0.01 \
  --tau_trainable \
  --use_improved_projection \
  --no-use_cross_attention \
  --use_multi_layer_text \
  --use_ema \
  --loss_type improved_infonce \
  --hard_negative_weight 0.2
```

---

## ⚠️ 常见问题

### 问题1：下载速度慢
**解决方案：**
```bash
# 使用代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 或使用aria2加速下载
aria2c -x 16 -s 16 https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip
```

### 问题2：解压失败
**解决方案：**
```bash
# 检查文件完整性
md5sum AUDIOCAPS.zip

# 重新下载
rm AUDIOCAPS.zip
wget https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip

# 使用unzip代替7z
unzip AUDIOCAPS.zip -d AUDIOCAPS
```

### 问题3：显存不足
**解决方案：**
```bash
# 减小batch size
--batch_size 16 \
--accumulate_grad_batches 4  # 保持有效batch=64
```

### 问题4：训练时间太长
**解决方案：**
```bash
# 减少epoch
--max_epochs 40

# 或先用小数据集验证
--audiocaps \
--max_epochs 10  # 快速验证
```

---

## 📈 预期性能提升

### 只用Clotho
- 训练样本：3,800
- 训练时间：2-3小时
- 预期mAP@10：30-33%

### Clotho + AudioCaps
- 训练样本：53,638
- 训练时间：8-12小时
- 预期mAP@10：35-40% ✨
- **提升：+5-7个百分点**

---

## 🔍 验证数据集是否正确加载

### 方法1：检查训练日志
训练开始时应该看到：
```
Loading Clotho dev: 3800 samples
Loading AudioCaps train: 49838 samples
Total training samples: 53638
```

### 方法2：运行测试脚本
```python
from aac_datasets import AudioCaps
from d25_t6.datasets.audio_loading import custom_loading

# 测试加载
ac = custom_loading(
    AudioCaps(subset="train", root="data", download=False, audio_format='mp3')
)
print(f"AudioCaps samples: {len(ac)}")
# 应该输出: AudioCaps samples: 49838
```

---

## 💾 磁盘空间要求

| 数据集 | 压缩包大小 | 解压后大小 | 总需求 |
|--------|-----------|-----------|--------|
| Clotho | ~1GB | ~2GB | ~3GB |
| AudioCaps | ~2GB | ~4GB | ~6GB |
| **总计** | **~3GB** | **~6GB** | **~9GB** |

**建议：** 至少预留15GB空间（包括模型checkpoint）

---

## 🎯 下一步

1. **启动下载和训练：**
```bash
python train.py --audiocaps --max_epochs 60
```

2. **监控训练：**
- 查看wandb面板
- 观察mAP@10是否上升

3. **预期结果：**
- Epoch 10: mAP@10 ≈ 28-30%
- Epoch 30: mAP@10 ≈ 33-36%
- Epoch 60: mAP@10 ≈ 36-40%

4. **如果效果好，继续添加WavCaps：**
```bash
python train.py --audiocaps --wavcaps --max_epochs 80
```

祝训练顺利！🚀

