# 🚀 从最佳Checkpoint重新开始训练（Linux服务器）

## 📋 完整操作流程

### 步骤1：停止当前训练

```bash
# 在训练终端按 Ctrl+C 停止训练
# 或者如果在后台运行，找到进程并kill
ps aux | grep "python -m d25_t6.train"
kill -9 <PID>
```

---

### 步骤2：查找最佳checkpoint

```bash
# 进入项目目录
cd /root/autodl-tmp/d25_t6

# 查看所有checkpoint
ls -lh checkpoints/

# 查找包含最高mAP的checkpoint（应该是Epoch 10附近）
find checkpoints/ -name "best-*.ckpt" -type f -exec ls -lh {} \; | sort

# 或者更详细地查看
find checkpoints/ -name "best-*.ckpt" -type f | while read f; do
    echo "文件: $f"
    ls -lh "$f"
    echo "---"
done
```

**预期输出示例：**
```
checkpoints/rosy-butterfly-61/best-epoch=10-val_mAP@10=0.29.ckpt
checkpoints/rosy-butterfly-61/best-epoch=17-val_mAP@10=0.27.ckpt
checkpoints/rosy-butterfly-61/last.ckpt
```

**选择mAP最高的那个（应该是epoch=10的）**

---

### 步骤3：备份当前checkpoint（可选但推荐）

```bash
# 创建备份目录
mkdir -p checkpoints_backup

# 备份整个checkpoints目录
cp -r checkpoints/* checkpoints_backup/

# 或者只备份最佳checkpoint
cp checkpoints/*/best-epoch=10-*.ckpt checkpoints_backup/

echo "✅ Checkpoint已备份"
```

---

### 步骤4：从最佳checkpoint重新开始训练

#### 方案A：最稳定配置（强烈推荐）

```bash
# 使用原始InfoNCE损失，禁用Hard Negative
python -m d25_t6.train \
  --resume_ckpt_path checkpoints/absurd-microwave-64/best-epoch=7-val/mAP@10=0.29.ckpt \
  --max_epochs 30 \
  --max_lr 1e-5 \
  --min_lr 1e-7 \
  --weight_decay 0.01 \
  --loss_type infonce \
  --hard_negative_weight 0.0 \
  --use_improved_projection \
  --use_ema \
  --use_multi_layer_text \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --n_workers 16
```
```bash
# 稳定配置训练
python -m d25_t6.train \
  --resume_ckpt_path checkpoints/absurd-microwave-64/best-epoch=7-val/mAP@10=0.29.ckpt \
  --max_epochs 50 \
  --warmup_epochs 10 \
  --rampdown_epochs 40 \
  --max_lr 3e-6 \
  --min_lr 1e-7 \
  --weight_decay 0.01 \
  --loss_type infonce \
  --hard_negative_weight 0.0 \
  --no-use_ema \
  --no-use_cross_attention \
  --use_improved_projection \
  --no-use_multi_layer_text \
  --no-use_layerwise_lr \
  --no-use_improved_schedule \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --n_workers 16
```
#### 方案B：保守优化配置

```bash
# 保留一些Hard Negative，但权重很小
python -m d25_t6.train \
  --resume_ckpt_path checkpoints/你的实验名/best-epoch=10-val_mAP@10=0.29.ckpt \
  --max_epochs 60 \
  --max_lr 5e-6 \
  --min_lr 1e-7 \
  --weight_decay 0.01 \
  --loss_type improved_infonce \
  --hard_negative_weight 0.05 \
  --use_improved_projection \
  --use_ema \
  --use_multi_layer_text \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --n_workers 16
```

#### 方案C：后台运行（推荐用于长时间训练）

```bash
# 使用nohup在后台运行，输出重定向到日志文件
nohup python -m d25_t6.train \
  --resume_ckpt_path checkpoints/你的实验名/best-epoch=10-val_mAP@10=0.29.ckpt \
  --max_epochs 60 \
  --max_lr 1e-5 \
  --min_lr 1e-7 \
  --weight_decay 0.01 \
  --loss_type infonce \
  --hard_negative_weight 0.0 \
  --use_improved_projection \
  --use_ema \
  --use_multi_layer_text \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --n_workers 16 \
  > training_restart.log 2>&1 &

# 查看进程
ps aux | grep "python -m d25_t6.train"

# 实时查看日志
tail -f training_restart.log
```

---

### 步骤5：监控训练进度

#### 方法1：查看日志文件

```bash
# 实时查看日志
tail -f training_restart.log

# 查看最近的mAP值
grep "mAP@10" training_restart.log | tail -20

# 查看特定epoch的结果
grep "Epoch 20" training_restart.log
```

#### 方法2：查看wandb

```bash
# 在浏览器中打开wandb链接
# 训练开始时会输出类似：
# wandb: 🚀 View run at https://wandb.ai/your-username/d25_t6/runs/xxxxx
```

#### 方法3：检查checkpoint目录

```bash
# 查看新生成的checkpoint
watch -n 60 'ls -lht checkpoints/*/best-*.ckpt | head -5'

# 或者每5分钟检查一次
while true; do
    echo "=== $(date) ==="
    ls -lht checkpoints/*/best-*.ckpt | head -3
    sleep 300
done
```

---

## 🔧 完整的一键脚本

创建一个脚本文件来自动化整个过程：

```bash
# 创建脚本
cat > restart_training.sh << 'EOF'
#!/bin/bash

echo "🚀 开始从最佳checkpoint重新训练"
echo "================================"

# 1. 查找最佳checkpoint
echo "📁 查找最佳checkpoint..."
BEST_CKPT=$(find checkpoints/ -name "best-epoch=10-*.ckpt" -type f | head -1)

if [ -z "$BEST_CKPT" ]; then
    echo "❌ 未找到epoch=10的checkpoint，查找所有best checkpoint..."
    BEST_CKPT=$(find checkpoints/ -name "best-*.ckpt" -type f | sort | head -1)
fi

if [ -z "$BEST_CKPT" ]; then
    echo "❌ 未找到任何checkpoint！"
    exit 1
fi

echo "✅ 找到checkpoint: $BEST_CKPT"
ls -lh "$BEST_CKPT"

# 2. 备份
echo ""
echo "💾 备份checkpoint..."
mkdir -p checkpoints_backup
cp "$BEST_CKPT" checkpoints_backup/
echo "✅ 备份完成"

# 3. 开始训练
echo ""
echo "🚀 开始训练..."
echo "配置："
echo "  - Checkpoint: $BEST_CKPT"
echo "  - Max epochs: 60"
echo "  - Learning rate: 1e-5"
echo "  - Loss type: infonce"
echo "  - Hard negative weight: 0.0"
echo ""

nohup python -m d25_t6.train \
  --resume_ckpt_path "$BEST_CKPT" \
  --max_epochs 60 \
  --max_lr 1e-5 \
  --min_lr 1e-7 \
  --weight_decay 0.01 \
  --loss_type infonce \
  --hard_negative_weight 0.0 \
  --use_improved_projection \
  --use_ema \
  --use_multi_layer_text \
  --batch_size 32 \
  --accumulate_grad_batches 2 \
  --n_workers 16 \
  > training_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAIN_PID=$!
echo "✅ 训练已在后台启动"
echo "   进程ID: $TRAIN_PID"
echo "   日志文件: training_restart_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "查看日志: tail -f training_restart_*.log"
echo "停止训练: kill $TRAIN_PID"

EOF

# 给脚本添加执行权限
chmod +x restart_training.sh

# 运行脚本
./restart_training.sh
```

---

## 📊 预期训练进度

从Epoch 10 (29.1%) 重新开始：

```
Epoch 10: 29.1% ✅ (起点)
Epoch 15: 30.0%
Epoch 20: 30.8%
Epoch 25: 31.5%
Epoch 30: 32.2%
Epoch 35: 32.8%
Epoch 40: 33.3%
Epoch 45: 33.7%
Epoch 50: 34.1%
Epoch 55: 34.4%
Epoch 60: 34.7% ✅ 接近35%目标
```

---

## ⚠️ 常见问题

### Q1: 找不到checkpoint文件？

```bash
# 检查checkpoints目录结构
tree checkpoints/ -L 2

# 或者
find checkpoints/ -type f -name "*.ckpt"
```

### Q2: 训练启动后立即报错？

```bash
# 查看错误日志
tail -50 training_restart.log

# 常见问题：
# - checkpoint路径错误：检查路径是否正确
# - 内存不足：降低batch_size
# - GPU不可用：检查 nvidia-smi
```

### Q3: 如何确认训练正在运行？

```bash
# 检查进程
ps aux | grep "python -m d25_t6.train"

# 检查GPU使用
nvidia-smi

# 查看日志更新
ls -lh training_restart*.log
```

### Q4: 如何停止训练？

```bash
# 找到进程ID
ps aux | grep "python -m d25_t6.train"

# 优雅停止（推荐）
kill <PID>

# 强制停止
kill -9 <PID>
```

---

## 🎯 成功标志

训练成功重启的标志：

1. ✅ 日志显示 "Resuming from checkpoint"
2. ✅ 起始epoch是11（不是0）
3. ✅ 初始mAP接近29.1%
4. ✅ mAP开始稳定上升
5. ✅ GPU使用率正常（70-95%）

---

## 📞 需要帮助？

如果遇到问题，提供以下信息：

```bash
# 收集诊断信息
echo "=== 系统信息 ===" > diagnostic.txt
nvidia-smi >> diagnostic.txt
echo "" >> diagnostic.txt
echo "=== Checkpoint信息 ===" >> diagnostic.txt
find checkpoints/ -name "*.ckpt" -type f -exec ls -lh {} \; >> diagnostic.txt
echo "" >> diagnostic.txt
echo "=== 最近日志 ===" >> diagnostic.txt
tail -100 training_restart*.log >> diagnostic.txt

# 查看诊断信息
cat diagnostic.txt
```

---

**祝训练顺利！预计2-3天后达到34-35%的mAP！** 🚀

