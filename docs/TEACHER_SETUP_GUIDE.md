# 🔧 预训练教师模型依赖安装指南

## 📦 需要安装的依赖

### 在服务器上执行

```bash
ssh root@your-server

# 激活环境
conda activate d25_t6

# 安装 transformers（用于 CLAP）
pip install transformers

# 安装音频处理库
pip install librosa soundfile

# 如果需要 AudioCLIP（可选）
pip install audioclip
```

---

## 🚀 上传并测试

### 步骤1：上传新文件

```bash
scp pretrained_teacher_loader.py root@your-server:/root/autodl-tmp/ProjectAR/d25_t6/
```

### 步骤2：测试教师模型加载

```bash
ssh root@your-server
cd /root/autodl-tmp/ProjectAR/d25_t6

# 测试 CLAP 加载
python -c "
from pretrained_teacher_loader import load_pretrained_teachers
import torch

configs = [
    {'type': 'clap', 'path': '/root/autodl-tmp/teacher_models/clap/clap-htsat-unfused'},
]

teachers = load_pretrained_teachers(configs)
print(f'Loaded {len(teachers)} teachers')

if len(teachers) > 0:
    batch = {
        'audio': torch.randn(2, 1, 32000),
        'caption': ['a dog barking', 'a cat meowing']
    }
    audio_emb, text_emb = teachers[0](batch)
    print(f'Audio embedding: {audio_emb.shape}')
    print(f'Text embedding: {text_emb.shape}')
    print('SUCCESS!')
"
```

### 步骤3：如果测试成功，启动训练

```bash
cd /root
python -m d25_t6.train
```

---

## ⚠️ 可能的问题和解决方案

### 问题1：transformers 安装失败

```bash
pip install --upgrade pip
pip install transformers --no-cache-dir
```

### 问题2：CLAP 模型加载失败

**症状**: "Error loading CLAP: ..."

**解决**: 
- 检查模型路径是否正确
- 确保模型文件完整（config.json, pytorch_model.bin等）

### 问题3：音频格式不匹配

**症状**: "CLAP forward error: ..."

**解决**: 
- 当前代码已经处理了音频格式转换
- 如果还有问题，检查采样率是否正确

---

## 📊 预期输出

### 成功加载时

```
============================================================
Loading PRETRAINED teacher models for ensemble distillation...
============================================================
Loading teacher 1: audioclip
  Loading audioclip from .../AudioCLIP-Full-Training.pt
     Checkpoint keys: ['state_dict', 'optimizer', ...]...
     SUCCESS
Loading teacher 2: clap
  Loading clap from .../clap-htsat-unfused
     SUCCESS
Loading teacher 3: clap
  Loading clap from .../clap-larger
     SUCCESS
Loading teacher 4: beats
  Loading beats from .../BEATs_iter3_plus_AS2M.pt
     SUCCESS
============================================================
Pretrained ensemble distillation enabled
   Teachers: 4
   Weight: 0.5
   Temperature: 2.0
============================================================
```

### 训练时

```
Epoch 0:   2%|██  | 64/2822 [02:06<1:30:55, 0.51it/s, 
    v_num=xxxx, 
    train/loss=2.150,
    train/ensemble_distill_loss=0.280]
```

**不应该再看到** "⚠️ 教师模型前向传播失败" 的错误！

---

## 💡 如果某些教师加载失败

代码会自动跳过失败的教师，使用成功加载的教师继续训练。

例如：
- AudioCLIP 加载失败 → 跳过
- CLAP 加载成功 → 使用
- BEATs 加载成功 → 使用

**至少需要1个教师成功加载才能进行蒸馏训练。**

---

## 🎯 下一步

1. **安装依赖** - `pip install transformers`
2. **上传文件** - `scp pretrained_teacher_loader.py ...`
3. **测试加载** - 运行测试脚本
4. **启动训练** - `python -m d25_t6.train`

告诉我测试结果！🚀
