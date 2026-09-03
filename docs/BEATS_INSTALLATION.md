# BEATs安装指南

## 问题说明

训练时出现错误：`No module named 'beats'`

这是因为BEATs需要单独安装。

---

## 解决方案1: 安装BEATs（推荐）

### 方法A: 从GitHub安装

```bash
# SSH登录服务器
ssh root@your-server

# 克隆BEATs仓库
cd /root/autodl-tmp
git clone https://github.com/microsoft/unilm.git

# 安装BEATs
cd unilm/beats
pip install -e .

# 或者直接安装
pip install git+https://github.com/microsoft/unilm.git#subdirectory=beats
```

### 方法B: 手动安装（如果无外网）

1. 在本地下载BEATs代码：
```bash
git clone https://github.com/microsoft/unilm.git
cd unilm/beats
```

2. 压缩beats目录：
```bash
tar -czf beats.tar.gz .
```

3. 上传到服务器：
```bash
scp beats.tar.gz root@your-server:/root/autodl-tmp/
```

4. 在服务器上安装：
```bash
ssh root@your-server
cd /root/autodl-tmp
mkdir beats_install
cd beats_install
tar -xzf ../beats.tar.gz
pip install -e .
```

---

## 解决方案2: 临时禁用BEATs（快速测试）

如果暂时无法安装BEATs，可以先只用PaSST + CLAP训练：

```bash
cd /root
python -m d25_t6.train --no-use_beats
```

这样会使用 PaSST + CLAP 双编码器，预期性能：mAP@10 = 0.31-0.32

---

## 解决方案3: 修改默认配置（永久禁用BEATs）

如果不想使用BEATs，可以修改train.py的默认配置：

找到这行：
```python
parser.add_argument('--use_beats', default=True, action=argparse.BooleanOptionalAction,
```

改为：
```python
parser.add_argument('--use_beats', default=False, action=argparse.BooleanOptionalAction,
```

然后直接运行：
```bash
python -m d25_t6.train
```

---

## 验证BEATs安装

```bash
python -c "from beats import BEATs, BEATsConfig; print('✅ BEATs安装成功')"
```

如果没有报错，说明安装成功。

---

## 推荐方案

### 方案1: PaSST + CLAP（当前可用）⭐⭐⭐⭐

```bash
python -m d25_t6.train --no-use_beats
```

**优点**:
- ✅ 立即可用（不需要安装BEATs）
- ✅ CLAP专为音频-文本检索优化
- ✅ 预期提升：mAP@10 = 0.31-0.32 (+7-10%)

### 方案2: PaSST + BEATs + CLAP（安装BEATs后）⭐⭐⭐⭐⭐

```bash
# 安装BEATs后
python -m d25_t6.train
```

**优点**:
- ✅ 最大化性能
- ✅ 三个编码器互补
- ✅ 预期提升：mAP@10 = 0.33-0.35 (+14-21%)

---

## 快速开始（推荐）

**立即开始训练（不等BEATs）**:

```bash
cd /root
python -m d25_t6.train --no-use_beats
```

**后续安装BEATs后再测试三者全开**:

```bash
# 安装BEATs
pip install git+https://github.com/microsoft/unilm.git#subdirectory=beats

# 重新训练
python -m d25_t6.train
```

---

## 性能对比

| 配置 | mAP@10 | 提升 | BEATs安装 |
|------|--------|------|----------|
| 仅PaSST | 0.29 | - | 不需要 |
| PaSST + CLAP | 0.31-0.32 | +7-10% | 不需要 ✅ |
| PaSST + BEATs | 0.31-0.32 | +7-10% | 需要 |
| 三者全开 | 0.33-0.35 | +14-21% | 需要 |

---

**建议**: 先用 PaSST + CLAP 开始训练，同时安装BEATs，后续再测试三者全开。

























