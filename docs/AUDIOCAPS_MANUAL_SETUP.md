# 🇨🇳 AudioCaps手动下载导入指南（中国境内服务器）

## 方案概述

由于服务器在中国境内无法访问外网，需要：
1. 在本地电脑（有外网）下载数据集
2. 上传到服务器
3. 在服务器上解压和配置

---

## 📥 步骤1：本地下载（在你的电脑上）

### 方法A：直接下载（推荐）

**下载链接：**
```
https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip
```

**操作步骤：**
1. 在浏览器中打开上面的链接
2. 下载 `AUDIOCAPS.zip`（约2GB）
3. 保存到本地（例如：`D:\Downloads\AUDIOCAPS.zip`）

### 方法B：使用下载工具（更稳定）

**使用IDM（Internet Download Manager）：**
```
1. 复制链接：https://cloud.cp.jku.at/index.php/s/9MiMcrNjJ3Z9FfH/download/AUDIOCAPS.zip
2. 打开IDM，点击"添加URL"
3. 粘贴链接，开始下载
```

**使用迅雷：**
```
1. 复制链接
2. 打开迅雷，自动识别下载任务
3. 开始下载
```

### 方法C：备用下载源

如果上面的链接无法访问，可以尝试：

**百度网盘分享（如果有）：**
- 搜索"AudioCaps dataset 百度网盘"
- 或在学术论坛寻找分享

**Google Drive（需要梯子）：**
- 搜索"AudioCaps Google Drive"

---

## 📤 步骤2：上传到服务器

### 方法A：使用SCP/SFTP（推荐）

**Windows用户 - 使用WinSCP：**

1. **下载WinSCP**：https://winscp.net/

2. **连接服务器**：
   ```
   主机名：你的服务器IP
   端口：22
   用户名：root（或你的用户名）
   密码：你的密码
   ```

3. **上传文件**：
   - 左侧：本地文件（找到AUDIOCAPS.zip）
   - 右侧：服务器路径（导航到项目目录）
   - 拖拽文件到右侧上传

4. **目标路径**：
   ```
   /d/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6/data/
   ```

**Linux/Mac用户 - 使用scp命令：**

```bash
# 在本地终端执行
scp AUDIOCAPS.zip root@你的服务器IP:/d/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6/data/

# 例如：
scp AUDIOCAPS.zip root@192.168.1.100:/d/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6/data/
```

### 方法B：使用FTP工具

**FileZilla：**
1. 下载FileZilla：https://filezilla-project.org/
2. 连接服务器（输入IP、用户名、密码）
3. 拖拽上传文件

### 方法C：使用云盘中转（适合大文件）

**如果直接上传太慢：**

1. **上传到百度网盘/阿里云盘**
2. **在服务器上安装下载工具**：
   ```bash
   # 安装BaiduPCS-Go（百度网盘命令行工具）
   wget https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/v3.9.5/BaiduPCS-Go-v3.9.5-linux-amd64.zip
   unzip BaiduPCS-Go-v3.9.5-linux-amd64.zip
   ./BaiduPCS-Go login
   ./BaiduPCS-Go download /AUDIOCAPS.zip
   ```

---

## 📦 步骤3：在服务器上解压

### 连接到服务器

```bash
# 使用SSH连接
ssh root@你的服务器IP
```

### 导航到项目目录

```bash
cd /d/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6/data
```

### 检查文件是否上传成功

```bash
ls -lh AUDIOCAPS.zip
# 应该显示：-rw-r--r-- 1 root root 2.0G ... AUDIOCAPS.zip
```

### 安装解压工具（如果没有）

```bash
# Ubuntu/Debian
apt-get update
apt-get install -y p7zip-full unzip

# CentOS/RHEL
yum install -y p7zip unzip
```

### 解压文件

```bash
# 方法1：使用7z（推荐）
7z x AUDIOCAPS.zip -oAUDIOCAPS

# 方法2：使用unzip
unzip AUDIOCAPS.zip -d AUDIOCAPS

# 解压后删除压缩包（可选，节省空间）
rm AUDIOCAPS.zip
```

### 验证解压结果

```bash
ls -lh AUDIOCAPS/
# 应该看到：
# drwxr-xr-x 2 root root 4.0K ... train/
# drwxr-xr-x 2 root root 4.0K ... val/
# drwxr-xr-x 2 root root 4.0K ... test/
# -rw-r--r-- 1 root root  XXM ... train.csv
# -rw-r--r-- 1 root root  XXK ... val.csv
# -rw-r--r-- 1 root root  XXK ... test.csv

# 检查音频文件数量
ls AUDIOCAPS/train/ | wc -l
# 应该显示：49838（或接近这个数字）
```

---

## 🔧 步骤4：修改代码（禁用自动下载）

由于已经手动下载，需要禁用自动下载功能：

### 编辑train.py

```bash
cd /d/GraduateDesign/Test/dcase2025_task6_baseline/ServerCodes/TestVersion/d25_t6
nano train.py  # 或使用 vim train.py
```

### 找到并注释掉下载代码

找到这几行：
```python
# download data sets; will be ignored if exists
download_clotho(args["data_path"])
# AudioCAps
if args['audiocaps']:
    download_audiocaps(args["data_path"])
```

改为：
```python
# download data sets; will be ignored if exists
# download_clotho(args["data_path"])  # 已手动下载，注释掉
# AudioCAps
# if args['audiocaps']:
#     download_audiocaps(args["data_path"])  # 已手动下载，注释掉
```

保存并退出（Ctrl+X，然后Y，然后Enter）

---

## ✅ 步骤5：验证配置

### 检查目录结构

```bash
tree -L 2 data/
# 应该显示：
# data/
# ├── AUDIOCAPS/
# │   ├── train/
# │   ├── val/
# │   ├── test/
# │   ├── train.csv
# │   ├── val.csv
# │   └── test.csv
# └── CLOTHO/
#     ├── dev/
#     ├── val/
#     └── eval/
```

### 测试数据加载

```bash
python -c "
from aac_datasets import AudioCaps
from d25_t6.datasets.audio_loading import custom_loading

ac = custom_loading(
    AudioCaps(subset='train', root='data', download=False, audio_format='mp3')
)
print(f'✅ AudioCaps加载成功！样本数：{len(ac)}')
"
```

如果看到 `✅ AudioCaps加载成功！样本数：49838`，说明配置正确！

---

## 🚀 步骤6：开始训练

```bash
python train.py \
  --audiocaps \
  --batch_size 32 \
  --max_epochs 60 \
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

---

## 📊 完整的目录结构

```
d25_t6/
├── data/
│   ├── AUDIOCAPS/
│   │   ├── train/
│   │   │   ├── Y0001.mp3
│   │   │   ├── Y0002.mp3
│   │   │   └── ... (49838个文件)
│   │   ├── val/
│   │   │   └── ... (494个文件)
│   │   ├── test/
│   │   │   └── ... (957个文件)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── CLOTHO/
│       ├── dev/
│       ├── val/
│       └── eval/
├── checkpoints/
├── train.py
├── retrieval_module.py
└── ...
```

---

## ⚠️ 常见问题

### 问题1：上传速度太慢

**解决方案：**
1. **压缩后再上传**：
   ```bash
   # 在本地压缩成更小的文件
   7z a -mx=9 AUDIOCAPS_compressed.7z AUDIOCAPS.zip
   ```

2. **分段上传**：
   ```bash
   # 分割成多个小文件
   split -b 500M AUDIOCAPS.zip AUDIOCAPS.zip.part
   # 上传所有part文件
   # 在服务器上合并
   cat AUDIOCAPS.zip.part* > AUDIOCAPS.zip
   ```

3. **使用断点续传**：
   ```bash
   rsync -avP AUDIOCAPS.zip root@服务器IP:/path/to/data/
   ```

### 问题2：服务器磁盘空间不足

**检查空间：**
```bash
df -h
```

**清理空间：**
```bash
# 清理apt缓存
apt-get clean

# 清理旧的日志
rm -rf /var/log/*.log

# 清理临时文件
rm -rf /tmp/*
```

### 问题3：解压失败

**错误：`7z: command not found`**
```bash
apt-get install p7zip-full
```

**错误：`End-of-central-directory signature not found`**
```bash
# 文件损坏，需要重新下载
rm AUDIOCAPS.zip
# 重新上传
```

### 问题4：权限问题

```bash
# 修改文件权限
chmod -R 755 data/AUDIOCAPS/

# 修改所有者
chown -R root:root data/AUDIOCAPS/
```

---

## 🎯 快速检查清单

- [ ] 本地下载完成（AUDIOCAPS.zip，~2GB）
- [ ] 上传到服务器（/path/to/d25_t6/data/）
- [ ] 解压成功（data/AUDIOCAPS/目录存在）
- [ ] 音频文件数量正确（train: 49838, val: 494, test: 957）
- [ ] CSV文件存在（train.csv, val.csv, test.csv）
- [ ] 注释掉自动下载代码
- [ ] 测试数据加载成功
- [ ] 开始训练

---

## 💡 推荐工作流程

### 时间估算

| 步骤 | 时间 | 说明 |
|------|------|------|
| 本地下载 | 10-30分钟 | 取决于网速 |
| 上传到服务器 | 30-120分钟 | 取决于带宽 |
| 解压 | 5-10分钟 | - |
| 配置和测试 | 5分钟 | - |
| **总计** | **1-3小时** | - |

### 最佳实践

1. **晚上下载**：利用夜间网速快的时间
2. **使用断点续传**：避免上传中断
3. **先测试小文件**：确保上传流程正确
4. **保留本地备份**：以防需要重新上传

---

## 📞 需要帮助？

如果遇到问题，提供以下信息：

1. 在哪一步卡住了？
2. 错误信息是什么？
3. 服务器系统版本：`cat /etc/os-release`
4. 磁盘空间：`df -h`
5. 文件是否存在：`ls -lh data/AUDIOCAPS.zip`

祝顺利完成！🚀

