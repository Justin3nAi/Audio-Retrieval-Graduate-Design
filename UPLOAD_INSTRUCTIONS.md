# 完整的上传指南

## ✅ 你的项目信息

**GitHub仓库：** https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design.git  
**用户名：** Justin3nAi  
**邮箱：** 1628575421@qq.com  
**当前分支：** TestVersion1

---

## 📦 模型上传方案

你的模型文件 `mAP@10=0.32.ckpt` (5.7GB) 太大，不能直接推送到GitHub。

### 推荐：上传到HuggingFace Hub

```bash
# 1. 安装工具
pip install huggingface-hub

# 2. 登录HuggingFace（需要先注册账号：https://huggingface.co/join）
huggingface-cli login

# 3. 上传模型
cd "d:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6"
huggingface-cli upload Justin3nAi/audio-text-retrieval "audio_app/Train4(0.32)/mAP@10=0.32.ckpt" model.ckpt
```

这样用户就可以通过以下命令自动下载：
```bash
python scripts/download_model.py
```

---

## 🚀 推送代码到GitHub

### 步骤1：添加和提交所有文件

```bash
cd "d:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6"

# 添加所有文件
git add .

# 查看状态（确认.gitignore生效，没有大文件）
git status

# 提交
git commit -m "Refactor: Clean project, add Gradio app and comprehensive documentation"
```

### 步骤2：推送到GitHub

```bash
# 推送到现有仓库的TestVersion1分支
git push origin TestVersion1
```

或者如果你想推送到main分支：

```bash
# 切换到main分支
git checkout -b main

# 推送到main
git push origin main
```

---

## 🔍 验证上传内容

在推送前，你可以验证哪些文件会被上传：

```bash
# 查看将要提交的文件
git status

# 查看被忽略的文件
git status --ignored

# 查看哪些大文件被.gitignore排除了
git status --ignored | findstr /i "ckpt pkl essay"
```

**确认以下文件被排除：**
- ✅ `audio_app/Train4(0.32)/mAP@10=0.32.ckpt` (5.7GB) - 被.gitignore排除
- ✅ `audio_app/caption_embeddings_cache_*.pkl` (313MB) - 被.gitignore排除  
- ✅ `essay/` 目录 - 被.gitignore排除
- ✅ `ChangeHistory/` 目录 - 被.gitignore排除

**确认以下文件会被上传：**
- ✅ `audio_app/candidate_captions.json` (5MB) - 必需的候选描述库
- ✅ `audio_app/audio_caption_app.py` - Gradio应用
- ✅ `README.md` - 项目说明
- ✅ `requirements.txt` - 依赖列表

---

## 📋 完整操作清单

### 操作1：推送代码到GitHub

```bash
cd "d:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6"
git add .
git commit -m "Refactor: Clean project structure and add deployment files"
git push origin TestVersion1
```

### 操作2：上传模型到HuggingFace

```bash
# 安装（如果还没装）
pip install huggingface-hub

# 登录
huggingface-cli login

# 上传模型
huggingface-cli upload Justin3nAi/audio-text-retrieval "audio_app/Train4(0.32)/mAP@10=0.32.ckpt" model.ckpt
```

### 操作3：测试完整流程

在另一个目录测试克隆和运行：

```bash
# 克隆仓库
git clone https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design.git
cd Audio-Retrieval-Graduate-Design

# 安装依赖
pip install -r requirements.txt

# 下载模型
python scripts/download_model.py

# 运行应用
python audio_app/run_app.py
```

---

## 💡 常见问题

### Q1: 如果不想用HuggingFace，可以用Google Drive吗？

可以！步骤如下：

1. 手动上传 `mAP@10=0.32.ckpt` 到Google Drive
2. 获取共享链接的file_id（链接格式：`https://drive.google.com/file/d/FILE_ID/view`）
3. 用户通过以下命令下载：

```bash
python scripts/download_model.py --source gdrive --file_id YOUR_FILE_ID
```

### Q2: 推送时显示文件太大怎么办？

检查 `.gitignore` 是否生效：

```bash
# 清除git缓存
git rm -r --cached .

# 重新添加（这次会遵守.gitignore）
git add .

# 提交
git commit -m "Fix: Apply .gitignore rules"

# 推送
git push origin TestVersion1
```

### Q3: 如何查看当前仓库大小？

```bash
# 查看工作目录大小
git count-objects -vH

# 查看将要推送的文件大小
git ls-files -z | xargs -0 du -ch 2>nul | find "total"
```

---

## ✅ 最终检查

推送前确认：

- [x] README.md已更新为你的信息（Justin3nAi）
- [x] scripts/download_model.py已更新repo_id
- [x] .gitignore正确配置（已验证）
- [x] 模型文件被排除（5.7GB不会上传）
- [ ] 模型已上传到HuggingFace（待操作）
- [ ] 代码已推送到GitHub（待操作）
- [ ] 测试完整克隆和运行流程（待操作）

---

**准备就绪！现在你可以安全地推送代码到GitHub了。** 🎉
