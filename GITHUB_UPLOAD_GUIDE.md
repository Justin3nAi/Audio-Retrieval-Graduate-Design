# GitHub 上传最终指南

## 检查结果总结

✅ **上传大小**: 6.4MB (63个文件)
✅ **排除大小**: 6.01GB (6360个文件)
✅ **没有超过10MB的单个文件**
✅ **核心文件都已包含**

---

## 上传前的最后清理

### 1. 删除临时文件

```bash
cd d:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6

# 删除临时脚本
del _scan_project.py
del _check_github_upload.py
del _github_upload_plan.md
del _github_upload_checklist.txt
del _tmp_template_as_pptx.pptx

# 或者让.gitignore自动排除它们
```

### 2. 确认.gitignore生效

```bash
git status
```

确保以下大文件/目录不会被追踪：
- `essay/` (论文文档)
- `ChangeHistory/` (实验记录)
- `data/` (数据文件)
- `*.ckpt` (模型权重)
- `*.pkl` (缓存文件)
- `audio_app/Train4(0.32)/` (训练checkpoint)
- `audio_app/caption_embeddings_cache_*.pkl` (320MB缓存)

---

## GitHub 上传步骤

### 方案A：首次上传（推荐）

```bash
cd d:\GraduateDesign\Test\dcase2025_task6_baseline\ServerCodes\TestVersion\d25_t6

# 1. 初始化Git仓库
git init

# 2. 添加所有文件（.gitignore会自动排除大文件）
git add .

# 3. 查看将要提交的文件
git status

# 4. 首次提交
git commit -m "Initial commit: Multi-encoder audio-text retrieval system"

# 5. 连接到GitHub远程仓库
git remote add origin https://github.com/yourusername/audio-text-retrieval.git

# 6. 推送到GitHub
git branch -M main
git push -u origin main
```

### 方案B：如果已有Git仓库

```bash
# 1. 确保.gitignore生效
git rm -r --cached .
git add .

# 2. 提交更新
git commit -m "Clean up project: Remove large files and organize structure"

# 3. 推送
git push origin main
```

---

## 模型权重托管

由于模型权重文件过大（>100MB），不能直接上传到GitHub。建议使用以下方案之一：

### 方案1：HuggingFace Hub（推荐）

```bash
# 1. 安装huggingface-cli
pip install huggingface-hub

# 2. 登录
huggingface-cli login

# 3. 上传模型
huggingface-cli upload yourusername/audio-text-retrieval \
    audio_app/Train4\(0.32\)/model.ckpt \
    model.ckpt

# 4. 更新scripts/download_model.py中的repo_id
# repo_id="yourusername/audio-text-retrieval"
```

### 方案2：GitHub Release

```bash
# 1. 在GitHub网页上创建Release
# 2. 上传model.ckpt到Release assets（最大2GB）
# 3. 在README中提供下载链接
```

### 方案3：Google Drive

```bash
# 1. 上传model.ckpt到Google Drive
# 2. 获取共享链接
# 3. 在README中提供下载链接
# 4. 更新scripts/download_model.py支持gdown
```

---

## README更新清单

在推送到GitHub前，更新README.md中的占位符：

- [ ] 替换 `yourusername` 为你的GitHub用户名
- [ ] 替换 `your.email@example.com` 为你的邮箱
- [ ] 添加模型下载链接（HuggingFace/Google Drive/GitHub Release）
- [ ] 更新screenshot（可选）
- [ ] 添加LICENSE文件（推荐MIT）
- [ ] 添加demo视频或GIF（可选）

---

## 用户快速测试流程

确保用户可以按以下步骤快速运行：

```bash
# 1. 克隆
git clone https://github.com/yourusername/audio-text-retrieval.git
cd audio-text-retrieval

# 2. 安装依赖（约5分钟）
pip install -r requirements.txt

# 3. 下载模型（约2-5分钟）
python scripts/download_model.py

# 4. 启动应用（立即）
python audio_app/run_app.py

# 5. 访问 http://localhost:7860
```

总时间：约10分钟内完成部署和运行。

---

## 最终检查清单

上传前确认：

- [x] .gitignore正确配置
- [x] 上传大小<10MB
- [x] README.md清晰易懂
- [ ] requirements.txt完整
- [ ] 模型下载脚本可用
- [ ] LICENSE文件添加（可选）
- [ ] 删除临时文件
- [ ] 测试git add .不会包含大文件

---

## 上传后的工作

1. **创建GitHub Release**
   - Tag: v1.0.0
   - Title: "Multi-Encoder Audio-Text Retrieval System v1.0"
   - 上传模型权重到Release assets（如果<2GB）

2. **添加Badges到README**
   - Build status
   - License
   - Python version
   - Downloads

3. **编写CONTRIBUTING.md**（可选）
   - 如何报告bug
   - 如何提交PR
   - 代码规范

4. **创建GitHub Issues模板**（可选）
   - Bug report template
   - Feature request template

5. **测试完整流程**
   - 在新机器上克隆仓库
   - 按README步骤安装和运行
   - 确保用户能在10分钟内运行起来

---

## 推荐的仓库设置

在GitHub仓库设置中：

1. **About区域**
   - Description: "Multi-encoder audio-text retrieval system combining PaSST and CLAP"
   - Website: 你的个人网站或项目主页
   - Topics: `audio`, `deep-learning`, `pytorch`, `retrieval`, `multimodal`, `gradio`

2. **社交预览图**
   - 上传一张项目截图作为social preview

3. **GitHub Pages**（可选）
   - 部署在线demo

---

## 预估用户体验时间

- 克隆仓库：30秒
- 安装依赖：3-5分钟
- 下载模型：2-5分钟
- 启动应用：5秒
- **总计：约6-11分钟**

目标：让用户在10分钟内从零到运行Gradio应用！
