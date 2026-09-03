# 🎵 Multi-Encoder Audio-Text Retrieval System

A retrieval-based audio-text matching system combining PaSST and CLAP encoders with attention-based fusion, achieving **mAP@10 = 0.32** on Clotho evaluation set.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [简体中文](README_CN.md)

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design.git
cd Audio-Retrieval-Graduate-Design
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

**Option A: From HuggingFace (Recommended)**
```bash
python scripts/download_model.py
```

**Option B: Manual Download**
- Download from: [Google Drive Link](https://drive.google.com/your-link)
- Place `model.ckpt` in `audio_app/checkpoints/`

### 4. Run Gradio Web Application
```bash
# Linux/Mac
bash audio_app/start_app.sh

# Windows
audio_app\start_app.bat

# Or directly
python audio_app/run_app.py
```

The application will launch at `http://localhost:7860`

---

## 📖 What is This?

This project implements a **retrieval-based audio-text matching system** that:

1. **Encodes audio** using complementary encoders (PaSST + CLAP)
2. **Fuses representations** using attention-based mechanism
3. **Retrieves descriptions** from a candidate library of 80,045 captions
4. **Deploys as web app** with 2-3 second processing time on CPU

### Key Features

- ✅ **Multi-encoder fusion**: Combines time-frequency patterns (PaSST) and audio-text semantics (CLAP)
- ✅ **Retrieval-based**: Fast similarity search without generative overhead
- ✅ **Real-time inference**: 2-3 seconds per audio clip on standard CPU
- ✅ **Web interface**: Easy-to-use Gradio application

---

## 🎯 Performance

| Model | mAP@10 | R@1 | R@10 |
|-------|--------|-----|------|
| PaSST-only | 0.290 | 0.181 | 0.588 |
| CLAP-only | 0.271 | 0.161 | 0.560 |
| **Multi-encoder fusion** | **0.325** | **0.203** | **0.623** |

- **+12.1%** improvement over PaSST-only
- **+20.0%** improvement over CLAP-only
- Evaluated on Clotho dataset

---

## 📁 Project Structure

```
Audio-Retrieval-Graduate-Design/
├── audio_app/                          # Gradio web application
│   ├── audio_caption_app.py           # Main application
│   ├── run_app.py                     # Application launcher
│   ├── candidate_captions.json        # 80,045 candidate descriptions
│   ├── requirements_app.txt           # Application dependencies
│   ├── start_app.sh                   # Linux/Mac launcher
│   └── start_app.bat                  # Windows launcher
│
├── retrieval_module.py                # Core retrieval model
├── multi_audio_encoder.py             # PaSST + CLAP fusion
├── passt.py                           # PaSST encoder wrapper
├── clap_encoder.py                    # CLAP encoder wrapper
├── train.py                           # Training script
├── predict.py                         # Inference script
│
├── datasets/                          # Data loading utilities
│   ├── audio_loading.py
│   └── __init__.py
│
├── scripts/                           # Helper scripts
│   ├── download_model.py              # Model weight downloader
│   └── extract_captions.py            # Caption extraction utility
│
├── requirements.txt                   # Project dependencies
├── .gitignore
└── README.md
```

---

## 💻 Usage

### Web Application (Recommended)

1. Launch the Gradio interface:
```bash
python audio_app/run_app.py
```

2. Open browser at `http://localhost:7860`

3. Upload audio file or record via microphone

4. Get top-10 retrieved descriptions with similarity scores

### Command-Line Inference

```bash
python predict.py --audio_path path/to/audio.wav --checkpoint path/to/model.ckpt
```

### Training

```bash
python train.py \
    --batch_size 32 \
    --max_epochs 50 \
    --learning_rate 2e-5
```

---

## 🔧 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- CPU-only supported

### Recommended Requirements
- Python 3.8+
- 8GB+ RAM
- NVIDIA GPU with 12GB+ VRAM (for training)

### Dependencies
- PyTorch 2.0+
- Transformers 4.30+
- Gradio 4.0+
- librosa 0.10+
- hear21passt (PaSST model)
- aac-datasets (CLAP model)

---

## 📊 Technical Details

### Architecture

**Audio Branch:**
- PaSST: Pre-trained on AudioSet, captures time-frequency patterns
- CLAP: Pre-trained with audio-text contrastive learning
- Attention fusion: Dynamically weights encoder outputs

**Text Branch:**
- RoBERTa-large: 1024-dimensional text embeddings
- Candidate library: 80,045 unique descriptions from Clotho + AudioCaps

**Retrieval:**
- Cosine similarity in joint embedding space
- Returns top-K most similar descriptions

### Training

- **Loss**: InfoNCE contrastive loss
- **Optimizer**: AdamW (lr=2e-5)
- **Augmentation**: SpecAugment (time mask 15 frames, freq mask 2 bins)
- **Batch size**: 32
- **Epochs**: 50

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{Audio-Retrieval-Graduate-Design-2026,
  author = {Justin},
  title = {Multi-Encoder Audio-Text Retrieval System},
  year = {2026},
  publisher = {Justin3nAi},
  url = {https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design}
}
```

---

## 🙏 Acknowledgments

This project builds upon:

- [PaSST](https://github.com/kkoutini/PaSST) - Patchout Spectrogram Transformer
- [CLAP](https://github.com/LAION-AI/CLAP) - Contrastive Language-Audio Pretraining
- [Clotho](https://zenodo.org/record/3490684) - Audio captioning dataset
- [AudioCaps](https://audiocaps.github.io/) - Audio captioning dataset

---

## 🐛 Issues and Contributions

- **Bug reports**: [Open an issue](https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design/issues)
- **Feature requests**: [Open an issue](https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design/issues)
- **Pull requests**: Welcome!

---

## 📧 Contact

For questions or discussions:

- **GitHub Issues**: [Project Issues](https://github.com/Justin3nAi/Audio-Retrieval-Graduate-Design/issues)
- **Email**: zhuyuanzhi09@126.com

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

---

**Last Updated**: March 2026  
**Status**: Active Development
