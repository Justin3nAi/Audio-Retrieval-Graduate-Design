"""
音频描述生成应用
用户上传音频文件，系统返回音频内容的文字描述
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import librosa
import gradio as gr
import numpy as np
from typing import List, Tuple

# 添加父目录到路径，以便导入模块
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# 确保父目录在路径中
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入模块（使用d25_t6包名）
from d25_t6.retrieval_module import AudioRetrievalModel
from d25_t6.datasets.audio_loading import _pad_or_subsample_audio


class AudioCaptionGenerator:
    """音频描述生成器"""
    
    def __init__(self, checkpoint_path: str, caption_file: str = None):
        """
        初始化模型
        
        Args:
            checkpoint_path: 模型checkpoint路径
            caption_file: 候选caption文件路径（JSON格式），如果为None则使用预定义列表
        """
        print("🔄 正在加载模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioRetrievalModel.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 模型加载成功！使用设备: {self.device}")
        
        # 加载候选描述
        if caption_file and os.path.exists(caption_file):
            print(f"📥 从文件加载候选描述: {caption_file}")
            import json
            with open(caption_file, 'r', encoding='utf-8') as f:
                self.candidate_captions = json.load(f)
            print(f"✅ 加载了 {len(self.candidate_captions)} 个候选描述")
        else:
            if caption_file:
                print(f"⚠️ 候选描述文件不存在: {caption_file}")
                print("   使用预定义的候选描述列表")
            
            # 预定义的候选描述（备用方案）
            self.candidate_captions = [
            # 自然环境
            "birds chirping in the forest",
            "rain falling on the ground",
            "ocean waves crashing on the shore",
            "wind blowing through trees",
            "thunder and lightning storm",
            "flowing water in a stream",
            "waterfall cascading down",
            
            # 城市环境
            "car engine running",
            "traffic on a busy street",
            "train passing by",
            "airplane flying overhead",
            "construction site with machinery",
            "people talking in a crowd",
            "footsteps on pavement",
            "door opening and closing",
            "elevator moving",
            
            # 室内声音
            "keyboard typing",
            "phone ringing",
            "clock ticking",
            "vacuum cleaner running",
            "washing machine operating",
            "microwave beeping",
            "refrigerator humming",
            "air conditioner running",
            
            # 动物声音
            "dog barking",
            "cat meowing",
            "horse neighing",
            "cow mooing",
            "rooster crowing",
            "insects buzzing",
            
            # 音乐和乐器
            "piano playing",
            "guitar strumming",
            "drums beating",
            "violin playing",
            "flute melody",
            "singing voice",
            "orchestra performing",
            
            # 人类活动
            "baby crying",
            "children playing",
            "people laughing",
            "someone coughing",
            "sneezing sound",
            "clapping hands",
            "footsteps running",
            
            # 工具和机械
            "hammer hitting nail",
            "saw cutting wood",
            "drill operating",
            "chainsaw running",
            "lawnmower cutting grass",
            
            # 厨房声音
            "water boiling",
            "food frying in pan",
            "knife chopping vegetables",
            "blender mixing",
            "dishes clanking",
            
            # 其他
            "fire crackling",
            "glass breaking",
            "paper rustling",
            "zipper opening",
            "coins jingling",
            "bell ringing",
            "whistle blowing",
        ]
        
        # 预计算候选描述的embeddings
        print("🔄 预计算候选描述embeddings...")
        self._precompute_caption_embeddings()
        print("✅ 初始化完成！")
    
    def _precompute_caption_embeddings(self):
        """预计算所有候选描述的embeddings - 带缓存和批处理优化"""
        import hashlib
        import pickle
        
        # 生成缓存文件名（基于caption列表的hash）
        captions_str = ''.join(self.candidate_captions)
        cache_hash = hashlib.md5(captions_str.encode()).hexdigest()[:8]
        cache_file = f'caption_embeddings_cache_{cache_hash}.pkl'
        
        # 尝试从缓存加载
        if os.path.exists(cache_file):
            print(f"📦 从缓存加载embeddings: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    self.caption_embeddings = pickle.load(f)
                print(f"✅ 缓存加载成功！跳过计算")
                return
            except Exception as e:
                print(f"⚠️ 缓存加载失败: {e}，重新计算...")
        
        # 批处理计算embeddings（更快）
        print(f"🔄 批量计算 {len(self.candidate_captions)} 个embeddings...")
        batch_size = 128  # 每批处理128个
        self.caption_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(self.candidate_captions), batch_size):
                batch_captions = self.candidate_captions[i:i+batch_size]
                
                # 批量处理
                batch_embeddings = []
                for caption in batch_captions:
                    embedding = self.model.forward_text({
                        'captions': [[caption]]
                    })[0].cpu()
                    batch_embeddings.append(embedding)
                
                self.caption_embeddings.extend(batch_embeddings)
                
                # 显示进度
                progress = min(i + batch_size, len(self.candidate_captions))
                print(f"   进度: {progress}/{len(self.candidate_captions)} ({progress*100//len(self.candidate_captions)}%)")
        
        self.caption_embeddings = torch.stack(self.caption_embeddings)  # (N, embed_dim)
        
        # 保存到缓存
        print(f"💾 保存embeddings到缓存: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.caption_embeddings, f)
            print(f"✅ 缓存保存成功！下次启动将直接加载")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")
    
    def process_audio(self, audio_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        处理音频文件，返回最匹配的描述
        
        Args:
            audio_path: 音频文件路径
            top_k: 返回前k个最匹配的描述
            
        Returns:
            List of (caption, similarity_score) tuples
        """
        try:
            # 1. 加载音频
            audio, sr = librosa.load(audio_path, sr=32000)
            duration = len(audio) / sr
            
            # 2. 预处理
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            audio_tensor = _pad_or_subsample_audio(audio_tensor, max_length=32000*30)
            audio_tensor = audio_tensor.to(self.device)
            
            # 3. 提取音频embedding
            with torch.no_grad():
                audio_embedding = self.model.forward_audio({
                    'audio': audio_tensor.unsqueeze(0),
                    'duration': [duration]
                })[0].cpu()  # (embed_dim,)
            
            # 4. 计算与所有候选描述的相似度
            # 使用余弦相似度
            audio_embedding_norm = audio_embedding / audio_embedding.norm()
            caption_embeddings_norm = self.caption_embeddings / self.caption_embeddings.norm(dim=1, keepdim=True)
            similarities = (audio_embedding_norm * caption_embeddings_norm).sum(dim=1)
            
            # 5. 获取top-k结果
            top_k_indices = similarities.argsort(descending=True)[:top_k]
            results = [
                (self.candidate_captions[idx], similarities[idx].item())
                for idx in top_k_indices
            ]
            
            return results, duration
            
        except Exception as e:
            raise Exception(f"处理音频时出错: {str(e)}")
    
    def format_results(self, results: List[Tuple[str, float]], duration: float) -> str:
        """格式化输出结果 - 现代科技风格"""
        # 渐变色方案
        gradients = [
            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
            "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
            "linear-gradient(135deg, #30cfd0 0%, #330867 100%)",
            "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
            "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)",
            "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
            "linear-gradient(135deg, #ff6e7f 0%, #bfe9ff 100%)",
        ]
        
        output = '<div style="padding: 10px;">\n'
        
        # 音频时长信息卡片
        output += f'''
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 20px; padding: 20px; margin-bottom: 25px;
                    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                    text-align: center;">
            <div style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 5px; 
                        letter-spacing: 2px; text-transform: uppercase;">Audio Duration</div>
            <div style="color: white; font-size: 36px; font-weight: 700; 
                        text-shadow: 0 2px 10px rgba(0,0,0,0.2);">{duration:.2f}s</div>
        </div>
        '''
        
        # 识别结果列表
        for i, (caption, score) in enumerate(results, 1):
            gradient = gradients[i-1] if i <= len(gradients) else gradients[0]
            
            # 计算透明度和动画延迟
            opacity = 1 - (i-1) * 0.05
            delay = (i-1) * 0.1
            
            output += f'''
            <div style="background: {gradient};
                        border-radius: 16px; 
                        padding: 18px 24px 18px 90px; 
                        margin-bottom: 12px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                        opacity: {opacity};
                        animation: slideIn {delay}s ease-out;
                        position: relative;
                        overflow: hidden;">
                <div style="position: absolute; top: 50%; left: 20px; 
                            transform: translateY(-50%);
                            color: rgba(255,255,255,0.3); font-size: 42px; 
                            font-weight: 900; line-height: 1;">#{i}</div>
                <div style="color: white; font-size: 17px; font-weight: 500; 
                            line-height: 1.6; 
                            text-shadow: 0 1px 3px rgba(0,0,0,0.2);">
                    {caption}
                </div>
            </div>
            '''
        
        output += '</div>'
        return output


def create_app(checkpoint_path: str, caption_file: str = None):
    """创建Gradio应用 - 现代科技风格"""
    
    # 设置Gradio为英文界面
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    
    # 尝试设置Gradio的默认语言
    try:
        import gradio
        if hasattr(gradio, 'set_static_paths'):
            # Gradio 4.x 的设置方法
            pass
    except:
        pass
    
    # 初始化生成器
    generator = AudioCaptionGenerator(checkpoint_path, caption_file)
    
    def predict_audio(audio_file, top_k):
        """Gradio回调函数"""
        if audio_file is None:
            return '''
            <div style="text-align: center; padding: 80px 20px;">
                <div style="font-size: 64px; margin-bottom: 20px;">🎵</div>
                <div style="font-size: 20px; color: #95a5a6;">Please upload audio file or record audio</div>
            </div>
            '''
        
        try:
            results, duration = generator.process_audio(audio_file, top_k=top_k)
            return generator.format_results(results, duration)
        except Exception as e:
            return f'''
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 48px; margin-bottom: 15px;">⚠️</div>
                <div style="font-size: 18px; color: #e74c3c;">Processing failed: {str(e)}</div>
            </div>
            '''
    
    # 自定义CSS样式 - 现代科技风格
    custom_css = """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); }
            50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        
        .gradio-container {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px);
            border-radius: 30px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;
        }
    </style>
    <script>
        // 强制设置Gradio为英文
        window.addEventListener('DOMContentLoaded', function() {
            // 尝试设置语言
            if (window.gradio && window.gradio.setLang) {
                window.gradio.setLang('en');
            }
            // 设置localStorage
            localStorage.setItem('gradio_language', 'en');
            localStorage.setItem('gradio-lang', 'en');
        });
    </script>
    """
    
    # 创建Gradio界面
    with gr.Blocks(
        title="AI Audio Recognition",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate",
            spacing_size="lg",
            radius_size="lg"
        ),
        css="""
            .gradio-container {
                max-width: 1600px !important;
                margin: 40px auto !important;
                padding: 50px 50px 150px 50px !important;
            }
            
            .main-row {
                gap: 60px !important;
                align-items: stretch !important;
            }
            
            #result-container {
                min-height: 500px;
                max-height: 650px;
                overflow-y: auto;
                overflow-x: hidden;
                padding: 10px;
            }
            
            #result-container::-webkit-scrollbar {
                width: 10px;
            }
            
            #result-container::-webkit-scrollbar-track {
                background: rgba(0,0,0,0.05);
                border-radius: 10px;
            }
            
            #result-container::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            /* 隐藏Gradio底部的设置按钮 */
            footer {
                display: none !important;
            }
            
            /* 隐藏所有设置相关的元素 */
            .gradio-dropdown, .dropdown-menu, .settings-button {
                display: none !important;
            }
        """
    ) as app:
        
        gr.HTML(custom_css)
        
        # 标题区域 - 极简现代风格
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 60px; animation: fadeIn 0.8s ease-out;">
                <div style="font-size: 56px; font-weight: 800; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            margin-bottom: 15px;
                            letter-spacing: -2px;">
                    AI Audio Recognition
                </div>
                <div style="font-size: 18px; color: #7f8c8d; font-weight: 400; letter-spacing: 1px;">
                    Powered by Deep Learning · Real-time Analysis
                </div>
            </div>
            """
        )
        
        # 主内容区域
        with gr.Row(elem_classes="main-row"):
            # 左侧：上传区域
            with gr.Column(scale=1):
                gr.HTML('''
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 20px; padding: 25px; margin-bottom: 25px;
                                box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
                        <div style="color: white; font-size: 24px; font-weight: 700; text-align: center;">
                            Upload Audio
                        </div>
                    </div>
                ''')
                
                # 添加英文提示
                gr.HTML('''
                    <div style="text-align: center; padding: 15px; margin-bottom: 10px;
                                background: rgba(102, 126, 234, 0.1); border-radius: 12px;">
                        <div style="font-size: 14px; color: #667eea; font-weight: 500;">
                            📁 Drag and drop audio file here or click to upload
                        </div>
                        <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                            🎤 Or use microphone to record
                        </div>
                    </div>
                ''')
                
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Results Count",
                    info="Number of recognition results to display"
                )
                
                predict_btn = gr.Button(
                    "Start Recognition", 
                    variant="primary", 
                    size="lg"
                )
                
                # 使用指南
                gr.HTML('''
                    <div style="margin-top: 30px; padding: 25px; 
                                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                border-radius: 16px;">
                        <div style="font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 15px;">
                            Quick Guide
                        </div>
                        <div style="font-size: 14px; color: #5a6c7d; line-height: 1.8;">
                            • Supported: WAV, MP3, FLAC<br>
                            • Duration: 1-30 seconds<br>
                            • Microphone recording available
                        </div>
                    </div>
                ''')
            
            # 右侧：结果区域
            with gr.Column(scale=1):
                gr.HTML('''
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                border-radius: 20px; padding: 25px; margin-bottom: 25px;
                                box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);">
                        <div style="color: white; font-size: 24px; font-weight: 700; text-align: center;">
                            Recognition Results
                        </div>
                    </div>
                ''')
                
                output_text = gr.HTML(
                    value='''
                    <div style="text-align: center; padding: 100px 20px;">
                        <div style="font-size: 72px; margin-bottom: 20px; animation: float 3s ease-in-out infinite;">🎵</div>
                        <div style="font-size: 20px; color: #95a5a6; font-weight: 500;">Waiting for audio input...</div>
                    </div>
                    ''',
                    elem_id="result-container"
                )
        
        # 底部信息栏 - 简洁版
        caption_count = len(generator.candidate_captions)
        gr.HTML(f'''
            <div style="margin-top: 50px; padding: 30px; 
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 20px; text-align: center;">
                <div style="font-size: 14px; color: #7f8c8d; letter-spacing: 1px;">
                    Database: {caption_count:,} Audio Descriptions | Training: Clotho + AudioCaps
                </div>
            </div>
        ''')
        
        # 绑定事件
        predict_btn.click(
            fn=predict_audio,
            inputs=[audio_input, top_k_slider],
            outputs=output_text
        )
    
    return app


if __name__ == "__main__":
    import argparse
    
    # 设置环境变量强制使用英文
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LANGUAGE'] = 'en_US:en'
    
    parser = argparse.ArgumentParser(description="音频描述生成应用")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='Train4(0.32)/mAP@10=0.32.ckpt',
        help='模型checkpoint路径（相对于audio_app目录）'
    )
    parser.add_argument(
        '--caption_file',
        type=str,
        default='candidate_captions.json',
        help='候选caption文件路径（JSON格式）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Web服务端口'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='是否创建公网链接'
    )
    
    args = parser.parse_args()
    
    # 🔥 切换到脚本所在目录（audio_app目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 工作目录: {os.getcwd()}")
    
    # 检查checkpoint是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误: 找不到checkpoint文件: {args.checkpoint}")
        print(f"   当前工作目录: {os.getcwd()}")
        print(f"   请确保在 audio_app 目录下运行，或模型文件存在")
        exit(1)
    
    # 检查caption文件
    caption_file_path = args.caption_file if os.path.exists(args.caption_file) else None
    if caption_file_path:
        print(f"✅ 找到候选caption文件: {caption_file_path}")
    else:
        print(f"⚠️ 未找到候选caption文件: {args.caption_file}")
        print(f"   将使用预定义的候选描述列表")
    
    # 创建并启动应用
    print("\n" + "="*60)
    print("🚀 启动音频内容识别系统...")
    print("="*60)
    
    app = create_app(args.checkpoint, caption_file_path)
    
    # 启动应用
    print("\n" + "="*60)
    print("📱 访问地址:")
    print(f"   英文界面: http://localhost:{args.port}/?__lang=en")
    print(f"   中文界面: http://localhost:{args.port}/?__lang=zh")
    print("="*60)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
        quiet=False
    )

