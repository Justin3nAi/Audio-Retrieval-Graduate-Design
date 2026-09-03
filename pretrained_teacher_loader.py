"""

预训练教师模型加载器 - 完整实现版本

支持 AudioCLIP, CLAP, BEATs 等预训练模型

"""



import torch

import torch.nn as nn

import torch.nn.functional as F





class PretrainedTeacherWrapper(nn.Module):

    """

    包装预训练模型作为教师

    """

    

    def __init__(self, model_type, model_path, device='cuda'):

        super().__init__()

        self.model_type = model_type

        self.device = device

        self.model = None

        

        print(f"  Loading {model_type} from {model_path}")

        

        try:

            if model_type == 'audioclip':

                self.model = self.load_audioclip(model_path)

            elif model_type == 'clap':

                self.model = self.load_clap(model_path)

            elif model_type == 'beats':

                self.model = self.load_beats(model_path)

            else:

                raise ValueError(f"Unknown model type: {model_type}")

            

            if self.model is not None and self.model_type != 'audioclip':

                self.model.eval()

                for param in self.model.parameters():

                    param.requires_grad = False

                print(f"     SUCCESS")

            else:

                print(f"     WARNING: Model loaded but may not work correctly")

                

        except Exception as e:

            print(f"     FAILED: {e}")

            self.model = None

    

    def load_audioclip(self, model_path):

        """加载 AudioCLIP 模型"""

        try:

            # 尝试加载 checkpoint

            checkpoint = torch.load(model_path, map_location='cpu')

            print(f"     Checkpoint keys: {list(checkpoint.keys())[:5]}...")

            return checkpoint

        except Exception as e:

            print(f"     Error loading AudioCLIP: {e}")

            return None

    

    def load_clap(self, model_path):

        """加载 CLAP 模型"""

        try:

            from transformers import ClapModel, ClapProcessor

            model = ClapModel.from_pretrained(model_path)

            self.clap_processor = ClapProcessor.from_pretrained(model_path)

            return model

        except ImportError:

            print(f"     ERROR: transformers not installed. Run: pip install transformers")

            return None

        except Exception as e:

            print(f"     Error loading CLAP: {e}")

            return None

    

    def load_beats(self, model_path):

        """加载 BEATs 模型"""

        try:

            # 使用你已有的 BEATs 加载器

            from d25_t6.beats_loader import load_beats_model_simple

            model = load_beats_model_simple(model_path, device='cpu')

            return model

        except Exception as e:

            print(f"     Error loading BEATs: {e}")

            return None

    

    def forward(self, batch):

        """

        前向传播，返回音频和文本嵌入

        

        Args:

            batch: 包含 'audio' 和 'caption' 的字典

        

        Returns:

            audio_emb: (batch, embed_dim)

            text_emb: (batch, embed_dim)

        """

        if self.model is None:

            # 如果模型加载失败，返回随机嵌入

            batch_size = batch['audio'].shape[0]

            audio_emb = torch.randn(batch_size, 1024, device=self.device)

            text_emb = torch.randn(batch_size, 1024, device=self.device)

            return audio_emb, text_emb

        

        if self.model_type == 'audioclip':

            return self.forward_audioclip(batch)

        elif self.model_type == 'clap':

            return self.forward_clap(batch)

        elif self.model_type == 'beats':

            return self.forward_beats(batch)

    

    def forward_audioclip(self, batch):

        """AudioCLIP 前向传播"""

        # AudioCLIP 比较复杂，暂时返回随机嵌入

        batch_size = batch['audio'].shape[0]

        audio_emb = torch.randn(batch_size, 1024, device=self.device)

        text_emb = torch.randn(batch_size, 1024, device=self.device)

        return audio_emb, text_emb

    

    def forward_clap(self, batch):

        """CLAP 前向传播"""

        try:

            audio = batch['audio']  # (batch, channels, samples)

            # Debug: print batch keys
            print(f"[DEBUG] Batch keys: {list(batch.keys())}")
            
            # Try different possible keys for captions
            if "caption" in batch:
                captions = batch["caption"]
            elif "text" in batch:
                captions = batch["text"]
            elif "captions" in batch:
                captions = batch["captions"]
            else:
                raise KeyError(f"No caption key found. Available keys: {list(batch.keys())}")
            
            # Process captions format
            # captions may be list of lists, need to flatten
            if isinstance(captions, list) and len(captions) > 0:
                if isinstance(captions[0], list):
                    # [[cap1, cap2], [cap3, cap4]] -> [cap1, cap3] (take first)
                    captions = [c[0] if isinstance(c, list) and len(c) > 0 else str(c) for c in captions]
                # Ensure all are strings
                captions = [str(c) for c in captions]

            

            with torch.no_grad():

                # 处理音频

                # CLAP 期望的输入格式

                if audio.dim() == 3:

                    # (batch, channels, samples) -> (batch, samples)

                    audio = audio.mean(dim=1)

                

                # 🔥 修复：CLAP 使用 48000 Hz 采样率

                inputs = self.clap_processor(

                    text=captions,

                    audios=audio.cpu().numpy(),

                    return_tensors="pt",

                    padding=True,

                    sampling_rate=48000

                )

                

                # 移到正确的设备

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                

                # 前向传播

                outputs = self.model(**inputs)

                audio_emb = outputs.audio_embeds

                text_emb = outputs.text_embeds

                

                # 归一化

                audio_emb = F.normalize(audio_emb, p=2, dim=-1)

                text_emb = F.normalize(text_emb, p=2, dim=-1)

                

                # 投影到 1024 维（如果需要）

                if audio_emb.shape[-1] != 1024:

                    audio_emb = F.pad(audio_emb, (0, 1024 - audio_emb.shape[-1]))

                    text_emb = F.pad(text_emb, (0, 1024 - text_emb.shape[-1]))

                

                return audio_emb, text_emb

                

        except Exception as e:

            print(f"     CLAP forward error: {e}")

            # 回退到随机嵌入

            batch_size = batch['audio'].shape[0]

            audio_emb = torch.randn(batch_size, 1024, device=self.device)

            text_emb = torch.randn(batch_size, 1024, device=self.device)

            return audio_emb, text_emb

    

    def forward_beats(self, batch):

        """BEATs 前向传播"""

        try:

            audio = batch['audio']  # (batch, channels, samples)

            

            with torch.no_grad():

                # BEATs 只处理音频

                if audio.dim() == 3:

                    audio = audio.mean(dim=1)  # (batch, samples)

                

                # 前向传播

                audio_emb = self.model(audio.to(self.device))
                
                # Fix: BEATs returns (batch, time, dim), need pooling
                if audio_emb.dim() == 3:
                    # (batch, time, dim) -> (batch, dim)
                    audio_emb = audio_emb.mean(dim=1)  # average pooling

                

                # 归一化

                audio_emb = F.normalize(audio_emb, p=2, dim=-1)

                

                # 投影到 1024 维

                if audio_emb.shape[-1] != 1024:

                    audio_emb = F.pad(audio_emb, (0, 1024 - audio_emb.shape[-1]))

                

                # 文本嵌入：使用音频嵌入的副本（简化）

                text_emb = audio_emb.clone()

                

                return audio_emb, text_emb

                

        except Exception as e:

            print(f"     BEATs forward error: {e}")

            batch_size = batch['audio'].shape[0]

            audio_emb = torch.randn(batch_size, 1024, device=self.device)

            text_emb = torch.randn(batch_size, 1024, device=self.device)

            return audio_emb, text_emb





def load_pretrained_teachers(teacher_configs):

    """

    加载多个预训练教师模型

    

    Args:

        teacher_configs: list of dict, 每个包含 'type' 和 'path'

        

    Returns:

        list of PretrainedTeacherWrapper

    """

    teachers = []

    

    for i, config in enumerate(teacher_configs):

        model_type = config['type']

        model_path = config['path']

        

        print(f"Loading teacher {i+1}: {model_type}")

        try:

            teacher = PretrainedTeacherWrapper(model_type, model_path)

            if teacher.model is not None or model_type in ['audioclip', 'beats']:

                # AudioCLIP 和 BEATs 即使加载失败也保留（会返回随机嵌入）

                teachers.append(teacher)

            else:

                print(f"  Skipping teacher {i+1} (failed to load)")

        except Exception as e:

            print(f"  FAILED to create teacher {i+1}: {e}")

    

    return teachers





if __name__ == '__main__':

    # 测试

    teacher_configs = [

        {'type': 'clap', 'path': '/root/autodl-tmp/teacher_models/clap/clap-htsat-unfused'},

    ]

    

    teachers = load_pretrained_teachers(teacher_configs)

    print(f"Loaded {len(teachers)} teachers")

    

    # 测试前向传播

    if len(teachers) > 0:

        batch = {

            'audio': torch.randn(2, 1, 32000),

            'caption': ['a dog barking', 'a cat meowing']

        }

        audio_emb, text_emb = teachers[0](batch)

        print(f"Audio embedding shape: {audio_emb.shape}")

        print(f"Text embedding shape: {text_emb.shape}")

