# 多音频编码器融合 - train.py 添加的参数和初始化代码

## 1. 在get_args()中添加参数（约在第210行附近）

```python
# 🔥 多音频编码器融合参数
parser.add_argument('--use_multi_encoder', default=False, action=argparse.BooleanOptionalAction, 
                    help='启用多音频编码器融合（PaSST+BEATs+CLAP）')
parser.add_argument('--use_passt', default=True, action=argparse.BooleanOptionalAction, 
                    help='在多编码器中使用PaSST')
parser.add_argument('--use_beats', default=True, action=argparse.BooleanOptionalAction, 
                    help='在多编码器中使用BEATs')
parser.add_argument('--use_clap', default=True, action=argparse.BooleanOptionalAction, 
                    help='在多编码器中使用CLAP')
parser.add_argument('--fusion_type', type=str, default='attention', 
                    choices=['concat', 'weighted', 'attention'],
                    help='多编码器融合策略：concat(拼接), weighted(加权), attention(注意力)')
parser.add_argument('--beats_model_path', type=str, 
                    default='/root/autodl-tmp/teacher_models/beats/BEATs_iter3_plus_AS2M.pt',
                    help='BEATs模型路径')
parser.add_argument('--clap_model_name', type=str, 
                    default='laion/clap-htsat-unfused',
                    help='CLAP模型名称或路径')
```

## 2. 在模型初始化后添加（约在第450行，model = AudioRetrievalModel(**model_args)之后）

```python
# 🔥 加载多音频编码器（如果启用）
if args['use_multi_encoder']:
    print("=" * 50)
    print("🎵 加载多音频编码器...")
    print("=" * 50)
    
    try:
        # 加载BEATs
        if args['use_beats']:
            print("📥 加载BEATs模型...")
            from d25_t6.multi_audio_encoder import load_beats_model
            
            beats_model = load_beats_model(
                model_path=args['beats_model_path'],
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if beats_model is not None:
                model.audio_embedding_model.set_beats_encoder(beats_model)
                print(f"✅ BEATs加载成功: {args['beats_model_path']}")
            else:
                print("⚠️ BEATs加载失败，将只使用PaSST")
        
        # 加载CLAP
        if args['use_clap']:
            print("📥 加载CLAP模型...")
            from d25_t6.multi_audio_encoder import load_clap_model
            
            clap_model = load_clap_model(
                model_name=args['clap_model_name'],
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if clap_model is not None:
                model.audio_embedding_model.set_clap_encoder(clap_model)
                print(f"✅ CLAP加载成功: {args['clap_model_name']}")
            else:
                print("⚠️ CLAP加载失败，将只使用PaSST")
        
        print("=" * 50)
        print("✅ 多音频编码器初始化完成！")
        print(f"   融合策略: {args['fusion_type']}")
        print(f"   使用编码器: ", end="")
        encoders = []
        if args['use_passt']: encoders.append("PaSST")
        if args['use_beats']: encoders.append("BEATs")
        if args['use_clap']: encoders.append("CLAP")
        print(" + ".join(encoders))
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 多音频编码器初始化失败: {e}")
        print("   将回退到单一PaSST编码器")
        import traceback
        traceback.print_exc()
        model.use_multi_encoder = False
        print("=" * 50)
```

## 3. 使用示例

### 方法1: 使用所有三个编码器（注意力融合）
```bash
python train.py \
    --use_multi_encoder \
    --use_passt \
    --use_beats \
    --use_clap \
    --fusion_type attention \
    --batch_size 16 \
    --max_lr 2e-5 \
    --max_epochs 40
```

### 方法2: 只使用PaSST+BEATs（加权融合）
```bash
python train.py \
    --use_multi_encoder \
    --use_passt \
    --use_beats \
    --no-use_clap \
    --fusion_type weighted \
    --batch_size 16 \
    --max_lr 2e-5
```

### 方法3: 只使用PaSST+CLAP（拼接融合）
```bash
python train.py \
    --use_multi_encoder \
    --use_passt \
    --no-use_beats \
    --use_clap \
    --fusion_type concat \
    --batch_size 16 \
    --max_lr 2e-5
```

## 4. 预期效果

| 配置 | 预期mAP@10 | 提升 | 说明 |
|------|-----------|------|------|
| 单一PaSST | 0.29 | 基线 | 当前配置 |
| PaSST+BEATs | 0.31-0.32 | +2-3% | 推荐优先尝试 |
| PaSST+CLAP | 0.32-0.33 | +3-4% | CLAP专门为音频-文本设计 |
| PaSST+BEATs+CLAP | 0.33-0.35 | +4-6% | 最强配置 |

## 5. 注意事项

### 显存需求
- 单一PaSST: ~10GB
- PaSST+BEATs: ~12GB
- PaSST+CLAP: ~13GB
- 三者全开: ~15GB

如果显存不足，降低batch size：
```bash
--batch_size 12  # 或更小
```

### 训练时间
- 多编码器会增加训练时间（约1.5-2倍）
- 但性能提升显著，值得投入

### 模型下载
需要提前下载：
1. BEATs: https://github.com/microsoft/unilm/tree/master/beats
2. CLAP: 会自动从HuggingFace下载（需要网络）

如果服务器无法连接外网，需要手动下载CLAP并指定本地路径。

























