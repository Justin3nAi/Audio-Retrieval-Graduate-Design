import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")

import os
from typing import Union, List, Mapping
import torch
import wandb
import argparse
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from aac_datasets import Clotho, WavCaps, AudioCaps
from torch.utils.data import DataLoader
from d25_t6.datasets.download_datasets import download_clotho, download_audiocaps, download_wavcaps_mp3
from d25_t6.datasets.audio_loading import custom_loading
from d25_t6.datasets.utils import exclude_broken_files, exclude_forbidden_files, exclude_forbidden_and_long_files
from d25_t6.datasets.batch_collate import CustomCollate

from d25_t6.retrieval_module import AudioRetrievalModel




def train(
        model: AudioRetrievalModel,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
):
    """
    Trains the AudioRetrievalModel using provided datasets, logger, and configuration arguments.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The model to be trained.
        train_ds (torch.utils.data.Dataset): The training dataset.
        val_ds (torch.utils.data.Dataset): The validation dataset.
        logger (Union[None, WandbLogger]): The logger for tracking training metrics.
        args (dict): A dictionary of configuration arguments for training.

    Returns:
        d25_t6.retrieval_module.AudioRetrievalModel: The trained model.
    """
    # get a unique experiment name for name of checkpoint
    if wandb.run is not None:
        experiment_name = wandb.run.name or wandb.run.id  # Use name if available, else use ID
    else:
        experiment_name = "experiment_" + wandb.util.generate_id()  # Random unique ID fallback

    # create path for the model checkpoints
    checkpoint_dir = os.path.join(args["checkpoints_path"], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    # checkpoint callback - 🔥 回到基线：每5轮验证并保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch}-{val/mAP@10:.2f}",
        save_top_k=1,
        monitor="val/mAP@10",
        mode="max",
        every_n_epochs=5,  # 🔥 回到基线：每5轮
        save_last=True
    )
    
    # 🔥 早停策略 - 防止过拟合
    early_stop_callback = EarlyStopping(
        monitor='val/mAP@10',
        patience=8,  # 8个验证周期（40个epoch）没有提升就停止
        mode='max',
        verbose=True,
        min_delta=0.001  # 至少提升0.001才算改进
    )

    # trainer
    trainer = pl.Trainer(
        devices=args['devices'],
        logger=logger if wandb.run else None,
        callbacks=[checkpoint_callback, early_stop_callback],  # 🔥 添加早停
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=5,  # 🔥 回到基线：每5轮验证
        fast_dev_run=False,
        gradient_clip_val=1.0,  # 梯度裁剪，稳定训练
        accumulate_grad_batches=args.get('accumulate_grad_batches', 1)
    )

    ### train on training set; monitor performance on val
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_ds, batch_size=args['batch_size'], num_workers=args['n_workers'], shuffle=True, drop_last=True,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        val_dataloaders=DataLoader(
            val_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        ),
        ckpt_path=args['resume_ckpt_path'] # should be none unless training is resumed
    )

    return model

def test(
        model: AudioRetrievalModel,
        test_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
) -> List[Mapping[str, float]]:
    """
    Tests the trained AudioRetrievalModel on a given test dataset.

    Args:
        model (d25_t6.retrieval_module.AudioRetrievalModel): The trained model to be evaluated.
        test_ds (torch.utils.data.Dataset): The test dataset.
        logger (Union[None, WandbLogger]): The logger for tracking test metrics.
        args (dict): A dictionary of configuration arguments for testing.

    Returns:
        dict: The result of the model evaluation on the test dataset.
    """
    trainer = pl.Trainer(
        devices=args['devices'],
        logger=logger if wandb.run else None,
        callbacks=None,
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### test on the eval set
    result = trainer.test(
        model,
        DataLoader(
            test_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate()
        )
    )

    return result


def get_args() -> dict:
    """
    Parses command-line arguments for configuring the training and testing process.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training configuration.")

    parser.add_argument('--devices', type=str, default='auto', help='Device selection (e.g., auto, cpu, cuda, etc.)')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction, help='Compile the model if GPU version >= 7.')
    parser.add_argument('--logging', default=True, action=argparse.BooleanOptionalAction, help='Log metrics in wandb or not.')

    # Parameter initialization & resume training
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--load_ckpt_path', type=str, default='/root/autodl-tmp/ProjectAR/teacher_checkpoints/best_checkpoint/mAP@10=0.32.ckpt', help='Path to checkpoint used as a weight initialization for training.')

    # Training parameters - 🔥 回归简单有效的配置（参考0.301基线）
    parser.add_argument('--seed', type=int, default=21208, help='Random seed of experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size matching best baseline')
    parser.add_argument('--batch_size_eval', type=int, default=32, help='Batch size for evaluation (can be larger)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=2, help='Gradient accumulation, effective batch=64')
    parser.add_argument('--max_epochs', type=int, default=25, help='Fine-tune epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='No warmup needed for fine-tuning')
    parser.add_argument('--rampdown_epochs', type=int, default=20, help='Long rampdown for fine-tuning')
    parser.add_argument('--max_lr', type=float, default=5e-6, help='Fine-tune learning rate (1/4 of baseline)')
    parser.add_argument('--min_lr', type=float, default=1e-8, help='Minimum learning rate')
    parser.add_argument('--use_cosine_restarts', default=False, action=argparse.BooleanOptionalAction, help='Disable learning rate restarts')
    parser.add_argument('--restart_period', type=int, default=15, help='Learning rate restart period (disabled)')
    parser.add_argument('--initial_tau', type=float, default=0.05, help='Initial tau value')
    parser.add_argument('--tau_trainable', default=False, action=argparse.BooleanOptionalAction, help='Tau not trainable')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--use_mlp_projection', default=True, action=argparse.BooleanOptionalAction, help='🔥 使用MLP投影头，提升表达能力')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='🔥 增加dropout到0.3防止过拟合')
    
    # 🔥 启用有效的优化
    parser.add_argument('--use_improved_projection', default=True, action=argparse.BooleanOptionalAction, help='🔥 启用改进投影头')
    parser.add_argument('--use_attentive_aggregation', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用注意力聚合，使用简单平均')
    parser.add_argument('--use_attention_pooling', default=True, action=argparse.BooleanOptionalAction, help='🔥 启用注意力池化')
    parser.add_argument('--use_cross_attention', default=False, action=argparse.BooleanOptionalAction, help='禁用交叉注意力')
    parser.add_argument('--cross_attn_warmup_epochs', type=int, default=100, help='Number of epochs before enabling cross attention')
    parser.add_argument('--use_multi_layer_text', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用多层融合，使用单层[CLS]')
    parser.add_argument('--use_ema', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用EMA，保持简单')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay（已禁用）')
    parser.add_argument('--use_layerwise_lr', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用分层学习率')
    parser.add_argument('--use_improved_schedule', default=False, action=argparse.BooleanOptionalAction, help='🔥 使用标准学习率调度')
    parser.add_argument('--loss_type', type=str, default='infonce', choices=['infonce', 'improved_infonce', 'focal'], help='🔥 使用标准InfoNCE（最稳定）')
    parser.add_argument('--hard_negative_weight', type=float, default=0.0, help='🔥 禁用Hard Negative Mining')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='禁用标签平滑')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    
    # 🔥 知识蒸馏参数（Kim论文核心技术，+4.3% mAP）
    parser.add_argument('--use_ensemble_distillation', default=True, action=argparse.BooleanOptionalAction,
                        help='启用集成知识蒸馏（Kim论文）')
    parser.add_argument('--use_pretrained_teachers', default=False, action=argparse.BooleanOptionalAction,
                        help='使用预训练模型作为教师（CLAP）')
    parser.add_argument('--teacher_checkpoints', type=str, nargs='+',
                        default=[
                            '/root/autodl-tmp/ProjectAR/teacher_checkpoints/teacher1_passt_only/last.ckpt',
                            '/root/autodl-tmp/ProjectAR/teacher_checkpoints/teacher2_passt_clap/mAP@10=0.32.ckpt'
                        ],
                        help='已训练的教师模型checkpoint路径（Kim论文方法）')
    parser.add_argument('--ensemble_distill_temperature', type=float, default=1.0,
                        help='蒸馏温度（Kim论文使用1.0）')
    parser.add_argument('--ensemble_distill_weight', type=float, default=1.0,
                        help='蒸馏损失权重（Kim论文λ1=1.0）')
    
    # 🔥 数据增强参数（高性价比优化，+3-5% mAP）
    parser.add_argument('--use_augmentation', default=True, action=argparse.BooleanOptionalAction,
                        help='启用数据增强（SpecAugment + Mixup）')
    parser.add_argument('--use_mixup', default=True, action=argparse.BooleanOptionalAction,
                        help='启用 Mixup 音频混合')
    
    # 🔥 改进损失函数参数（高性价比优化，+2-3% mAP）
    parser.add_argument('--use_improved_loss', default=True, action=argparse.BooleanOptionalAction,
                        help='启用改进损失函数（Focal Loss + Hard Negative Mining）')
    
    # 🔥 禁用所有其他蒸馏，只用Kim的方法
    parser.add_argument('--use_online_distillation', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用在线蒸馏')
    parser.add_argument('--distill_temperature', type=float, default=2.0, help='蒸馏温度（已禁用）')
    parser.add_argument('--distill_alpha', type=float, default=0.5, help='在线蒸馏权重（已禁用）')
    
    # 🔥 禁用聚类分类
    parser.add_argument('--use_clustering_classification', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用聚类分类')
    parser.add_argument('--num_clusters', type=int, default=50, help='聚类数量（已禁用）')
    parser.add_argument('--clustering_weight', type=float, default=0.05, help='聚类损失权重（已禁用）')
    
    # 🔥 禁用AudioCLIP蒸馏
    parser.add_argument('--use_audioclip_distillation', default=False, action=argparse.BooleanOptionalAction, help='🔥 禁用AudioCLIP蒸馏')
    parser.add_argument('--audioclip_model_path', type=str, default='/root/autodl-tmp/teacher_models/audioclip/AudioCLIP-Full-Training.pt', help='AudioCLIP模型路径（已禁用）')
    parser.add_argument('--audioclip_temperature', type=float, default=2.0, help='AudioCLIP蒸馏温度（已禁用）')
    parser.add_argument('--audioclip_alpha', type=float, default=0.7, help='AudioCLIP蒸馏权重（已禁用）')
    
    # 🔥 CLIP蒸馏参数 (已弃用)
    parser.add_argument('--use_multi_teacher_distillation', default=False, action=argparse.BooleanOptionalAction, help='禁用CLIP蒸馏（效果不佳）')
    parser.add_argument('--teacher_model_dir', type=str, default='/root/autodl-tmp/teacher_models', help='教师模型目录（可选）')
    parser.add_argument('--clip_model_path', type=str, default='/root/autodl-tmp/huggingface_cache/clip-vit-base-patch32', help='本地CLIP模型路径')
    
    # 🔥 MoE参数 (Mixture of Experts)
    parser.add_argument('--use_moe', default=False, action=argparse.BooleanOptionalAction, help='在交叉注意力的FFN中使用MoE')
    parser.add_argument('--num_experts', type=int, default=4, help='MoE专家数量')
    parser.add_argument('--top_k', type=int, default=2, help='MoE每次激活的专家数量')
    parser.add_argument('--moe_load_balance_weight', type=float, default=0.01, help='MoE负载均衡损失权重')

    # ============ 多音频编码器融合参数 ============
    parser.add_argument('--use_multi_encoder', default=True, action=argparse.BooleanOptionalAction,
                        help='🔥 默认启用多音频编码器融合')
    parser.add_argument('--use_passt', default=True, action=argparse.BooleanOptionalAction,
                        help='🔥 使用PaSST编码器')
    parser.add_argument('--use_beats', default=False, action=argparse.BooleanOptionalAction,
                        help='🔥 禁用BEATs（PaSST+CLAP效果更好）')
    parser.add_argument('--use_clap', default=True, action=argparse.BooleanOptionalAction,
                        help='🔥 使用CLAP编码器')
    parser.add_argument('--fusion_type', type=str, default='attention',
                        choices=['concat', 'weighted', 'attention'],
                        help='🔥 默认使用attention融合策略')
    parser.add_argument('--beats_model_path', type=str,
                        default='/root/autodl-tmp/teacher_models/beats/BEATs_iter3_plus_AS2M.pt',
                        help='BEATs模型路径')
    parser.add_argument('--clap_model_name', type=str,
                        default='/root/autodl-tmp/teacher_models/clap/clap-htsat-unfused',
                        help='CLAP模型名称或路径')

    # ============ 多粒度语义对齐参数 ============
    parser.add_argument('--use_multi_granularity', default=False, action=argparse.BooleanOptionalAction,
                        help='🔥 禁用多粒度对齐（显存不足）')
    parser.add_argument('--mg_global_weight', type=float, default=1.0,
                        help='多粒度对齐中全局相似度的权重')
    parser.add_argument('--mg_local_weight', type=float, default=0.0,
                        help='多粒度对齐中局部相似度的权重')

    # PaSST parameters - 🔥 回到基线的数据增强
    parser.add_argument('--s_patchout_t', type=int, default=15, help='🔥 回到基线的15')
    parser.add_argument('--s_patchout_f', type=int, default=2, help='🔥 回到基线的2')
    parser.add_argument('--mel_freqm', type=int, default=48, help='频率mask')
    parser.add_argument('--mel_timem', type=int, default=192, help='时间mask')

    # RoBERTa parameters - 🔥 回到基线：使用roberta-large
    parser.add_argument('--roberta_base', default=False, action=argparse.BooleanOptionalAction,  help='🔥 回到基线：使用roberta-large')
    parser.add_argument('--roberta_model_path', type=str, default='/root/autodl-tmp/huggingface_cache', help='本地RoBERTa模型路径')
    
    # use additional data sets...
    parser.add_argument('--wavcaps', default=False, action=argparse.BooleanOptionalAction, help='Include WavCaps in the training or not.')
    parser.add_argument('--audiocaps', default=True, action=argparse.BooleanOptionalAction, help='🔥 默认使用AudioCaps增加训练数据')
    parser.add_argument('--ablate_clean_setup', default=True, action=argparse.BooleanOptionalAction, help='Include ClothoV2.1 eval, test in the training or not.')

    # Paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset; dataset will be downloaded into this folder.')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='Path to save checkpoints to.')

    # run training / test
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help='Run training or not.')
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction, help='Run testing or not.')

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    try:
        """
        Entry point for training and testing the model.
        - Downloads datasets if necessary.
        - Initializes logging and model.
        - Runs training and/or testing based on arguments.
        """
        # ============================================
        # 🚀 激进学习率重启策略 - 突破0.29平台期
        # ============================================
        
        args = get_args()
        roberta_model_path=args['roberta_model_path']
        print("Data path:", args["data_path"])

        print("Contents of data path:", os.listdir(args["data_path"]))
        os.makedirs(args["data_path"], exist_ok=True)
        
        # 数据集已手动下载，禁用自动下载
        # download data sets; will be ignored if exists
        # ClothoV2.1
        # download_clotho(args["data_path"])
        # AudioCAps
        # if args['audiocaps']:
        #     download_audiocaps(args["data_path"])
        # WavCaps
        # if args['wavcaps']:
        #     download_wavcaps_mp3(args["data_path"])
        
        # 验证数据集是否存在
        if args['audiocaps']:
            audiocaps_path = os.path.join(args["data_path"], "AUDIOCAPS")
            if not os.path.exists(audiocaps_path):
                raise FileNotFoundError(f"❌ AudioCaps数据集未找到！请确保已解压到: {audiocaps_path}")
            else:
                print(f"✅ 找到AudioCaps数据集: {audiocaps_path}")
                # 检查关键文件和目录
                audio_dir = os.path.join(audiocaps_path, "audio_32000Hz")
                train_dir = os.path.join(audio_dir, "train")
                train_csv = os.path.join(audiocaps_path, "train.csv")
                
                print(f"   📁 检查目录结构:")
                print(f"      - audio_32000Hz/: {'✅ 存在' if os.path.exists(audio_dir) else '❌ 不存在'}")
                print(f"      - audio_32000Hz/train/: {'✅ 存在' if os.path.exists(train_dir) else '❌ 不存在'}")
                print(f"      - train.csv: {'✅ 存在' if os.path.exists(train_csv) else '❌ 不存在'}")
                
                if os.path.exists(train_dir):
                    audio_files = [f for f in os.listdir(train_dir) if f.endswith('.mp3')]
                    print(f"      - 训练音频文件数: {len(audio_files)} 个")

        # set a seed to make experiments reproducible
        
         # 检查并创建模型缓存目录
        model_cache_dir = "/root/autodl-tmp/huggingface_cache"
        os.makedirs(model_cache_dir, exist_ok=True)
        print(f"模型缓存目录内容: {os.listdir(model_cache_dir)}")
        
        # 设置RoBERTa模型路径
        roberta_model_name = "roberta-base" if args['roberta_base'] else "roberta-large"
        roberta_model_path = os.path.join(model_cache_dir, roberta_model_name)
        
        # 如果模型不存在，提示下载
        if not os.path.exists(os.path.join(roberta_model_path, "pytorch_model.bin")):
            print(f"⚠️ {roberta_model_name} 模型未找到在缓存目录中")
            print(f"请手动下载模型文件到: {roberta_model_path}")
            print(f"下载命令示例: ")
            print(f"mkdir -p {roberta_model_path}")
            print(f"wget https://hf-mirror.com/{roberta_model_name}/resolve/main/pytorch_model.bin -P {roberta_model_path}/")
            print(f"wget https://hf-mirror.com/{roberta_model_name}/resolve/main/config.json -P {roberta_model_path}/")
            print(f"wget https://hf-mirror.com/{roberta_model_name}/resolve/main/tokenizer.json -P {roberta_model_path}/")
            print(f"wget https://hf-mirror.com/{roberta_model_name}/resolve/main/tokenizer_config.json -P {roberta_model_path}/")
            print(f"wget https://hf-mirror.com/{roberta_model_name}/resolve/main/vocab.json -P {roberta_model_path}/")
        else:
            print(f"✅ 找到本地缓存的{roberta_model_name}模型")

        if args['seed'] > 0:
            seed_everything(args['seed'], workers=True)
        else:
            print("Not seeding experiment.")

        # initialize wandb, i.e., the logging framework
        if args['logging']:
            wandb.init(project="d25_t6")
            logger = WandbLogger()
        else:
            logger = None

        # initialize the model
        if args['load_ckpt_path']:
            model = AudioRetrievalModel.load_from_checkpoint(args['load_ckpt_path'])
        else:
            # 添加本地模型路径参数
            model_args = dict(args)
            model_args['roberta_model_path'] = roberta_model_path  # 传递本地路径
            model = AudioRetrievalModel(**model_args)
        
        # ============ 加载多音频编码器（如果启用）============
        if args['use_multi_encoder']:
            print("=" * 60)
            print("🎵 初始化多音频编码器...")
            print("=" * 60)
            
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
                        args['use_beats'] = False
                
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
                        args['use_clap'] = False
                
                # 打印配置摘要
                print("=" * 60)
                print("📊 多编码器配置:")
                print(f"   - PaSST: {'✅' if args['use_passt'] else '❌'}")
                print(f"   - BEATs: {'✅' if args['use_beats'] else '❌'}")
                print(f"   - CLAP: {'✅' if args['use_clap'] else '❌'}")
                print(f"   - 融合策略: {args['fusion_type']}")
                print("=" * 60)
                
            except Exception as e:
                print(f"❌ 多音频编码器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️ 回退到单编码器模式（仅PaSST）")
                model.use_multi_encoder = False
        
        # 🔥 加载AudioCLIP教师蒸馏模型（方案C - 最强优化）
        if args['use_audioclip_distillation']:
            print("=" * 50)
            print("🎓 加载AudioCLIP教师蒸馏模型...")
            print("=" * 50)
            try:
                from d25_t6.audioclip_distillation import load_audioclip_teacher
                
                teacher, distill_loss_fn = load_audioclip_teacher(
                    model_path=args['audioclip_model_path'],
                    temperature=args['audioclip_temperature'],
                    alpha=args['audioclip_alpha'],
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                if teacher is not None:
                    # 将教师模型和蒸馏损失函数添加到模型
                    model.audioclip_teacher = teacher
                    model.audioclip_distillation_loss_fn = distill_loss_fn
                    model.use_audioclip_distillation = True
                    
                    print("✅ AudioCLIP教师蒸馏模型加载成功！")
                    print(f"   - 模型路径: {args['audioclip_model_path']}")
                    print(f"   - 蒸馏温度: {args['audioclip_temperature']}")
                    print(f"   - 蒸馏权重: {args['audioclip_alpha']}")
                else:
                    print("⚠️ AudioCLIP教师模型加载失败，将继续使用标准训练")
                    model.use_audioclip_distillation = False
                    model.audioclip_teacher = None
                    
                print("=" * 50)
            except Exception as e:
                print(f"❌ AudioCLIP教师蒸馏模型加载失败: {e}")
                print("   将继续使用标准训练（无AudioCLIP蒸馏）")
                import traceback
                traceback.print_exc()
                model.use_audioclip_distillation = False
                model.audioclip_teacher = None
                print("=" * 50)
        else:
            model.use_audioclip_distillation = False
            model.audioclip_teacher = None
        
        # 🔥 加载CLIP教师蒸馏模型（已弃用，保留兼容性）
        if args['use_multi_teacher_distillation']:
            print("=" * 50)
            print("🎓 加载单教师蒸馏模型（AudioCLIP）...")
            print("=" * 50)
            try:
                teacher_ensemble = load_teacher_ensemble(
                    teacher_model_dir=args.get('teacher_model_dir'),
                    clip_model_path=args.get('clip_model_path'),  # 传递本地CLIP路径
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # 将教师模型和蒸馏损失函数添加到模型
                model.teacher_ensemble = teacher_ensemble
                model.multi_teacher_distillation_loss_fn = MultiTeacherDistillationLoss(
                    temperature=args['distill_temperature'],
                    alpha=args['distill_alpha']
                )
                model.use_multi_teacher_distillation = True
                
                print("✅ 单教师蒸馏模型加载成功！")
                print(f"   - 蒸馏温度: {args['distill_temperature']}")
                print(f"   - 蒸馏权重: {args['distill_alpha']}")
                print("=" * 50)
            except Exception as e:
                print(f"❌ 单教师蒸馏模型加载失败: {e}")
                print("   将继续使用标准训练（无蒸馏）")
                model.use_multi_teacher_distillation = False
                model.teacher_ensemble = None
                print("=" * 50)
        else:
            model.use_multi_teacher_distillation = False
            model.teacher_ensemble = None
        
        # 🔥 初始化聚类器（用于聚类引导分类）
        if args['use_clustering_classification']:
            print("=" * 50)
            print("🎯 初始化聚类器（Clustering-Guided Classification）...")
            print("=" * 50)
            try:
                from d25_t6.clustering_classification import prepare_clustering
                
                # 🔥 先将模型移到GPU
                if torch.cuda.is_available():
                    model = model.cuda()
                    print(f"✅ 模型已移到GPU: {next(model.parameters()).device}")
                
                # 需要先加载训练数据集
                temp_train_ds = custom_loading(Clotho(subset="dev", root=args["data_path"], flat_captions=True))
                
                # 准备聚类
                cache_path = os.path.join(args['checkpoints_path'], 'cluster_cache.pkl')
                clusterer = prepare_clustering(
                    temp_train_ds, 
                    model, 
                    num_clusters=args['num_clusters'],
                    cache_path=cache_path
                )
                
                # 将聚类器添加到模型
                model.clusterer = clusterer
                
                # 🔥 清理显存：聚类完成后释放GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"✅ GPU缓存已清理")
                    # 显示当前显存使用
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"   当前显存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
                
                print("✅ 聚类器初始化成功！")
                print(f"   - 聚类数量: {args['num_clusters']}")
                print(f"   - 聚类权重: {args['clustering_weight']}")
                print("=" * 50)
            except Exception as e:
                print(f"❌ 聚类器初始化失败: {e}")
                print("   将继续训练（无聚类分类）")
                import traceback
                traceback.print_exc()
                model.use_clustering_classification = False
                model.clusterer = None
                # 即使失败也清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("=" * 50)
        else:
            model.use_clustering_classification = False
            model.clusterer = None

        # train
        if args['train']:
            # get training ad validation data sets; add the resampling transformation
            print("=" * 50)
            print("📊 加载训练数据集...")
            print("=" * 50)
            
            train_ds = custom_loading(Clotho(subset="dev", root=args["data_path"], flat_captions=True))
            print(f"✅ Clotho dev: {len(train_ds)} 样本")

            if args['audiocaps']:
                print("📥 加载AudioCaps数据集...")
                try:
                    ac = custom_loading(
                        AudioCaps(
                            subset="train", 
                            root=args["data_path"], 
                            download=False,  # 禁用自动下载
                            download_audio=False, 
                            audio_format='mp3'
                        )
                    )
                    print(f"✅ AudioCaps train 加载成功: {len(ac)} 样本")
                    
                    # 显示第一个样本的详细信息来验证
                    if len(ac) > 0:
                        sample = ac[0]
                        print(f"   📄 第一个样本信息:")
                        print(f"      - 音频文件: {sample.get('fname', 'N/A')}")
                        print(f"      - 音频形状: {sample.get('audio', torch.tensor([])).shape if 'audio' in sample else 'N/A'}")
                        print(f"      - 标注文本: {sample.get('captions', ['N/A'])[0] if 'captions' in sample else 'N/A'}")
                        print(f"      - 数据集来源: {sample.get('dataset', 'N/A')}")
                    
                    # 合并数据集
                    clotho_count = len(train_ds)
                    train_ds = torch.utils.data.ConcatDataset([train_ds, ac])
                    print(f"✅ 数据集合并成功!")
                    print(f"   - Clotho: {clotho_count} 样本")
                    print(f"   - AudioCaps: {len(ac)} 样本")
                    print(f"   - 总计: {len(train_ds)} 样本")
                    print(f"   - 数据增加: {len(train_ds) - clotho_count} 样本 ({(len(ac)/clotho_count)*100:.1f}% 增长)")
                    
                except Exception as e:
                    print(f"❌ AudioCaps加载失败: {e}")
                    print(f"   错误类型: {type(e).__name__}")
                    print(f"   错误详情: {str(e)}")
                    print("提示：请检查数据集路径和文件结构")
                    print(f"   预期路径: {os.path.join(args['data_path'], 'AUDIOCAPS')}")
                    raise

            if args['wavcaps']:
                # load the subsets
                wc_f = exclude_forbidden_files(custom_loading(WavCaps(subset="freesound", root=args["data_path"])))
                wc_b = custom_loading(WavCaps(subset="bbc", root=args["data_path"]))
                wc_s = custom_loading(WavCaps(subset="soundbible", root=args["data_path"]))
                wc_a = exclude_broken_files(custom_loading(WavCaps(subset="audioset_no_audiocaps" if not args["ablate_clean_setup"] else "audioset", root=args["data_path"])))
                train_ds = torch.utils.data.ConcatDataset([train_ds, wc_f, wc_b, wc_s, wc_a])

            val_ds = custom_loading(Clotho(subset="val", root=args["data_path"], flat_captions=True))
            print(f"✅ Clotho val: {len(val_ds)} 样本")
            print("=" * 50)

            model = train(model, train_ds, val_ds, logger, args)

        # test
        if args['test']:
            test_ds = custom_loading(Clotho(subset="eval", root=args["data_path"], flat_captions=True))

            results = test(model, test_ds, logger, args)
            print(results)

    except SystemError as e:
        if "error return without exception set" in str(e):
            print("检测到系统错误，尝试保存当前进度...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 0,
                'loss': 1.390
            }, 'emergency_checkpoint.pth')
            print("紧急检查点已保存")
