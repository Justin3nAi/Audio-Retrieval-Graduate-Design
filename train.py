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

    # checkpoint callback - 每轮验证并保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch}-{val/mAP@10:.2f}",
        save_top_k=1,
        monitor="val/mAP@10",
        mode="max",
        every_n_epochs=1,  # 每轮都检查，避免错过最佳模型
        save_last=True
    )
    
    # 🔥 早停策略 - 防止过拟合
    early_stop_callback = EarlyStopping(
        monitor='val/mAP@10',
        patience=8,  # 8轮不提升就停止
        mode='max',
        verbose=True,
        min_delta=0.001  # 至少提升0.1%才算改进
    )

    # trainer
    trainer = pl.Trainer(
        devices=args['devices'],
        logger=logger if wandb.run else None,
        callbacks=[checkpoint_callback, early_stop_callback],  # 🔥 添加早停
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,  # 每轮验证，更好监控训练
        fast_dev_run=False,
        gradient_clip_val=1.0,  # 梯度裁剪，稳定训练
        accumulate_grad_batches=args.get('accumulate_grad_batches', 1)  # 梯度累积，增大有效 batch
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
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path to checkpoint used as a weight initialization for training.')

    # Training parameters - 🔥 修复过拟合：增加正则化 + 早停策略
    parser.add_argument('--seed', type=int, default=21208, help='Random seed of experiment')
    parser.add_argument('--batch_size', type=int, default=48, help='roberta-base更小，可以增加batch size')
    parser.add_argument('--batch_size_eval', type=int, default=24, help='Batch size for evaluation')
    parser.add_argument('--accumulate_grad_batches', type=int, default=2, help='梯度累积2次，有效batch=96')
    parser.add_argument('--max_epochs', type=int, default=50, help='🔥 减少到50 epochs，配合早停策略')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='🔥 增加warmup到5轮，更稳定')
    parser.add_argument('--rampdown_epochs', type=int, default=40, help='40轮衰减期')
    parser.add_argument('--max_lr', type=float, default=6e-5, help='🔥 平衡：提高到6e-5，在稳定和性能间平衡')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率1e-6')
    parser.add_argument('--use_cosine_restarts', default=True, action=argparse.BooleanOptionalAction, help='启用学习率重启')
    parser.add_argument('--restart_period', type=int, default=10, help='🔥 每10轮重启学习率')
    parser.add_argument('--initial_tau', type=float, default=0.07, help='初始tau为0.07')
    parser.add_argument('--tau_trainable', default=True, action=argparse.BooleanOptionalAction, help='Temperature parameter is trainable or not.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='🔥 平衡：降低到0.01，在过拟合和欠拟合间平衡')
    parser.add_argument('--use_mlp_projection', default=False, action=argparse.BooleanOptionalAction, help='Use MLP projection head instead of linear')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='🔥 平衡：降低到0.1，在过拟合和欠拟合间平衡')
    
    # 启用所有有效的优化技术
    parser.add_argument('--use_improved_projection', default=True, action=argparse.BooleanOptionalAction, help='使用改进的投影头')
    parser.add_argument('--use_cross_attention', default=False, action=argparse.BooleanOptionalAction, help='暂时禁用交叉注意力')
    parser.add_argument('--cross_attn_warmup_epochs', type=int, default=100, help='Number of epochs before enabling cross attention')
    parser.add_argument('--use_multi_layer_text', default=True, action=argparse.BooleanOptionalAction, help='启用多层文本特征融合')
    parser.add_argument('--use_ema', default=True, action=argparse.BooleanOptionalAction, help='启用EMA提高泛化')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='降低EMA decay到0.999')
    parser.add_argument('--use_layerwise_lr', default=True, action=argparse.BooleanOptionalAction, help='启用分层学习率')
    parser.add_argument('--use_improved_schedule', default=True, action=argparse.BooleanOptionalAction, help='使用改进的学习率调度')
    parser.add_argument('--loss_type', type=str, default='infonce', choices=['infonce', 'improved_infonce', 'focal'], help='🔥 禁用Hard Negative Mining，使用标准InfoNCE')
    parser.add_argument('--hard_negative_weight', type=float, default=0.0, help='🔥 完全禁用Hard Negative Mining')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='禁用标签平滑')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')

    # PaSST parameters
    parser.add_argument('--s_patchout_t', type=int, default=15, help='Temporal patchout size')
    parser.add_argument('--s_patchout_f', type=int, default=2, help='Frequency patchout size')
    parser.add_argument('--mel_freqm', type=int, default=24, help='Mel SpecAugment freq mask (0=disable)')
    parser.add_argument('--mel_timem', type=int, default=96, help='Mel SpecAugment time mask (0=disable)')

    # RoBERTa parameters - 切换到roberta-base
    parser.add_argument('--roberta_base', default=True, action=argparse.BooleanOptionalAction,  help='切换到Roberta base（更适合小数据集）')
    parser.add_argument('--roberta_model_path', type=str, default='/root/autodl-tmp/huggingface_cache', help='本地RoBERTa模型路径')
    
    # use additional data sets...
    parser.add_argument('--wavcaps', default=False, action=argparse.BooleanOptionalAction, help='Include WavCaps in the training or not.')
    parser.add_argument('--audiocaps', default=True, action=argparse.BooleanOptionalAction, help='Include AudioCaps in the training or not. (默认启用)')
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
        
        # 设置CUDA内存优化
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("✅ 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print()
        
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
