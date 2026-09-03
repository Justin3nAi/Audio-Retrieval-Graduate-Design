"""
Download model weights from HuggingFace Hub or Google Drive

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --source huggingface
    python scripts/download_model.py --source gdrive --file_id YOUR_FILE_ID
"""

import argparse
import os
from pathlib import Path


def download_from_huggingface(repo_id, filename, local_dir):
    """Download model from HuggingFace Hub"""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"📥 Downloading from HuggingFace Hub...")
        print(f"   Repository: {repo_id}")
        print(f"   File: {filename}")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Location: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading from HuggingFace: {e}")
        return None


def download_from_gdrive(file_id, output_path):
    """Download model from Google Drive"""
    try:
        import gdown
        
        print(f"📥 Downloading from Google Drive...")
        print(f"   File ID: {file_id}")
        
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        
        print(f"✅ Model downloaded successfully!")
        print(f"   Location: {output_path}")
        return output_path
        
    except ImportError:
        print("❌ gdown not installed. Install with: pip install gdown")
        return None
    except Exception as e:
        print(f"❌ Error downloading from Google Drive: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "gdrive"],
        help="Download source (default: huggingface)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Justinzhu09/Audio-Retrieval-Graduate-Design",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="model.ckpt",
        help="Model filename"
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default=None,
        help="Google Drive file ID (for gdrive source)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="audio_app/checkpoints",
        help="Output directory (default: audio_app/checkpoints)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🎵 Audio-Text Retrieval Model Downloader")
    print("=" * 60)
    
    if args.source == "huggingface":
        result = download_from_huggingface(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(output_dir)
        )
    elif args.source == "gdrive":
        if not args.file_id:
            print("❌ Error: --file_id required for Google Drive download")
            print("   Example: python scripts/download_model.py --source gdrive --file_id YOUR_FILE_ID")
            return
        
        output_path = output_dir / args.filename
        result = download_from_gdrive(
            file_id=args.file_id,
            output_path=output_path
        )
    
    if result:
        print("\n" + "=" * 60)
        print("✅ Setup complete! You can now run the application:")
        print("   python audio_app/run_app.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Download failed. Please try:")
        print("   1. Check your internet connection")
        print("   2. Verify the repository ID or file ID")
        print("   3. Download manually from:")
        print("      HuggingFace: https://huggingface.co/Justinzhu09/Audio-Retrieval-Graduate-Design")
        print("      Google Drive: [your-link-here]")
        print("=" * 60)


if __name__ == "__main__":
    main()
