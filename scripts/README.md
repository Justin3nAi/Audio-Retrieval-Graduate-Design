# Scripts Directory

Helper scripts for the audio-text retrieval project.

## Available Scripts

### `download_model.py`

Downloads pre-trained model weights from HuggingFace Hub or Google Drive.

**Usage:**
```bash
# From HuggingFace (default)
python scripts/download_model.py

# From Google Drive
python scripts/download_model.py --source gdrive --file_id YOUR_FILE_ID
```

### `extract_captions.py`

Extracts and deduplicates audio captions from Clotho and AudioCaps datasets.

**Usage:**
```bash
python scripts/extract_captions.py --clotho_dir path/to/clotho --audiocaps_csv path/to/audiocaps.csv
```
