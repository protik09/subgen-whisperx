# SubGen-WhisperX

A powerful subtitle generation tool using WhisperX for accurate speech-to-text transcription with precise timestamp alignment.

## Features

- 🎯 Accurate speech recognition using WhisperX
- ⚡ GPU acceleration support (CUDA)
- 🎵 Handles both video and audio files
- 📁 Batch processing support
- ⏱️ Performance timing and logging
- 🔧 Multiple language model options
- 🎛️ Configurable compute device selection

## Prerequisites

- Python 3.10 or later
- NVIDIA GPU with CUDA 12 support (optional, for GPU acceleration)
- Latest driver from nVidia
- FFmpeg
- Git

## Installation

### The easy way

1. Clone the repository:

```bash
git clone https://github.com/protik09/subgen-whisperx.git
cd subgen-whisperx
```

2. Create and activate a conda environment:

```bash
.\activate_venv.ps1
```

or

```bash
.\activate_venv.sh
```

### The manual way

1. Clone the repository:

```bash
git clone https://github.com/protik09/subgen-whisperx.git
cd subgen-whisperx
```

2. Create and activate a conda environment:

```bash
conda create -n whisperx python=3.10 -y
conda activate whisperx
```

3. Install PyTorch with CUDA support:

```bash
conda install -y pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install required packages:

```bash
pip install whisperx ffmpeg coloredlogs halo
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir
```

## Usage

### Basic Usage

Generate subtitles for a single video file:

```bash
python subgen_whisperx.py -f path/to/video.mp4
```

### Advanced Options

Process all media files in a directory:

```bash
python subgen_whisperx.py -d path/to/directory
```

Specify compute device and model size:

```bash
python subgen_whisperx.py -f video.mp4 -c cuda -m medium
```

Set logging level:

```bash
python subgen_whisperx.py -f video.mp4 -l DEBUG
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-f`, `--file` | Path to input media file | None |
| `-d`, `--directory` | Path to directory containing media files | None |
| `-c`, `--compute_device` | Device for computation (`cuda` or `cpu`) | Auto-detect |
| `-m`, `--model_size` | WhisperX model size | `base.en` |
| `-l`, `--log-level` | Logging level | `ERROR` |
| `-t-`, `--txt` | Text file with file/folder paths | None |

## Output

The script generates SRT subtitle files in the same directory as the input media:

- Format: `filename.{language}-ai.srt`
- Example: `Meetings-0822.en-ai.srt` for a video called `Meetings-0822.mp4`

## Troubleshooting

If the automatic model selection leads to the CUDA Out of Memory Issue, just manually select 
the next smaller model using the `-m` flag.

## Performance

- GPU acceleration provides significantly faster processing
- There is CPU fallback if GPU access fails
- Progress indicators show real-time status
- Performance timing information displayed after completion

## License

This project operates under the MIT Open Source License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for the core transcription technology
- [FFmpeg](https://ffmpeg.org/) for media processing capabilities
