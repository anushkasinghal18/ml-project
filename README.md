# Voice Trial (Speaker Identification)

A simple machine learning project that identifies **who is speaking** from an audio clip.

It uses:
- **MFCC** features from audio
- **SVM** classifier for training
- Basic evaluation with accuracy, confusion matrix, and classification report

## Project Structure

- `scriptnew.py` - main training + evaluation + prediction script
- `dataset/` - speaker-wise folders (each folder = one person)
- `Input_dataset/` / `Voices/` - sample audio files
- `run.bat`, `run.ps1` - helper run scripts

## How It Works (Short)

1. Load audio files from `dataset/`
2. Convert audio to MFCC-based numeric features
3. Split into train/test sets
4. Train SVM model
5. Evaluate performance
6. Optionally predict speaker from file or microphone recording

## Run

Use your virtual environment, then run:

`python scriptnew.py`

## Notes

- More recordings per speaker usually give better accuracy.
- Keep recording conditions consistent (mic/noise/room).
- For some formats like `.aac`, FFmpeg may be needed.

---

