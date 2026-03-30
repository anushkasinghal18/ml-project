import os
import sys
import shutil
from collections import Counter

import librosa
import numpy as np
import sounddevice as sd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from scipy.io.wavfile import write as write_wav



def check_venv():
	"""
	Warn if script is not running in the project's .venv.
	Does NOT block execution — just warns the user.
	"""
	venv_path = os.path.join(os.path.dirname(__file__), ".venv")
	executable_lower = sys.executable.lower()
	venv_path_lower = venv_path.lower()

	if not executable_lower.startswith(venv_path_lower):
		print("\n[WARNING] Running with system Python (not venv)")
		print(f"Current: {sys.executable}")
		print(f"Ideal:   {os.path.join(venv_path, 'Scripts', 'python.exe')}")
		print("If you encounter errors, use the venv interpreter.\n")


check_venv()


# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "dataset"
SUPPORTED_EXTENSIONS = (".wav", ".ogg", ".flac", ".aac", ".mp3", ".m4a")
SAMPLE_RATE = 16000
N_MFCC = 13
RANDOM_STATE = 42

'''
Basically firstly model ko smjhane ke liye audio ko numbers me convert krna hota hai. MFCC : ek popular technique hai jo audio signal ko kuch features mei convert krega which is MFCC.
MFCC audio ko aise numerical features mein convert karta hai jo voice ke tone, pitch, timbre aur speaking pattern ko represent karte hain.
'''
def extract_mfcc_features(file_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
	"""
	Step 1 + Step 2: Audio ko fixed-size MFCC feature vector me convert karta hai.
	Har coefficient ka mean + std lekar final vector banata hai.
	"""
	try:
		signal, sr = librosa.load(file_path, sr=sample_rate, mono=True)

		if signal.size == 0:
			return None

		mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

		mfcc_mean = np.mean(mfcc, axis=1)
		mfcc_std = np.std(mfcc, axis=1)

		feature_vector = np.concatenate([mfcc_mean, mfcc_std])
		return feature_vector

	except Exception as exc:
		ext = os.path.splitext(file_path)[1].lower()
		error_name = type(exc).__name__
		error_detail = repr(exc)

		if ext == ".aac" and error_name == "NoBackendError":
			ffmpeg_installed = shutil.which("ffmpeg") is not None
			if not ffmpeg_installed:
				print(
					f"[WARN] Could not decode AAC file: {file_path} -> {error_name}: {error_detail}"
				)
				print(
					"       Reason: FFmpeg is not installed or not in PATH. "
					"Install FFmpeg or convert .aac files to .wav/.ogg."
				)
			else:
				print(
					f"[WARN] Could not decode AAC file: {file_path} -> {error_name}: {error_detail}"
				)
				print(
					"       FFmpeg seems available, so this AAC file may be corrupted "
					"or encoded with an unsupported codec profile."
				)
		else:
			print(f"[WARN] Could not process file: {file_path} -> {error_name}: {error_detail}")
		return None

'''
Data ko catogorize karna ki konsi voice kiski hai, 
uske liye hume labels ki zarurat hoti hai. 
Jese ki hamare dataset me har speaker ke liye alag folder hai, 
to hum folder ke naam ko label me use karenge.
'''
def load_dataset(dataset_dir=DATASET_DIR):
	"""
	Step 1 + Step 3:
	Dataset folders se features aur labels load karta hai.
	"""
	features = []
	labels = []
	skipped_files = 0

	if not os.path.isdir(dataset_dir):
		raise FileNotFoundError(
			f"Dataset folder not found: '{dataset_dir}'. "
			"Create it and add speaker subfolders with .wav files."
		)

	speaker_folders = sorted(
		[
			name
			for name in os.listdir(dataset_dir)
			if os.path.isdir(os.path.join(dataset_dir, name))
		]
	)

	if not speaker_folders:
		raise ValueError("No speaker folders found inside dataset directory.")

	print("\n=== Step 1: Data Preprocessing ===")
	print(f"Dataset path: {dataset_dir}")

	for speaker in speaker_folders:
		speaker_dir = os.path.join(dataset_dir, speaker)
		audio_files = sorted(
			[
				f
				for f in os.listdir(speaker_dir)
				if os.path.isfile(os.path.join(speaker_dir, f))
				and f.lower().endswith(SUPPORTED_EXTENSIONS)
			]
		)

		for audio_file in audio_files:
			file_path = os.path.join(speaker_dir, audio_file)
			feature_vector = extract_mfcc_features(file_path)
			if feature_vector is None:
				skipped_files += 1
				continue

			features.append(feature_vector)
			labels.append(speaker)
			print(f"Loaded: {file_path} -> label='{speaker}'")

	if not features:
		raise ValueError(
			"No valid audio features extracted from dataset. "
			"If your files are .aac, install FFmpeg so librosa/audioread can decode them."
		)

	X = np.array(features, dtype=np.float32)
	y = np.array(labels)

	print("\n=== Step 2: Feature Handling ===")
	print(f"MFCC count per frame: {N_MFCC}")
	print(f"Feature vector size per audio: {X.shape[1]} (mean+std)")

	print("\n=== Step 3: Dataset Creation ===")
	print(f"X shape: {X.shape}")
	print(f"y shape: {y.shape}")
	print("Samples per class:", dict(Counter(y)))
	if skipped_files:
		print(f"Skipped unreadable files: {skipped_files}")

	if len(np.unique(y)) < 2:
		raise ValueError(
			"Need at least 2 speaker classes with readable audio to train a classifier."
		)

	return X, y


# Test size decide karta hai ki total data me se kitna hissa testing ke liye jayega.
# Agar per-class samples bahut kam ho to split ko safer banaya jata hai.
def choose_test_size(y_labels):
	"""
	Step 4 helper:
	Small dataset me classes train/test dono me aayein, isliye dynamic test_size choose karta hai.
	"""
	class_counts = Counter(y_labels)
	min_per_class = min(class_counts.values())

	if min_per_class <= 2:
		return 0.5
	return 0.2

# Ab model train karke uski performance evaluate karte hain.
def train_and_evaluate_svm(X, y, kernel="linear"):
	"""
	Step 4 + Step 5 + Step 6:
	- Encode labels
	- Stratified split
	- Scale features
	- Train SVM
	- Evaluate with accuracy + confusion matrix
	"""
	print("\n=== Step 4: Train-Test Split ===")

	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y)

	print("Label mapping (number -> speaker):")
	for idx, class_name in enumerate(label_encoder.classes_):
		print(f"  {idx} -> {class_name}")

	test_size = choose_test_size(y)
	print(f"Using stratified split with test_size={test_size}")

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y_encoded,
		test_size=test_size,
		random_state=RANDOM_STATE,
		stratify=y_encoded,
	)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	print(f"Training samples: {len(X_train_scaled)}")
	print(f"Testing samples: {len(X_test_scaled)}")

	print("\n=== Step 5: Model Training ===")
	print("Kernel explanation:")
	print("- linear: simpler boundary, good first baseline for tiny datasets")
	print("- rbf: nonlinear boundary, often powerful but needs more data/tuning")

	model = SVC(kernel=kernel, C=1.0, gamma="scale")
	model.fit(X_train_scaled, y_train)
	print(f"SVM trained successfully with kernel='{kernel}'")

	print("\n=== Step 6: Evaluation ===")
	y_pred = model.predict(X_test_scaled)

	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc * 100:.2f}%")

	cm = confusion_matrix(y_test, y_pred)
	print("\nConfusion Matrix (rows=true, cols=predicted):")
	print(cm)

	print("\nClassification Report:")
	print(
		classification_report(
			y_test,
			y_pred,
			target_names=label_encoder.classes_,
			zero_division=0,
		)
	)

	print("How to read confusion matrix:")
	print("- Diagonal numbers = correct predictions")
	print("- Off-diagonal numbers = mistakes (speaker confusion)")

	artifacts = {
		"model": model,
		"scaler": scaler,
		"label_encoder": label_encoder,
	}
	return artifacts


def predict_speaker(audio_path, model, scaler, label_encoder):
	"""
	Step 7: Ek naye audio file ka speaker predict karta hai.
	"""
	feature_vector = extract_mfcc_features(audio_path)
	if feature_vector is None:
		raise ValueError(f"Could not extract features from: {audio_path}")

	feature_vector = feature_vector.reshape(1, -1)
	feature_vector_scaled = scaler.transform(feature_vector)
	pred_numeric = model.predict(feature_vector_scaled)[0]
	pred_label = label_encoder.inverse_transform([pred_numeric])[0]
	return pred_label


def record_from_microphone(output_path, duration_seconds=5, sample_rate=SAMPLE_RATE):
	"""
	Mic se audio record karke WAV file save karta hai prediction ke liye.
	"""
	if duration_seconds <= 0:
		raise ValueError("Recording duration must be greater than 0 seconds.")

	print(f"\nRecording starts now... Speak for {duration_seconds} seconds.")
	audio = sd.rec(
		int(duration_seconds * sample_rate),
		samplerate=sample_rate,
		channels=1,
		dtype="float32",
	)
	sd.wait()

	# WAV save karne ke liye float audio ko int16 format me convert karte hain.
	audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
	write_wav(output_path, sample_rate, audio_int16)
	print(f"Recording saved to: {output_path}")
	return output_path


def interactive_prediction(artifacts):
	"""
	Training ke baad interactive prediction menu chalata hai.
	TTY na ho to safely skip kar deta hai.
	"""
	if not sys.stdin.isatty():
		print("\n=== Step 7: Prediction (Interactive) ===")
		print("Skipping interactive prediction (non-interactive mode detected).")
		print("To use mic recording or file prediction, run in an interactive terminal:")
		print(f"  {os.path.join('.venv', 'Scripts', 'python.exe')} {os.path.basename(__file__)}")
		return

	print("\n=== Step 7: Prediction (Interactive) ===")
	print("Choose input mode:")
	print("1) Predict using an existing audio file")
	print("2) Record from microphone and predict")

	try:
		choice = input("Enter 1 or 2 (or press Enter to skip): ").strip()
	except (KeyboardInterrupt, EOFError):
		print("Skipping interactive prediction (no input received).")
		return

	if choice == "":
		print("Skipping interactive prediction.")
		return

	if choice == "1":
		try:
			audio_path = input("Enter full path to audio file (.wav/.ogg/.aac/...): ").strip().strip('"')
		except (KeyboardInterrupt, EOFError):
			print("Skipping interactive prediction.")
			return

		if not os.path.isfile(audio_path):
			print(f"[ERROR] File not found: {audio_path}")
			return

		predicted = predict_speaker(
			audio_path,
			artifacts["model"],
			artifacts["scaler"],
			artifacts["label_encoder"],
		)
		print(f"Predicted speaker: {predicted}")
		return

	if choice == "2":
		try:
			duration_input = input("Enter recording duration in seconds (default 5): ").strip()
		except (KeyboardInterrupt, EOFError):
			print("Skipping interactive prediction.")
			return

		if duration_input == "":
			duration_seconds = 5
		else:
			try:
				duration_seconds = int(duration_input)
			except ValueError:
				print("[ERROR] Duration must be an integer.")
				return

		recorded_file = os.path.join(os.getcwd(), "live_test_recording.wav")
		try:
			record_from_microphone(
				output_path=recorded_file,
				duration_seconds=duration_seconds,
				sample_rate=SAMPLE_RATE,
			)
		except Exception as exc:
			print(f"[ERROR] Microphone recording failed: {type(exc).__name__}: {exc}")
			print("        Check mic permission/device and try again.")
			return

		predicted = predict_speaker(
			recorded_file,
			artifacts["model"],
			artifacts["scaler"],
			artifacts["label_encoder"],
		)
		print(f"Predicted speaker for live recording: {predicted}")
		return

	print("[ERROR] Invalid choice. Enter only 1 or 2.")


def main():
	print("Speaker Identification Project (SVM + MFCC)")
	print("Input: audio file | Output: speaker label")

	X, y = load_dataset(DATASET_DIR)
	artifacts = train_and_evaluate_svm(X, y, kernel="linear")
	interactive_prediction(artifacts)

	# Optional example: apni audio file ka path dekar direct predict kar sakte ho.
	# example_file = r"dataset\\person 1\\sample.wav"
	# predicted = predict_speaker(
	#     example_file,
	#     artifacts["model"],
	#     artifacts["scaler"],
	#     artifacts["label_encoder"],
	# )
	# print(f"\nPredicted speaker for '{example_file}': {predicted}")

	print("\n=== Notes: Small Dataset Limitations ===")
	print("- With very few recordings per speaker, model can overfit.")
	print("- High accuracy on one split may not generalize to new recordings.")

	print("\n=== How to Improve Accuracy ===")
	print("1) Collect more recordings per speaker (different sessions/times)")
	print("2) Keep microphone and room noise conditions consistent")
	print("3) Remove long silence segments from audio")
	print("4) Try data augmentation (noise, pitch/time shift)")
	print("5) Tune SVM hyperparameters (C, gamma) and compare kernels")

	print("\n=== Future Improvements ===")
	print("- Use speaker embeddings (x-vectors, ECAPA-TDNN, wav2vec features)")
	print("- Try deep learning (CNN/CRNN) on spectrograms")
	print("- Use transfer learning from pretrained audio models")


if __name__ == "__main__":
	main()
