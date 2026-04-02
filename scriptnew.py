import os
import sys
import shutil
import csv
from datetime import datetime
from collections import Counter

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



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
OUTPUT_DIR = "outputs"
MFCC_EXCEL_FILE = "mfcc_features.xlsx"
EVAL_EXCEL_FILE = "evaluation_metrics.xlsx"
COMPARISON_EXCEL_FILE = "model_comparison.xlsx"
MANUAL_EVAL_CSV_FILE = "manual_prediction_accuracy.csv"
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
	feature_records = []
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

			record = {
				"speaker": speaker,
				"file_name": audio_file,
				"file_path": file_path,
			}
			for idx, value in enumerate(feature_vector, start=1):
				record[f"mfcc_feature_{idx}"] = float(value)
			feature_records.append(record)

	if not features:
		raise ValueError(
			"No valid audio features extracted from dataset. "
			"If your files are .aac, install FFmpeg so librosa/audioread can decode them."
		)

	X = np.array(features, dtype=np.float32)
	y = np.array(labels)

	print(f"Loaded dataset: {len(y)} samples, {len(np.unique(y))} speakers")
	if skipped_files:
		print(f"Skipped unreadable files: {skipped_files}")

	if len(np.unique(y)) < 2:
		raise ValueError(
			"Need at least 2 speaker classes with readable audio to train a classifier."
		)

	return X, y, feature_records


def save_mfcc_features_to_excel(feature_records, output_path):
	"""
	Per-audio MFCC feature vectors ko Excel me save karta hai.
	"""
	if not feature_records:
		print("[WARN] No MFCC feature records found to save.")
		return

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	df = pd.DataFrame(feature_records)
	try:
		df.to_excel(output_path, index=False)
		print(f"MFCC feature Excel generated: {output_path}")
	except PermissionError:
		base, ext = os.path.splitext(output_path)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		fallback_path = f"{base}_{timestamp}{ext}"
		df.to_excel(fallback_path, index=False)
		print(f"[WARN] File locked: {output_path}")
		print(f"MFCC feature Excel generated at fallback path: {fallback_path}")


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

def evaluate_model_predictions(y_test, y_pred, label_encoder, model_name):
	"""
	Given predictions, standard evaluation metrics compute karta hai aur print bhi karta hai.
	"""
	acc = accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)

	classification_text = classification_report(
		y_test,
		y_pred,
		target_names=label_encoder.classes_,
		zero_division=0,
	)

	report_dict = classification_report(
		y_test,
		y_pred,
		target_names=label_encoder.classes_,
		zero_division=0,
		output_dict=True,
	)

	print(f"\n--- {model_name} Evaluation ---")
	print(f"Accuracy: {acc * 100:.2f}%")
	print("Confusion Matrix (rows=true, cols=predicted):")
	print(cm)
	print("\nClassification Report:")
	print(classification_text)

	return {
		"accuracy": acc,
		"confusion_matrix": cm,
		"classification_report_text": classification_text,
		"classification_report_dict": report_dict,
	}


# Ab models train karke unki performance evaluate karte hain.
def prepare_train_test_data(X, y):
	"""
	Step 4 helper:
	- Encode labels
	- Stratified split
	- Scale features
	"""
	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y)

	test_size = choose_test_size(y)

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

	return {
		"label_encoder": label_encoder,
		"scaler": scaler,
		"X_train_raw": X_train,
		"X_test_raw": X_test,
		"X_train_scaled": X_train_scaled,
		"X_test_scaled": X_test_scaled,
		"y_train": y_train,
		"y_test": y_test,
	}


def train_and_evaluate_svm(prepared_data, kernel="linear"):
	"""
	Step 5 + Step 6 (SVM):
	SVM ko train karke metrics return karta hai.
	"""
	model = SVC(kernel=kernel, C=1.0, gamma="scale")
	model.fit(prepared_data["X_train_scaled"], prepared_data["y_train"])
	y_pred = model.predict(prepared_data["X_test_scaled"])
	metrics = evaluate_model_predictions(
		prepared_data["y_test"],
		y_pred,
		prepared_data["label_encoder"],
		"SVM",
	)

	return {
		"model": model,
		"metrics": metrics,
	}


def train_and_evaluate_knn(prepared_data, n_neighbors=3):
	"""
	Step 5 + Step 6 (KNN):
	KNN ko train karke metrics return karta hai.
	"""
	model = KNeighborsClassifier(n_neighbors=n_neighbors)
	model.fit(prepared_data["X_train_scaled"], prepared_data["y_train"])
	y_pred = model.predict(prepared_data["X_test_scaled"])
	metrics = evaluate_model_predictions(
		prepared_data["y_test"],
		y_pred,
		prepared_data["label_encoder"],
		"KNN",
	)

	return {
		"model": model,
		"metrics": metrics,
	}


def train_and_evaluate_decision_tree(prepared_data, max_depth=None):
	"""
	Step 5 + Step 6 (Decision Tree):
	Decision Tree ko train karke metrics return karta hai.
	"""
	model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
	model.fit(prepared_data["X_train_raw"], prepared_data["y_train"])
	y_pred = model.predict(prepared_data["X_test_raw"])
	metrics = evaluate_model_predictions(
		prepared_data["y_test"],
		y_pred,
		prepared_data["label_encoder"],
		"Decision Tree",
	)

	return {
		"model": model,
		"metrics": metrics,
	}


def train_and_evaluate_logistic_regression(prepared_data, max_iter=1000):
	"""
	Step 5 + Step 6 (Logistic Regression):
	Logistic Regression ko train karke metrics return karta hai.
	"""
	model = LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE)
	model.fit(prepared_data["X_train_scaled"], prepared_data["y_train"])
	y_pred = model.predict(prepared_data["X_test_scaled"])
	metrics = evaluate_model_predictions(
		prepared_data["y_test"],
		y_pred,
		prepared_data["label_encoder"],
		"Logistic Regression",
	)

	return {
		"model": model,
		"metrics": metrics,
	}


def train_and_evaluate_models(X, y, svm_kernel="linear", knn_neighbors=3):
	"""
	Orchestrator:
	- Shared data prep
	- Separate SVM, KNN, Decision Tree and Logistic Regression functions
	- Comparison rows build
	"""
	prepared_data = prepare_train_test_data(X, y)

	svm_result = train_and_evaluate_svm(prepared_data, kernel=svm_kernel)
	knn_result = train_and_evaluate_knn(prepared_data, n_neighbors=knn_neighbors)
	dt_result = train_and_evaluate_decision_tree(prepared_data, max_depth=None)
	lr_result = train_and_evaluate_logistic_regression(prepared_data, max_iter=1000)
	svm_metrics = svm_result["metrics"]
	knn_metrics = knn_result["metrics"]
	dt_metrics = dt_result["metrics"]
	lr_metrics = lr_result["metrics"]

	comparison_rows = [
		{
			"model": "SVM",
			"accuracy": float(svm_metrics["accuracy"]),
			"accuracy_percent": float(svm_metrics["accuracy"]) * 100.0,
			"macro_f1": float(svm_metrics["classification_report_dict"]["macro avg"]["f1-score"]),
			"weighted_f1": float(svm_metrics["classification_report_dict"]["weighted avg"]["f1-score"]),
		},
		{
			"model": "KNN",
			"accuracy": float(knn_metrics["accuracy"]),
			"accuracy_percent": float(knn_metrics["accuracy"]) * 100.0,
			"macro_f1": float(knn_metrics["classification_report_dict"]["macro avg"]["f1-score"]),
			"weighted_f1": float(knn_metrics["classification_report_dict"]["weighted avg"]["f1-score"]),
		},
		{
			"model": "Decision Tree",
			"accuracy": float(dt_metrics["accuracy"]),
			"accuracy_percent": float(dt_metrics["accuracy"]) * 100.0,
			"macro_f1": float(dt_metrics["classification_report_dict"]["macro avg"]["f1-score"]),
			"weighted_f1": float(dt_metrics["classification_report_dict"]["weighted avg"]["f1-score"]),
		},
		{
			"model": "Logistic Regression",
			"accuracy": float(lr_metrics["accuracy"]),
			"accuracy_percent": float(lr_metrics["accuracy"]) * 100.0,
			"macro_f1": float(lr_metrics["classification_report_dict"]["macro avg"]["f1-score"]),
			"weighted_f1": float(lr_metrics["classification_report_dict"]["weighted avg"]["f1-score"]),
		},
	]

	artifacts = {
		"model": svm_result["model"],
		"scaler": prepared_data["scaler"],
		"label_encoder": prepared_data["label_encoder"],
		"primary_model_name": "SVM",
		"primary_model_params": {
			"svm_kernel": svm_kernel,
			"knn_neighbors": knn_neighbors,
		},
		"adaptive_X": prepared_data["X_train_raw"].copy(),
		"adaptive_y": prepared_data["y_train"].copy(),
		"metrics": svm_metrics,
		"all_metrics": {
			"SVM": svm_metrics,
			"KNN": knn_metrics,
			"Decision Tree": dt_metrics,
			"Logistic Regression": lr_metrics,
		},
		"all_models": {
			"SVM": svm_result["model"],
			"KNN": knn_result["model"],
			"Decision Tree": dt_result["model"],
			"Logistic Regression": lr_result["model"],
		},
		"comparison_rows": comparison_rows,
	}
	return artifacts


def save_model_comparison_to_excel(comparison_rows, output_path):
	"""
	SVM vs KNN ka summary comparison Excel me save karta hai.
	"""
	if not comparison_rows:
		print("[WARN] No comparison rows found to save.")
		return

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	comparison_df = pd.DataFrame(comparison_rows)
	try:
		comparison_df.to_excel(output_path, sheet_name="comparison", index=False)
		print(f"Model comparison Excel generated: {output_path}")
	except PermissionError:
		base, ext = os.path.splitext(output_path)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		fallback_path = f"{base}_{timestamp}{ext}"
		comparison_df.to_excel(fallback_path, sheet_name="comparison", index=False)
		print(f"[WARN] File locked: {output_path}")
		print(f"Model comparison Excel generated at fallback path: {fallback_path}")


def save_evaluation_to_excel(metrics, label_encoder, output_path):
	"""
	Accuracy, confusion matrix aur classification report ko multi-sheet Excel me save karta hai.
	"""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	accuracy_df = pd.DataFrame(
		[
			{
				"metric": "accuracy",
				"value": float(metrics["accuracy"]),
				"value_percent": float(metrics["accuracy"]) * 100.0,
			}
		]
	)

	labels = list(label_encoder.classes_)
	cm_df = pd.DataFrame(
		metrics["confusion_matrix"],
		index=[f"true_{name}" for name in labels],
		columns=[f"pred_{name}" for name in labels],
	)

	report_df = pd.DataFrame(metrics["classification_report_dict"]).transpose()
	report_df.index.name = "class_or_avg"

	try:
		with pd.ExcelWriter(output_path) as writer:
			accuracy_df.to_excel(writer, sheet_name="accuracy", index=False)
			cm_df.to_excel(writer, sheet_name="confusion_matrix")
			report_df.to_excel(writer, sheet_name="classification_report")
		print(f"Evaluation Excel generated: {output_path}")
	except PermissionError:
		base, ext = os.path.splitext(output_path)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		fallback_path = f"{base}_{timestamp}{ext}"
		with pd.ExcelWriter(fallback_path) as writer:
			accuracy_df.to_excel(writer, sheet_name="accuracy", index=False)
			cm_df.to_excel(writer, sheet_name="confusion_matrix")
			report_df.to_excel(writer, sheet_name="classification_report")
		print(f"[WARN] File locked: {output_path}")
		print(f"Evaluation Excel generated at fallback path: {fallback_path}")


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


def predict_speaker_with_features(audio_path, model, scaler, label_encoder):
	"""
	Prediction ke saath raw feature vector bhi return karta hai,
	taaki feedback se model update kiya ja sake.
	"""
	feature_vector = extract_mfcc_features(audio_path)
	if feature_vector is None:
		raise ValueError(f"Could not extract features from: {audio_path}")

	feature_vector_2d = feature_vector.reshape(1, -1)
	feature_vector_scaled = scaler.transform(feature_vector_2d)
	pred_numeric = model.predict(feature_vector_scaled)[0]
	pred_label = label_encoder.inverse_transform([pred_numeric])[0]
	return pred_label, feature_vector


def retrain_model_from_feedback(artifacts):
	"""
	Adaptive dataset (feedback-added) se model ko re-train karta hai.
	Isse next prediction updated model se hota hai.
	"""
	model_name = artifacts["primary_model_name"]
	params = artifacts["primary_model_params"]

	new_scaler = StandardScaler()
	X_scaled = new_scaler.fit_transform(artifacts["adaptive_X"])

	if model_name == "SVM":
		new_model = SVC(kernel=params["svm_kernel"], C=1.0, gamma="scale")
	elif model_name == "KNN":
		new_model = KNeighborsClassifier(n_neighbors=params["knn_neighbors"])
	else:
		raise ValueError(f"Unsupported primary_model_name: {model_name}")

	new_model.fit(X_scaled, artifacts["adaptive_y"])
	artifacts["scaler"] = new_scaler
	artifacts["model"] = new_model


def resolve_user_label(user_input, label_encoder):
	"""
	User-entered label ko known class labels se case-insensitive match karta hai.
	Match na mile to None return karta hai.
	"""
	cleaned = user_input.strip()
	if cleaned == "":
		return None

	label_map = {label.lower(): label for label in label_encoder.classes_}
	return label_map.get(cleaned.lower())


def save_manual_evaluation_rows(rows, output_path):
	"""
	Manual prediction checks ko CSV me append karta hai.
	"""
	if not rows:
		return

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	file_exists = os.path.isfile(output_path)
	fieldnames = [
		"timestamp",
		"input_mode",
		"input_path",
		"predicted_speaker",
		"true_speaker",
		"is_correct",
		"running_accuracy_percent",
		"model_updated",
		"adaptive_training_samples",
	]

	try:
		with open(output_path, mode="a", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			if not file_exists:
				writer.writeheader()
			writer.writerows(rows)
		print(f"Saved manual log: {output_path}")
	except PermissionError:
		base, ext = os.path.splitext(output_path)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		fallback_path = f"{base}_{timestamp}{ext}"
		with open(fallback_path, mode="w", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(rows)
		print(f"[WARN] File locked: {output_path}")
		print(f"Saved manual log (fallback): {fallback_path}")


def interactive_prediction(artifacts):
	"""
	Training ke baad interactive prediction menu chalata hai.
	TTY na ho to safely skip kar deta hai.
	"""
	if not sys.stdin.isatty():
		print("Interactive prediction skipped (non-interactive terminal).")
		print("Run in an interactive terminal:")
		print(f"  {os.path.join('.venv', 'Scripts', 'python.exe')} {os.path.basename(__file__)}")
		return

	print("\nFile prediction mode")
	print("Known speakers:", ", ".join(artifacts["label_encoder"].classes_))

	manual_correct = 0
	manual_total = 0
	manual_rows = []

	while True:
		try:
			audio_path = input("\nEnter audio file path (or press Enter/q to quit): ").strip().strip('"')
		except (KeyboardInterrupt, EOFError):
			print("\nPrediction stopped.")
			break

		if audio_path.lower() in {"", "q"}:
			print("Prediction stopped.")
			break

		if not os.path.isfile(audio_path):
			print(f"[ERROR] File not found: {audio_path}")
			continue

		predicted, sample_features = predict_speaker_with_features(
			audio_path,
			artifacts["model"],
			artifacts["scaler"],
			artifacts["label_encoder"],
		)
		input_mode = "file"
		input_path = audio_path
		print(f"Predicted: {predicted}")

		try:
			true_label_input = input(
				"Enter TRUE speaker label for this sample (press Enter to skip): "
			).strip()
		except (KeyboardInterrupt, EOFError):
			print("Skipping manual scoring for this sample.")
			continue

		if true_label_input == "":
			print("Manual scoring skipped for this sample.")
			continue

		resolved_true_label = resolve_user_label(
			true_label_input,
			artifacts["label_encoder"],
		)
		if resolved_true_label is None:
			print("[ERROR] Unknown speaker label.")
			print("Use one of:", ", ".join(artifacts["label_encoder"].classes_))
			continue

		is_correct = int(predicted == resolved_true_label)
		manual_total += 1
		manual_correct += is_correct
		running_accuracy = (manual_correct / manual_total) * 100.0

		# Feedback sample ko adaptive training set me add karke model update karte hain.
		true_numeric = artifacts["label_encoder"].transform([resolved_true_label])[0]
		artifacts["adaptive_X"] = np.vstack(
			[artifacts["adaptive_X"], sample_features.reshape(1, -1)]
		)
		artifacts["adaptive_y"] = np.append(artifacts["adaptive_y"], true_numeric)
		retrain_model_from_feedback(artifacts)

		manual_rows.append(
			{
				"timestamp": datetime.now().isoformat(timespec="seconds"),
				"input_mode": input_mode,
				"input_path": input_path,
				"predicted_speaker": predicted,
				"true_speaker": resolved_true_label,
				"is_correct": is_correct,
				"running_accuracy_percent": round(running_accuracy, 2),
				"model_updated": 1,
				"adaptive_training_samples": int(len(artifacts["adaptive_y"])),
			}
		)

		status_text = "Correct ✅" if is_correct else "Wrong ❌"
		print(
			f"{status_text} | Accuracy: "
			f"{manual_correct}/{manual_total} = {running_accuracy:.2f}%"
		)
		print("Model updated.")

	if manual_total > 0:
		manual_csv_path = os.path.join(OUTPUT_DIR, MANUAL_EVAL_CSV_FILE)
		save_manual_evaluation_rows(manual_rows, manual_csv_path)
		print(
			f"Final accuracy: "
			f"{manual_correct}/{manual_total} = {(manual_correct / manual_total) * 100:.2f}%"
		)
	else:
		print("No labeled predictions entered.")


def main():
	print("Speaker Identification (file input)")

	X, y, feature_records = load_dataset(DATASET_DIR)

	mfcc_excel_path = os.path.join(OUTPUT_DIR, MFCC_EXCEL_FILE)
	save_mfcc_features_to_excel(feature_records, mfcc_excel_path)

	artifacts = train_and_evaluate_models(X, y, svm_kernel="linear", knn_neighbors=3)
	print(
		f"Model accuracy (test split) - SVM: {artifacts['all_metrics']['SVM']['accuracy'] * 100:.2f}% | "
		f"KNN: {artifacts['all_metrics']['KNN']['accuracy'] * 100:.2f}% | "
		f"Decision Tree: {artifacts['all_metrics']['Decision Tree']['accuracy'] * 100:.2f}% | "
		f"Logistic Regression: {artifacts['all_metrics']['Logistic Regression']['accuracy'] * 100:.2f}%"
	)

	eval_excel_path = os.path.join(OUTPUT_DIR, EVAL_EXCEL_FILE)
	save_evaluation_to_excel(
		metrics=artifacts["metrics"],
		label_encoder=artifacts["label_encoder"],
		output_path=eval_excel_path,
	)

	comparison_excel_path = os.path.join(OUTPUT_DIR, COMPARISON_EXCEL_FILE)
	save_model_comparison_to_excel(
		comparison_rows=artifacts["comparison_rows"],
		output_path=comparison_excel_path,
	)

	interactive_prediction(artifacts)


if __name__ == "__main__":
	main()
