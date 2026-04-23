import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	roc_auc_score,
	f1_score,
	roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = "outputs"
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
MFCC_EXCEL = os.path.join(OUTPUT_DIR, "mfcc_features.xlsx")
EVAL_EXCEL = os.path.join(OUTPUT_DIR, "evaluation_metrics.xlsx")


def choose_test_size(y_labels):
	class_counts = Counter(y_labels)
	min_per_class = min(class_counts.values())
	if min_per_class <= 2:
		return 0.5
	return 0.2


def choose_cv_splits(y_labels):
	class_counts = Counter(y_labels)
	min_per_class = min(class_counts.values())
	return max(2, min(5, int(min_per_class)))


def latest_model_comparison_file(output_dir=OUTPUT_DIR):
	pattern = os.path.join(output_dir, "model_comparison*.xlsx")
	matches = glob.glob(pattern)
	if not matches:
		return None
	return max(matches, key=os.path.getmtime)


def load_feature_table(path=MFCC_EXCEL):
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Missing file: {path}")

	df = pd.read_excel(path)
	feature_cols = [c for c in df.columns if str(c).startswith("mfcc_feature_")]
	if "speaker" not in df.columns or not feature_cols:
		raise ValueError("mfcc_features.xlsx must contain speaker + mfcc_feature_* columns")

	X = df[feature_cols].to_numpy(dtype=np.float32)
	y = df["speaker"].astype(str).to_numpy()
	return X, y


def train_models_and_collect_metrics(X, y):
	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y)
	test_size = choose_test_size(y)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y_encoded,
		test_size=test_size,
		random_state=42,
		stratify=y_encoded,
	)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	cv = StratifiedKFold(
		n_splits=choose_cv_splits(y_train),
		shuffle=True,
		random_state=42,
	)

	svm_search = GridSearchCV(
		estimator=SVC(probability=True, class_weight="balanced", random_state=42),
		param_grid=[{"kernel": ["linear"], "C": [0.5, 1.0, 5.0], "gamma": ["scale"]}],
		scoring="f1_weighted",
		cv=cv,
		n_jobs=-1,
	)
	svm_search.fit(X_train_scaled, y_train)
	svm_model = svm_search.best_estimator_

	models = {
		"SVM": (svm_model, X_test_scaled),
		"KNN": (KNeighborsClassifier(n_neighbors=3).fit(X_train_scaled, y_train), X_test_scaled),
		"Decision Tree": (DecisionTreeClassifier(random_state=42).fit(X_train, y_train), X_test),
		"Logistic Regression": (
			LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train),
			X_test_scaled,
		),
	}

	n_classes = len(label_encoder.classes_)
	y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

	rows = []
	roc_curves = {}
	best_by_f1 = (None, -1.0)

	for model_name, (model, X_eval) in models.items():
		y_pred = model.predict(X_eval)
		y_prob = model.predict_proba(X_eval)

		acc = accuracy_score(y_test, y_pred)
		f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
		f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

		# Multi-class One-vs-Rest ROC AUC
		roc_auc_macro_ovr = roc_auc_score(
			y_test,
			y_prob,
			multi_class="ovr",
			average="macro",
		)

		# Micro-average ROC curve for visual comparison in one chart.
		fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
		roc_curves[model_name] = {
			"fpr": fpr,
			"tpr": tpr,
			"auc": roc_auc_macro_ovr,
		}

		rows.append(
			{
				"model": model_name,
				"accuracy_percent": acc * 100,
				"f1_weighted": f1_weighted,
				"f1_macro": f1_macro,
				"roc_auc_macro_ovr": roc_auc_macro_ovr,
			}
		)

		if f1_weighted > best_by_f1[1]:
			best_by_f1 = (model_name, f1_weighted)

	metrics_df = pd.DataFrame(rows).sort_values(
		by=["f1_weighted", "roc_auc_macro_ovr"],
		ascending=False,
	)

	best_model_name = best_by_f1[0]
	best_model, best_X_eval = models[best_model_name]
	cm_display = ConfusionMatrixDisplay.from_predictions(
		y_test,
		best_model.predict(best_X_eval),
		cmap="Blues",
		colorbar=False,
	)

	return {
		"metrics_df": metrics_df,
		"roc_curves": roc_curves,
		"best_model_name": best_model_name,
		"cm_display": cm_display,
	}


def plot_model_metric_bars(metrics_df, save_path):
	plot_df = metrics_df.copy()
	x = np.arange(len(plot_df))
	width = 0.24

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.bar(x - width, plot_df["f1_weighted"], width, label="F1 (Weighted)")
	ax.bar(x, plot_df["f1_macro"], width, label="F1 (Macro)")
	ax.bar(x + width, plot_df["roc_auc_macro_ovr"], width, label="ROC AUC (Macro OVR)")

	ax.set_ylim(0, 1.0)
	ax.set_xticks(x)
	ax.set_xticklabels(plot_df["model"], rotation=15)
	ax.set_title("Model Comparison: F1 and ROC-AUC (Primary Presentation Metrics)")
	ax.set_ylabel("Score (0 to 1)")
	ax.grid(axis="y", alpha=0.2)
	ax.legend(loc="lower right")

	for idx, value in enumerate(plot_df["f1_weighted"]):
		ax.text(idx - width, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
	for idx, value in enumerate(plot_df["roc_auc_macro_ovr"]):
		ax.text(idx + width, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)

	fig.tight_layout()
	fig.savefig(save_path, dpi=200)
	plt.close(fig)


def plot_roc_curves(roc_curves, save_path):
	fig, ax = plt.subplots(figsize=(10, 7))
	for model_name, curve_data in roc_curves.items():
		ax.plot(
			curve_data["fpr"],
			curve_data["tpr"],
			label=f"{model_name} (AUC={curve_data['auc']:.3f})",
			linewidth=2,
		)

	ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC Curve Comparison (Micro-Curve, Macro-OVR AUC in Legend)")
	ax.grid(alpha=0.3)
	ax.legend(loc="lower right")

	fig.tight_layout()
	fig.savefig(save_path, dpi=200)
	plt.close(fig)


def plot_svm_class_f1_from_eval_excel(eval_excel_path, save_path):
	if not os.path.isfile(eval_excel_path):
		print(f"[WARN] Could not find {eval_excel_path}; skipping per-class SVM F1 plot.")
		return

	report_df = pd.read_excel(eval_excel_path, sheet_name="classification_report")
	if "class_or_avg" not in report_df.columns or "f1-score" not in report_df.columns:
		print("[WARN] evaluation_metrics.xlsx schema unexpected; skipping per-class SVM F1 plot.")
		return

	ignore_rows = {"accuracy", "macro avg", "weighted avg"}
	class_df = report_df[~report_df["class_or_avg"].isin(ignore_rows)].copy()
	if class_df.empty:
		print("[WARN] No per-class rows found in evaluation report; skipping per-class SVM F1 plot.")
		return

	class_df = class_df.sort_values("f1-score", ascending=False)

	fig, ax = plt.subplots(figsize=(14, 7))
	ax.bar(class_df["class_or_avg"].astype(str), class_df["f1-score"].astype(float), color="#4C78A8")
	ax.set_ylim(0, 1)
	ax.set_title("Per-Class F1 Score (SVM from evaluation_metrics.xlsx)")
	ax.set_xlabel("Speaker")
	ax.set_ylabel("F1 Score")
	ax.tick_params(axis="x", rotation=75)
	ax.grid(axis="y", alpha=0.2)

	fig.tight_layout()
	fig.savefig(save_path, dpi=200)
	plt.close(fig)


def save_metrics_table(metrics_df, save_path):
	metrics_df.to_csv(save_path, index=False)


def main():
	os.makedirs(GRAPHS_DIR, exist_ok=True)

	X, y = load_feature_table(MFCC_EXCEL)
	artifacts = train_models_and_collect_metrics(X, y)

	metrics_df = artifacts["metrics_df"]
	print("\n=== Presentation Metrics (sorted by weighted F1) ===")
	print(metrics_df.to_string(index=False))

	plot_model_metric_bars(
		metrics_df,
		os.path.join(GRAPHS_DIR, "model_f1_roc_comparison.png"),
	)
	plot_roc_curves(
		artifacts["roc_curves"],
		os.path.join(GRAPHS_DIR, "model_roc_curves.png"),
	)

	# Save confusion matrix for best model by weighted F1.
	cm_fig = artifacts["cm_display"].figure_
	cm_fig.set_size_inches(10, 8)
	cm_fig.suptitle(f"Confusion Matrix (Best by F1: {artifacts['best_model_name']})")
	cm_fig.tight_layout()
	cm_fig.savefig(os.path.join(GRAPHS_DIR, "best_model_confusion_matrix.png"), dpi=200)
	plt.close(cm_fig)

	plot_svm_class_f1_from_eval_excel(
		EVAL_EXCEL,
		os.path.join(GRAPHS_DIR, "svm_per_class_f1.png"),
	)

	save_metrics_table(metrics_df, os.path.join(GRAPHS_DIR, "presentation_metrics_summary.csv"))

	latest_comparison = latest_model_comparison_file(OUTPUT_DIR)
	if latest_comparison:
		print(f"Source comparison file found: {latest_comparison}")

	print(f"\nGraphs generated in: {GRAPHS_DIR}")


if __name__ == "__main__":
	main()
