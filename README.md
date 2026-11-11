# Hierarchical Multimodal Biometric Verification System

## 1. Project Overview

This project implements a traditional-machine-learning-based biometric verification system using two biometric modalities:

- **Face Verification**
- **Voice Verification**

The core system also implements:

- Early Multimodal Fusion
- Hierarchical Verification
- Model Evaluation and Comparison

The main objective is to investigate whether combining face and voice biometrics can improve verification performance compared with using each modality independently.

**Important:** Deep learning is not used. The implemented core verification models are traditional machine learning models.

---

## 2. Project Structure

```text
Biometric_Verification/
│
├── data/
│   ├── fingerprint/
│   │   └── SOCOFing/
│   │       ├── Altered/
│   │       └── Real/
│   │
│   ├── custom_dataset/
│   │   ├── face/
│   │   └── voice/
│   │
│   └── vidtimit/
│       ├── videos/
│       ├── manifest.csv
│       └── splits.csv
│
├── features/
│   ├── face/
│   ├── voice/
│   ├── fusion/
│   ├── fingerprint/
│   ├── commands/
│   └── custom_dataset/
│
├── models/
│   ├── face/
│   ├── voice/
│   ├── fusion/
│   ├── hierarchical/
│   ├── fingerprint/
│   ├── commands/
│   └── custom_dataset/
│
├── results/
│   ├── face/
│   ├── voice/
│   ├── fusion/
│   ├── hierarchical/
│   ├── fingerprint/
│   ├── commands/
│   ├── custom_dataset/
│   └── comparison/
│
├── archive/
│   ├── old_voice_models/
│   └── old_voice_results/
│
├── backup_before_refactor/
│
├── src/
│   ├── data_preparation.py
│   ├── create_splits.py
│   ├── verification_utils.py
│   ├── face_features.py
│   ├── face_verification.py
│   ├── voice_features.py
│   ├── voice_verification.py
│   ├── fusion.py
│   ├── fusion_verification.py
│   ├── hierarchical.py
│   ├── final_comparison.py
│   ├── evaluation.py
│   ├── fingerprint_features.py
│   ├── fingerprint_verification.py
│   ├── realtime.py
│   └── speech_commands.py
│
├── main.py
├── requirements.txt
└── README.md
```

**Project structure notes:**

- `data/vidtimit/` contains the main dataset used by the implemented face and voice pipeline.
- `data/fingerprint/` contains SOCOFing, reserved for the optional fingerprint extension.
- `data/custom_dataset/` contains separate locations for custom face and voice data.
- `features/`, `models/`, and `results/` contain the corresponding generated outputs.
- The fingerprint, speech-command, real-time, and custom-dataset folders are reserved for optional extensions.
- The four optional source modules are currently placeholders and should not be considered implemented functionality.
- `archive/` preserves older voice models and results that are not part of the active experiments.
- `backup_before_refactor/` preserves distinct historical backup files.
- The `.venv/` virtual environment is local to the development machine and is not included in the documented source structure.
- Generated feature arrays, trained models, and evaluation results are created by running the corresponding scripts. They should not be manually created as empty files.

---

## 3. Datasets

### 3.1 VIDTIMIT

The main dataset used for face and voice verification is VIDTIMIT.

Current dataset audit:

- 43 unique identities
- 10 paired recordings per identity
- 430 paired audio/video sequences
- 430 valid WAV files
- 45,661 valid JPEG frames
- Audio sample rate: 32 kHz
- Audio channels: Mono

Each audio sequence is matched with its corresponding video sequence using the same identity and sequence name.

The original audio files are located inside the corresponding identity folders. The current feature extraction pipeline reads these original audio files directly; a separate `extracted_audio/` directory is not required.

### 3.2 SOCOFing

SOCOFing is stored separately for a possible fingerprint verification extension.

Current dataset audit:

- 6,000 files in `Real`
- 49,270 files in `Altered`
- 55,270 valid BMP images

**Important limitation:** SOCOFing identities do not correspond to the identities in VIDTIMIT. Therefore, fingerprint features cannot currently be fused with the VIDTIMIT face and voice features as a valid identity-level multimodal system.

Fingerprint verification should be evaluated independently unless a properly identity-matched fingerprint dataset becomes available.

### 3.3 Custom Dataset

The project includes separate directories for a possible custom dataset:

- `data/custom_dataset/face/`
- `data/custom_dataset/voice/`

This dataset is separate from VIDTIMIT. It is reserved for an optional extension and is not part of the current reported experiments.

---

## 4. Data Preparation and Verification Protocol

### 4.1 Manifest

`src/data_preparation.py` creates:

`data/vidtimit/manifest.csv`

The manifest contains:

- Person_ID
- Sequence_ID
- Audio_Path
- Face_Path
- Frame_Count

Each row represents one paired audio/video recording.

The data preparation stage successfully generated 430 paired sequences from 43 identities, with no missing video sequences.

### 4.2 Dataset Splitting

`src/create_splits.py` creates:

`data/vidtimit/splits.csv`

The dataset contains 430 recordings distributed as follows:

| Split | Recordings |
|---|---:|
| Training | 172 |
| Enrollment | 86 |
| Testing | 172 |
| **Total** | **430** |

For each of the 43 identities:

- 4 recordings are used for training.
- 2 recordings are used for enrollment.
- 4 recordings are reserved for testing.

The random seed is 42.

The enrollment recordings represent the registered biometric samples. Test recordings are held out as probes for verification.

The verification system compares genuine pairs (same identity) and impostor pairs (different identities).

### 4.3 Pair Construction

The system represents each verification comparison using the absolute difference between two biometric feature vectors.

- Genuine pairs contain two samples from the same identity.
- Impostor pairs contain samples from different identities.

Training, validation, enrollment, and testing are handled according to the verification protocol implemented in the corresponding scripts.

Thresholds are selected using validation data rather than the final test set. The test recordings remain separate from training and validation recordings.

---

## 5. Feature Extraction

### 5.1 Face Features

Implemented in:

`src/face_features.py`

Processing pipeline:

1. Read video frames.
2. Sample 10 frames uniformly from each sequence.
3. Detect the face using OpenCV Haar Cascade.
4. Crop the detected face.
5. Convert the crop to grayscale.
6. Resize to 128 × 128.
7. Extract HOG features.
8. Extract LBP histogram features.
9. Aggregate the successful frame features using mean and standard deviation.

Feature configuration:

- HOG: 9 orientations
- Pixels per cell: 16 × 16
- Cells per block: 2 × 2
- LBP: Uniform pattern, P = 8, R = 1
- HOG + LBP per frame: 1,774 features
- Aggregated sequence representation: 3,548 features

Output files:

- `features/face/face_features.npy`
- `features/face/face_metadata.csv`

Extraction status:

- 430 sequences processed
- 0 completely failed sequences
- 0 partial detection sequences
- Feature matrix: 430 × 3,548

### 5.2 Voice Features

Implemented in:

`src/voice_features.py`

Audio is loaded as mono at 32 kHz.

Extracted features:

- MFCC: 13 coefficients
- Delta MFCC
- Delta-Delta MFCC
- Zero Crossing Rate (ZCR)
- Root Mean Square Energy (RMS)

Mean and standard deviation statistics are calculated for the feature sequences.

The complete extracted representation contains 82 features per recording.

Output files:

- `features/voice/voice_features.npy`
- `features/voice/voice_metadata.csv`

Extraction status:

- 430 sequences processed
- 0 failed sequences
- Feature matrix: 430 × 82

The current voice verification experiments use a selected 30-dimensional subset:

- 26 static MFCC statistics
- 2 ZCR statistics
- 2 RMS statistics

### 5.3 Early Fusion Features

Implemented in:

`src/fusion.py`

Early fusion combines the face and selected voice feature vectors by concatenation.

- Face features: 3,548
- Selected voice features: 30
- Fused representation: 3,578 features per recording

Output files:

- `features/fusion/fusion_features.npy`
- `features/fusion/fusion_metadata.csv`

The feature rows are checked for identity, sequence, and split alignment before fusion.

---

## 6. Verification Models

The project uses traditional machine learning models:

- Support Vector Machine (SVM)
- Random Forest (RF)

The verification process constructs pairs of biometric feature vectors.

For each pair, the absolute difference between the two feature vectors is used as the pair representation.

The models learn to distinguish:

- Genuine pairs: both samples belong to the same person.
- Impostor pairs: samples belong to different people.

For SVM, feature scaling is fitted using training data only.

Shared verification utilities are implemented in:

`src/verification_utils.py`

These utilities support pair construction, data splitting, threshold selection, and metric calculation across the verification experiments.

### 6.1 Face Verification

Implemented in:

`src/face_verification.py`

Evaluates face-only verification using HOG and LBP representations.

Two models are evaluated:

- SVM
- Random Forest

### 6.2 Voice Verification

Implemented in:

`src/voice_verification.py`

Evaluates voice-only verification using the selected MFCC, ZCR, and RMS features.

Two models are evaluated:

- SVM
- Random Forest

### 6.3 Early Fusion Verification

Implemented in:

`src/fusion_verification.py`

Evaluates the concatenated face and voice representation using:

- SVM
- Random Forest

### 6.4 Hierarchical Verification

Implemented in:

`src/hierarchical.py`

The hierarchical pipeline uses:

1. Face Random Forest as the primary verification stage.
2. A confidence-based decision.
3. Voice SVM as a fallback when face confidence is below the selected cutoff.

The voice verification threshold and face confidence cutoff are selected using validation data.

The test set is kept separate from threshold selection.

The goal is to use the second modality when the primary modality is uncertain, rather than always requiring both modalities for every comparison.

---

## 7. Evaluation Metrics

The system reports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Confusion Matrix

The confusion matrix follows this order:

```text
                 Predicted
                 Impostor   Genuine
Actual Impostor     TN         FP
Actual Genuine      FN         TP
```

Where:

- TP: Genuine pairs correctly accepted.
- TN: Impostor pairs correctly rejected.
- FP: Impostor pairs incorrectly accepted.
- FN: Genuine pairs incorrectly rejected.

Important formulas:

- FAR = FP / (FP + TN)
- FRR = FN / (FN + TP)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 × Precision × Recall / (Precision + Recall)

Accuracy alone is not sufficient because the test set contains substantially more impostor pairs than genuine pairs.

Balanced Accuracy and FAR/FRR are therefore important when interpreting results.

---

## 8. Current Experimental Results

The final test evaluation contains:

- 14,792 total comparison pairs
- 344 genuine pairs
- 14,448 impostor pairs

The following table summarizes the latest experimental results generated by `src/final_comparison.py`.

All values are percentages.

| Modality | Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | FAR | FRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Face-only | SVM | 99.85 | 98.08 | 97.35 | 96.22 | 96.78 | 0.06 | 3.78 |
| Face-only | Random Forest | 99.89 | 97.96 | 99.40 | 95.93 | 97.63 | 0.01 | 4.07 |
| Voice-only | SVM | 72.49 | 75.27 | 6.31 | 78.20 | 11.68 | 27.65 | 21.80 |
| Voice-only | Random Forest | 66.47 | 72.34 | 5.24 | 78.49 | 9.82 | 33.82 | 21.51 |
| Early Fusion | SVM | 99.80 | 98.76 | 94.12 | 97.67 | 95.86 | 0.15 | 2.33 |
| Early Fusion | Random Forest | 99.28 | 99.21 | 76.80 | 99.13 | 86.55 | 0.71 | 0.87 |
| Hierarchical | Face RF baseline | 99.23 | 83.43 | 100.00 | 66.86 | 80.14 | 0.00 | 33.14 |
| Hierarchical | Voice SVM baseline | 72.49 | 75.28 | 6.31 | 78.20 | 11.68 | 27.64 | 21.80 |
| Hierarchical | Hierarchical pipeline | 99.54 | 90.54 | 98.94 | 81.10 | 89.14 | 0.02 | 18.90 |

These results correspond to the current dataset, feature representations, model configurations, verification protocol, and selected thresholds.

### 8.1 Results Interpretation

- **Face-only verification:** Face RF achieves the highest F1-score among the individual and multimodal model configurations shown in the comparison table, at 97.63%. This indicates strong overall genuine/impostor classification performance under the current protocol.
- **Voice-only verification:** Voice performance is substantially weaker, particularly in precision and F1-score. Its relatively low precision is associated with a high number of false acceptances among the much larger impostor class.
- **Early Fusion SVM:** Fusion SVM provides a strong balance between genuine acceptance and impostor rejection, with 98.76% Balanced Accuracy and 95.86% F1-score.
- **Early Fusion Random Forest:** Fusion RF achieves the highest Balanced Accuracy (99.21%) and Recall (99.13%). However, its precision and F1-score are lower than those of Face RF, demonstrating a trade-off between accepting genuine pairs and rejecting impostor pairs.
- **Hierarchical verification:** The cascade achieves 90.54% Balanced Accuracy and 89.14% F1-score. Compared with its internal face baseline, it reduces FRR from 33.14% to 18.90%, while FAR increases slightly from 0.00% to approximately 0.02%.

The hierarchical baseline models are trained under the same internal training/validation setup as the cascade. They should not be treated as identical experiments to the separately trained face-only and voice-only models.

**Overall finding:** No single model is best across all evaluation metrics. Face RF provides the strongest F1-score, Fusion RF provides the highest Balanced Accuracy and Recall, and the hierarchical pipeline offers a different trade-off by using voice as a fallback when face confidence is low.

---

## 9. Validation and Test Considerations

The verification protocol uses enrollment recordings and held-out test recordings.

Thresholds are selected using validation data rather than the final test set.

However, the test comparisons are pair-based:

- A test recording may be compared with multiple enrollment recordings.
- The same probe and enrollment recordings can therefore appear in multiple pairs.

Consequently, comparison pairs are not statistically independent observations.

This limitation should be considered when interpreting the metrics and when making claims about generalization.

The current results should be treated as experimental dataset results, not as evidence of production-level biometric security.

---

## 10. Running the Project

### 10.1 Requirements

Recommended environment:

- Python 3.12.5
- Windows
- PowerShell
- Virtual environment (`.venv`)

Install the dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

### 10.2 Run the Interactive Main Menu

The project includes an interactive menu that allows users to select and run individual project stages.

From the project root directory, activate the virtual environment and run:

```powershell
python main.py
```

The main menu displays the available core pipeline stages:

1. Data Preparation - Create Manifest
2. Create Dataset Splits
3. Face Feature Extraction
4. Face Verification
5. Voice Feature Extraction
6. Voice Verification
7. Early Fusion Feature Creation
8. Early Fusion Verification
9. Hierarchical Verification
10. Final Model Comparison
11. Evaluation Audit

Enter the corresponding stage number to select a task. The program asks for confirmation before executing the selected stage.

Enter `0` to exit the menu.

### 10.3 Run the Core Pipeline Individually

Alternatively, run the scripts from the project root in the following order.

**Step 1 — Create the manifest**

```powershell
python -m src.data_preparation
```

**Step 2 — Create dataset splits**

```powershell
python -m src.create_splits
```

**Step 3 — Extract face features**

```powershell
python -m src.face_features
```

**Step 4 — Train and evaluate face models**

```powershell
python -m src.face_verification
```

**Step 5 — Extract voice features**

```powershell
python -m src.voice_features
```

**Step 6 — Train and evaluate voice models**

```powershell
python -m src.voice_verification
```

**Step 7 — Create early fusion features**

```powershell
python -m src.fusion
```

**Step 8 — Train and evaluate fusion models**

```powershell
python -m src.fusion_verification
```

**Step 9 — Run hierarchical verification**

```powershell
python -m src.hierarchical
```

**Step 10 — Generate final model comparison**

```powershell
python -m src.final_comparison
```

**Step 11 — Run the evaluation audit**

```powershell
python -m src.evaluation
```

Run each stage only after its required previous outputs have been generated.

Feature extraction and model training can take time. Existing output files may be overwritten when scripts are rerun.

---

## 11. Important Output Files

### 11.1 Feature Files

- `features/face/face_features.npy`
- `features/face/face_metadata.csv`
- `features/voice/voice_features.npy`
- `features/voice/voice_metadata.csv`
- `features/fusion/fusion_features.npy`
- `features/fusion/fusion_metadata.csv`

### 11.2 Model Files

Models are saved under the corresponding folders in:

- `models/face/`
- `models/voice/`
- `models/fusion/`
- `models/hierarchical/`

The current core model artifacts include:

- `models/face/face_svm.pkl`
- `models/face/face_random_forest.pkl`
- `models/voice/voice_static_mfcc_rms_zcr_svm.pkl`
- `models/voice/voice_static_mfcc_rms_zcr_random_forest.pkl`
- `models/fusion/fusion_svm.pkl`
- `models/fusion/fusion_random_forest.pkl`
- `models/hierarchical/hierarchical_face_random_forest.pkl`
- `models/hierarchical/hierarchical_voice_svm.pkl`
- `models/hierarchical/hierarchical_configuration.pkl`

### 11.3 Results

Results are saved under:

- `results/face/`
- `results/voice/`
- `results/fusion/`
- `results/hierarchical/`
- `results/comparison/`

The final comparison is saved as:

`results/comparison/final_model_comparison.csv`

The evaluation audit is saved as:

`results/comparison/evaluation_audit.csv`

The evaluation audit checks that the reported metrics are consistent with their corresponding confusion matrices.

---

## 12. Optional Extensions and Current Limitations

The project structure reserves locations for optional extensions:

| Extension | Reserved files/folders | Current status |
|---|---|---|
| Fingerprint verification | `src/fingerprint_features.py`, `src/fingerprint_verification.py`, `data/fingerprint/`, `features/fingerprint/`, `models/fingerprint/`, `results/fingerprint/` | Dataset available; implementation reserved |
| Real-time verification | `src/realtime.py` | Placeholder |
| Speech command recognition | `src/speech_commands.py`, `features/commands/`, `models/commands/`, `results/commands/` | Placeholder |
| Custom dataset | `data/custom_dataset/`, `features/custom_dataset/`, `models/custom_dataset/`, `results/custom_dataset/` | Reserved for optional data and experiments |

These extensions are not part of the completed core pipeline and should not be considered implemented functionality until their implementations and evaluations are added.

Fingerprint data is currently independent of VIDTIMIT and cannot be used for identity-matched fusion with the current face and voice data.

Other limitations include:

- Voice verification currently performs substantially worse than face verification.
- The experimental results are based on one dataset and the current split configuration.
- Pair-based test comparisons are not statistically independent.
- The current system is an experimental verification implementation, not a production-ready biometric security solution.

---

## 13. Project Status

The following core components have been implemented and experimentally evaluated:

- VIDTIMIT data preparation
- Identity-based enrollment, training, and testing splits
- Face feature extraction using HOG and LBP
- Voice feature extraction using MFCC, Delta MFCC, Delta-Delta MFCC, ZCR, and RMS
- Face-only verification using SVM and Random Forest
- Voice-only verification using SVM and Random Forest
- Early feature fusion
- Early fusion verification using SVM and Random Forest
- Hierarchical verification
- Final model comparison
- Evaluation audit

The latest evaluation audit checked nine model configurations:

- 9 passed
- 0 require checking

All reported metrics match their corresponding confusion matrices.

The optional extension folders and source placeholders are reserved for future work and are not included in the completed core evaluation.

Further work may include improving voice features, evaluating additional verification strategies, completing optional extensions, and preparing the final report and presentation.