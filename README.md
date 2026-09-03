# Gmail Multimodal Dataset (GMD)
## Overview
The Gmail Multimodal Dataset (GMD) is a time-series behavioral biometrics dataset designed to support research on user authentication and multimodal authentication systems. It contains interaction data from 56 participants completing naturalistic, prescribed tasks in a desktop Gmail environment.
This repository contains the code used for the baseline evaluations presented in [paper]. GBM, SVM, RF and biLSTM classifiers were trained in for user authentication. Fusion was evaluated across modalities using a weighted average score-level fusion approach.

### Structure
* feature_extraction/
  * traditional/: feature extraction for keystroke, mouse, scroll, and widget features
  * lstm/: sequence-based feature extraction for LSTM training
* utils/
  * train_utils.py: train/validation split logic
  * lstm_train_utils.py: LSTM train/validation split logic
* train_traditional.py: trains classical ML classifiers
* train_lstm.py: trains bidirectional LSTM models
* adversarial_attack.py: evaluates robustness using random/naive adversarial vectors
* demographic_bias.py: demographic/bias analysis
* sig_diff.py: significance testing / statistical comparisons
* time_window_fusion.py: feature/window fusion experiments
* test_lstm.py: LSTM evaluation/testing
* model_scores/, models/, results/, adversarial_results/: generated output

## Modalities
The dataset contains four behavioral modalities:
Mouse: ~4.6 million events
Widget interactions: ~490,000 events
Keystrokes: ~559,000 events
Scrolling: ~2.3 million events
Data were collected over 4–8 sessions per participant using a Firefox extension that recorded timestamped interaction events.

## Tasks
Participants performed eight Gmail activities, including browsing and reading email, forwarding and replying to messages, deleting and restoring emails, searching for specific messages, attaching files, and reporting suspicious email as spam.
## Data Organization
The dataset is organized by modality, user, and session:
Each CSV contains one session of data. Session numbers are aligned across modalities when available, although some modalities may be absent from a session because the corresponding behavior did not occur.

## Data Schemas
Mouse: user ID, X/Y coordinates, event type, timestamp.
Widget: user ID, widget ID, event type, widget dimensions, timestamp.
Keystroke: user ID, key, key up/down event, timestamp.
Scroll: user ID, mouse/page coordinates, scroll deltas, timestamp.

## Intended Use
GMD is intended for developing and evaluating behavioral biometric authentication and multimodal fusion methods. The paper demonstrates baseline evaluations using GBM, SVM, Random Forest, and bidirectional LSTM classifiers, including single-modality and multimodal experiments.

## Access and Citation
The dataset is available upon request to the authors. Instructions for access can be found at: https://doi.org/10.5281/zenodo.22257851
