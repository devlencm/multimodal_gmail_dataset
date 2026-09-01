import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os
import tqdm

def window_and_fuse(m, weights, window_min=2, required_modalities=None, split="test"):
    # Get sample start times, model scores, labels, and corresponding modality IDs
    times = np.array(m[split]["start"])
    scores = np.array(m[split]["scores"])
    labels = np.array(m[split]["labels"])
    m_id = np.array(m[split]["m_id"])

    all_labels = []
    all_scores = []

    modality_labels = {mod: [] for mod in weights}
    modality_scores = {mod: [] for mod in weights}

    # Convert window size from minutes to ms
    window_size = window_min * 60 * 1000

    # Get unique classes
    unique_labels = np.unique(labels)

    # Group samples by label (genuine and impostor do not occur at same time)
    for label in unique_labels:
        # Get only scores and times for current label
        mask = labels == label
        label_times = times[mask]
        label_scores = scores[mask]
        label_m_id = m_id[mask]

        # Sort scores + time samples by time
        sort_idx = np.argsort(label_times)
        label_times = label_times[sort_idx]
        label_scores = label_scores[sort_idx]
        label_m_id = label_m_id[sort_idx]

        current_start = label_times[0]
        window_scores = [label_scores[0]]
        window_m_ids = [label_m_id[0]]

        # From the starting time, segment non overlapping windows of N-minutes
        for time, score, m_id_val in zip(label_times[1:], label_scores[1:], label_m_id[1:]):
            if time - current_start <= window_size:
                window_scores.append(score)
                window_m_ids.append(m_id_val)
            else:
                unique_m_ids = np.unique(window_m_ids)

                # Store modality-specific scores for this window
                for mod in unique_m_ids:
                    mask_mod = np.array(window_m_ids) == mod
                    mod_scores = np.array(window_scores)[mask_mod]

                    if mod in modality_scores:
                        modality_labels[mod].append(label)
                        modality_scores[mod].append(np.mean(mod_scores))

                # Calculate multimodal fusion only if >=2 modalities are present
                if len(unique_m_ids) >= 2:
                    mod_means = []
                    mod_weights = []

                    for mod in unique_m_ids:
                        mask_mod = np.array(window_m_ids) == mod
                        mod_scores = np.array(window_scores)[mask_mod]

                        mod_means.append(np.mean(mod_scores))
                        mod_weights.append(weights.get(mod, 0))

                    mod_means = np.array(mod_means)
                    mod_weights = np.array(mod_weights)

                    if np.sum(mod_weights) > 0:
                        per_mod_mean = np.sum(mod_weights * mod_means) / np.sum(
                            mod_weights
                        )

                        all_labels.append(label)
                        all_scores.append(per_mod_mean)

                # Set window bound as new start
                current_start = time
                window_scores = [score]
                window_m_ids = [m_id_val]

        # Handle last window
        if len(window_scores) > 0:
            unique_m_ids = np.unique(window_m_ids)

            # Store modality-specific scores for this window
            for mod in unique_m_ids:
                mask_mod = np.array(window_m_ids) == mod
                mod_scores = np.array(window_scores)[mask_mod]

                if mod in modality_scores:
                    modality_labels[mod].append(label)
                    modality_scores[mod].append(np.mean(mod_scores))

            # Calculate fusion only if >=2 modalities are present
            if len(unique_m_ids) >= 2:
                mod_means = []
                mod_weights = []

                for mod in unique_m_ids:
                    mask_mod = np.array(window_m_ids) == mod
                    mod_scores = np.array(window_scores)[mask_mod]

                    mod_means.append(np.mean(mod_scores))
                    mod_weights.append(weights.get(mod, 0))

                mod_means = np.array(mod_means)
                mod_weights = np.array(mod_weights)

                if np.sum(mod_weights) > 0:
                    per_mod_mean = np.sum(mod_weights * mod_means) / np.sum(mod_weights)

                    all_labels.append(label)
                    all_scores.append(per_mod_mean)

    return (
        np.array(all_labels),
        np.array(all_scores),
        {mod: np.array(modality_labels[mod]) for mod in weights},
        {mod: np.array(modality_scores[mod]) for mod in weights},
    )


def get_windows(m, window_min=10, split="test"):
    # For the input modality, get start times, scores, and labels
    times = np.array(m[split]["start"])
    scores = np.array(m[split]["scores"])
    labels = np.array(m[split]["labels"])

    all_labels = []
    all_scores = []

    # Convert window size into ms
    window_size = window_min * 60 * 1000

    # Get the unique labels (user IDs)
    unique_labels = np.unique(labels)

    # Group samples by label (genuine and impostor do not occur at same time)
    for label in unique_labels:
        # Get only scores and times for current label
        mask = labels == label
        label_times = times[mask]
        label_scores = scores[mask]

        # Sort scores + time samples by time
        sort_idx = np.argsort(label_times)
        label_times = label_times[sort_idx]
        label_scores = label_scores[sort_idx]

        current_start = label_times[0]
        window_scores = [label_scores[0]]

        # From the starting time, segment non overlapping windows of N-minutes
        for time, score in zip(label_times[1:], label_scores[1:]):
            if time - current_start <= window_size:
                window_scores.append(score)
            else:
                # Time difference is > window _size, get the mean score per window, assign the label
                all_labels.append(label)
                all_scores.append(np.mean(window_scores))

                # Set window bound as new start
                current_start = time
                window_scores = [score]

        # Save last window
        if len(window_scores) > 0:
            all_labels.append(label)
            all_scores.append(np.mean(window_scores))

    return np.array(all_labels), np.array(all_scores)


def compute_eer_auc(labels, scores):
    # Get false positive rate, true positive rate
    fpr, tpr, _ = roc_curve(labels, scores)

    # Calculate EER by finding the intersection of FAR and FRR
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    # Calculate AUC
    auc = roc_auc_score(labels, scores)
    return eer, auc


def plot_roc(y, s, label):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")


def merge_modalities(user, modalities):
    # Define a dictionary that will represent our "merged" modalities
    merged = {"test": {"start": [], "end": [], "scores": [], "labels": [], "m_id": []}}

    # Iterate over modalities
    for modality_name, modality_data in modalities.items():
        # For each modality, simply extend their scores, labels, start times, end times into the new modality
        m = modality_data[user]

        merged["test"]["start"].extend(m["test"]["start"])
        merged["test"]["end"].extend(m["test"]["end"])
        merged["test"]["scores"].extend(np.array(m["test"]["scores"]))
        merged["test"]["labels"].extend(m["test"]["labels"])

        # m_id represents which modality the scores labels etc. belong to
        merged["test"]["m_id"].extend(len(m["test"]["labels"]) * [modality_name])

    # Convert each field in 'test' to numpy array
    for key in merged["test"]:
        merged["test"][key] = np.array(merged["test"][key])

    return merged


# Inter-session fusion

def multi_modal_fusion_inter(window_min=1, model="GBM"):
    # Load scores for each modality
    mouse = np.load(f"model_scores/{model}/mouse/user_scores.npy", allow_pickle=True).item()
    keystroke = np.load(f"model_scores/{model}/keystroke/user_scores.npy", allow_pickle=True).item()
    widget = np.load(f"model_scores/{model}/widget/user_scores.npy", allow_pickle=True).item()
    scroll = np.load(f"model_scores/{model}/scroll/user_scores.npy", allow_pickle=True).item()

    # Define a dictionary containing all modality dicts
    modalities = {
        "Mouse": mouse,
        "Keystroke": keystroke,
        "Widget": widget,
        "Scroll": scroll,
    }

    # Get intersecting users across modalities
    all_users = sorted(set.intersection(*[set(modality_data.keys()) for modality_data in modalities.values()]))

    global_modality = {m: {"y": [], "s": []} for m in modalities.keys()}
    global_fusion = {"y": [], "s": []}
    modality_results = {modality_name: [] for modality_name in modalities.keys()}

    fusion_results = []

    # Iterate over each user and fuse scores for both single and multiple modalities
    for user in tqdm.tqdm(all_users, desc="Inter-session fusion"):
        user_int = int(user)

        user_weight_dict = {}

        # Create weights inverse to single modality EER on training set
        for modality_name, modality_data in modalities.items():
            m = modality_data[user]

            labels_train, scores_train = get_windows(m, window_min, split="test")
            binary_labels_train = (labels_train == user_int).astype(int)

            eer, _ = compute_eer_auc(binary_labels_train, scores_train)
            user_weight_dict[modality_name] = 1 / (eer + 1e-6)

        # Normalize weights to sum to 1
        total = sum(user_weight_dict.values())
        for mod in user_weight_dict:
            user_weight_dict[mod] /= total

        # Combine score lists for modalities (sorted in chronological order)
        fused_data = merge_modalities(user, modalities)

        # Get the scores + labels for each N-minute window (single and multi modality)
        labels_test, scores_test, modality_labels_test, modality_scores_test = (
            window_and_fuse(fused_data, user_weight_dict, window_min, split="test")
        )

        # For each individual modality, get their scores and labels, fuse
        for modality_name in modalities:
            labels_modality = modality_labels_test[modality_name]
            scores_modality = modality_scores_test[modality_name]

            # Create genuine and impostor binary labels based on current user
            binary_labels = (labels_modality == user_int).astype(int)

            # Extend the global modality lists for later ROC computation
            global_modality[modality_name]["y"].extend(binary_labels)
            global_modality[modality_name]["s"].extend(scores_modality)

            # Compute EER and AUC
            eer, auc = compute_eer_auc(binary_labels, scores_modality)

            # Store EER and AUC in modality_results
            modality_results[modality_name].append({"User": user, "EER": eer, "AUC": auc})

        # Create genuine and impostor binary labels based on current user across all modalities
        binary_labels_multi = (labels_test == user_int).astype(int)

        # Extend global_fusion for ROC
        global_fusion["y"].extend(binary_labels_multi)
        global_fusion["s"].extend(scores_test)

        # Compute fusion EER and AUC
        eer, auc = compute_eer_auc(binary_labels_multi, scores_test)

        # Append fusion results
        fusion_results.append({"User": user, "EER": eer, "AUC": auc})

    # Save results for single and multiple modalities
    for modality_name, results in modality_results.items():
        df = pd.DataFrame(results)

        if len(df) == 0:
            continue

        mean_row = pd.DataFrame(
            [{"User": "Mean", "EER": df["EER"].mean(), "AUC": df["AUC"].mean()}]
        )

        df = pd.concat([df, mean_row], ignore_index=True)

        df.to_csv(f"{modality_name}_results_{model}_{window_min}_inter_test.csv", index=False)

    df_fusion = pd.DataFrame(fusion_results)

    if len(df_fusion) > 0:
        mean_row = pd.DataFrame(
            [
                {
                    "User": "Mean",
                    "EER": df_fusion["EER"].mean(),
                    "AUC": df_fusion["AUC"].mean(),
                }
            ]
        )

        df_fusion = pd.concat([df_fusion, mean_row], ignore_index=True)

    df_fusion.to_csv(f"Fusion_results_{model}_{window_min}_inter_test.csv", index=False)

    plt.figure(figsize=(7, 6))

    for mod in global_modality:
        y = np.array(global_modality[mod]["y"])
        s = np.array(global_modality[mod]["s"])
        plot_roc(y, s, mod)

    y = np.array(global_fusion["y"])
    s = np.array(global_fusion["s"])

    plot_roc(y, s, "Fusion")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model} Inter-Session ROC Curves ({window_min} min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model}_roc_curves_{window_min}min_inter.png")
    plt.close()


# Intra-session loading

def load_intra_modalities(model="RF"):
    modalities = {}

    # Iterate over 4 modalities
    for modality_name in ["mouse", "keystroke", "scroll"]:
        modality_dir = os.path.join("model_scores", model, modality_name)

        upper_name = modality_name.capitalize()
        modalities[upper_name] = {}

        if not os.path.exists(modality_dir):
            continue

        # Iterate over each user directory under the modality model scores directory
        for user_dir in os.listdir(modality_dir):
            if not user_dir.startswith("user_") or user_dir.startswith("user_scores"):
                continue

            user = int(user_dir.replace("user_", ""))
            user_path = os.path.join(modality_dir, user_dir)

            modalities[upper_name][user] = {}

            # Iterate over sessions for a given user
            for sess_file in os.listdir(user_path):
                if not sess_file.startswith("session_") or not sess_file.endswith(
                    ".npy"
                ):
                    continue

                session = int(
                    float(sess_file.replace("session_", "").replace(".npy", ""))
                )

                sess_path = os.path.join(user_path, sess_file)

                # Load the scores for the modality/user/session, store in modalities dict
                modalities[upper_name][user][session] = np.load(
                    sess_path, allow_pickle=True
                ).item()

    return modalities


def merge_modalities_intra(user, session, modalities):
    # Define a dictionary that will represent our "merged" modalities
    merged = {"test": {"start": [], "end": [], "scores": [], "labels": [], "m_id": []}}

    # Iterate over modalities
    for modality_name, modality_data in modalities.items():
        if user not in modality_data:
            continue

        if session not in modality_data[user]:
            continue

        m = modality_data[user][session]

        # For each modality, simply extend their scores, labels, start times, end times into the new modality
        merged["test"]["start"].extend(m["test"]["start"])
        merged["test"]["end"].extend(m["test"]["end"])
        merged["test"]["scores"].extend(np.array(m["test"]["scores"]))
        merged["test"]["labels"].extend(m["test"]["labels"])

        # m_id represents which modality the scores labels etc. belong to
        merged["test"]["m_id"].extend(len(m["test"]["labels"]) * [modality_name])

    # Convert each field in 'test' to numpy array
    for key in merged["test"]:
        merged["test"][key] = np.array(merged["test"][key])

    return merged


# Intra-session fusion
def multi_modal_fusion_intra(window_min=1, model="SVM"):
    # Load the model scores for each experiment
    modalities = load_intra_modalities(model=model)

    # Get all users across all modalities
    all_users = sorted(
        set.intersection(
            *[set(modality_data.keys()) for modality_data in modalities.values()]
        )
    )

    global_modality = {m: {"y": [], "s": []} for m in modalities.keys()}
    global_fusion = {"y": [], "s": []}

    # Single-modality results
    modality_results = {modality_name: [] for modality_name in modalities.keys()}

    # Multimodal fusion results
    fusion_results = []

    # Iterate over users
    for user in tqdm.tqdm(all_users, desc="Intra-session fusion"):
        user_int = int(user)

        # Sessions shared by all modalities for this user
        common_sessions = set.intersection(
            *[set(modality_data[user].keys()) for modality_data in modalities.values()]
        )

        # Iterate over sessions shared by all modalities
        for session in sorted(common_sessions):

            # Calculate modality fusion weights from train
            user_weight_dict = {}

            # Iterate over modalities
            for modality_name, modality_data in modalities.items():

                m = modality_data[user][session]

                # Get the scores + labels for each N-minute window in the training data
                labels_train, scores_train = get_windows(m, window_min, split="test")

                # Create genuine and impostor binary labels based on current user
                binary_labels_train = (labels_train == user_int).astype(int)

                # Compute the single-modality EER used to determine the fusion weight
                eer, _ = compute_eer_auc(binary_labels_train, scores_train)

                user_weight_dict[modality_name] = 1 / (eer + 1e-6)


            # Normalize weights to sum to 1
            total = sum(user_weight_dict.values())

            for mod in user_weight_dict:
                user_weight_dict[mod] /= total

            # Merge and fuse THIS session only
            fused_data = merge_modalities_intra(user, session, modalities)

            # Get the scores + labels for each N-minute window (single and multi modality)
            labels_test, scores_test, modality_labels_test, modality_scores_test = (
                window_and_fuse(
                    fused_data, user_weight_dict, window_min, split="test"
                )
            )

            # For each individual modality, get their scores and labels, fuse
            for modality_name in modalities:
                labels_modality = modality_labels_test[modality_name]
                scores_modality = modality_scores_test[modality_name]

                # Create genuine and impostor binary labels based on current user
                binary_labels = (labels_modality == user_int).astype(int)

                if len(np.unique(binary_labels)) < 2:
                    continue

                # Extend the global modality lists for later ROC computation
                global_modality[modality_name]["y"].extend(binary_labels)
                global_modality[modality_name]["s"].extend(scores_modality)

                # Compute EER and AUC
                eer, auc = compute_eer_auc(binary_labels, scores_modality)

                # Store EER and AUC in modality_results
                modality_results[modality_name].append(
                    {"User": user, "Session": session, "EER": eer, "AUC": auc}
                )

            # Create genuine and impostor binary labels based on current user across all modalities
            binary_labels_test = (labels_test == user_int).astype(int)

            if len(np.unique(binary_labels_test)) < 2:
                continue

            # Extend global_fusion for ROC
            global_fusion["y"].extend(binary_labels_test)
            global_fusion["s"].extend(scores_test)

            # Compute fusion EER and AUC
            eer, auc = compute_eer_auc(binary_labels_test, scores_test)

            # Store session-level fusion results
            fusion_results.append(
                {"User": user, "Session": session, "EER": eer, "AUC": auc}
            )

    # Save single-modality results
    for modality_name, results in modality_results.items():
        df = pd.DataFrame(results)
        mean_row = pd.DataFrame(
            [
                {
                    "User": "Mean",
                    "Session": "",
                    "EER": df["EER"].mean(),
                    "AUC": df["AUC"].mean(),
                }
            ]
        )
        df = pd.concat([df, mean_row], ignore_index=True)
        df.to_csv(f"results/{modality_name}_fusion/{modality_name}_results_{model}_{window_min}_intra_test.csv", index=False)

    # Save session-level fusion results
    df_fusion = pd.DataFrame(fusion_results)
    mean_row = pd.DataFrame(
        [
            {
                "User": "Mean",
                "Session": "",
                "EER": df_fusion["EER"].mean(),
                "AUC": df_fusion["AUC"].mean(),
            }
        ]
    )
    df_fusion = pd.concat([df_fusion, mean_row], ignore_index=True)
    df_fusion.to_csv(f"results/multimodal/Fusion_results_{model}_{window_min}_intra_test.csv", index=False)

    # ROC Plot
    plt.figure(figsize=(7, 6))

    for mod in global_modality:
        y = np.array(global_modality[mod]["y"])
        s = np.array(global_modality[mod]["s"])
        plot_roc(y, s, mod)

    y = np.array(global_fusion["y"])
    s = np.array(global_fusion["s"])

    plot_roc(y, s, "Fusion")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model} Intra-Session ROC Curves ({window_min} min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model}_roc_curves_{window_min}min_intra.png")
    plt.close()


# RUN FUSION EXPERIMENTS

# Inter-session
for i in range(1, 6):
    multi_modal_fusion_inter(window_min=i, model="RF")

# Intra-session
for i in range(1, 6):
    multi_modal_fusion_intra(window_min=i, model="RF")