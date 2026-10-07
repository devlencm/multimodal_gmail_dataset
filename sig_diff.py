import os
import numpy as np
import pandas as pd
import pingouin as pg

from scipy.stats import ttest_rel, shapiro
from statsmodels.stats.multitest import multipletests


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.05
METRIC = "EER"

# Display name -> filename name
MODEL_FILE_NAMES = {
    "GBM": "GBM",
    "SVM": "SVM",
    "RF": "RF",
    "biLSTM": "LSTM",
}

# Available modalities for each model
MODEL_MODALITIES = {
    "GBM": ["mouse", "widget", "keystroke", "scroll"],
    "SVM": ["mouse", "widget", "keystroke", "scroll"],
    "RF": ["mouse", "widget", "keystroke", "scroll"],
    "biLSTM": ["mouse", "keystroke", "scroll"],
}

TRIALS = {
    "GBM": [1, 2, 3, 4, 5],
    "SVM": [1, 2, 3, 4, 5],
    "RF": [1, 2, 3, 4, 5],
    "biLSTM": [1, 2, 3, 4, 5],
}

SESSIONS = ["intra", "inter"]


# Helpers
def modality_display(modality):
    return modality.capitalize()


def load_dataframe(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nFile not found:\n{filepath}"
        )

    df = pd.read_csv(filepath)

    # Remove Mean row
    if "User" in df.columns:
        df = df[df["User"] != "Mean"].copy()

    # Convert User to numeric
    df["User"] = pd.to_numeric(df["User"], errors="coerce")

    # Sort
    if "Session" in df.columns:
        df["Session"] = pd.to_numeric(df["Session"], errors="coerce")
        df = df.sort_values(["User", "Session"], na_position="last")
    else:
        df = df.sort_values("User")

    return df.reset_index(drop=True)


def get_paired_values(file_1, file_2, metric):

    df_1 = load_dataframe(file_1)
    df_2 = load_dataframe(file_2)

    users_1 = df_1["User"].values
    users_2 = df_2["User"].values

    if not np.array_equal(users_1, users_2):

        only_1 = np.setdiff1d(users_1, users_2)
        only_2 = np.setdiff1d(users_2, users_1)

        print("\nUSER MISMATCH")
        print("File 1:", file_1)
        print("File 2:", file_2)
        print("Only in file 1:", only_1)
        print("Only in file 2:", only_2)

        raise ValueError(
            "User IDs/order do not match."
        )

    x = df_1[metric].astype(float).values
    y = df_2[metric].astype(float).values

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        raise ValueError(
            f"Only {len(x)} valid pairs: "
            f"{file_1} vs {file_2}"
        )

    return x, y


# Statistical tests
def run_paired_test(x, y):

    diff = x - y

    shapiro_stat, shapiro_p = shapiro(diff)

    # Paired t-test (normal)
    if shapiro_p > ALPHA:
        stat, p = ttest_rel(x, y)

        test_name = "Paired t-test"
        sd_diff = diff.std(ddof=1)

        if sd_diff == 0:
            effect_size = np.nan
        else:
            effect_size = diff.mean() / sd_diff

        effect_type = "Cohen's dz"

        ad = abs(effect_size)

        if ad < 0.2:
            effect = "Negligible"
        elif ad < 0.5:
            effect = "Small"
        elif ad < 0.8:
            effect = "Medium"
        else:
            effect = "Large"

    # Wilcoxon Signed-Rank
    else:
        res = pg.wilcoxon(x, y)
        stat = res["W_val"].iloc[0]
        p = res["p_val"].iloc[0]

        effect_size = res["RBC"].iloc[0]
        test_name = "Wilcoxon signed-rank test"
        effect_type = "Rank-biserial r"

        ad = abs(effect_size)

        if ad < 0.1:
            effect = "Negligible"
        elif ad < 0.3:
            effect = "Small"
        elif ad < 0.5:
            effect = "Medium"
        else:
            effect = "Large"

    return {
        "N": len(x),
        "Test": test_name,
        "Statistic": stat,
        "Shapiro_p": shapiro_p,
        "Raw_p": p,
        "Effect_size": effect_size,
        "Effect_size_type": effect_type,
        "Effect": effect,
        "Mean_1": np.mean(x),
        "Mean_2": np.mean(y),
        "Mean_difference": np.mean(diff),
    }


# Make comparisons
def make_comparison(file_1, file_2, comparison, family, session, modality=None, trial=None):
    x, y = get_paired_values(file_1, file_2, METRIC)

    result = run_paired_test(x, y)

    result.update({
        "Family": family,
        "Session": session,
        "Trial": trial,
        "Modality": modality,
        "Comparison": comparison,
        "File_1": file_1,
        "File_2": file_2,
    })

    return result


# Model level comparisons (e.g., GBM vs SVM)
def generate_model_comparisons():

    results = []

    models = list(MODEL_FILE_NAMES.keys())
    for session in SESSIONS:

        family = f"Model_{session}"

        # Each modality separately
        for modality in [
            "mouse",
            "widget",
            "keystroke",
            "scroll"
        ]:

            # Only include models that actually have this modality
            available_models = [
                model
                for model in models
                if modality in MODEL_MODALITIES[model]
            ]

            # Pairwise model comparisons
            for i in range(len(available_models)):
                for j in range(i + 1, len(available_models)):

                    model_1 = available_models[i]
                    model_2 = available_models[j]

                    file_model_1 = MODEL_FILE_NAMES[model_1]
                    file_model_2 = MODEL_FILE_NAMES[model_2]

                    file_1 = (
                        "results/single_sample/"
                        f"user_metrics_{file_model_1}_"
                        f"{modality}_{session}.csv"
                    )

                    file_2 = (
                        "results/single_sample/"
                        f"user_metrics_{file_model_2}_"
                        f"{modality}_{session}.csv"
                    )

                    comparison = (
                        f"{model_1} vs {model_2}"
                    )

                    result = make_comparison(
                        file_1,
                        file_2,
                        comparison=comparison,
                        family=family,
                        session=session,
                        modality=modality_display(modality)
                    )

                    results.append(result)

    return results


# Fusion Comparisons
def generate_fusion_comparisons():

    results = []

    for session in SESSIONS:
        family = f"Fusion_{session}"
        for model in MODEL_FILE_NAMES:
            filename_model = MODEL_FILE_NAMES[model]

            # Only modalities that exist for this model
            for modality in MODEL_MODALITIES[model]:

                display_modality = modality_display(
                    modality
                )

                for trial in TRIALS[model]:
                    # Single modality file
                    modality_file = (
                        f"results/{modality}_fusion/"
                        f"{display_modality}_results_"
                        f"{filename_model}_{trial}_"
                        f"{session}_test.csv"
                    )

                    # Fusion file
                    fusion_file = (
                        "results/multimodal/"
                        f"Fusion_results_"
                        f"{filename_model}_{trial}_"
                        f"{session}_test.csv"
                    )

                    comparison = (
                        f"{model} {trial}: "
                        f"{display_modality} vs Fusion"
                    )

                    result = make_comparison(
                        modality_file,
                        fusion_file,
                        comparison=comparison,
                        family=family,
                        session=session,
                        modality=display_modality,
                        trial=f"{model} {trial}"
                    )

                    results.append(result)

    return results


# Run
all_results = []

print("=" * 80)
print("MODEL COMPARISONS")
print("=" * 80)

all_results.extend(
    generate_model_comparisons()
)

print("=" * 80)
print("FUSION COMPARISONS")
print("=" * 80)

all_results.extend(
    generate_fusion_comparisons()
)

results_df = pd.DataFrame(all_results)

# Holm Correction
results_df["Holm_p"] = np.nan
results_df["Significant"] = False

for family in results_df["Family"].unique():
    mask = results_df["Family"] == family
    raw_p = results_df.loc[mask, "Raw_p"].values
    reject, corrected_p, _, _ = multipletests(raw_p, alpha=ALPHA, method="holm")

    results_df.loc[mask, "Holm_p"] = corrected_p
    results_df.loc[mask, "Significant"] = reject

# Organize results df
results_df = results_df[
    [
        "Family",
        "Session",
        "Trial",
        "Modality",
        "Comparison",
        "N",
        "Test",
        "Statistic",
        "Shapiro_p",
        "Raw_p",
        "Holm_p",
        "Significant",
        "Effect_size",
        "Effect_size_type",
        "Effect",
        "Mean_1",
        "Mean_2",
        "Mean_difference",
        "File_1",
        "File_2",
    ]
]


results_df = results_df.sort_values(
    [
        "Family",
        "Modality",
        "Trial",
        "Comparison"
    ],
    na_position="last"
).reset_index(drop=True)


print("\n")
print("=" * 120)
print("HOLM-CORRECTED RESULTS")
print("=" * 120)

print(
    results_df[
        [
            "Family",
            "Session",
            "Trial",
            "Modality",
            "Comparison",
            "Test",
            "Raw_p",
            "Holm_p",
            "Significant",
            "Effect_size",
            "Effect"
        ]
    ].to_string(index=False)
)


# Save results
os.makedirs(
    "results/statistics",
    exist_ok=True
)

results_df.to_csv(
    "results/statistics/holm_results.csv",
    index=False
)