import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import shapiro


# ID groups
ASIAN_IDS = [
    5957492, 5286985, 8746874, 7316795, 6700716, 5150547,
    2886837, 7114382, 8348481, 7688576, 2693592, 7373019,
    4179044, 6638682, 8573429, 2633840, 3084978, 4372737,
    7310944, 9707394, 9333707, 9417614, 4542213, 7847405,
    2346653, 2722493, 9259884, 3695167, 1244132
]

WHITE_IDS = [
    1165247, 5254922, 1105159, 2750057, 8555844, 9733371,
    6200741, 8635407, 3726809, 9242542, 5865581, 1988613,
    6289350, 9274838, 7620603, 3130601, 5362628, 6362291,
    4148061
]

MALE_IDS = [
    5957492, 5286985, 1165247, 8746874, 2886837, 7114382,
    5254922, 1105159, 8822529, 8555844, 7688576, 9733371,
    7373019, 1174404, 8824899, 6638682, 8635407, 5577693,
    3084978, 4372737, 1961295, 8503192, 3726809, 5865581,
    1988613, 7310944, 7620603, 3130601, 9333707, 5362628,
    9417614, 7847405, 2722493, 3695167, 1244132, 4148061
]

FEMALE_IDS = [
    7316795, 6700716, 8348481, 2693592, 1865189, 4179044,
    8573429, 6200741, 2633840, 9707394, 4542213, 2346653,
    9259884, 6362291
]

MOUSE_IDS = [
    5150547, 2886837, 8822529, 1865189, 6708958, 3084978,
    6289350, 5362628, 7847405, 4148061
]

TOUCHPAD_IDS = [
    7114382, 5254922, 1105159, 8348481, 2750057, 8555844,
    7688576, 2693592, 9733371, 7373019, 1174404, 8824899,
    6638682, 8573429, 6200741, 8635407, 2633840, 5577693,
    4372737, 1961295, 8503192, 3726809, 9242542, 5865581,
    1988613, 7310944, 9274838, 7620603, 3130601, 9707394,
    9333707, 9417614, 4542213, 2346653, 2722493, 9259884,
    3695167, 6362291
]


# Group definitions
RACE_GROUPS = {
    "Asian": ASIAN_IDS,
    "White": WHITE_IDS,
}

GENDER_GROUPS = {
    "Male": MALE_IDS,
    "Female": FEMALE_IDS,
}

DEVICE_GROUPS = {
    "Mouse": MOUSE_IDS,
    "Touchpad": TOUCHPAD_IDS,
}


# Assign demographic group
def assign_group(results, groups, column_name):

    results[column_name] = "Unknown"

    for group_name, ids in groups.items():
        results.loc[
            results["User"].isin(ids),
            column_name
        ] = group_name

    return results


# Check normality using Shapiro-Wilk
def shapiro_normality(data, alpha=0.05):

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return np.nan, False

    statistic, p_value = shapiro(data)

    return p_value, p_value >= alpha


# Normal distributions: Welch's t-test + Cohen's d
# Non-normal distributions: Mann-Whitney U + rank-biserial correlation
def compare_groups(results, group_column, alpha=0.05):

    rows = []

    groups = [
        g for g in results[group_column].dropna().unique()
        if g != "Unknown"
    ]

    for group in groups:
        # Group EER
        group_eer = results.loc[results[group_column] == group, "EER"].dropna()

        # Everyone else in the defined groups
        rest_eer = results.loc[(results[group_column] != group) & (results[group_column] != "Unknown"), "EER"].dropna()

        n1 = len(group_eer)
        n2 = len(rest_eer)

        mean1 = group_eer.mean()
        mean2 = rest_eer.mean()

        std1 = group_eer.std(ddof=1)
        std2 = rest_eer.std(ddof=1)

        difference = mean1 - mean2

        # Shapiro-Wilk normality tests
        group_shapiro_p, group_normal = shapiro_normality(group_eer, alpha)
        rest_shapiro_p, rest_normal = shapiro_normality(rest_eer, alpha)

        # PARAMETRIC WELCH'S T-TEST
        if group_normal and rest_normal:

            welch = pg.ttest(
                group_eer,
                rest_eer,
                paired=False,
                correction=True
            )

            p_value = welch["p_val"].iloc[0]

            effect_size = welch["cohen_d"].iloc[0]

            test_name = "Welch's t-test"
            effect_size_type = "Cohen's d"

        # NON-PARAMETRIC MANN-WHITNEY U
        else:

            mw = pg.mwu(
                group_eer,
                rest_eer,
                alternative="two-sided"
            )

            p_value = mw["p_val"].iloc[0]

            effect_size = mw["RBC"].iloc[0]

            test_name = "Mann-Whitney U"
            effect_size_type = "Rank-biserial correlation"

        # Statistical significance
        significant = p_value < alpha

        # Store results
        rows.append({
            "group": group,
            "n_group": n1,
            "n_rest": n2,
            "group_mean": mean1,
            "group_std": std1,
            "rest_mean": mean2,
            "rest_std": std2,
            "difference": difference,
            "group_shapiro_p": group_shapiro_p,
            "rest_shapiro_p": rest_shapiro_p,
            "test": test_name,
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_size_type": effect_size_type,
            "significant": significant
        })

    return pd.DataFrame(rows)


# MAIN ANALYSIS
def bias_corr(results_path):
    # Load results CSV
    results = pd.read_csv(results_path)

    # Remove Mean row
    results = results[results["User"].astype(str) != "Mean"].copy()

    # Convert User and EER to numeric
    results["User"] = pd.to_numeric(results["User"], errors="coerce")
    results["EER"] = pd.to_numeric(results["EER"], errors="coerce")

    # Assign demographic groups
    results = assign_group(results, RACE_GROUPS, "Race")
    results = assign_group(results, GENDER_GROUPS, "Gender")
    results = assign_group(results, DEVICE_GROUPS, "Device")

    # Statistical tests
    race_tests = compare_groups(results, "Race")
    gender_tests = compare_groups(results, "Gender")
    device_tests = compare_groups(results, "Device")

    # PRINT RESULTS
    print("\n")
    print("=" * 90)
    print("RACE — STATISTICAL SIGNIFICANCE + EFFECT SIZE")
    print("=" * 90)

    print(race_tests.to_string(index=False))

    print("\n")
    print("=" * 90)
    print("GENDER — STATISTICAL SIGNIFICANCE + EFFECT SIZE")
    print("=" * 90)

    print(gender_tests.to_string(index=False))

    print("\n")
    print("=" * 90)
    print("DEVICE — STATISTICAL SIGNIFICANCE + EFFECT SIZE")
    print("=" * 90)

    print(device_tests.to_string(index=False))

    # RETURN RESULTS
    return {
        "results": results,
        "race_tests": race_tests,
        "gender_tests": gender_tests,
        "device_tests": device_tests,
    }


# RUN ALL MODALITIES
modalities = [
    "mouse",
    "widget",
    "keystroke",
    "scroll",
]

analyses = {}

for modality in modalities:

    filename = (
        f"results/"
        f"user_metrics_GBM_{modality}_inter.csv"
    )

    print("\n")
    print("#" * 90)
    print(f"# RUNNING ANALYSIS: {modality.upper()}")
    print(f"# FILE: {filename}")
    print("#" * 90)

    analyses[modality] = bias_corr(filename)