import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, shapiro

# Load CSVs
df_1 = pd.read_csv("results/mouse_fusion/Mouse_results_RF_1_intra_test.csv")
df_2 = pd.read_csv("results/multimodal/Fusion_results_RF_1_intra_test.csv")

# Remove the 'Mean' row
df_1 = df_1[df_1["User"] != "Mean"]
df_2 = df_2[df_2["User"] != "Mean"]

# Convert to numeric
df_1["User"] = pd.to_numeric(df_1["User"], errors="coerce")
df_2["User"] = pd.to_numeric(df_2["User"], errors="coerce")

# Sort by User + Session if Session exists,
# otherwise sort only by User
if "Session" in df_1.columns:
    df_1["Session"] = pd.to_numeric(df_1["Session"], errors="coerce")
    df_1 = df_1.sort_values(["User", "Session"], na_position="last")
else:
    df_1 = df_1.sort_values("User")

if "Session" in df_2.columns:
    df_2["Session"] = pd.to_numeric(df_2["Session"], errors="coerce")
    df_2 = df_2.sort_values(["User", "Session"], na_position="last")
else:
    df_2 = df_2.sort_values("User")

df_1 = df_1.reset_index(drop=True)
df_2 = df_2.reset_index(drop=True)
print(np.setdiff1d(df_1['User'], df_2['User']))

# Check order and users match
assert all(df_1["User"].values == df_2["User"].values)

# Extract metrics
eer_1 = df_1["EER"].values
eer_2 = df_2["EER"].values

# Paired differences
diff = eer_1 - eer_2

# Normality test
shapiro_stat, shapiro_p = shapiro(diff)
alpha = 0.05

print(f"Shapiro-Wilk p-value: {shapiro_p:.6f}")

# Normally distributed differences
if shapiro_p > alpha:
    stat, p = ttest_rel(eer_1, eer_2)
    test_name = "Paired t-test"

    # Cohen's d
    cohens_d = diff.mean() / diff.std(ddof=1)

    ad = abs(cohens_d)

    if ad < 0.2:
        interpretation = "negligible"
    elif ad < 0.5:
        interpretation = "small"
    elif ad < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    effect_size_label = f"Cohen's d (dz): {cohens_d:.6f}"

# Non-Normal differences
else:
    test_name = "Wilcoxon signed-rank test"

    res = pg.wilcoxon(eer_1, eer_2)

    stat = res["W_val"].values[0]
    p = res["p_val"].values[0]
    rank_biserial = res["RBC"].values[0]

    ad = abs(rank_biserial)
    if ad < 0.1:
        interpretation = "negligible"
    elif ad < 0.3:
        interpretation = "small"
    elif ad < 0.5:
        interpretation = "medium"
    else:
        interpretation = "large"

    effect_size_label = f"Rank-biserial r: {rank_biserial:.6f}"

print(f"Using: {test_name}")
print(f"Statistic: {stat:.6f}")
print(f"p-value: {p:.6f}")
print(f"Statistically significant: {p <= alpha}")

print(f"\n{effect_size_label}")
print(f"Effect size: {interpretation}")
