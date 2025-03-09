import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


# 1. Data standardization
def standardize_data(df, column):
    mean = df[column].mean()
    std = df[column].std()
    df[column + '_standardized'] = (df[column] - mean) / std
    return df


# 2. bootstrap method
def bootstrap_expected_mean_difference(group1, group2, num_iterations=1000):
    mean_diffs = []
    n1 = len(group1)
    n2 = len(group2)


    for _ in range(num_iterations):
        # Сэмплирование с возвращением
        general = np.concatenate((group1, group2))
        sample1 = np.random.choice(general, size=n1, replace=True)
        sample2 = np.random.choice(general, size=n2, replace=True)
        mean_diff = np.mean(sample1) - np.mean(sample2)
        mean_diffs.append(mean_diff)

    return np.array(mean_diffs)


def full_bootstrap_mean_diff(df, col, text_add=''):
    # Selecting standardized data
    df = standardize_data(df, col)
    group1 = df[df['type'] != 'senior academics'][col + '_standardized'].values
    group2 = df[df['type'] == 'senior academics'][col + '_standardized'].values


    mean_expected_diffs = bootstrap_expected_mean_difference(group1, group2)

    # Calculate p-value and confidence interval
    observed_diff = np.mean(group1) - np.mean(group2)
    p_value = np.mean(np.abs(mean_expected_diffs) >= np.abs(observed_diff))
    ci = np.percentile(mean_expected_diffs, [2.5, 97.5])

    # Calculate effect size (Cohen's d)
    s_pooled = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / 
                       (len(group1) + len(group2) - 2))
    cohens_d = observed_diff / s_pooled


    # Output of results
    print("Observed mean difference (standardized):", observed_diff)
    print("bootstrap p-value:", p_value)
    print("Confidence interval (95%):", ci)
    print("Cohen's d (effect size):", cohens_d)

    # Plot a histogram of differences in means between groups 
    y, x, _ =plt.hist(mean_expected_diffs, bins=30, edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=1, label=f'Obs. difference in means')
    plt.axvline(ci[0], color='grey', linestyle='dashed', linewidth=1, label=f'95% Confidence Interval')
    plt.axvline(ci[1], color='grey', linestyle='dashed', linewidth=1)
    plt.title('Distribution of exp. differences in means')
    plt.xlabel('Exp. differences in means\n(expected if there is no difference between groups)')
    plt.ylabel('frequency')
    plt.text(min(mean_expected_diffs), -0.4*y.max(), f'\n{text_add}"{col}" is compared between students and senior academics', fontsize=10)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    return ci, p_value, observed_diff, cohens_d
