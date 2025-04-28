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
def bootstrap_expected_mean_difference(group1, group2, num_iterations=10000):
    mean_diffs = []
    n1 = len(group1)
    n2 = len(group2)


    for _ in range(num_iterations):
        # Сэмплирование с возвращением True, без - False
        general = np.concatenate((group1, group2))
        sample1 = np.random.choice(general, size=n1, replace=False)
        sample2 = np.random.choice(general, size=n2, replace=False)
        mean_diff = np.mean(sample1) - np.mean(sample2)
        mean_diffs.append(mean_diff)

    shap, p = stats.shapiro(mean_diffs)
    return np.array(mean_diffs), shap, p


def full_bootstrap_mean_diff(df, col, text_add='', num_it=10000):
    # Selecting standardized data
    df = standardize_data(df, col)
    group1 = df[df['type'] != 'senior academics'][col + '_standardized'].values
    group2 = df[df['type'] == 'senior academics'][col + '_standardized'].values


    mean_expected_diffs, shap, pshap = bootstrap_expected_mean_difference(group1, group2, num_iterations=num_it)

    # Calculate p-value and confidence interval
    observed_diff = np.mean(group1) - np.mean(group2)
    p_value = np.mean(np.abs(mean_expected_diffs) >= np.abs(observed_diff))
    ci = np.percentile(mean_expected_diffs, [2.5, 97.5])

    # Calculate effect size (Cohen's d)
    s_pooled = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + (len(group2) - 1) * np.var(group2, ddof=1)) / 
                       (len(group1) + len(group2) - 2))
    delta = observed_diff
    cohens_d = observed_diff / s_pooled
    s_standard_diff_deviation = np.sqrt( (np.var(group1)/len(group1)) + (np.var(group2)/len(group2))
                                        )
    mdelta_lower = delta - 1.96*s_standard_diff_deviation
    mdelta_upper = delta + 1.96*s_standard_diff_deviation
    mdelta_ = [mdelta_lower, mdelta_upper]


    # Output of results
    print("Observed difference (standardized):", observed_diff)
    print("CI for observed difference", mdelta_)
    print("Cohen's d (effect size):", cohens_d)

    print("bootstrap p-value:", p_value)
    print("H0 Confidence interval (95%):", ci)
    if pshap<0.05:
        print('Expected mean diffs are not normal disrtibuted')
    else:
        print('Expected mean diffs are normal disrtibuted')

    # Plot a histogram of differences in means between groups 
    y, x, _ =plt.hist(mean_expected_diffs, bins=30, edgecolor='black')
    plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=1, label=f'Obs. difference')
    plt.axvline(ci[0], color='grey', linestyle='dashed', linewidth=1, label=f'H0 95% Confidence Interval')
    plt.axvline(ci[1], color='grey', linestyle='dashed', linewidth=1)
    plt.title('Distribution of exp. differences in means')
    plt.xlabel('Exp. differences in means\n(expected if there is no difference between groups)')
    plt.ylabel('frequency')
    plt.text(min(mean_expected_diffs), -0.4*y.max(), f'\n{text_add}"{col}" is compared between students and senior academics', fontsize=10)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    return ci, p_value, observed_diff, mdelta_, cohens_d
