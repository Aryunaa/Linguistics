import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import chi2


# 1. Data standardization
def standardize_data(df, column):
    mean = df[column].mean()
    std = np.std(df[column].values, ddof=1)
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


# def bootstrap_test_for_variance(df_students, df_seniors, col):
#     """
#     Функция проводит бутстрэппинг для проверки гипотезы равенства дисперсий между двумя группами студентов и старших преподавателей.
#     Возвращает значение p-value на основе бутстрэпа и сравнение с критическим значением хи-квадрат.
#     """
#     var_students = np.var(df_students[col])
#     var_seniors = np.var(df_seniors[col])
#     n_students = len(df_students)
#     n_seniors = len(df_seniors)
    
#     # Число степеней свободы
#     dof_students = n_students - 1
#     dof_seniors = n_seniors - 1
    
#     # Общее число степеней свободы
#     total_dof = dof_students + dof_seniors
    
#     # Создаем пустые списки для хранения оцененных дисперсий
#     bootstrapped_student_vars = []
#     bootstrapped_senior_vars = []
    
#     # Количество итераций бутстрапа
#     num_bootstraps = 10000
    
#     # Проведение бутстрэпа
#     for i in range(num_bootstraps):
#         # student_sample = resample(df_students[col], replace=True, random_state=i)
#         student_sample = df_students[col].sample(n=len(df_students), replace=True, random_state=i)
#         senior_sample = df_seniors[col].sample(n=len(df_seniors), replace=True, random_state=i)
#         # senior_sample = resample(df_seniors[col], replace=True, random_state=i)
        
#         bootstrapped_student_vars.append(np.var(student_sample))
#         bootstrapped_senior_vars.append(np.var(senior_sample))
    
#     # Вычисляем наблюденную разницу дисперсий
#     observed_diff_in_vars = abs(var_students - var_seniors)
    
#     # Разница в бутстрэпированных дисперсиях
#     diff_in_vars = np.abs(np.array(bootstrapped_student_vars) - np.array(bootstrapped_senior_vars))
    
#     # Подсчет числа случаев, когда разница больше или равна наблюденной
#     count_exceeding_observed = sum(diff_in_vars >= observed_diff_in_vars)
    
#     # P-значение (доля случаев, превышающих наблюденный уровень)
#     p_value = count_exceeding_observed / num_bootstraps
    
#     # Критическое значение Хи-квадрата для уровня значимости альфа = 0.05
#     critical_value = chi2.ppf(q=0.95, df=total_dof)
    
#     return p_value, critical_value




import numpy as np
from scipy import stats

def bootstrap_expected_mean_and_variance_difference(group1, group2, num_iterations=10000):
    """Получает распределения различий средних значений и дисперсий методом бутстрапа."""
    n1, n2 = len(group1), len(group2)
    general = np.concatenate([group1, group2])
    
    mean_diffs = []
    var_diffs = []
    t_stats = []
    
    for _ in range(num_iterations):
        # Формируем новые выборки с повторениями
        sample1 = np.random.choice(general, size=n1, replace=True)
        sample2 = np.random.choice(general, size=n2, replace=True)
        
        # Вычисляем разницу средних
        mean_diff = np.mean(sample1) - np.mean(sample2)
        mean_diffs.append(mean_diff)
        
        # Вычисляем разницу дисперсий
        var_diff = np.var(sample1, ddof=1) - np.var(sample2, ddof=1)
        var_diffs.append(var_diff)
        
        # Рассчитываем объединённую стандартную ошибку
        s_pooled = np.sqrt(
            ((n1 - 1) * np.var(sample1, ddof=1) +
             (n2 - 1) * np.var(sample2, ddof=1)) /
            (n1 + n2 - 2)
        )
        
        # Т-статистика для проверки гипотезы равенства средних
        t_stat = mean_diff / (s_pooled * np.sqrt(1/n1 + 1/n2))
        t_stats.append(t_stat)
    
    # Тест нормальности распределений разницы средних и дисперсий по хи-квадрат
    _, p_mean = stats.shapiro(mean_diffs)
    total_dof = n1+n2-2
    chi_critical = chi2.ppf(q=0.95, df=total_dof)

    # Число интервалов (например, используем правило Стёрджеса)
    k = int(np.ceil(1 + np.log2(len(var_diffs))))
    # Интервализация данных
    hist, bins = np.histogram(var_diffs, bins=k, density=True)
    expected_freqs = len(var_diffs) * hist / sum(hist)
    observed_freqs, _ = np.histogram(var_diffs, bins=bins)

    # Тестирование с использованием chi-square test
    statistic, p_var = stats.chisquare(f_obs=observed_freqs, f_exp=expected_freqs)

    print("Статистика Хи-квадрат:", statistic)
    print("P-значение:", p_var)

    # _, p_var = stats.shapiro(var_diffs)
    
    return (
        np.array(mean_diffs),
        np.array(var_diffs),
        np.array(t_stats),
        p_mean,
        p_var
    )

def calculate_p_values(mean_diffs, var_diffs, group1, group2):
    """Вычисляет р-значения для среднего и дисперсии."""
    observed_mean_diff = np.mean(group1) - np.mean(group2)
    observed_var_diff = np.var(group1, ddof=1) - np.var(group2, ddof=1)
    
    # P-значение для различия средних
    p_value_mean = np.mean(np.abs(mean_diffs) >= abs(observed_mean_diff))
    
    # P-значение для различия дисперсий
    p_value_var = np.mean(np.abs(var_diffs) >= abs(observed_var_diff))
    
    return p_value_mean, p_value_var

def calculate_confidence_intervals(diffs, alpha=0.05):
    """Рассчитывает доверительные интервалы."""
    lower_bound = np.percentile(diffs, alpha*50)
    upper_bound = np.percentile(diffs, 100-alpha*50)
    return lower_bound, upper_bound

def cohen_d(group1, group2):
    """Рассчитывает эффект размером Cohen's D."""
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1, ddof=1)*len(group1)+np.var(group2, ddof=1)*len(group2))/(len(group1)+len(group2)-2))
    return diff / pooled_std

def full_bootstrap_analysis(df, col, type_col="type", senior_type="senior academics", num_it=10000):
    """Выполняет полное бутстрап-анализ для заданного столбца."""
    # Стандартизируем данные (необходимо заранее иметь функцию standardize_data())
    df = standardize_data(df, col)
    group1 = df[df[type_col] != senior_type][col + "_standardized"].values
    group2 = df[df[type_col] == senior_type][col + "_standardized"].values
    
    # Получаем распределение разниц методом бутстрапа
    mean_diffs, var_diffs, t_stats, p_mean, p_var = bootstrap_expected_mean_and_variance_difference(group1, group2, num_it)
    
    # Рассчитываем p-значения
    p_value_mean, p_value_var = calculate_p_values(mean_diffs, var_diffs, group1, group2)
    
    # Доверительный интервал для средней разницы
    ci_mean = calculate_confidence_intervals(mean_diffs)
    
    # Эффект размерности (Cohen's D)
    cohens_d = cohen_d(group1, group2)
    
    # Выводы и метрики
    print(f"Нормальность среднего: p={p_mean:.4f}")
    print(f"Нормальность дисперсии: p={p_var:.4f}")
    print(f"P-значение для разницы средних: {p_value_mean:.4f}")
    print(f"П-значение для разницы дисперсий: {p_value_var:.4f}")
    print(f"Доверительный интервал для средней разницы: ({ci_mean[0]:.4f}, {ci_mean[1]:.4f})")
    print(f"Эффект (Cohen's D): {cohens_d:.4f}")
    
    return mean_diffs, var_diffs, t_stats, p_value_mean, p_value_var, ci_mean, cohens_d
