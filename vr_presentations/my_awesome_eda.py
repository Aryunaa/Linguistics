from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # для треугольной матрицы в хитмапе, чтобы красиво


def change_types(df):
    '''
    changes types
    '''
    for i in df.columns:
        a = df[[i]].nunique(dropna=True)
        if a[0] < 10:
            df[i] = df[i].astype("category")
    tostr_cols = list(df.select_dtypes(include=["object"]).columns)
    df[tostr_cols] = df[tostr_cols].astype("string")
    return df


def run_viz(df, num_cols, cat_cols):
    '''
    makes vizualization

    '''

    # barplot
    na_table = pd.DataFrame(df.isna().sum())[df.isna().sum() != 0]
    sns.set_theme(style="whitegrid")
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    sns.barplot(x=na_table.iloc[:, 0], y=na_table.index, data=na_table,
                label="Total", color="b")
    plt.title("Number of missing values per column", size=10, weight="bold")
    plt.show()

    # heatmap
    sns.set_theme(style="white")
    heatmap_cols = []
    heatmap_cols.extend(num_cols)
    heatmap_cols.extend(cat_cols)
    df[cat_cols] = df[cat_cols].apply(lambda x: pd.factorize(x)[0])
    corr = df[heatmap_cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.9, center=0, vmin=-0.9,
                square=True, linewidths=.25, cbar_kws={"shrink": .25})
    plt.title("Heatmap for numerical and categorical columns", size=10, weight="bold")
    plt.show()

    # numericals barplots with histo

    print(num_cols)
    print('asbsdf', df.shape, df.columns)
    for i in num_cols:
        try:

            print(i)
            sns.set(style="whitegrid")
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

            # assigning a graph to each ax
            sns.boxplot(pd.to_numeric(df[i].dropna()), orient="h", ax=ax_box)
            try:
                print(pd.to_numeric(df[i].dropna()).values.dtype)
                sns.histplot(data=pd.to_numeric(df[i].dropna()).values, ax=ax_hist)
                ax_hist.set(xlabel=str(i), ylabel='Counts')
            except NotImplementedError:
                print('error')
            plt.show()


        except IndexError:

            print('error, I am sorry')

    # plt.show()


def run_eda(df):
    """
    makes eda of the dataframe
    """
    print("Hello! EDA of your dataframe is running now!")
    print("number of rows = " + str(df.shape[0]))
    print("number of cols = " + str(df.shape[1]))

    df = change_types(df)
    str_cols = list(df.select_dtypes(include=["string"]).columns)
    num_cols = list(df.select_dtypes(include=["float64", "int64"]).columns)
    cat_cols = list(df.select_dtypes(include=["category"]).columns)
    print('Datatypes: \n')
    print(df.dtypes)

    # work with category data
    print("_________________________")
    print("\nWorking with category data:")
    for i in cat_cols:
        freqs = pd.DataFrame(df[i].value_counts())
        freqs['frequencies'] = df[i].value_counts() / sum(df[i].value_counts())
        display(freqs)

    # work with numerical
    print("_________________________")
    print("\nWorking with numerical data:")
    num_desc = df[num_cols].describe().loc[['min', 'max', 'std', '25%', '50%', '75%']]
    display(num_desc)

    q1 = df[num_cols].quantile(0.25)
    q3 = df[num_cols].quantile(0.75)
    iqr = q3 - q1

    print("Outliers beyond the +-1.5 iqr rule:")
    outliers = pd.DataFrame(((df[num_cols] < (q1 - 1.5 * iqr)) | (df[num_cols] > (q3 + 1.5 * iqr))).sum(),
                            columns=['Outliers num'])
    display(outliers)

    print("_________________________\n")
    print("Working with missing values:")
    print("Number of missing values = " + str(df.isna().sum().sum()))
    print("Number of rows with missing values = " + str(df[df.isna().any(axis=1)].shape[0]))
    print("Columns that contain missing values: ")
    nacols = dict(df.isna().sum() > 0)
    na_columns = [i for i in nacols if nacols[i] is True]
    print(*na_columns)

    print("_________________________")
    print("\nNumber of duplicated rows = " + str(df.duplicated(keep=False).sum()))

    run_viz(df, num_cols, cat_cols)
