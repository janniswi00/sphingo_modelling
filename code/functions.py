import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(rows, cols, main_title, df, df_cols, bins):
    fig, axs = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axs[row, col].hist(df[df_cols[row][col]].values, bins=bins)
            axs[row, col].set_title(df_cols[row][col])

    plt.suptitle(main_title)
    plt.tight_layout()

    return fig, axs

def preprocessing(df:pd.DataFrame, add_unlabeled_labeled:bool=True, replace_zero:str="half_min_glob", relative_values:bool=True, log_transformation:bool=True) -> pd.DataFrame:
    """Application of different preprocessing steps to metabolomics pd.DataFrame

    Args:
        df (pd.DataFrame): pd.DataFrame that should be preprocessed
        add_unlabeled_labeled (bool, optional): Sum of unlabeled and labeled values. Defaults to True.
        replace_zero (str, optional): Replace zero values. Defaults to "half_min_glob".
        relative_values (bool, optional): Relative values (value / sum(values per experiment and per condition)). Defaults to True.
        log_transformation (bool, optional): log2-transformation. Defaults to True.

    Returns:
        pd.DataFrame: preprocessed DataFrame
    """

    ## add unlabeled and labeled values
    if add_unlabeled_labeled == True:
        for col in [c for c in df.columns[1:] if "_l" not in c]:
            col_l = col + "_l"
            df[f"{"sum_" + col}"] = df[[col, col_l]].sum(axis=1)

    ## drop not detected compounds
    rows_to_drop = []

    for i, r in df.filter(regex="sum").iterrows():
        if np.all(r.values == 0):
            rows_to_drop.append(i)

    df = df.drop(index=rows_to_drop, axis=1)

    ## replace zero-values
    if replace_zero == "half_min_glob": # half minimum of all values
        half_min = np.min([val for val in df.filter(regex=r"^(?!.*compound).*$").values.flatten() if val != 0]) / 2

        for col in df.columns:
            if col == "compound":
                continue
            
            if "sum" in col:
                df[col] = df[col].replace(0, half_min)

    if replace_zero == "half_min_loc": # half minimum value of both experiments and conditions per compound
        sum_cols = df.columns.str.contains("sum")

        for i, row in df.filter(regex="sum").iterrows():
            half_min = np.min([val for val in row.values if val != 0]) / 2
            df.loc[i, sum_cols] = df.loc[i, sum_cols].replace(0, half_min)

    ## relative values
    if relative_values == True:
        cols = [c for c in df.columns[1:] if "sum" in c]

        for exp in ["exp2", "exp4"]:
            for deg in ["22deg", "37deg"]:
                df_exp = df.filter(regex=rf"sum_{exp}_{deg}")
                summe = np.sum(df_exp.values.flatten())

                for col in df_exp.columns:
                    df[col] = [val / summe for val in df[col].values]

    ## log-transformation
    if log_transformation == True:
        rel_cols = [c for c in df.columns if "sum" in c]

        for col in rel_cols:
            df[col] = np.log(df[col].values)

    return df