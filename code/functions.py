import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

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
    ## change type of all columns to numeric
    for col in df.columns[1:]:
        df[col] = df[col].astype("float32")

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
        half_min = np.min([val for val in df.filter(regex=r"sum").values.flatten() if val != 0]) / 2

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

def plot_volcano(df, compounds_col, fcr_col, fc_col, significance=0.05, fc_level=1, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
    ax.scatter(x=df[fc_col], y=df[fcr_col].apply(lambda x:-np.log10(x)), s=10, label="Not significant", color="grey")

    # highlight down- or up- regulated metabolites
    down = df[(df[fc_col] <= fc_level * -1) & (df[fcr_col] <= significance)]
    up = df[(df[fc_col] >= fc_level) & (df[fcr_col] <= significance)]
    ax.scatter(x=down[fc_col], y=down[fcr_col].apply(lambda x:-np.log10(x)), s=10, label="Down-regulated", color="blue")
    ax.scatter(x=up[fc_col], y=up[fcr_col].apply(lambda x:-np.log10(x)), s=10, label="Up-regulated", color="red")


    # add texts
    texts=[]
    for _,r in up.iterrows():
        texts.append(ax.text(x=r[fc_col],y=-np.log10(r[fcr_col]),s=r[compounds_col], fontsize=8))

    for _,r in down.iterrows():
        texts.append(ax.text(x=r[fc_col],y=-np.log10(r[fcr_col]),s=r[compounds_col], fontsize=8))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='black', lw=0.5),
                expand_points=(1.2,1.2),
                expand_text=(1.2,1.2),
                force_points=0.5,
                force_text=0.5,
                lim=100)

    ax.set_title(f"{title}")
    ax.set_xlabel("logFC")
    ax.set_ylabel("-logFDR")
    ax.axvline(fc_level*-1, color="grey", linestyle="--")
    ax.axvline(fc_level, color="grey", linestyle="--")
    ax.axhline(-np.log10(significance), color="grey", linestyle="--")
    ax.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc="upper center")

    return fig, ax

def format_compound_names(name):
    name = str(name)
    name = name.replace("(", "_").replace(")", "").replace("/", "_")
    return name

def format_dataframe(data_name, save=False):
    df_org = pd.read_excel(f"../data/unformated/{data_name}.xlsx", sheet_name="formatted")
    rows_unl = [i for i in range(len(df_org)) if i % 2 == 0]
    rows_l = [i for i in range(len(df_org)) if i % 2 != 0]
    df = df_org.drop(rows_l, axis=0)
    df_l = df_org.drop(rows_unl, axis=0)

    # add labeled values as new columns
    cols = df_org.columns[1:]
    for col in cols:
        df[f"{col}_l"] = df_l[col].values

    # format compound name
    df["compound"] = df["compound"].apply(format_compound_names)

    # save
    if save == True:
        df.to_csv(f"../data/formated/{data_name}.csv", index=False)

    return df

def ttest_for_df(df, cols1, cols2, label):
    vals1 = df[cols1]
    vals2 = df[cols2]

    t_stat, p_val = ttest_ind(vals1, vals2, axis=1, equal_var=False)

    vars1 = vals1.var(axis=1).values
    vars2 = vals2.var(axis=1).values

    # invalid tests
    invalid = (
        (vars1 == 0) |
        (vars2 == 0) |
        np.isinf(t_stat) |
        np.isnan(t_stat)
    )

    # set invalid tests to NaN
    t_stat[invalid] = np.nan
    p_val[invalid] = np.nan

    # BH correction ONLY on valid p-values
    p_val_adj = np.full_like(p_val, np.nan)
    valid_mask = ~np.isnan(p_val)

    if valid_mask.sum() > 0:
        p_val_adj[valid_mask] = multipletests(
            p_val[valid_mask],
            method="fdr_bh"
        )[1]

    df[f"t_stat_{label}"] = t_stat
    df[f"p_val_{label}"] = p_val
    df[f"p_val_adj_{label}"] = p_val_adj

    return df


def batch_correction(df):
    df_t = df.filter(regex="sum").T
    df_t.columns = df["compound"].values

    samples = df.filter(regex="sum").columns
    results = []

    for metabolite in df_t.columns:
        
        df_model = pd.DataFrame({
            "sample": samples,
            "y": df_t[metabolite].values
        })
        
        df_model["batch"] = df_model["sample"].str.extract(r"(exp\d+)")
        df_model["condition"] = df_model["sample"].str.extract(r"_(\d+deg)")
        
        model = smf.ols("y ~ C(condition) + C(batch)", data=df_model).fit()
        
        # Extract condition effect
        condition_param = [p for p in model.params.index if "C(condition)" in p]
        
        if len(condition_param) > 0:
            coef = model.params[condition_param[0]]
            pval = model.pvalues[condition_param[0]]
        else:
            coef = np.nan
            pval = np.nan
        
        # Extract batch effect
        batch_param = [p for p in model.params.index if "C(batch)" in p]
        
        if len(batch_param) > 0:
            batch_pval = model.pvalues[batch_param[0]]
        else:
            batch_pval = np.nan
        
        results.append({
            "compound": metabolite,
            "EffectSize_condition": coef,
            "p_val_condition": pval,
            "p_val_batch": batch_pval,
            "R2": model.rsquared
        })

    results_df = pd.DataFrame(results)

    # FDR-correction
    results_df["p_val_adj_condition"] = multipletests(
        results_df["p_val_condition"],
        method="fdr_bh"
    )[1]

    return results_df