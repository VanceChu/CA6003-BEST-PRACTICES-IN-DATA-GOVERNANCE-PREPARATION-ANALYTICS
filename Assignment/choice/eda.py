import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "application_train.csv"
OUT_DIR = "eda_output"
ID_COLS = {"SK_ID_CURR"}
TARGET_COLS = {"TARGET"}
TOP_K_CATEGORIES = 10
TOP_K_NUMERIC_HIST = 12
HIGH_MISSING_PCT = 50.0
HIGH_CARDINALITY_CATEGORICAL = 50


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def iqr_outlier_stats(series):
    clean = series.dropna()
    if clean.empty:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return 0, q1, q3, iqr, np.nan, np.nan
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((clean < lower) | (clean > upper)).sum()
    return int(count), q1, q3, iqr, lower, upper


def format_value(value, max_len=80):
    if pd.isna(value):
        return "<NA>"
    text = str(value).replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def plot_hist_grid(df, columns, out_path, bins=50):
    if not columns:
        return
    cols = columns[:]
    n = len(cols)
    rows = int(np.ceil(n / 3))
    plt.figure(figsize=(15, rows * 3))
    for i, col in enumerate(cols, 1):
        plt.subplot(rows, 3, i)
        df[col].hist(bins=bins)
        plt.title(col)
        plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ensure_out_dir()

    df = pd.read_csv(DATA_PATH)
    n_rows = len(df)
    target_col = next(iter(TARGET_COLS)) if TARGET_COLS else None
    has_target = target_col in df.columns

    basic_lines = []
    basic_lines.append(f"shape: {df.shape}")
    basic_lines.append(f"rows: {len(df)}")
    basic_lines.append(f"columns: {len(df.columns)}")
    basic_lines.append(f"duplicates: {df.duplicated().sum()}")
    basic_lines.append(f"memory_mb: {df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    save_text(os.path.join(OUT_DIR, "basic_info.txt"), "\n".join(basic_lines) + "\n")

    dtype_counts = df.dtypes.value_counts().rename_axis("dtype").reset_index(name="count")
    dtype_counts.to_csv(os.path.join(OUT_DIR, "dtype_summary.csv"), index=False)

    missing_count = df.isna().sum()
    missing_rate = (missing_count / len(df) * 100).round(2)
    missing_df = (
        pd.DataFrame({"missing_count": missing_count, "missing_rate_pct": missing_rate})
        .sort_values("missing_count", ascending=False)
    )
    missing_df.to_csv(os.path.join(OUT_DIR, "missing_summary.csv"))

    quality_rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        non_null = int(series.notna().sum())
        missing = int(n_rows - non_null)
        missing_rate_pct = round(missing / n_rows * 100, 2) if n_rows else 0
        nunique = int(series.nunique(dropna=True))
        unique_rate_pct = round(nunique / n_rows * 100, 2) if n_rows else 0
        vc = series.value_counts(dropna=False)
        if len(vc) > 0:
            top_value = format_value(vc.index[0])
            top_count = int(vc.iloc[0])
        else:
            top_value = "<NA>"
            top_count = 0
        top_rate_pct = round(top_count / n_rows * 100, 2) if n_rows else 0
        is_constant = nunique == 1
        is_id_like = nunique == n_rows
        is_cat = pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)
        is_high_cardinality = is_cat and nunique > HIGH_CARDINALITY_CATEGORICAL
        if pd.api.types.is_numeric_dtype(series):
            inf_count = int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
        else:
            inf_count = 0
        quality_rows.append(
            {
                "column": col,
                "dtype": dtype,
                "non_null": non_null,
                "missing_count": missing,
                "missing_rate_pct": missing_rate_pct,
                "nunique": nunique,
                "unique_rate_pct": unique_rate_pct,
                "top_value": top_value,
                "top_count": top_count,
                "top_rate_pct": top_rate_pct,
                "inf_count": inf_count,
                "is_constant": is_constant,
                "is_id_like": is_id_like,
                "is_high_cardinality": is_high_cardinality,
            }
        )
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(os.path.join(OUT_DIR, "quality_report.csv"), index=False)

    top_missing = missing_df[missing_df["missing_count"] > 0].head(30)
    if not top_missing.empty:
        plt.figure(figsize=(10, 8))
        plt.barh(top_missing.index[::-1], top_missing["missing_rate_pct"][::-1])
        plt.xlabel("Missing Rate (%)")
        plt.title("Top 30 Missing Rate Columns")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "missing_top30.png"))
        plt.close()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ID_COLS and c not in TARGET_COLS]
    if num_cols:
        df[num_cols].describe().T.to_csv(os.path.join(OUT_DIR, "numeric_summary.csv"))

        numeric_rows = []
        for col in num_cols:
            series = df[col]
            non_null = int(series.notna().sum())
            missing_num = int(n_rows - non_null)
            if non_null > 0:
                quant = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                mean_val = series.mean()
                median_val = series.median()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
                skew_val = series.skew()
                kurt_val = series.kurt()
                zero_count = int((series == 0).sum())
                neg_count = int((series < 0).sum())
                zero_rate = round(zero_count / non_null * 100, 2)
                neg_rate = round(neg_count / non_null * 100, 2)
            else:
                quant = pd.Series(
                    [np.nan] * 7, index=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                )
                mean_val = np.nan
                median_val = np.nan
                std_val = np.nan
                min_val = np.nan
                max_val = np.nan
                skew_val = np.nan
                kurt_val = np.nan
                zero_count = 0
                neg_count = 0
                zero_rate = np.nan
                neg_rate = np.nan
            numeric_rows.append(
                {
                    "column": col,
                    "non_null": non_null,
                    "missing_count": missing_num,
                    "missing_rate_pct": round(missing_num / n_rows * 100, 2) if n_rows else 0,
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "skew": skew_val,
                    "kurtosis": kurt_val,
                    "zero_count": zero_count,
                    "zero_rate_pct": zero_rate,
                    "negative_count": neg_count,
                    "negative_rate_pct": neg_rate,
                    "p01": quant.loc[0.01],
                    "p05": quant.loc[0.05],
                    "p25": quant.loc[0.25],
                    "p50": quant.loc[0.5],
                    "p75": quant.loc[0.75],
                    "p95": quant.loc[0.95],
                    "p99": quant.loc[0.99],
                }
            )
        numeric_dist = pd.DataFrame(numeric_rows).set_index("column")
        numeric_dist.to_csv(os.path.join(OUT_DIR, "numeric_distribution.csv"))

        skew_top = (
            numeric_dist["skew"].abs().sort_values(ascending=False).head(TOP_K_NUMERIC_HIST).index.tolist()
        )
        plot_hist_grid(df, skew_top, os.path.join(OUT_DIR, "numeric_hist_top_skew.png"))

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        cat_summary = []
        cat_dist_rows = []
        for col in cat_cols:
            series = df[col]
            top = series.value_counts(dropna=False).head(1)
            if len(top) > 0:
                top_value = top.index[0]
                top_count = int(top.iloc[0])
            else:
                top_value = np.nan
                top_count = 0
            missing = int(series.isna().sum())
            cat_summary.append(
                {
                    "column": col,
                    "nunique": series.nunique(dropna=True),
                    "top": top_value,
                    "top_count": top_count,
                    "top_rate_pct": round(top_count / len(series) * 100, 2),
                    "missing_count": missing,
                    "missing_rate_pct": round(missing / len(series) * 100, 2),
                }
            )
            series_filled = series.astype("object").where(series.notna(), "__MISSING__")
            vc = series_filled.value_counts().head(TOP_K_CATEGORIES)
            for cat_value, count in vc.items():
                cat_dist_rows.append(
                    {
                        "column": col,
                        "category": format_value(cat_value),
                        "count": int(count),
                        "rate_pct": round(count / n_rows * 100, 2),
                    }
                )
            other_count = n_rows - int(vc.sum())
            if other_count > 0:
                cat_dist_rows.append(
                    {
                        "column": col,
                        "category": "__OTHER__",
                        "count": int(other_count),
                        "rate_pct": round(other_count / n_rows * 100, 2),
                    }
                )
        pd.DataFrame(cat_summary).to_csv(os.path.join(OUT_DIR, "categorical_summary.csv"), index=False)
        pd.DataFrame(cat_dist_rows).to_csv(
            os.path.join(OUT_DIR, "categorical_distribution_top10.csv"), index=False
        )

    outlier_rows = []
    for col in num_cols:
        count, q1, q3, iqr, lower, upper = iqr_outlier_stats(df[col])
        non_null = df[col].notna().sum()
        outlier_rate = (count / non_null * 100) if non_null else 0
        outlier_rows.append(
            {
                "column": col,
                "non_null": int(non_null),
                "outlier_count": int(count),
                "outlier_rate_pct": round(outlier_rate, 2),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
            }
        )
    outlier_df = pd.DataFrame(outlier_rows).sort_values("outlier_rate_pct", ascending=False)
    outlier_df.to_csv(os.path.join(OUT_DIR, "outlier_summary.csv"), index=False)

    top_outliers = outlier_df[outlier_df["outlier_count"] > 0].head(10)
    if not top_outliers.empty:
        cols = top_outliers["column"].tolist()
        plt.figure(figsize=(12, 6))
        df[cols].boxplot(rot=45)
        plt.title("Top 10 Outlier Columns (IQR Method)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "outlier_top10_boxplot.png"))
        plt.close()

    if has_target:
        target_series = df[target_col]
        target_counts = target_series.value_counts(dropna=False).sort_index()
        target_dist = pd.DataFrame(
            {
                "target_value": target_counts.index,
                "count": target_counts.values,
                "rate_pct": (target_counts.values / n_rows * 100).round(2),
            }
        )
        target_dist.to_csv(os.path.join(OUT_DIR, "target_distribution.csv"), index=False)

        plt.figure(figsize=(6, 4))
        plt.bar(target_dist["target_value"].astype(str), target_dist["count"])
        plt.title("Target Distribution")
        plt.xlabel("Target")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "target_distribution.png"))
        plt.close()

        if num_cols:
            target_corr_rows = []
            target_values = sorted(target_series.dropna().unique().tolist())
            binary_target = len(target_values) == 2
            t0, t1 = (target_values + [None, None])[:2]
            diff_rows = []
            for col in num_cols:
                series = df[col]
                pair_mask = series.notna() & target_series.notna()
                pair_count = int(pair_mask.sum())
                if pair_count > 0:
                    corr_val = series[pair_mask].corr(target_series[pair_mask])
                else:
                    corr_val = np.nan
                target_corr_rows.append(
                    {
                        "column": col,
                        "corr_pearson": corr_val,
                        "abs_corr": abs(corr_val) if pd.notna(corr_val) else np.nan,
                        "pair_count": pair_count,
                        "missing_rate_pct": missing_df.loc[col, "missing_rate_pct"],
                    }
                )
                if binary_target:
                    s0 = series[target_series == t0]
                    s1 = series[target_series == t1]
                    diff_rows.append(
                        {
                            "column": col,
                            "mean_target_0": s0.mean(),
                            "mean_target_1": s1.mean(),
                            "mean_diff_1_minus_0": s1.mean() - s0.mean(),
                            "median_target_0": s0.median(),
                            "median_target_1": s1.median(),
                            "median_diff_1_minus_0": s1.median() - s0.median(),
                        }
                    )
            target_corr = pd.DataFrame(target_corr_rows).sort_values("abs_corr", ascending=False)
            target_corr.to_csv(os.path.join(OUT_DIR, "target_numeric_correlation.csv"), index=False)
            if binary_target:
                pd.DataFrame(diff_rows).set_index("column").to_csv(
                    os.path.join(OUT_DIR, "target_numeric_diff.csv")
                )

        if cat_cols:
            cat_target_detail = []
            cat_target_summary = []
            target_non_null = target_series.notna()
            for col in cat_cols:
                series = df[col].astype("object").where(df[col].notna(), "__MISSING__")
                tmp = pd.DataFrame({"category": series, "target": target_series})
                tmp = tmp[target_non_null]
                if tmp.empty:
                    continue
                counts = tmp["category"].value_counts()
                rates = tmp.groupby("category")["target"].mean()
                top = counts.head(TOP_K_CATEGORIES)
                for cat_value, count in top.items():
                    rate = rates.get(cat_value, np.nan)
                    cat_target_detail.append(
                        {
                            "column": col,
                            "category": format_value(cat_value),
                            "count": int(count),
                            "rate_pct": round(count / len(tmp) * 100, 2),
                            "target_rate_pct": round(rate * 100, 2) if pd.notna(rate) else np.nan,
                        }
                    )
                other_count = int(len(tmp) - top.sum())
                if other_count > 0:
                    other_cats = counts.index[TOP_K_CATEGORIES:]
                    other_rate = tmp[tmp["category"].isin(other_cats)]["target"].mean()
                    cat_target_detail.append(
                        {
                            "column": col,
                            "category": "__OTHER__",
                            "count": other_count,
                            "rate_pct": round(other_count / len(tmp) * 100, 2),
                            "target_rate_pct": round(other_rate * 100, 2) if pd.notna(other_rate) else np.nan,
                        }
                    )
                cat_target_summary.append(
                    {
                        "column": col,
                        "n_categories": int(rates.shape[0]),
                        "missing_rate_pct": missing_df.loc[col, "missing_rate_pct"],
                        "target_rate_min": rates.min(),
                        "target_rate_max": rates.max(),
                        "target_rate_std": rates.std(),
                        "target_rate_gap": rates.max() - rates.min(),
                        "top_category": format_value(counts.index[0]),
                        "top_category_rate_pct": round(counts.iloc[0] / len(tmp) * 100, 2),
                    }
                )
            pd.DataFrame(cat_target_detail).to_csv(
                os.path.join(OUT_DIR, "target_rate_by_category_top10.csv"), index=False
            )
            pd.DataFrame(cat_target_summary).to_csv(
                os.path.join(OUT_DIR, "categorical_target_summary.csv"), index=False
            )

    high_missing_cols = missing_df[missing_df["missing_rate_pct"] >= HIGH_MISSING_PCT].index.tolist()
    constant_cols = quality_df[quality_df["is_constant"]]["column"].tolist()
    id_like_cols = quality_df[quality_df["is_id_like"]]["column"].tolist()
    high_card_cat = quality_df[quality_df["is_high_cardinality"]]["column"].tolist()
    inf_cols = quality_df[quality_df["inf_count"] > 0]["column"].tolist()

    save_text(os.path.join(OUT_DIR, "columns_missing_over_50_pct.txt"), "\n".join(high_missing_cols))
    save_text(os.path.join(OUT_DIR, "constant_columns.txt"), "\n".join(constant_cols))
    save_text(os.path.join(OUT_DIR, "id_like_columns.txt"), "\n".join(id_like_cols))
    save_text(os.path.join(OUT_DIR, "high_cardinality_categorical.txt"), "\n".join(high_card_cat))
    save_text(os.path.join(OUT_DIR, "infinite_columns.txt"), "\n".join(inf_cols))

    summary_text = textwrap.dedent(
        f"""
        Dataset: {DATA_PATH}
        Shape: {df.shape}
        Duplicate rows: {df.duplicated().sum()}
        Columns with missing values: {(missing_df['missing_count'] > 0).sum()}
        Columns with >= {HIGH_MISSING_PCT:.0f}% missing: {len(high_missing_cols)}
        Constant columns: {len(constant_cols)}
        ID-like columns: {len(id_like_cols)}
        High-cardinality categorical columns: {len(high_card_cat)}
        Columns with infinite values: {len(inf_cols)}
        Numeric columns (excluded ID/TARGET): {len(num_cols)}
        Categorical columns: {len(cat_cols)}
        Target present: {has_target}
        """
    ).strip()
    save_text(os.path.join(OUT_DIR, "eda_summary.txt"), summary_text + "\n")


if __name__ == "__main__":
    main()
