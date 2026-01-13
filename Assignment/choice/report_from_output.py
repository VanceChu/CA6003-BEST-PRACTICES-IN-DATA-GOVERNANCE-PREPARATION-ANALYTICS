import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EDA_DIR = "eda_output"
REPORT_DIR = "report_output"

TOP_K_NUMERIC = 10
TOP_K_CAT_GAP = 10
TOP_K_CAT_PLOT_FEATURES = 5
TOP_K_CATEGORY_PER_FEATURE = 8

HIGH_MISSING_PCT = 50.0
MEDIUM_MISSING_PCT = 20.0
NEAR_CONSTANT_TOP_RATE = 99.5
ZERO_RATE_THRESHOLD = 99.5
SKEW_THRESHOLD = 10.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_list(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_basic_info(path):
    info = {}
    if not os.path.exists(path):
        return info
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                info[key.strip()] = value.strip()
    return info


def slugify(text):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return slug or "feature"


def plot_top_numeric_correlations(corr_df, out_path, top_k):
    if corr_df.empty:
        return
    corr_top = corr_df.dropna(subset=["corr_pearson"]).head(top_k).copy()
    if corr_top.empty:
        return
    corr_top = corr_top.sort_values("corr_pearson")
    colors = ["#d95f02" if v < 0 else "#1b9e77" for v in corr_top["corr_pearson"]]
    plt.figure(figsize=(10, 6))
    plt.barh(corr_top["column"], corr_top["corr_pearson"], color=colors)
    plt.axvline(0, color="#444444", linewidth=1)
    plt.title("Top Numeric Correlations with Target")
    plt.xlabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_categorical_gap_ranges(cat_summary, out_path, top_k):
    if cat_summary.empty:
        return
    gap_top = cat_summary.sort_values("target_rate_gap", ascending=False).head(top_k)
    if gap_top.empty:
        return
    gap_top = gap_top.sort_values("target_rate_gap")
    y = np.arange(len(gap_top))
    plt.figure(figsize=(10, 6))
    for idx, row in enumerate(gap_top.itertuples(index=False)):
        plt.plot([row.target_rate_min, row.target_rate_max], [idx, idx], color="#1f77b4")
        plt.scatter([row.target_rate_min, row.target_rate_max], [idx, idx], color="#1f77b4")
    plt.yticks(y, gap_top["column"])
    plt.xlabel("Target Rate")
    plt.title("Top Categorical Features by Target Rate Gap (Min–Max)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_category_target_rates(cat_rates, feature, out_path, top_k):
    sub = cat_rates[cat_rates["column"] == feature].copy()
    if sub.empty:
        return
    sub = sub.sort_values("target_rate_pct", ascending=False).head(top_k)
    if sub.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.barh(sub["category"][::-1], sub["target_rate_pct"][::-1], color="#4c78a8")
    plt.xlabel("Target Rate (%)")
    plt.title(f"Target Rate by Category: {feature}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ensure_dir(REPORT_DIR)

    required = [
        "basic_info.txt",
        "missing_summary.csv",
        "quality_report.csv",
        "numeric_distribution.csv",
        "target_distribution.csv",
        "target_numeric_correlation.csv",
        "categorical_target_summary.csv",
        "target_rate_by_category_top10.csv",
    ]
    missing_files = [f for f in required if not os.path.exists(os.path.join(EDA_DIR, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing required files in {EDA_DIR}: {missing_files}")

    basic = parse_basic_info(os.path.join(EDA_DIR, "basic_info.txt"))
    missing = pd.read_csv(os.path.join(EDA_DIR, "missing_summary.csv"), index_col=0)
    quality = pd.read_csv(os.path.join(EDA_DIR, "quality_report.csv"))
    numeric_dist = pd.read_csv(os.path.join(EDA_DIR, "numeric_distribution.csv"), index_col=0)
    target_dist = pd.read_csv(os.path.join(EDA_DIR, "target_distribution.csv"))
    corr = pd.read_csv(os.path.join(EDA_DIR, "target_numeric_correlation.csv"))
    cat_summary = pd.read_csv(os.path.join(EDA_DIR, "categorical_target_summary.csv"))
    cat_rates = pd.read_csv(os.path.join(EDA_DIR, "target_rate_by_category_top10.csv"))

    missing_cols = missing[missing["missing_count"] > 0]
    high_missing = missing[missing["missing_rate_pct"] >= HIGH_MISSING_PCT].index.tolist()
    mid_missing = missing[
        (missing["missing_rate_pct"] >= MEDIUM_MISSING_PCT)
        & (missing["missing_rate_pct"] < HIGH_MISSING_PCT)
    ].index.tolist()

    id_like = quality[quality["is_id_like"]]["column"].tolist()
    high_card = quality[quality["is_high_cardinality"]]["column"].tolist()
    near_constant = quality[quality["top_rate_pct"] >= NEAR_CONSTANT_TOP_RATE]["column"].tolist()

    zero_inflated = (
        numeric_dist[numeric_dist["zero_rate_pct"] >= ZERO_RATE_THRESHOLD].index.tolist()
        if "zero_rate_pct" in numeric_dist.columns
        else []
    )
    skewed = (
        numeric_dist[numeric_dist["skew"].abs() >= SKEW_THRESHOLD].index.tolist()
        if "skew" in numeric_dist.columns
        else []
    )

    corr_top = corr.dropna(subset=["corr_pearson"]).head(TOP_K_NUMERIC)
    cat_gap_top = cat_summary.sort_values("target_rate_gap", ascending=False).head(TOP_K_CAT_GAP)

    plot_top_numeric_correlations(
        corr_top, os.path.join(REPORT_DIR, "top_numeric_correlations.png"), TOP_K_NUMERIC
    )
    plot_categorical_gap_ranges(
        cat_summary, os.path.join(REPORT_DIR, "top_categorical_gap.png"), TOP_K_CAT_GAP
    )

    for feature in cat_gap_top["column"].head(TOP_K_CAT_PLOT_FEATURES):
        out_name = f"category_target_rates_{slugify(feature)}.png"
        plot_category_target_rates(
            cat_rates, feature, os.path.join(REPORT_DIR, out_name), TOP_K_CATEGORY_PER_FEATURE
        )

    rows = basic.get("rows", "NA")
    cols = basic.get("columns", "NA")
    duplicates = basic.get("duplicates", "NA")
    memory_mb = basic.get("memory_mb", "NA")

    target_text = ""
    if not target_dist.empty:
        target_dist = target_dist.sort_values("target_value")
        target_lines = []
        for row in target_dist.itertuples(index=False):
            target_lines.append(f"TARGET={row.target_value}: {row.count} ({row.rate_pct:.2f}%)")
        target_text = "；".join(target_lines)

    corr_list = []
    for row in corr_top.itertuples(index=False):
        corr_list.append(f"{row.column}({row.corr_pearson:.3f})")
    corr_text = "、".join(corr_list) if corr_list else "无"

    cat_gap_list = []
    for row in cat_gap_top.itertuples(index=False):
        cat_gap_list.append(f"{row.column}(gap={row.target_rate_gap:.3f})")
    cat_gap_text = "、".join(cat_gap_list) if cat_gap_list else "无"

    narrative = textwrap.dedent(
        f"""
        数据集质量报告（基于已有 EDA 输出，未进行缺失/异常/清洗处理）

        一、数据概览
        数据规模为 {rows} 行 × {cols} 列，重复行 {duplicates}，内存占用约 {memory_mb} MB。
        目标变量分布为：{target_text}。该分布体现出明显类别不平衡。

        二、缺失与结构性质量问题
        含缺失值的字段共 {len(missing_cols)} 个，其中缺失率≥{HIGH_MISSING_PCT:.0f}% 的字段有 {len(high_missing)} 个。
        高缺失字段主要集中在房屋/居住类相关变量（例如 COMMONAREA_*、LIVINGAPARTMENTS_*、NONLIVINGAPARTMENTS_*）。
        ID-like 字段：{", ".join(id_like) if id_like else "无"}；高基数类别字段：{", ".join(high_card) if high_card else "无"}。

        三、分布特征与数据形态
        数值变量中存在明显的偏态与长尾，|skew|≥{SKEW_THRESHOLD:.0f} 的字段包括：{", ".join(skewed[:10]) if skewed else "无"}。
        近零方差/高集中度字段（top 占比≥{NEAR_CONSTANT_TOP_RATE:.1f}%）包括：{", ".join(near_constant[:10]) if near_constant else "无"}。
        高零值占比（零值≥{ZERO_RATE_THRESHOLD:.1f}%）字段包括：{", ".join(zero_inflated[:10]) if zero_inflated else "无"}。

        四、目标关联特征概览
        与目标变量相关性（绝对值）较高的数值特征为：{corr_text}。
        目标率差异较大的类别特征为：{cat_gap_text}。

        五、说明
        本报告仅基于已生成的统计与可视化结果进行总结，未做任何缺失值填补、异常值处理或字段清洗。
        """
    ).strip()
    save_text(os.path.join(REPORT_DIR, "report_narrative.txt"), narrative + "\n")

    priority_lines = []
    priority_lines.append("P0_HIGH")
    priority_lines.append(
        f"missing>= {HIGH_MISSING_PCT:.0f}%: count={len(high_missing)}; columns="
        + (", ".join(high_missing[:20]) if high_missing else "none")
    )
    priority_lines.append(
        f"id_like: count={len(id_like)}; columns=" + (", ".join(id_like) if id_like else "none")
    )
    if target_text:
        priority_lines.append(f"target_imbalance: {target_text}")

    priority_lines.append("P1_MEDIUM")
    priority_lines.append(
        "high_cardinality_categorical: count="
        + str(len(high_card))
        + "; columns="
        + (", ".join(high_card[:20]) if high_card else "none")
    )
    priority_lines.append(
        f"near_constant(top_rate>={NEAR_CONSTANT_TOP_RATE:.1f}%): count={len(near_constant)}; "
        + ("columns=" + ", ".join(near_constant[:20]) if near_constant else "columns=none")
    )
    priority_lines.append(
        f"zero_inflated(zero_rate>={ZERO_RATE_THRESHOLD:.1f}%): count={len(zero_inflated)}; "
        + ("columns=" + ", ".join(zero_inflated[:20]) if zero_inflated else "columns=none")
    )

    priority_lines.append("P2_LOW")
    priority_lines.append(
        f"missing {MEDIUM_MISSING_PCT:.0f}%-{HIGH_MISSING_PCT:.0f}%: count={len(mid_missing)}; "
        + ("columns=" + ", ".join(mid_missing[:20]) if mid_missing else "columns=none")
    )
    priority_lines.append(
        f"high_skew(|skew|>={SKEW_THRESHOLD:.0f}): count={len(skewed)}; "
        + ("columns=" + ", ".join(skewed[:20]) if skewed else "columns=none")
    )

    save_text(os.path.join(REPORT_DIR, "quality_issue_priority.txt"), "\n".join(priority_lines) + "\n")


if __name__ == "__main__":
    main()
