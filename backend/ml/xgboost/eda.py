import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ml.data_merge import create_merged_dataset
from .constant import CURRENCIES

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def calculate_vif(df):
    """VIF 계산"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data


def main():
    df = create_merged_dataset()
    if df is None:
        print("데이터 로딩 실패")
        return

    df["kr_us_diff"] = df["kr_rate"] - df["us_rate"]

    features = ["wti", "dxy", "vix", "kr_rate", "us_rate", "kr_us_diff", "dgs10"]

    df_subset = df[CURRENCIES + features].dropna()

    corr_matrix = df_subset.corr(method="pearson")
    print("피어슨 상관계수:")
    print(corr_matrix)

    plt.figure(figsize=(15, 12))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 9},
    )
    plt.title("전체 변수 상관관계 히트맵 (상단 삼각형)", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    vif_df = calculate_vif(df_subset[features])
    print("\n독립변수 VIF 점수:")
    print(vif_df)

    important_features = ["dxy", "vix", "dgs10", "wti", "kr_us_diff"]

    for target in CURRENCIES:
        fig, axes = plt.subplots(1, len(important_features), figsize=(18, 4))
        fig.suptitle(f"{target.upper()}와 주요 독립 변수 간 산점도", fontsize=16)

        for j, feat in enumerate(important_features):
            sns.scatterplot(
                x=df_subset[feat], y=df_subset[target], alpha=0.6, ax=axes[j]
            )
            sns.regplot(
                x=df_subset[feat],
                y=df_subset[target],
                scatter=False,
                color="red",
                ax=axes[j],
            )
            axes[j].set_title(f"{target.upper()} vs {feat.upper()}")
            axes[j].set_xlabel(feat)
            axes[j].set_ylabel(target)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    df_sync = df[["usd", "dxy"]].dropna()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color_usd = "tab:blue"
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("USD 환율", color=color_usd)
    ax1.plot(df_sync.index, df_sync["usd"], color=color_usd, label="USD 환율")
    ax1.tick_params(axis="y", labelcolor=color_usd)

    ax2 = ax1.twinx()
    color_dxy = "tab:red"
    ax2.set_ylabel("DXY 지수", color=color_dxy)
    ax2.plot(df_sync.index, df_sync["dxy"], color=color_dxy, label="DXY 지수")
    ax2.tick_params(axis="y", labelcolor=color_dxy)

    plt.title("USD 환율과 DXY 지수 동조화 시각화", fontsize=16)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
