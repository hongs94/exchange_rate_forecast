import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ..data_merge import create_merged_dataset

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def calculate_vif(df):
    """VIF 계산"""
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)


def main():
    df = create_merged_dataset()
    if df is None:
        print("데이터 로딩 실패")
        return

    df["kr_us_diff"] = df["kr_rate"] - df["us_rate"]
    df["dgs10_jpy10_diff"] = df["dgs10"] - df["jpy10"]
    df["dgs10_eur10_diff"] = df["dgs10"] - df["eur10"]

    targets = ["usd", "cny", "jpy", "eur"]
    feature_map = {
        "usd": ["dgs10", "vix", "dxy", "kr_us_diff", "kr_rate", "us_rate"],
        "cny": ["cny_fx_reserves", "cny_trade_bal", "wti", "vix"],
        "jpy": ["jpy10", "dgs10", "dgs10_jpy10_diff", "vix"],
        "eur": ["eur10", "dxy", "dgs10_eur10_diff", "vix"],
    }

    for target in targets:
        print(f"\n{'='*25}")
        print(f"✅ {target.upper()} 모델 Feature 분석 시작")
        print(f"{'='*25}")

        features = feature_map.get(target)
        if not features:
            print(f"{target.upper()}에 대한 Feature가 정의되지 않았습니다.")
            continue

        df_subset = df[[target] + features].dropna()

        corr_matrix = df_subset.corr(method="pearson")
        print(f"\n[{target.upper()} 모델] 피어슨 상관계수:")
        print(corr_matrix)

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
        )
        plt.title(f"{target.upper()} 모델 Feature 상관관계 히트맵", fontsize=16)
        plt.show()

        vif_df = calculate_vif(df_subset[features])
        print(f"\n[{target.upper()} 모델] 독립변수 VIF 점수:")
        print(vif_df)

        num_features = len(features)
        fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
        if num_features == 1: 
            axes = [axes]
            
        fig.suptitle(f"{target.upper()}와 독립 변수 간 산점도", fontsize=16)

        for j, feat in enumerate(features):
            sns.scatterplot(x=df_subset[feat], y=df_subset[target], alpha=0.6, ax=axes[j])
            sns.regplot(x=df_subset[feat], y=df_subset[target], scatter=False, color="red", ax=axes[j])
            axes[j].set_title(f"{target.upper()} vs {feat.upper()}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    main()