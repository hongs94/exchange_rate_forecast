import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ml.data_merge import create_merged_dataset

# 한글 폰트 설정 (Windows 설정)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False 

# VIF 계산
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i) for i in range(df.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

# 데이터 로드
def main():
    df = create_merged_dataset()
    if df is None:
        print("데이터 로딩 실패")
        return

    # 파생변수 추가
    df["kr_us_diff"] = df["kr_rate"] - df["us_rate"]
    df["us_jp_diff"] = df["dgs10"] - df["jpy10"]
    df["us_eu_diff"] = df["dgs10"] -df["eur10"]
    
    # 통화별 분석 변수 정의
    analysis_targets = {
        "usd": ["dgs10", "vix", "dxy", "kr_us_diff", "kr_rate", "us_rate"],
        "cny": ["cny_fx_reserves", "cny_trade_bal", "wti", "vix"],
        "jpy": ["jpy10", "dgs10", "us_jp_diff", "vix"],
        "eur": ["eur10", "dxy", "us_eu_diff", "vix"],
    }
    
    for target, features in analysis_targets.items():
        available_features = [f for f in features if f in df.columns]
        subset_cols = [target] + available_features
        df_subset = df[subset_cols].dropna()
    
        if df_subset.empty:
            print(f"{target.upper()} 데이터 없음 (선택한 변수: {features})")
            continue
        
        # 기초 통계, 이상치/결측치 파악
        print(f"\n {target.upper()}/KRW 데이터 기초 통계")
        print("\n📌 기초 통계량:")
        print(df_subset.describe().T)
        
        print("\n📌 결측치 개수:")
        print(df[subset_cols].isna().sum())
        
        print("\n📌 이상치 (IQR 기준):")
        Q1 = df_subset.quantile(0.25)
        Q3 = df_subset.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df_subset < (Q1 - 1.5 * IQR)) | (df_subset > (Q3 + 1.5 * IQR))).sum()
        print(outliers)
        
        # 피어슨 상관계수 계산
        print(f"\n📊 {target.upper()}/KRW 상관계수 분석 (변수: {available_features})")
        corr_matrix = df_subset.corr(method="pearson")
        print(corr_matrix[target].sort_values(ascending=False))
        
        # 히트맵
        plt.figure(figsize=(15,12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # 색상은 mask 적용
        sns.heatmap(
            corr_matrix, mask=mask, cmap="coolwarm", center=0,
            linewidths=0.5, linecolor="black", cbar=True, annot=False
        )

        # 숫자 따로 추가
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if not mask[i, j]:
                    plt.text(
                        j + 0.5, i + 0.5, f"{corr_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=9
                    )

        plt.title(f"{target.upper()}/KRW 상관관계 히트맵", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        
        # VIF 점수 출력
        X = sm.add_constant(df_subset[features])
        vif_df = calculate_vif(X.drop(columns=["const"]))
        print(f"\n{target.upper()}/KRW 독립변수 VIF 점수:")
        print(vif_df)
        
    # USD, DXY 동조화
    df_sync = df[["usd","dxy"]].dropna()
    fig, ax1 = plt.subplots(figsize=(14,6))

    # 왼쪽 y축: USD
    color_usd = "tab:blue"
    ax1.set_xlabel("날짜")
    ax1.set_ylabel("USD/KRW", color=color_usd)
    ax1.plot(df_sync.index, df_sync["usd"], color=color_usd, label="USD/KRW")
    ax1.tick_params(axis="y", labelcolor=color_usd)

    # 오른쪽 y축: DXY
    ax2 = ax1.twinx()
    color_dxy = "tab:red"
    ax2.set_ylabel("DXY", color=color_dxy)
    ax2.plot(df_sync.index, df_sync["dxy"], color=color_dxy, label="DXY")
    ax2.tick_params(axis="y", labelcolor=color_dxy)

    # 레전드
    plt.title("USD/KRW와 DXY 동조화", fontsize=16)
    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
