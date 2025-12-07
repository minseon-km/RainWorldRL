# evaluate.py
import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="pastel")

def load_results(result_glob_pattern="results_*.csv"):
    paths = sorted(glob.glob(result_glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No result CSV found with pattern {result_glob_pattern}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        # 파일명에서 알고리즘 추출(예: results_PPO.csv -> 'PPO')
        alg = os.path.basename(p).split(".")[0].split("results_")[-1]
        if "Algorithm" in df.columns:
            # 이미 알고리즘 컬럼이 있으면 유지
            pass
        else:
            df["Algorithm"] = alg
        dfs.append(df)
    results = pd.concat(dfs, ignore_index=True)
    return results

def summarize_by_algorithm(df):
    """
    각 알고리즘별 mean, std, 95% CI를 계산
    기대 컬럼: Algorithm, Score
    """
    group = df.groupby("Algorithm")["Score"].agg(["mean", "std", "count"])
    group = group.rename(columns={"mean": "mean_score", "std": "std_score", "count": "n"})
    group["sem"] = group["std_score"] / np.sqrt(group["n"])
    group["ci_low"] = group["mean_score"] - 1.96 * group["sem"]
    group["ci_high"] = group["mean_score"] + 1.96 * group["sem"]
    return group.reset_index()

def plot_bar_ci(summary_df, out_dir="plots"):
    """
    알고리즘별 평균 보상 + 95% CI를 바 차트로 그려 PNG/SVG로 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    sns.barplot(data=summary_df, x="Algorithm", y="mean_score", yerr=0, ax=ax,
                palette="pastel", capsize=0.1)
    # 95% CI를 error bar로 추가
    ax.errorbar(summary_df["Algorithm"],
                summary_df["mean_score"],
                yerr=1.96 * summary_df["sem"],
                fmt='none', c='black', capsize=5)
    plt.title("Mean Score by Algorithm with 95% CI")
    plt.ylabel("Mean Score")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "algo_mean_ci.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "algo_mean_ci.svg"), format="svg")
    plt.close()

def plot_learning_curves(df, out_dir="plots"):
    """
    에피소드별 평균 보상을 시각화. df에 columns: Episode, Algorithm, Score
    이 함수는 opt-in 형태로, 파일이 없으면 스킵.
    """
    if "Episode" not in df.columns:
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df, x="Episode", y="Score", hue="Algorithm", estimator=None)
    plt.title("Episode-wise Score by Algorithm")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "learning_curves.svg"), format="svg")
    plt.close()

def save_results_csv(results_df, path="results_combined.csv"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f" Combined results saved to {path}")

def main():
    # 기본적으로 기존 결과 CSV를 합쳐 요약/시각화를 수행
    results = load_results("results_*.csv")
    # 만약 로그에 Algorithm 컬럼이 없다면 보정
    if "Algorithm" not in results.columns:
        # 파일명에서 추출하는 방법으로 간단 보정
        results["Algorithm"] = results.get("Algorithm", "Unknown")

    summary = summarize_by_algorithm(results)

    # 저장
    save_results_csv(results, path="results_combined.csv")

    # 시각화
    plot_bar_ci(summary, out_dir="plots")
    # 필요시 학습 곡선도 시각화
    if "Episode" in results.columns:
        # Episode별 Score를 알고리즘별로 묶어 그래프
        plot_learning_curves(results, out_dir="plots")

if __name__ == "__main__":
    main()
