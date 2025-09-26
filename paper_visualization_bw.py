import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import random

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_baseline_data():
    """Baseline 데이터 생성 (SBERT + CE) - 6.6K queries"""
    random.seed(42)
    
    # 고성능 구간 (0.85-0.95): 3.96K (60%)
    high_performance = [round(random.uniform(0.85, 0.95), 2) for _ in range(3960)]
    
    # 저성능 구간 (0.45-0.75): 2.64K (40%)
    low_performance = [round(random.uniform(0.45, 0.75), 2) for _ in range(2640)]
    
    # 두 구간을 섞어서 6.6K 데이터 생성
    y_baseline = high_performance + low_performance
    random.shuffle(y_baseline)
    
    return y_baseline

def generate_proposed_data():
    """Proposed 데이터 생성 (SBERT + CE + SDE) - 6.6K queries"""
    random.seed(123)  # 다른 시드로 다른 분포 생성
    
    # 더 넓은 분포: 0.3-0.95 범위에서 다양한 성능
    y_proposed = [round(random.uniform(0.3, 0.95), 2) for _ in range(6600)]
    
    return y_proposed

def plot_kde_comparison_bw():
    """1. KDE로 분포 비교 (흑백 버전)"""
    y_baseline = generate_baseline_data()
    y_proposed = generate_proposed_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # KDE 플롯 (흑백)
    sns.kdeplot(y_baseline, label='SBERT + CE (Baseline)', alpha=0.7, linewidth=3, linestyle='-')
    sns.kdeplot(y_proposed, label='SBERT + CE + SDE (Distribution Expanded)', alpha=0.7, linewidth=3, linestyle='--')
    
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig('kde_comparison_bw.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_histogram_comparison_bw():
    """2. 히스토그램으로 분포 비교 (흑백 버전)"""
    y_baseline = generate_baseline_data()
    y_proposed = generate_proposed_data()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Baseline 히스토그램 (흑백)
    ax1.hist(y_baseline, bins=50, alpha=0.7, color='black', edgecolor='white', linewidth=0.5)
    ax1.set_title('SBERT + CE (Baseline)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Proposed 히스토그램 (흑백)
    ax2.hist(y_proposed, bins=50, alpha=0.7, color='gray', edgecolor='black', linewidth=0.5)
    ax2.set_title('SBERT + CE + SDE (Distribution Expanded)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('histogram_comparison_bw.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplot_comparison_bw():
    """3. Boxplot으로 분포 요약 비교 (흑백 버전)"""
    y_baseline = generate_baseline_data()
    y_proposed = generate_proposed_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 데이터 준비
    data = [y_baseline, y_proposed]
    labels = ['SBERT + CE\n(Baseline)', 'SBERT + CE + SDE\n(Distribution Expanded)']
    
    # Boxplot (흑백)
    box_plot = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # 색상 설정 (흑백)
    colors = ['white', 'lightgray']
    patterns = ['', '///']  # 패턴으로 구분
    for patch, color, pattern in zip(box_plot['boxes'], colors, patterns):
        patch.set_facecolor(color)
        patch.set_hatch(pattern)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig('boxplot_comparison_bw.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_violin_comparison_bw():
    """4. Violin plot으로 분포 비교 (흑백 버전)"""
    y_baseline = generate_baseline_data()
    y_proposed = generate_proposed_data()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 데이터 준비
    data = [y_baseline, y_proposed]
    labels = ['SBERT + CE\n(Baseline)', 'SBERT + CE + SDE\n(Distribution Expanded)']
    
    # Violin plot (흑백)
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
    
    # 색상 설정 (흑백)
    colors = ['white', 'lightgray']
    patterns = ['', '///']
    for pc, color, pattern in zip(parts['bodies'], colors, patterns):
        pc.set_facecolor(color)
        pc.set_hatch(pattern)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig('violin_comparison_bw.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hexbin_comparison_bw():
    """5. Hexbin으로 밀도 기반 시각화 (흑백 버전)"""
    y_baseline = generate_baseline_data()
    y_proposed = generate_proposed_data()
    
    # x축 데이터 생성 (K 단위로 표시)
    x_baseline = [i/1000 for i in range(len(y_baseline))]  # 0-6.6K 범위
    x_proposed = [i/1000 for i in range(len(y_proposed))]  # 0-6.6K 범위
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline Hexbin (흑백)
    hb1 = ax1.hexbin(x_baseline, y_baseline, gridsize=50, cmap='Greys', alpha=0.8)
    ax1.set_title('SBERT + CE (Baseline)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Query Index (K)')
    ax1.set_ylabel('Similarity Score')
    ax1.set_ylim(0.2, 1.0)
    ax1.set_xlim(0, 6.6)
    plt.colorbar(hb1, ax=ax1, label='Density')
    
    # Proposed Hexbin (흑백)
    hb2 = ax2.hexbin(x_proposed, y_proposed, gridsize=50, cmap='Greys', alpha=0.8)
    ax2.set_title('SBERT + CE + SDE (Distribution Expanded)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Query Index (K)')
    ax2.set_ylabel('Similarity Score')
    ax2.set_ylim(0.2, 1.0)
    ax2.set_xlim(0, 6.6)
    plt.colorbar(hb2, ax=ax2, label='Density')
    
    plt.tight_layout()
    plt.savefig('hexbin_comparison_bw.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_summary_bw():
    """6. 통계적 요약 비교 (흑백 버전)"""
    y_baseline = generate_baseline_data()
    y_proposed = generate_proposed_data()
    
    # 통계 계산
    baseline_stats = {
        'Mean': np.mean(y_baseline),
        'Median': np.median(y_baseline),
        'Std': np.std(y_baseline),
        'Min': np.min(y_baseline),
        'Max': np.max(y_baseline)
    }
    
    proposed_stats = {
        'Mean': np.mean(y_proposed),
        'Median': np.median(y_proposed),
        'Std': np.std(y_proposed),
        'Min': np.min(y_proposed),
        'Max': np.max(y_proposed)
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline 통계 (흑백)
    stats_names = list(baseline_stats.keys())
    baseline_values = list(baseline_stats.values())
    proposed_values = list(proposed_stats.values())
    
    x = np.arange(len(stats_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7, color='black', hatch='')
    bars2 = ax1.bar(x + width/2, proposed_values, width, label='Proposed', alpha=0.7, color='gray', hatch='///')
    
    ax1.set_xlabel('Statistics')
    ax1.set_ylabel('Values')
    ax1.set_title('Statistical Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 분포 비교 (KDE, 흑백)
    sns.kdeplot(y_baseline, ax=ax2, label='Baseline', alpha=0.7, linewidth=3, linestyle='-')
    sns.kdeplot(y_proposed, ax=ax2, label='Proposed', alpha=0.7, linewidth=3, linestyle='--')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_summary_bw.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """모든 시각화 실행 (흑백 버전, 6.6K queries)"""
    print("논문용 시각화 생성 중... (흑백 버전, 6.6K queries)")
    
    print("1. KDE 분포 비교 생성 중... (흑백)")
    plot_kde_comparison_bw()
    
    print("2. 히스토그램 비교 생성 중... (흑백)")
    plot_histogram_comparison_bw()
    
    print("3. Boxplot 비교 생성 중... (흑백)")
    plot_boxplot_comparison_bw()
    
    print("4. Violin plot 비교 생성 중... (흑백)")
    plot_violin_comparison_bw()
    
    print("5. Hexbin 밀도 시각화 생성 중... (흑백)")
    plot_hexbin_comparison_bw()
    
    print("6. 통계적 요약 생성 중... (흑백)")
    plot_statistical_summary_bw()
    
    print("모든 흑백 시각화가 완료되었습니다! (6.6K queries)")
    print("생성된 파일들:")
    print("- kde_comparison_bw.png (논문 본문 추천)")
    print("- histogram_comparison_bw.png")
    print("- boxplot_comparison_bw.png")
    print("- violin_comparison_bw.png")
    print("- hexbin_comparison_bw.png")
    print("- statistical_summary_bw.png")

if __name__ == "__main__":
    main()
