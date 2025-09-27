import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_performance_data():
    """성능평가 데이터 생성 (MTEB FiQA dataset) - Drop 지표만"""
    data = {
        'Metric': ['Drop Precision', 'Drop Recall', 'Drop F1'],
        'SBERT+CE (Baseline)': [0.0, 0.0, 0.0],
        'SBERT+CE+SDE': [12.0, 4.5, 6.9],
        'SBERT+CE+CBC': [18.5, 10.2, 13.8],
        'SBERT+CE+SDE+CBC (Proposed)': [85.3, 36.7, 51.1],
        'Improvement (%p)': [85.3, 36.7, 51.1]
    }
    return pd.DataFrame(data)

def plot_performance_comparison():
    """성능평가 결과 시각화 (방법별 비교)"""
    df = create_performance_data()
    
    # 첫 번째 그래프만 유지
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    methods = ['SBERT+CE (Baseline)', 'SBERT+CE+SDE', 'SBERT+CE+CBC', 'SBERT+CE+SDE+CBC (Proposed)']
    method_labels = ['Baseline', 'SDE', 'CBC', 'Proposed']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#8c8c8c']  # Baseline 초록, Proposed 회색
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2.5, 2.5, 2.5, 2.5]
    metrics = df['Metric'].tolist()
    
    # 지표별 선형 그래프 (방법별 비교)
    x = np.arange(len(metrics))  # x축은 지표들
    
    for i, method in enumerate(methods):
        values = [df.iloc[j][method] for j in range(len(metrics))]
        
        # 선형 그래프로 표시 (색상 추가, 학술적 스타일)
        ax.plot(x, values, color=colors[i], linestyle=line_styles[i], 
                linewidth=line_widths[i], label=method_labels[i], alpha=0.9)
    
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontsize=9)
    ax.legend(fontsize=8, loc='upper right', labelspacing=0.3, handlelength=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 100)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_relative_improvement():
    """성능 점수 막대 그래프"""
    df = create_performance_data()
    
    # 성능 점수 데이터
    baseline_values = [df.iloc[i]['SBERT+CE (Baseline)'] for i in range(len(df))]
    sde_values = [df.iloc[i]['SBERT+CE+SDE'] for i in range(len(df))]
    cbc_values = [df.iloc[i]['SBERT+CE+CBC'] for i in range(len(df))]
    proposed_values = [df.iloc[i]['SBERT+CE+SDE+CBC (Proposed)'] for i in range(len(df))]
    
    metrics = df['Metric'].tolist()
    x = np.arange(len(metrics))
    width = 0.15
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # 막대 그래프 생성 (Baseline 포함) - Baseline 초록, Proposed 회색
    bars0 = ax.bar(x - 1.5*width, baseline_values, width, label='Baseline', 
                   color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars1 = ax.bar(x - 0.5*width, sde_values, width, label='SDE', 
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + 0.5*width, cbc_values, width, label='CBC', 
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 1.5*width, proposed_values, width, label='Proposed', 
                   color='#8c8c8c', alpha=0.8, edgecolor='black', linewidth=0.5, hatch='///', hatch_linewidth=0.3)
    
    # 수치 라벨 추가 (Baseline 포함)
    all_bars = [bars0, bars1, bars2, bars3]
    all_values = [baseline_values, sde_values, cbc_values, proposed_values]
    
    for bars, values in zip(all_bars, all_values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # 모든 값 표시
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}%', ha='center', va='bottom', 
                   fontsize=6, fontweight='normal')
    
    ax.set_ylabel('Performance (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, fontsize=9)
    ax.legend(fontsize=8, loc='upper right', labelspacing=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-5, 100)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig('relative_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_improvement_analysis():
    """개선도 분석 (단계별 비교)"""
    df = create_performance_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    metrics = df['Metric'].tolist()
    improvements = df['Improvement (%p)'].tolist()
    
    # 개선도 막대그래프
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax1.bar(metrics, improvements, color=colors, alpha=0.7)
    ax1.set_ylabel('Improvement (%p)', fontsize=12)
    ax1.set_xticks(np.arange(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=0, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.tick_params(axis='y', labelsize=10)
    
    
    # Baseline vs Proposed 비교
    baseline_values = [df.iloc[i]['SBERT+CE (Baseline)'] for i in range(len(df))]
    proposed_values = [df.iloc[i]['SBERT+CE+SDE+CBC (Proposed)'] for i in range(len(df))]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7, color='lightblue')
    bars2 = ax2.bar(x + width/2, proposed_values, width, label='Proposed', alpha=0.7, color='gold')
    
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=0, fontsize=11)
    ax2.legend(fontsize=6, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metric_focus():
    """메트릭별 집중 분석 (단계별)"""
    df = create_performance_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    axes = axes.flatten()
    
    methods = ['SBERT+CE (Baseline)', 'SBERT+CE+SDE', 'SBERT+CE+CBC', 'SBERT+CE+SDE+CBC (Proposed)']
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
    
    for i, metric in enumerate(df['Metric']):
        values = [df.iloc[i][method] for method in methods]
        
        bars = axes[i].bar(methods, values, color=colors, alpha=0.8)
        axes[i].set_ylabel('Percentage (%)', fontsize=11)
        axes[i].set_xticks(np.arange(len(methods)))
        axes[i].set_xticklabels(methods, rotation=0, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='y', labelsize=10)
    
    # 마지막 subplot은 개선도 요약
    improvements = df['Improvement (%p)'].tolist()
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]
    bars = axes[3].bar(df['Metric'], improvements, color=colors_imp, alpha=0.7)
    axes[3].set_ylabel('Improvement (%p)', fontsize=11)
    axes[3].set_xticks(np.arange(len(df['Metric'])))
    axes[3].set_xticklabels(df['Metric'], rotation=0, fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[3].tick_params(axis='y', labelsize=10)
    
    
    plt.tight_layout()
    plt.savefig('metric_focus_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_heatmap():
    """성능평가 히트맵"""
    df = create_performance_data()
    
    # 히트맵용 데이터 준비
    methods = ['SBERT+CE (Baseline)', 'SBERT+CE+SDE', 'SBERT+CE+CBC', 'SBERT+CE+SDE+CBC (Proposed)']
    metrics = df['Metric'].tolist()
    
    # 데이터 매트릭스 생성
    data_matrix = []
    for i, metric in enumerate(metrics):
        row = []
        for method in methods:
            row.append(df.iloc[i][method])
        data_matrix.append(row)
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    
    # 축 레이블 설정
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(methods, rotation=0, fontsize=11)
    ax.set_yticklabels(metrics, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.colorbar(im, ax=ax, label='Percentage (%)')
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """성능평가 시각화 실행"""
    print("성능평가 결과 시각화 생성 중...")
    
    print("1. 성능평가 비교 선형그래프 생성 중...")
    plot_performance_comparison()
    
    print("2. 성능 점수 막대그래프 생성 중...")
    plot_relative_improvement()
    
    print("성능평가 시각화가 완료되었습니다!")
    print("생성된 파일들:")
    print("- performance_comparison.png (성능평가 비교)")
    print("- relative_improvement.png (성능 점수 비교)")

if __name__ == "__main__":
    main()
