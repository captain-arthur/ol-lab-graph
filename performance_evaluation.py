import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_performance_data():
    """성능평가 데이터 생성"""
    data = {
        'Metric': ['Drop Precision', 'Drop Recall', 'Drop F1', 'P@1', 'P@10'],
        'SBERT+CE (Baseline)': [0.0, 0.0, 0.0, 5.0, 1.0],
        'SBERT+CE+SDE': [0.0, 0.0, 0.0, 5.0, 1.0],
        'SBERT+CE+CBC': [0.0, 0.0, 0.0, 0.0, 0.0],
        'SBERT+CE+SDE+CBC (Proposed)': [99.8, 41.8, 58.9, 0.0, 0.0],
        'Improvement (%p)': [99.8, 41.8, 58.9, -5.0, -1.0]
    }
    return pd.DataFrame(data)

def plot_performance_comparison():
    """성능평가 결과 막대그래프 비교 (4가지 방법을 하나의 그래프에)"""
    df = create_performance_data()
    
    # 하나의 그래프에 모든 방법과 메트릭 표시
    fig, ax = plt.subplots(figsize=(14, 8))
    
    methods = ['SBERT+CE', 'SBERT+CE+SDE', 'SBERT+CE+CBC', 'SBERT+CE+SDE+CBC']
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
    metrics = df['Metric'].tolist()
    
    # 데이터 준비
    x = np.arange(len(metrics))
    width = 0.2  # 막대 너비
    
    # 각 방법별로 막대그래프 그리기
    for i, method in enumerate(methods):
        if method == 'SBERT+CE':
            values = [df.iloc[j]['SBERT+CE (Baseline)'] for j in range(len(metrics))]
        elif method == 'SBERT+CE+SDE':
            values = [df.iloc[j]['SBERT+CE+SDE'] for j in range(len(metrics))]
        elif method == 'SBERT+CE+CBC':
            values = [df.iloc[j]['SBERT+CE+CBC'] for j in range(len(metrics))]
        else:  # SBERT+CE+SDE+CBC
            values = [df.iloc[j]['SBERT+CE+SDE+CBC (Proposed)'] for j in range(len(metrics))]
        
        ax.bar(x + i * width, values, width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Performance Evaluation Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 값 표시
    for i, method in enumerate(methods):
        if method == 'SBERT+CE':
            values = [df.iloc[j]['SBERT+CE (Baseline)'] for j in range(len(metrics))]
        elif method == 'SBERT+CE+SDE':
            values = [df.iloc[j]['SBERT+CE+SDE'] for j in range(len(metrics))]
        elif method == 'SBERT+CE+CBC':
            values = [df.iloc[j]['SBERT+CE+CBC'] for j in range(len(metrics))]
        else:  # SBERT+CE+SDE+CBC
            values = [df.iloc[j]['SBERT+CE+SDE+CBC (Proposed)'] for j in range(len(metrics))]
        
        for j, value in enumerate(values):
            if value > 0:
                ax.text(j + i * width, value + 0.5, f'{value:.1f}%', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_improvement_analysis():
    """개선도 분석 막대그래프"""
    df = create_performance_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    metrics = df['Metric'].tolist()
    improvements = df['Improvement (%p)'].tolist()
    
    # 개선도 막대그래프
    colors = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax1.bar(metrics, improvements, color=colors, alpha=0.7)
    ax1.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Improvement (%p)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 값 표시
    for bar, value in zip(bars, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:+.1f}%p', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Proposed vs Baseline 비교
    proposed_values = [df.iloc[i]['SBERT+CE+SDE+CBC (Proposed)'] for i in range(len(df))]
    baseline_values = [df.iloc[i]['SBERT+CE (Baseline)'] for i in range(len(df))]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.7, color='lightblue')
    bars2 = ax2.bar(x + width/2, proposed_values, width, label='Proposed', alpha=0.7, color='gold')
    
    ax2.set_title('Baseline vs Proposed Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metric_focus():
    """메트릭별 집중 분석"""
    df = create_performance_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    methods = ['SBERT+CE (Baseline)', 'SBERT+CE+SDE', 'SBERT+CE+CBC', 'SBERT+CE+SDE+CBC (Proposed)']
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
    
    for i, metric in enumerate(df['Metric']):
        values = [df.iloc[i][method] for method in methods]
        
        bars = axes[i].bar(methods, values, color=colors, alpha=0.8)
        axes[i].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Percentage')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, values):
            if value > 0:
                axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 마지막 subplot은 개선도 요약
    improvements = df['Improvement (%p)'].tolist()
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]
    bars = axes[5].bar(df['Metric'], improvements, color=colors_imp, alpha=0.7)
    axes[5].set_title('Improvement Summary', fontsize=12, fontweight='bold')
    axes[5].set_ylabel('Improvement (%p)')
    axes[5].tick_params(axis='x', rotation=45)
    axes[5].grid(True, alpha=0.3)
    axes[5].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 개선도 값 표시
    for bar, value in zip(bars, improvements):
        height = bar.get_height()
        axes[5].text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{value:+.1f}%p', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('metric_focus_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_heatmap():
    """성능평가 히트맵"""
    df = create_performance_data()
    
    # 히트맵용 데이터 준비
    methods = ['SBERT+CE', 'SBERT+CE+SDE', 'SBERT+CE+CBC', 'SBERT+CE+SDE+CBC']
    metrics = df['Metric'].tolist()
    
    # 데이터 매트릭스 생성
    data_matrix = []
    for i, metric in enumerate(metrics):
        row = []
        for method in methods:
            if method == 'SBERT+CE':
                row.append(df.iloc[i]['SBERT+CE (Baseline)'])
            elif method == 'SBERT+CE+SDE':
                row.append(df.iloc[i]['SBERT+CE+SDE'])
            elif method == 'SBERT+CE+CBC':
                row.append(df.iloc[i]['SBERT+CE+CBC'])
            else:  # SBERT+CE+SDE+CBC
                row.append(df.iloc[i]['SBERT+CE+SDE+CBC (Proposed)'])
        data_matrix.append(row)
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    
    # 축 레이블 설정
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(methods, rotation=45)
    ax.set_yticklabels(metrics)
    
    # 값 표시
    for i in range(len(metrics)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{data_matrix[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Evaluation Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Percentage')
    plt.tight_layout()
    plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """성능평가 시각화 실행"""
    print("성능평가 결과 시각화 생성 중...")
    
    print("1. 성능평가 비교 막대그래프 생성 중...")
    plot_performance_comparison()
    
    print("2. 개선도 분석 막대그래프 생성 중...")
    plot_improvement_analysis()
    
    print("3. 메트릭별 집중 분석 생성 중...")
    plot_metric_focus()
    
    print("4. 성능평가 히트맵 생성 중...")
    plot_performance_heatmap()
    
    print("모든 성능평가 시각화가 완료되었습니다!")
    print("생성된 파일들:")
    print("- performance_comparison.png (성능평가 비교)")
    print("- improvement_analysis.png (개선도 분석)")
    print("- metric_focus_analysis.png (메트릭별 집중 분석)")
    print("- performance_heatmap.png (성능평가 히트맵)")

if __name__ == "__main__":
    main()
