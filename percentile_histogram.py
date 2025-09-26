import matplotlib.pyplot as plt
import numpy as np
import random

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_concentrated_data():
    """집중된 분포 데이터 생성 (10% 퍼센타일 적용)"""
    random.seed(42)
    
    # 집중된 분포: 대부분이 0.8-0.95 범위에 몰림
    data = []
    for _ in range(1000):
        if random.random() < 0.8:  # 80%는 고성능
            data.append(random.uniform(0.85, 0.95))
        else:  # 20%는 저성능
            data.append(random.uniform(0.3, 0.6))
    
    return sorted(data)

def generate_dispersed_data():
    """분산된 분포 데이터 생성 (30% 퍼센타일 적용)"""
    random.seed(123)
    
    # 분산된 분포: 0.2-0.95 범위에 고르게 분포
    data = []
    for _ in range(1000):
        data.append(random.uniform(0.2, 0.95))
    
    return sorted(data)

def generate_balanced_data():
    """균형적인 분포 데이터 생성 (20% 퍼센타일 적용)"""
    random.seed(456)
    
    # 균형적인 분포: 두 개의 피크 (0.4-0.6, 0.8-0.9)
    data = []
    for _ in range(1000):
        if random.random() < 0.5:  # 50%는 중간 성능
            data.append(random.uniform(0.4, 0.6))
        else:  # 50%는 고성능
            data.append(random.uniform(0.8, 0.9))
    
    return sorted(data)

def calculate_percentile_threshold(data, percentile):
    """퍼센타일 임계값 계산"""
    return np.percentile(data, percentile)

def plot_percentile_histograms():
    """퍼센타일 적용 상황별 히스토그램"""
    
    # 데이터 생성
    concentrated = generate_concentrated_data()
    dispersed = generate_dispersed_data()
    balanced = generate_balanced_data()
    
    # 퍼센타일 임계값 계산
    conc_threshold = calculate_percentile_threshold(concentrated, 10)
    disp_threshold = calculate_percentile_threshold(dispersed, 30)
    bal_threshold = calculate_percentile_threshold(balanced, 20)
    
    # 3개 서브플롯 생성 (크기 축소)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # 1. 집중된 분포 (10% 퍼센타일)
    ax1.hist(concentrated, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(conc_threshold, color='red', linestyle='--', linewidth=2)
    ax1.set_title('Concentrated Distribution (10th Percentile)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.text(0.02, 0.95, f'10th Percentile: {conc_threshold:.3f}', 
             transform=ax1.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.grid(True, alpha=0.3)
    
    # 2. 분산된 분포 (30% 퍼센타일)
    ax2.hist(dispersed, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(disp_threshold, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Dispersed Distribution (30th Percentile)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.text(0.02, 0.95, f'30th Percentile: {disp_threshold:.3f}', 
             transform=ax2.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax2.grid(True, alpha=0.3)
    
    # 3. 균형적인 분포 (20% 퍼센타일)
    ax3.hist(balanced, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(bal_threshold, color='red', linestyle='--', linewidth=2)
    ax3.set_title('Balanced Distribution (20th Percentile)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Similarity Score')
    ax3.set_ylabel('Frequency')
    ax3.text(0.02, 0.95, f'20th Percentile: {bal_threshold:.3f}', 
             transform=ax3.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('percentile_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 통계 정보 출력
    print("=== 퍼센타일 적용 상황별 분석 ===")
    print(f"집중된 분포 (10% 퍼센타일): {conc_threshold:.3f}")
    print(f"분산된 분포 (30% 퍼센타일): {disp_threshold:.3f}")
    print(f"균형적인 분포 (20% 퍼센타일): {bal_threshold:.3f}")
    
    return concentrated, dispersed, balanced, conc_threshold, disp_threshold, bal_threshold

def plot_percentile_comparison():
    """퍼센타일 적용 전후 비교"""
    
    # 데이터 생성
    concentrated = generate_concentrated_data()
    dispersed = generate_dispersed_data()
    balanced = generate_balanced_data()
    
    # 퍼센타일 임계값 계산
    conc_threshold = calculate_percentile_threshold(concentrated, 10)
    disp_threshold = calculate_percentile_threshold(dispersed, 30)
    bal_threshold = calculate_percentile_threshold(balanced, 20)
    
    # 2x2 서브플롯 생성 (크기 축소)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 집중된 분포 - 적용 전
    ax1.hist(concentrated, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('Concentrated Distribution\n(Before 10th Percentile)', fontweight='bold')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. 집중된 분포 - 적용 후
    filtered_conc = [x for x in concentrated if x >= conc_threshold]
    ax2.hist(filtered_conc, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(conc_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {conc_threshold:.3f}')
    ax2.set_title('Concentrated Distribution\n(After 10th Percentile)', fontweight='bold')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 분산된 분포 - 적용 전
    ax3.hist(dispersed, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Dispersed Distribution\n(Before 30th Percentile)', fontweight='bold')
    ax3.set_xlabel('Similarity Score')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 4. 분산된 분포 - 적용 후
    filtered_disp = [x for x in dispersed if x >= disp_threshold]
    ax4.hist(filtered_disp, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(disp_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {disp_threshold:.3f}')
    ax4.set_title('Dispersed Distribution\n(After 30th Percentile)', fontweight='bold')
    ax4.set_xlabel('Similarity Score')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('percentile_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_percentile_statistics():
    """퍼센타일 적용 통계 분석"""
    
    # 데이터 생성
    concentrated = generate_concentrated_data()
    dispersed = generate_dispersed_data()
    balanced = generate_balanced_data()
    
    # 퍼센타일 임계값 계산
    conc_threshold = calculate_percentile_threshold(concentrated, 10)
    disp_threshold = calculate_percentile_threshold(dispersed, 30)
    bal_threshold = calculate_percentile_threshold(balanced, 20)
    
    # 필터링된 데이터
    filtered_conc = [x for x in concentrated if x >= conc_threshold]
    filtered_disp = [x for x in dispersed if x >= disp_threshold]
    filtered_bal = [x for x in balanced if x >= bal_threshold]
    
    # 통계 계산
    stats_data = {
        'Concentrated (10%)': {
            'Original': {'Mean': np.mean(concentrated), 'Std': np.std(concentrated), 'Count': len(concentrated)},
            'Filtered': {'Mean': np.mean(filtered_conc), 'Std': np.std(filtered_conc), 'Count': len(filtered_conc)}
        },
        'Dispersed (30%)': {
            'Original': {'Mean': np.mean(dispersed), 'Std': np.std(dispersed), 'Count': len(dispersed)},
            'Filtered': {'Mean': np.mean(filtered_disp), 'Std': np.std(filtered_disp), 'Count': len(filtered_disp)}
        },
        'Balanced (20%)': {
            'Original': {'Mean': np.mean(balanced), 'Std': np.std(balanced), 'Count': len(balanced)},
            'Filtered': {'Mean': np.mean(filtered_bal), 'Std': np.std(filtered_bal), 'Count': len(filtered_bal)}
        }
    }
    
    # 막대그래프로 통계 비교 (크기 축소)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 평균 비교
    categories = ['Concentrated', 'Dispersed', 'Balanced']
    original_means = [stats_data['Concentrated (10%)']['Original']['Mean'],
                     stats_data['Dispersed (30%)']['Original']['Mean'],
                     stats_data['Balanced (20%)']['Original']['Mean']]
    filtered_means = [stats_data['Concentrated (10%)']['Filtered']['Mean'],
                     stats_data['Dispersed (30%)']['Filtered']['Mean'],
                     stats_data['Balanced (20%)']['Filtered']['Mean']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, original_means, width, label='Original', alpha=0.7, color='lightblue')
    ax1.bar(x + width/2, filtered_means, width, label='Filtered', alpha=0.7, color='blue')
    ax1.set_xlabel('Distribution Type')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Score Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 표준편차 비교
    original_stds = [stats_data['Concentrated (10%)']['Original']['Std'],
                    stats_data['Dispersed (30%)']['Original']['Std'],
                    stats_data['Balanced (20%)']['Original']['Std']]
    filtered_stds = [stats_data['Concentrated (10%)']['Filtered']['Std'],
                    stats_data['Dispersed (30%)']['Filtered']['Std'],
                    stats_data['Balanced (20%)']['Filtered']['Std']]
    
    ax2.bar(x - width/2, original_stds, width, label='Original', alpha=0.7, color='lightgreen')
    ax2.bar(x + width/2, filtered_stds, width, label='Filtered', alpha=0.7, color='green')
    ax2.set_xlabel('Distribution Type')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 데이터 개수 비교
    original_counts = [stats_data['Concentrated (10%)']['Original']['Count'],
                      stats_data['Dispersed (30%)']['Original']['Count'],
                      stats_data['Balanced (20%)']['Original']['Count']]
    filtered_counts = [stats_data['Concentrated (10%)']['Filtered']['Count'],
                       stats_data['Dispersed (30%)']['Filtered']['Count'],
                       stats_data['Balanced (20%)']['Filtered']['Count']]
    
    ax3.bar(x - width/2, original_counts, width, label='Original', alpha=0.7, color='lightcoral')
    ax3.bar(x + width/2, filtered_counts, width, label='Filtered', alpha=0.7, color='red')
    ax3.set_xlabel('Distribution Type')
    ax3.set_ylabel('Data Count')
    ax3.set_title('Data Count Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 퍼센타일 임계값 비교
    thresholds = [conc_threshold, disp_threshold, bal_threshold]
    ax4.bar(categories, thresholds, alpha=0.7, color='orange')
    ax4.set_xlabel('Distribution Type')
    ax4.set_ylabel('Percentile Threshold')
    ax4.set_title('Percentile Thresholds', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 값 표시
    for i, v in enumerate(thresholds):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('percentile_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 통계 정보 출력
    print("\n=== 퍼센타일 적용 통계 분석 ===")
    for dist_type, data in stats_data.items():
        print(f"\n{dist_type}:")
        print(f"  Original - Mean: {data['Original']['Mean']:.3f}, Std: {data['Original']['Std']:.3f}, Count: {data['Original']['Count']}")
        print(f"  Filtered - Mean: {data['Filtered']['Mean']:.3f}, Std: {data['Filtered']['Std']:.3f}, Count: {data['Filtered']['Count']}")
        print(f"  Retention Rate: {data['Filtered']['Count']/data['Original']['Count']*100:.1f}%")

def main():
    """퍼센타일 적용 상황별 히스토그램 생성"""
    print("퍼센타일 적용 상황별 히스토그램 생성 중...")
    
    print("1. 퍼센타일 적용 상황별 히스토그램 생성 중...")
    plot_percentile_histograms()
    
    print("2. 퍼센타일 적용 전후 비교 생성 중...")
    plot_percentile_comparison()
    
    print("3. 퍼센타일 적용 통계 분석 생성 중...")
    plot_percentile_statistics()
    
    print("모든 퍼센타일 히스토그램이 완료되었습니다!")
    print("생성된 파일들:")
    print("- percentile_histograms.png (상황별 히스토그램)")
    print("- percentile_comparison.png (적용 전후 비교)")
    print("- percentile_statistics.png (통계 분석)")

if __name__ == "__main__":
    main()
