import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 设置图表风格
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 创建图形和子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 第一个子图：AUC指标
    ax1.plot(df['iteration'], df['overall_auc'], '-o', label='AUC', markersize=4, zorder=5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('AUC Score', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.legend(loc='upper left')
    ax1.set_title('AUC Metrics')
    
    # 第二个子图：DCG和Recall指标
    ax2_1 = ax2
    ax2_2 = ax2.twinx()  # 创建双y轴
    
    # DCG指标
    lines1 = ax2_1.plot(df['iteration'], df['dcg@2'], '-o', label='DCG@2', markersize=4, zorder=5)
    lines2 = ax2_1.plot(df['iteration'], df['dcg@4'], '-o', label='DCG@4', markersize=4, zorder=5)
    lines3 = ax2_1.plot(df['iteration'], df['dcg@6'], '-o', label='DCG@6', markersize=4, zorder=5)
    ax2_1.set_xlabel('Iteration')
    ax2_1.set_ylabel('DCG Score', color='black')
    ax2_1.tick_params(axis='y', labelcolor='black')
    
    # Recall指标
    lines4 = ax2_2.plot(df['iteration'], df['recall@2'], '--o', label='Recall@2', markersize=4, alpha=0.7, zorder=5)
    lines5 = ax2_2.plot(df['iteration'], df['recall@4'], '--o', label='Recall@4', markersize=4, alpha=0.7, zorder=5)
    lines6 = ax2_2.plot(df['iteration'], df['recall@6'], '--o', label='Recall@6', markersize=4, alpha=0.7, zorder=5)
    ax2_2.set_ylabel('Recall Score', color='black')
    ax2_2.tick_params(axis='y', labelcolor='black')
    
    # 合并两个y轴的图例
    lines = lines1 + lines2 + lines3 + lines4 + lines5 + lines6
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.set_title('DCG and Recall Metrics')
    ax2_1.grid(True, alpha=0.3, zorder=1)
    
    # 第三个子图：Calibration指标
    ax3_1 = ax3
    ax3_2 = ax3.twinx()  # 创建双y轴
    
    # NLL
    line1 = ax3_1.plot(df['iteration'], df['nll'], '-o', color='blue', label='NLL', markersize=4, zorder=5)
    ax3_1.set_xlabel('Iteration')
    ax3_1.set_ylabel('NLL', color='blue')
    ax3_1.tick_params(axis='y', labelcolor='blue')
    
    # ECE
    line2 = ax3_2.plot(df['iteration'], df['ece'], '-o', color='red', label='ECE', markersize=4, zorder=5)
    ax3_2.set_ylabel('ECE', color='red')
    ax3_2.tick_params(axis='y', labelcolor='red')
    
    # 合并两个y轴的图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    ax3.set_title('Calibration Metrics')
    ax3.grid(True, alpha=0.3, zorder=1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('metrics_visualization_no_ls.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="propensity_evaluation_results_coat.csv")
    args = parser.parse_args()
    
    plot_metrics(args.csv_file)