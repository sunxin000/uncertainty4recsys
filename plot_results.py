import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 设置字体和大小
    plt.rc('font', size=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', family='Times New Roman')
    
    # 获取Dark2颜色方案
    colors = plt.cm.get_cmap("Dark2")
    markers = ['o', 's', '^', 'p', 'v', 'd', 'h', '2', '8', '6']
    
    # 创建图形和子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # 第一个子图：AUC指标
    ax1.plot(df['iteration'], df['overall_auc'], marker=markers[0], 
            color=colors(0), label='AUC', markersize=5, zorder=5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('AUC Score')
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.legend(loc='upper left')
    ax1.set_title('AUC Metrics')
    
    # 第二个子图：DCG和Recall指标
    ax2_1 = ax2
    ax2_2 = ax2.twinx()
    
    # DCG指标
    lines1 = ax2_1.plot(df['iteration'], df['dcg@2'], marker=markers[0], 
                      color=colors(0), label='DCG@2', markersize=5, zorder=5)
    lines2 = ax2_1.plot(df['iteration'], df['dcg@4'], marker=markers[1], 
                      color=colors(0), label='DCG@4', markersize=5, zorder=5, alpha=0.7)
    lines3 = ax2_1.plot(df['iteration'], df['dcg@6'], marker=markers[2], 
                      color=colors(0), label='DCG@6', markersize=5, zorder=5, alpha=0.4)
    ax2_1.set_xlabel('Iteration')
    ax2_1.set_ylabel('DCG Score')
    ax2_1.grid(True, alpha=0.3, zorder=1)
    
    # Recall指标
    lines4 = ax2_2.plot(df['iteration'], df['recall@2'], marker=markers[3], 
                      color=colors(1), label='Recall@2', markersize=5, zorder=5)
    lines5 = ax2_2.plot(df['iteration'], df['recall@4'], marker=markers[4], 
                      color=colors(1), label='Recall@4', markersize=5, zorder=5, alpha=0.7)
    lines6 = ax2_2.plot(df['iteration'], df['recall@6'], marker=markers[5], 
                      color=colors(1), label='Recall@6', markersize=5, zorder=5, alpha=0.4)
    ax2_2.set_ylabel('Recall Score')
    ax2_2.grid(False)
    
    # 合并两个y轴的图例
    lines = lines1 + lines2 + lines3 + lines4 + lines5 + lines6
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.set_title('DCG and Recall Metrics')
    
    # 第三个子图：Calibration指标
    ax3_1 = ax3
    ax3_2 = ax3.twinx()
    
    # NLL
    line1 = ax3_1.plot(df['iteration'], df['nll'], marker=markers[0], 
                     color=colors(0), label='NLL', markersize=5, zorder=5)
    ax3_1.set_xlabel('Iteration')
    ax3_1.set_ylabel('NLL')
    ax3_1.grid(True, alpha=0.3, zorder=1)
    
    # ECE
    line2 = ax3_2.plot(df['iteration'], df['ece'], marker=markers[1], 
                     color=colors(1), label='ECE', markersize=5, zorder=5)
    ax3_2.set_ylabel('ECE')
    ax3_2.grid(False)
    
    # 合并两个y轴的图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    ax3.set_title('Calibration Metrics')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('metrics_visualization_no_ls.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="propensity_evaluation_results_coat.csv")
    args = parser.parse_args()
    
    plot_metrics(args.csv_file)