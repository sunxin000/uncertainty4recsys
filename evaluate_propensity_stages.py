import os
import glob
import torch
import pandas as pd
import subprocess
from collections import defaultdict
import re

def evaluate_propensity_checkpoints(data="coat", seed=42):
    # 匹配特定后缀的文件
    suffix = "_no_ls"  # 或者其他后缀
    checkpoint_files = glob.glob(f'platt_calibration_plots/checkpoint_logits_*{suffix}.pt')
    
    def extract_number(filename):
        # 修改正则表达式以匹配 checkpoint_logits_数字_后缀.pt 格式
        match = re.search(r'checkpoint_logits_(\d+)_no_ls\.pt$', filename)
        if match:
            print(f"Found number {match.group(1)} in filename {filename}")
        else:
            print(f"No match found in filename {filename}")
        return int(match.group(1)) if match else 0
    
    # 测试输出
    print("Found files:", checkpoint_files)
    checkpoint_files.sort(key=extract_number)
    
    # 存储所有结果
    results = defaultdict(list)
    
    for checkpoint_file in checkpoint_files:
        print(checkpoint_file)
        iteration = extract_number(checkpoint_file)
        print(f"\nEvaluating propensity scores from iteration {iteration}")
        
        # 临时保存当前checkpoint的logits用于评估
        checkpoint = torch.load(checkpoint_file)
        temp_propensity_path = f'temp_propensity_{iteration}.pt'
        print(type(checkpoint))
        print(checkpoint.keys())
        torch.save(checkpoint[iteration]['logits'], temp_propensity_path)
        
        # 运行推荐模型训练脚本
        cmd = [
            "python", "train_rec_with_ips.py",
            "--dataset", data,
            "--seed", str(seed),
            "--path", temp_propensity_path,
            "--epoch", "200" if data == "coat" else "4"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # 读取结果文件
            with open(f"new_results/{data}_ips_raw.txt", "r") as f:
                last_line = f.readlines()[-1].strip().split()
                
                # 解析结果
                results['iteration'].append(iteration)
                results['overall_auc'].append(float(last_line[0]))
                results['user_auc'].append(float(last_line[1]))
                results['dcg@2'].append(float(last_line[2]))
                results['dcg@4'].append(float(last_line[3]))
                results['dcg@6'].append(float(last_line[4]))
                results['recall@2'].append(float(last_line[5]))
                results['recall@4'].append(float(last_line[6]))
                results['recall@6'].append(float(last_line[7]))
                
                # 添加calibration指标
                results['nll'].append(checkpoint[iteration]['nll'])
                results['ece'].append(checkpoint[iteration]['ece'])
                
        except subprocess.CalledProcessError as e:
            print(f"Error running recommendation for iteration {iteration}: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_propensity_path):
                os.remove(temp_propensity_path)
    
    # 将结果保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(f'propensity_evaluation_results_{data}_no_ls.csv', index=False)
    
    # 打印结果摘要
    print("\nEvaluation Summary:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coat")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    results_df = evaluate_propensity_checkpoints(args.dataset, args.seed)