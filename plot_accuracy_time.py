#!/usr/bin/env python3
import csv
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("错误: 未找到matplotlib库")
    print("请安装matplotlib: pip3 install matplotlib")
    print("或者使用HTML版本: plot_accuracy_time.html")
    sys.exit(1)

# 读取日志文件
log_file = 'nccl_tree_bert_5_16.log'
epoch_times = []  # 每个epoch的时间
accuracies = []

with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epoch_times.append(float(row['Time']))
        accuracies.append(float(row['Train Acc']))

# 计算累计时间
cumulative_times = []
cumulative_time = 0
for epoch_time in epoch_times:
    cumulative_time += epoch_time
    cumulative_times.append(cumulative_time)

# 绘制 accuracy vs time
plt.figure(figsize=(10, 6))
plt.plot(cumulative_times, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
plt.xlabel('Cumulative Time (seconds)', fontsize=12)
plt.ylabel('Train Accuracy', fontsize=12)
plt.title('BERT Training: Accuracy vs Cumulative Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
output_file = 'nccl_tree_bert_5_16_accuracy_time.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'图表已保存到: {output_file}')

# 显示图表
plt.show()

