import matplotlib.pyplot as plt
import numpy as np

color1 = '#1661ab'
color2 = 'indianred'
linestyle = [':', '--']
marker = ['o', 's']
nums = [1, 2, 3, 4]
ratios = [25, 50, 75, 100]
datasets = ['Beauty', 'Book', 'Yelp']
bar_width = 10  # 调整柱状图的宽度
ylabel_font_size = 14  # y轴标签字体大小
xlabel_font_size = 14  # x轴标签字体大小
ratio_font_size = 14  # ratio字体大小
legend_font_size = 12  # ratio字体大小
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 设置子图和布局

bar_positions = np.array(ratios)

# ======================== Beauty 子图 =========================== #
recall_values = [8.11, 9.26, 8.81, 9.07]
ndcg_values = [5.39, 6.14, 5.83, 5.96]

# 绘制Recall@10折线图
axes[0].bar(bar_positions, recall_values, width=bar_width, alpha=0.7, label='Recall@10', color=color1)
axes[0].set_ylim(8.0, 9.5)
axes[0].set_yticks(np.arange(8.0, 9.5, 0.25))
axes[0].set_xticks(ratios)
axes[0].set_xticklabels([f'{num}' for num in nums])
# axes[0].set_xlabel('# Layer', fontdict={'size': ratio_font_size})
axes[0].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})
# 创建第二个Y轴用于绘制NDCG折线图
ax2 = axes[0].twinx()
ax2.plot(ratios, ndcg_values, marker=marker[1], linestyle=linestyle[1], linewidth=2, label="NDCG@10", color=color2)
ax2.set_ylim(5.3, 6.26)
ax2.set_yticks(np.arange(5.3, 6.26, 0.16))
# ax2.set_ylabel('NDCG@10', fontdict={'size': ylabel_font_size})

# 设置标题
axes[0].set_title(f'{datasets[0]}')
# ======================== Beauty 子图 =========================== #

# ======================== Book 子图 =========================== #
recall_values = [7.81, 8.02, 8.13, 8.16]
ndcg_values = [8.83, 9.11, 9.18, 9.26]

axes[1].bar(bar_positions, recall_values, width=bar_width, alpha=0.7, label='Recall@10', color=color1)
axes[1].set_ylim(7.5, 8.5)
axes[1].set_yticks(np.arange(7.0, 8.5, 0.25))
axes[1].set_xticks(ratios)
axes[1].set_xticklabels([f'{num}' for num in nums])
axes[1].set_xlabel('# The Number of Propagation Layers', fontdict={'size': ratio_font_size})
# axes[0].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})

# axes[1].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})
# 创建第二个Y轴用于绘制NDCG折线图
ax2 = axes[1].twinx()
ax2.plot(ratios, ndcg_values, marker=marker[1], linestyle=linestyle[1], linewidth=2, label="NDCG@10", color=color2)
ax2.set_ylim(8.71, 9.29)
ax2.set_yticks(np.arange(8.7, 9.29, 0.1))
ax2.set_xticklabels(nums)
# ax2.set_ylabel('NDCG@10', fontdict={'size': ylabel_font_size})

# 设置标题
axes[1].set_title(f'{datasets[1]}')
# ======================== Book 子图 =========================== #


# ======================== Yelp 子图 =========================== #
recall_values = [5.01, 5.51, 5.23, 5.15]
ndcg_values = [5.65, 6.11, 5.83, 5.74]

# 绘制Recall@10折线图
axes[2].bar(bar_positions, recall_values, width=bar_width, alpha=0.7, label='Recall@10', color=color1)
axes[2].set_ylim(4.95, 5.55)
axes[2].set_yticks(np.arange(4.95, 5.55, 0.1))
axes[2].set_xticks(ratios)
axes[2].set_xticklabels([f'{num}' for num in nums])
# axes[2].set_xlabel('The Number of Propagation Layers', fontdict={'size': ratio_font_size})
# axes[2].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})
# 创建第二个Y轴用于绘制NDCG折线图
ax2 = axes[2].twinx()
ax2.plot(ratios, ndcg_values, marker=marker[1], linestyle=linestyle[1], linewidth=2, label="NDCG@10", color=color2)
ax2.set_ylim(5.6, 6.2)
ax2.set_yticks(np.arange(5.6, 6.19, 0.1))
ax2.set_ylabel('NDCG@10', fontdict={'size': ylabel_font_size})

# 设置标题
axes[2].set_title(f'{datasets[2]}')
# ======================== Yelp 子图 =========================== #


# 图例汇集到一起，居中显示
lines = []
labels = []
i = 0
for ax in fig.axes:
    if i == 0 or i == 5:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    i = i + 1
plt.legend(lines, labels, loc='upper center', ncol=4, fontsize=legend_font_size, bbox_to_anchor=(-0.21, 1.22))

plt.subplots_adjust(wspace=0.4)
plt.savefig('layer.pdf', bbox_inches='tight')
plt.show()
