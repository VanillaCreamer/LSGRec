import matplotlib.pyplot as plt
import numpy as np

color1 = '#1661ab'
color2 = 'indianred'
linestyle = [':', '--']
marker = ['o', 's']
ratios = [20, 40, 60, 80, 100]
variants = ["$\mathbf{LSGRec}$", "$-\mathcal{L}_{BPR}^{-}$", "$-\mathcal{L}_{MSE}$", "$-\mathcal{L}_{ortho}$", "$-filter$"]
datasets = ['Beauty', 'Book', 'Yelp']
models = []
bar_width = 8  # 调整柱状图的宽度
ylabel_font_size = 14  # y轴标签字体大小
xlabel_font_size = 14  # x轴标签字体大小
ratio_font_size = 14  # ratio字体大小
legend_font_size = 12  # ratio字体大小
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 设置子图和布局


# ======================== Beauty 子图 =========================== #
recall_values = [9.26, 9.23, 9.21, 9.09, 9.17]
ndcg_values = [6.14, 6.09, 6.06, 6.04, 6.05]

# 绘制Recall@10柱状图
# bar_positions = np.array(ratios) + (j - 0.5) * 1.5 * bar_width  # 两个柱有间隙
bar_positions = np.array(ratios) + (0 - 0.5) * bar_width
axes[0].bar(bar_positions, recall_values, width=bar_width, alpha=0.7, label='Recall@10', color=color1)
axes[0].set_ylim(9.0, 9.3)
axes[0].set_yticks(np.arange(9.0, 9.27, 0.05))
axes[0].set_xticks(ratios)
axes[0].set_xticklabels(variants, rotation=-30)
axes[0].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})

ax2 = axes[0].twinx()
bar_positions = np.array(ratios) + (0 + 0.5) * bar_width
ax2.bar(bar_positions, ndcg_values, width=bar_width, alpha=0.7, label='NDCG@10', color=color2)
ax2.set_ylim(6.0, 6.18)
ax2.set_yticks(np.arange(6.0, 6.18, 0.03))
# ax2.set_ylabel('NDCG@10', fontdict={'size': ylabel_font_size})

# 设置标题
axes[0].set_title(f'{datasets[0]}')

# ======================== Book 子图 =========================== #
recall_values = [8.16, 7.88, 7.86, 7.89, 7.88]
ndcg_values = [9.26, 8.91, 8.88, 8.93, 8.93]

# 绘制Recall@10柱状图
bar_positions = np.array(ratios) + (0 - 0.5) * bar_width
axes[1].bar(bar_positions, recall_values, width=bar_width, alpha=0.7, label='Recall@10', color=color1)
axes[1].set_ylim(7.8, 8.22)
axes[1].set_yticks(np.arange(7.8, 8.2, 0.07))
axes[1].set_xticks(ratios)
axes[1].set_xticklabels(variants, rotation=-30)
axes[1].set_xlabel('Variant of LSGRec', fontdict={'size': ratio_font_size})
# axes[1].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})

ax2 = axes[1].twinx()
bar_positions = np.array(ratios) + (0 + 0.5) * bar_width
ax2.bar(bar_positions, ndcg_values, width=bar_width, alpha=0.7, label='NDCG@10', color=color2)
ax2.set_ylim(8.8, 9.34)
ax2.set_yticks(np.arange(8.8, 9.34, 0.09))
# ax2.set_ylabel('NDCG@10', fontdict={'size': ylabel_font_size})

# 设置标题
axes[1].set_title(f'{datasets[1]}')
# ======================== Amazon 子图 =========================== #

# ======================== Yelp 子图 =========================== #
recall_values = [5.51, 5.17, 5.39, 5.01, 5.04]
ndcg_values = [6.11, 5.72, 5.99, 5.55, 5.61]

bar_positions = np.array(ratios) + (0 - 0.5) * bar_width
axes[2].bar(bar_positions, recall_values, width=bar_width, alpha=0.7, label='Recall@10', color=color1)
axes[2].set_ylim(5.0, 5.6)
axes[2].set_yticks(np.arange(5.0, 5.6, 0.1))
axes[2].set_xticks(ratios)
axes[2].set_xticklabels(variants, rotation=-30)
# axes[2].set_ylabel('Recall@10', fontdict={'size': ylabel_font_size})

ax2 = axes[2].twinx()
bar_positions = np.array(ratios) + (0 + 0.5) * bar_width
ax2.bar(bar_positions, ndcg_values, width=bar_width, alpha=0.7, label='NDCG@10', color=color2)
ax2.set_ylim(5.5, 6.16)
ax2.set_yticks(np.arange(5.5, 6.16, 0.11))
ax2.set_ylabel('NDCG@10', fontdict={'size': ylabel_font_size})

axes[2].set_title(f'{datasets[2]}')

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
plt.savefig('variant.pdf', bbox_inches='tight')
plt.show()
