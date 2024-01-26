import matplotlib.pyplot as plt
import numpy as np

color1 = '#63B2EE'
color2 = '#F89588'
linestyle = [':', '--']
marker = ['o', 's']
ratios = [25, 50, 75, 100]
datasets = ['Yelp2021', 'Amazon-Book']
models = ['RecEraser', 'GFEraser']
bar_width = 6  # 调整柱状图的宽度
ylabel_font_size = 14  # y轴标签字体大小
xlabel_font_size = 14  # x轴标签字体大小
ratio_font_size = 14  # ratio字体大小
legend_font_size = 12  # ratio字体大小
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 设置子图和布局


# ======================== Yelp 子图 =========================== #
rdr_values = [[2.2917, 3.3282, 3.5531, 5.2533], [4.1052, 5.8668, 6.3343, 7.0149]]
recall_values = [[0.0747, 0.0734, 0.0726, 0.0762], [0.1167, 0.1143, 0.1025, 0.1073]]

# 绘制Recall@20柱状图
for j, model in enumerate(models):
    # bar_positions = np.array(ratios) + (j - 0.5) * 1.5 * bar_width  # 两个柱有间隙
    bar_positions = np.array(ratios) + (j - 0.5) * bar_width
    axes[0].bar(bar_positions, recall_values[j], width=bar_width, alpha=0.7,
                label=f'{model} (Recall@20)', color=['#eacdd0', '#a39aae'][j])
    axes[0].set_ylim(0.06, 0.135)
    axes[0].set_yticks(np.arange(0.05, 0.135, 0.015))
axes[0].set_xticks(ratios)
axes[0].set_xticklabels([f'{ratio}%' for ratio in ratios])
axes[0].set_xlabel('PC Interactions Ratio', fontdict={'size': ratio_font_size})
axes[0].set_ylabel('Recall@20', fontdict={'size': ylabel_font_size})

# 创建第二个Y轴用于绘制RDR折线图
ax2 = axes[0].twinx()
for j, model in enumerate(models):
    ax2.plot(ratios, rdr_values[j], marker=marker[j], linestyle=linestyle[j], linewidth=2,
             label=f'{model} (RDR)', color=[color1, color2][j])
    ax2.set_ylim(1.5, 10)
    ax2.set_yticks(np.arange(1.5, 10, 1.5))
ax2.set_ylabel('RDR', fontdict={'size': ylabel_font_size})

# 设置标题
axes[0].set_title(f'{datasets[0]}')
# ======================== Yelp 子图 =========================== #


# ======================== Amazon 子图 =========================== #
recall_values = [[0.0855, 0.0831, 0.0845, 0.0874], [0.1403, 0.1358, 0.1340, 0.1322]]
rdr_values = [[6.0007, 7.2687, 7.8827, 11.2257], [8.2324, 8.8665, 9.1109, 13.1252]]

# 绘制Recall@20柱状图
for j, model in enumerate(models):
    # bar_positions = np.array(ratios) + (j - 0.5) * 1.5 * bar_width  # 两个柱有间隙
    bar_positions = np.array(ratios) + (j - 0.5) * bar_width
    axes[1].bar(bar_positions, recall_values[j], width=bar_width, alpha=0.7,
                label=f'{model} (Recall@20)', color=['#eacdd0', '#a39aae'][j])
    axes[1].set_ylim(0.06, 0.16)
    axes[1].set_yticks(np.arange(0.06, 0.16, 0.015))
axes[1].set_xticks(ratios)
axes[1].set_xticklabels([f'{ratio}%' for ratio in ratios])
axes[1].set_xlabel('PC Interactions Ratio', fontdict={'size': ratio_font_size})
axes[1].set_ylabel('Recall@20', fontdict={'size': ylabel_font_size})

# 创建第二个Y轴用于绘制RDR折线图
ax2 = axes[1].twinx()
for j, model in enumerate(models):
    ax2.plot(ratios, rdr_values[j], marker=marker[j], linestyle=linestyle[j], linewidth=2,
             label=f'{model} (RDR)', color=[color1, color2][j])
    ax2.set_ylim(5, 15)
    ax2.set_yticks(np.arange(5, 15, 1.5))
ax2.set_ylabel('RDR', fontdict={'size': ylabel_font_size})

# 设置标题
axes[1].set_title(f'{datasets[1]}')
# ======================== Amazon 子图 =========================== #


# 图例汇集到一起，居中显示
lines = []
labels = []
i = 0
for ax in fig.axes:
    if i == 0 or i == 2:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    i = i + 1
plt.legend(lines, labels, loc='upper center', ncol=4, fontsize=legend_font_size, bbox_to_anchor=(-0.21, 1.22))

plt.subplots_adjust(wspace=0.4)
plt.savefig('5-data_ratio.pdf', bbox_inches='tight')
plt.show()
