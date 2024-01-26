import matplotlib.pyplot as plt
import numpy as np

color1 = '#f8aa1b'
color2 = '#1f8975'
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
fig, axes = plt.subplots(1, 2, figsize=(10, 3))  # 设置子图和布局


# ======================== Yelp 子图 =========================== #
time_values = [[247, 197, 235, 219], [10, 10, 14, 16]]

# 绘制Recall@20柱状图
for j, model in enumerate(models):
    axes[0].plot(ratios, time_values[j], marker=marker[j], linestyle=linestyle[j], linewidth=2,
                 label=f'{model}', color=[color1, color2][j])
    axes[0].set_ylim(0, 300)
    axes[0].set_yticks(np.arange(0, 305, 60))

axes[0].set_xticks(ratios)
axes[0].set_xticklabels([f'{ratio}%' for ratio in ratios])
axes[0].set_xlabel('PC Interactions Ratio', fontdict={'size': ratio_font_size})
axes[0].set_ylabel('Time(min)', fontdict={'size': ylabel_font_size})
axes[0].set_title(f'{datasets[0]}')
# ======================== Yelp 子图 =========================== #


# ======================== Amazon 子图 =========================== #
time_values = [[394, 414, 425, 382], [11, 14, 12, 17]]
# 绘制Recall@20柱状图
for j, model in enumerate(models):
    axes[1].plot(ratios, time_values[j], marker=marker[j], linestyle=linestyle[j], linewidth=2,
                 label=f'{model}', color=[color1, color2][j])
    axes[1].set_ylim(0, 480)
    axes[1].set_yticks(np.arange(0, 480, 70))

axes[1].set_xticks(ratios)
axes[1].set_xticklabels([f'{ratio}%' for ratio in ratios])
axes[1].set_xlabel('PC Interactions Ratio', fontdict={'size': ratio_font_size})
axes[1].set_ylabel('Time(min)', fontdict={'size': ylabel_font_size})
axes[1].set_title(f'{datasets[1]}')
# ======================== Amazon 子图 =========================== #


# 图例汇集到一起，居中显示
lines = []
labels = []
i = 0
for ax in fig.axes:
    if i == 0:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    i = i + 1
plt.subplots_adjust(wspace=0.2)
plt.legend(lines, labels, loc='upper center', ncol=2, fontsize=legend_font_size, bbox_to_anchor=(-0.13, 1.29))
plt.savefig('6-data_ratio(Time).pdf', bbox_inches='tight')
plt.show()
