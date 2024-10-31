
import os
import pathlib
import pickle
import sys

import matplotlib.pyplot as plt

home_path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
print(home_path)
if home_path not in sys.path:
    sys.path.append(home_path)

import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# setting font globally
font_path = home_path + '/fonts/Times New Roman.ttf'
font = fm.FontProperties(fname=font_path, size=12)

# move to parent path of home_path
home_path = os.path.dirname(home_path)
pathlib.Path(home_path + '/experiments/visualizations/figures/competing').mkdir(parents=True, exist_ok=True)

dataset_name = "KITTI360_18"

groups = ['CrossI2PLoc_{}'.format(dataset_name),
            'CrossI2PLoc1_4_{}'.format(dataset_name),
            'LIPLoc_{}'.format(dataset_name),
            # 'LIPLoc1_4_{}'.format(dataset_name),
            'baseline_{}'.format(dataset_name)]

labels = ['SaliencyI2PLoc(Ours)', 
            'SaliencyI2PLoc-1/4(Ours)', 
            'LIP-Loc',
            # 'LIP-Loc-1/4',
            'AE-Spherical*']

mask = [0, 1, 2, 3]# 4, 5, 6, 7, 8]
groups = [groups[m] for m in mask]
labels = [labels[m] for m in mask]
markers = ['o', 'p', 's', '^', 'o', 'p', 's', '^', 'o', 'p', 's', 'o']
dot_p = 0.3
dot_n = 1.5
line_p = 2
line_n = 1.5
linestyles = [
    'solid', (0, (1, 1.5, 1, 1.5)), (0, (1, 3, 1, 3)), (0, (3, 3)), (0, (line_p, line_n, dot_p, dot_n)), (0, (line_p, line_n, line_p, line_n, dot_p, dot_n)),
    (0, (dot_p, dot_n, dot_p, dot_n, line_p, line_n)), (0, (dot_p, dot_n)), (0, (dot_p, dot_n, dot_p, dot_n, dot_p, dot_n, line_p, line_n))
]

recalls_list = []
precisions_list = []
recalls_at_n_list = []

for index, g in enumerate(groups):
    with open(home_path + '/experiments/results/{}/{}_results.pickle'.format(g, g), 'rb') as f:
        feature = pickle.load(f)
        precisions_list.append(feature['precisions'])
        recalls_list.append(feature['recalls'])
        recalls_at_n_list.append(feature['recalls_at_n'])
# print(precisions_list[0])
# print("")
# print(recalls_list[0])
# ---------------------------------------------- PR ---------------------------------------------- #
plt.style.use('ggplot')
fig = plt.figure(figsize=(4.8, 4.8))
ax = fig.add_subplot(111)
for index in range(len(recalls_list)):
    # ax.plot(recalls_list[index], precisions_list[index], linestyle=linestyles[index], label=labels[index], dash_capstyle='round', solid_capstyle="round")
    ax.plot(recalls_list[index], precisions_list[index], linestyle=linestyles[index], label=labels[index], dash_capstyle='round', solid_capstyle="round", lw=2)
ax.set_xlabel('Recall', fontproperties=font)
ax.set_ylabel('Precision', fontproperties=font)
# https://www.coder.work/article/93874
# legend = ax.legend(loc='lower left', handlelength=2, prop=font)
# legend = ax.legend(loc='upper center', ncol=3, handlelength=1.5, bbox_to_anchor=(0.5, -0.15), borderpad=0.5, labelspacing=0.3, columnspacing=0.3, prop=font)
legend = ax.legend(loc='lower left', ncol=1, handlelength=1.8, borderpad=0.5, labelspacing=0.5, columnspacing=1, prop=font)
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none')

ax.grid('on', color='#e6e6e6')
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
ax.set_facecolor('white')
bwith = 1
ax.spines['top'].set_color('grey')
ax.spines['right'].set_color('grey')
ax.spines['bottom'].set_color('grey')
ax.spines['left'].set_color('grey')
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
ax.xaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='y', colors='black')
# ax.set_ylim([0.7, 1])
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
# ax.set_aspect(0.7 / ax.get_data_ratio(), adjustable='box')
# ax.set_aspect('equal', 'box')

plt.tight_layout()
plt.savefig(home_path + '/experiments/visualizations/figures/competing/PR_competing_{}.svg'.format(dataset_name), format="svg")
plt.savefig(home_path + '/experiments/visualizations/figures/competing/PR_competing_{}.png'.format(dataset_name), bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.savefig(home_path + '/experiments/visualizations/figures/competing/PR_competing_{}.eps'.format(dataset_name), format="eps")
pp = PdfPages(home_path + '/experiments/visualizations/figures/competing/PR_competing_{}.pdf'.format(dataset_name))
pp.savefig()
pp.close()


# ------------------------------------------- Recall@N ------------------------------------------- #
# cmap = plt.get_cmap('cubehelix')
# colors = [cmap(i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
plt.style.use('ggplot')
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
for index, recall in enumerate(recalls_at_n_list):
    ax.plot([1, 5, 10, 15, 20], [recall[1], recall[5], recall[10], recall[15], recall[20]], markersize=5, marker=markers[index], linestyle=linestyles[index], label=labels[index])
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel('N', fontproperties=font)
ax.set_ylabel('Recall@N', fontproperties=font)
# ax.set_ylim([0, 90]) # this will make the y-axis from 0-90
ax.grid('on')
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.15), prop=font)
plt.tight_layout()
plt.savefig(home_path + '/experiments/visualizations/figures/competing/recall_at_n_competing_{}.svg'.format(dataset_name), format='svg')
plt.savefig(home_path + '/experiments/visualizations/figures/competing/recall_at_n_competing_{}.png'.format(dataset_name), bbox_inches='tight', dpi=300, pad_inches=0.1)
pp = PdfPages(home_path + '/experiments/visualizations/figures/competing/recall_at_n_competing_{}.pdf'.format(dataset_name))
pp.savefig()
pp.close()