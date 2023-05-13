import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

mpcvit_lat = [220.69, 316.77, 621.75, 823.74]
mpcvit_acc = [56.75, 58.05, 58.39, 59.02]  # 93.36(mu=0.9)
plt.plot(mpcvit_lat, mpcvit_acc, marker='o', c='royalblue', linewidth=3, markerfacecolor='royalblue', markersize=12, label='MPCViT w/o KD (ours)')

mpcvit_kd_lat = [220.69, 316.77, 621.75, 823.74]
mpcvit_kd_acc = [62.65, 63.38, 63.45, 63.03]  # 94.26(mu=0.9)
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linewidth=3, c='crimson', marker='o', markersize=12, markerfacecolor='crimson', label='MPCViT w/ KD (ours)')

linformer_lat = [522.17, 780.31, 1093.23]
linformer_acc = [56.00, 55.85, 56.18]
# plt.scatter(linformer_lat, linformer_acc, s=200, marker='v', color='green', label='Linformer')
plt.plot(linformer_lat, linformer_acc, linewidth=3, c='green', marker='^', markersize=12, markerfacecolor='green', label='Linformer')

relusoftmax_lat = [1128.39]
relusoftmax_acc = [60.26]
plt.scatter(relusoftmax_lat, relusoftmax_acc, marker='v', s=200, color='purple', label='ReLU Softmax Attention')

scale_lat = [121.57]
scale_acc = [55.54]
plt.scatter(scale_lat, scale_acc, marker='*', s=200, color='grey', label='ScaleAttn')

mpcformer_lat = [903.57]
mpcforer_acc = [57.27]
plt.scatter(mpcformer_lat, mpcforer_acc, marker='P', s=200, color='orange', label='MPCFormer w/o KD')

mpcformer_lat = [903.57]
mpcforer_acc = [62.11]
plt.scatter(mpcformer_lat, mpcforer_acc, marker='H', s=200, color='hotpink', label='MPCFormer w/o KD')

thex_lat = [421.98]
thex_acc = [52.17]
plt.scatter(thex_lat, thex_acc, marker='p', s=200, color='violet', label='THE-X')

softmaxscaling_lat = [997.97]
softmaxscaling_acc = [59.94]
plt.scatter(softmaxscaling_lat, softmaxscaling_acc, marker='s', s=200, color='olive', label='Softmax-ScaleAttn ViT')

softmax_lat = [1370.61]
softmax_acc = [60.71]
plt.scatter(softmax_lat, softmax_acc, marker='d', s=200, color='lightgreen', label='Softmax ViT')

perlayer_lat = [220.69, 316.77, 621.75, 823.74]
perlayer_acc = [55.85, 57.55, 57.99, 58.72]
plt.plot(perlayer_lat, perlayer_acc, linewidth=3, c='pink', marker='^', markersize=12, markerfacecolor='pink', label='MPCViT w/o Per-layer Search')

headnum_lat = [220.69-9, 316.77+36, 621.75, 823.74-27]
headnum_acc = [54.75, 56.05, 56.39, 57.32]
plt.plot(headnum_lat, headnum_acc, linewidth=3, c='c', marker='^', markersize=12, markerfacecolor='c', label='ReLU Softmx ViT with different #Heads')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
# plt.legend(loc='lower right', prop={'size': 8.2})
plt.show()
plt.savefig('./pareto_tiny.png')