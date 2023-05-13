import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

mpcvit_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_acc = [92.86, 93.01, 93.21, 93.38]  # 93.36(mu=0.9)
plt.plot(mpcvit_lat, mpcvit_acc, marker='o', c='royalblue', linewidth=3, markerfacecolor='royalblue', markersize=12, label='MPCViT w/o KD (ours)')

mpcvit_kd_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_kd_acc = [93.59, 94.08, 94.22, 94.27]  # 94.26(mu=0.9)
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linewidth=3, c='crimson', marker='o', markersize=12, markerfacecolor='crimson', label='MPCViT w/ KD (ours)')

linformer_lat = [102.78, 109.32, 119.94]
linformer_acc = [92.32, 91.85, 92.33]
# plt.scatter(linformer_lat, linformer_acc, s=200, marker='v', color='green', label='Linformer')
plt.plot(linformer_lat, linformer_acc, linewidth=3, c='green', marker='^', markersize=12, markerfacecolor='green', label='Linformer')

relusoftmax_lat = [71.82]
relusoftmax_acc = [93.52]
plt.scatter(relusoftmax_lat, relusoftmax_acc, marker='v', s=200, color='purple', label='ReLU Softmax ViT')

scale_lat = [44.63]
scale_acc = [92.23]
plt.scatter(scale_lat, scale_acc, marker='*', s=200, color='grey', label='ScaleAttn ViT')

mpcformer_lat = [66.98]
mpcforer_acc = [92.93]
plt.scatter(mpcformer_lat, mpcforer_acc, marker='P', s=200, color='orange', label='MPCFormer w/o KD')

mpcformer_lat = [66.98]
mpcforer_acc = [93.97]
plt.scatter(mpcformer_lat, mpcforer_acc, marker='H', s=200, color='hotpink', label='MPCFormer w/ KD')

thex_lat = [52.39]
thex_acc = [89.66]
plt.scatter(thex_lat, thex_acc, marker='p', s=200, color='violet', label='THE-X')

softmaxscaling_lat = [91.05]
softmaxscaling_acc = [93.85]
plt.scatter(softmaxscaling_lat, softmaxscaling_acc, marker='s', s=200, color='olive', label='Softmax-ScaleAttn ViT')

softmax_lat = [112.23]
softmax_acc = [93.97]
plt.scatter(softmax_lat, softmax_acc, marker='d', s=200, color='lightgreen', label='Softmax ViT')

perlayer_lat = [48.03, 52.88, 58.66, 63.56]
perlayer_acc = [92.62, 92.83, 93.05, 93.31]
plt.plot(perlayer_lat, perlayer_acc, linewidth=3, c='pink', marker='^', markersize=12, markerfacecolor='pink', label='MPCViT w/o Per-layer Search')

headnum_lat = [51.88-1, 57.65, 64.01+2]
headnum_acc = [92.48, 92.82, 93.23]
plt.plot(headnum_lat, headnum_acc, linewidth=3, c='c', marker='^', markersize=12, markerfacecolor='c', label='ReLU Softmx ViT with different #Heads')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
# plt.legend(loc='lower center', prop={'size': 10}, ncol=2)
plt.show()
plt.savefig('./pareto_cifar10.png')