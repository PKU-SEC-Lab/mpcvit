import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

mpcvit_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_acc = [73.17, 74.51, 74.45, 75.38]
plt.plot(mpcvit_lat, mpcvit_acc, marker='o', linewidth=3, c='royalblue', markerfacecolor='royalblue', markersize=12, label='MPCViT w/o KD (ours)')

mpcvit_kd_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_kd_acc = [76.40, 76.93, 76.92, 77.76]
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linewidth=3, c='crimson', marker='o', markersize=12, markerfacecolor='crimson', label='MPCViT w/ KD (ours)')

linformer_lat = [102.78, 109.32, 119.94]
linformer_acc = [72.64, 73.10, 73.55]
# plt.scatter(linformer_lat, linformer_acc, s=200, marker='v', color='green', label='Linformer')
plt.plot(linformer_lat, linformer_acc, linewidth=3, c='green', marker='^', markersize=12, markerfacecolor='green', label='Linformer')

relusoftmax_lat = [72.55]
relusoftmax_acc = [75.27]
plt.scatter(relusoftmax_lat, relusoftmax_acc, marker='v', s=200, color='purple', label='ReLU Softmax Attention')

scale_lat = [45.13]
scale_acc = [73.57]
plt.scatter(scale_lat, scale_acc, marker='*', s=200, color='grey', label='ScaleAttn')

mpcformer_lat = [67.45]
mpcforer_acc = [73.98]
plt.scatter(mpcformer_lat, mpcforer_acc, marker='P', s=200, color='orange', label='MPCFormer w/o KD')

mpcformer_lat = [67.45]
mpcforer_acc = [77.01]
plt.scatter(mpcformer_lat, mpcforer_acc, marker='H', s=200, color='hotpink', label='MPCFormer w/ KD')

thex_lat = [53.24]
thex_acc = [68.73]
plt.scatter(thex_lat, thex_acc, marker='p', s=200, color='violet', label='THE-X')

softmaxscaling_lat = [91.98]
softmaxscaling_acc = [75.42]
plt.scatter(softmaxscaling_lat, softmaxscaling_acc, marker='s', s=200, color='olive', label='Softmax-ScaleAttn ViT')

softmax_lat = [112.23]
softmax_acc = [76.34]
plt.scatter(softmax_lat, softmax_acc, marker='d', s=200, color='lightgreen', label='Softmax ViT')

perlayer_lat = [48.03, 52.88, 58.66, 63.56]
perlayer_acc = [72.73, 74.27, 74.25, 75.14]
plt.plot(perlayer_lat, perlayer_acc, linewidth=3, c='pink', marker='^', markersize=12, markerfacecolor='pink', label='MPCViT w/o Per-layer Search')

headnum_lat = [51.88-1, 57.65, 64.01+2]
headnum_acc = [73.2, 73.99, 74.53]
plt.plot(headnum_lat, headnum_acc, linewidth=3, c='c', marker='^', markersize=12, markerfacecolor='c', label='ReLU Softmx ViT with different #Heads')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
# plt.legend(loc='lower center', prop={'size': 10})
plt.show()
plt.savefig('./pareto_cifar100.png')