import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


mpcvit_kd_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_kd_acc = [93.59, 94.08, 94.22, 94.27]  # 94.26(mu=0.9)
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linestyle='-', linewidth=5, c='orange', marker='o', markersize=18, markerfacecolor='orange', markeredgewidth=3, label='MPCViT')

mpcvitplus_token_lat = [48.03-9, 52.88-9, 58.66-6.7, 63.56-9]
mpcvitplus_token_acc = [93.28, 93.92, 94.2, 94.27]
plt.plot(mpcvitplus_token_lat, mpcvitplus_token_acc, linewidth=5, linestyle='-', c='crimson', marker='^', markersize=18, markerfacecolor='crimson', markeredgewidth=3, label='Token-wise MPCViT+')

mpcvitplus_layer_lat = [48.03-11.9, 52.88-11.9, 58.66-11.9, 63.56-11.9]
mpcvitplus_layer_acc = [93.38, 93.76, 94.11, 94.26]
plt.plot(mpcvitplus_layer_lat, mpcvitplus_layer_acc, linewidth=5, linestyle='-', c='royalblue', marker='o', markersize=18, markerfacecolor='royalblue', markeredgewidth=3, label='Layer-wise MPCViT+')


plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
plt.legend(labelspacing=1, prop={'size': 16})
plt.show()
plt.savefig('./mpcvit+_cifar10_token_layer.png')