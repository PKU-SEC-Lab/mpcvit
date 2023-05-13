import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# crimson royalblue

mpcvit_kd_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_kd_acc = [76.40, 76.93, 76.92, 77.76]
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linewidth=5, linestyle='-', c='orange', marker='o', markersize=18, markerfacecolor='orange', markeredgewidth=3, label='MPCViT')

mpcvitplus_token_lat = [48.03-8, 52.88-8, 58.66-10, 63.56-6.5]
mpcvitplus_token_acc = [76.24, 76.37, 76.87, 77.21]
plt.plot(mpcvitplus_token_lat, mpcvitplus_token_acc, linewidth=5, linestyle='-', c='crimson', marker='^', markersize=18, markeredgewidth=3, markerfacecolor='crimson', label='Token-wise MPCViT+')

mpcvitplus_layer_lat = [48.03-6.4, 52.88-6.4, 58.66-6.4, 63.56-6.4]
mpcvitplus_layer_acc = [75.81, 76.19, 76.64, 77.11]
plt.plot(mpcvitplus_layer_lat, mpcvitplus_layer_acc, linewidth=5, linestyle='-', c='royalblue', marker='^', markersize=18, markeredgewidth=3, markerfacecolor='royalblue', label='Layer-wise MPCViT+')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
plt.legend(labelspacing=1, prop={'size': 16})
plt.show()
plt.savefig('./mpcvit+_cifar100_token_layer.png')