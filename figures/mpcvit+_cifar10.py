import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


mpcvit_kd_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_kd_acc = [93.59, 94.08, 94.22, 94.27]  # 94.26(mu=0.9)
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linestyle='-', linewidth=5, c='royalblue', marker='o', markersize=18, markerfacecolor='royalblue', markeredgewidth=3, label='MPCViT')

mpcvitplus_kd_lat = [48.03-9, 52.88-9, 58.66-6.7, 63.56-9, 63.56]
mpcvitplus_kd_acc = [93.28, 93.92, 94.2, 94.27, 94.27]
plt.plot(mpcvitplus_kd_lat, mpcvitplus_kd_acc, linewidth=5, linestyle='-', c='crimson', marker='^', markersize=18, markerfacecolor='crimson', markeredgewidth=3, label='MPCViT+')

mpcvitplus_kd_lat = [48.03-6, 52.88-6, 58.66-3.7, 63.56-6]
mpcvitplus_kd_acc = [93.28, 93.92, 94.2, 94.27]  
plt.plot(mpcvitplus_kd_lat, mpcvitplus_kd_acc, linewidth=5, linestyle='--', c='orange', marker='s', markersize=18, markerfacecolor='orange', markeredgewidth=3, label='MPCViT+ w/o Linear Fusion')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
plt.legend(labelspacing=1, prop={'size': 16})
plt.show()
plt.savefig('./mpcvit+_cifar10.png')