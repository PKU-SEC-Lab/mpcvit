import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# crimson royalblue

mpcvit_kd_lat = [48.03, 52.88, 58.66, 63.56]
mpcvit_kd_acc = [76.40, 76.93, 76.92, 77.76]
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linewidth=5, linestyle='-', c='royalblue', marker='o', markersize=18, markerfacecolor='royalblue', markeredgewidth=3, label='MPCViT')

mpcvitplus_kd_lat = [48.03-8, 52.88-8, 58.66-10, 63.56-6.5, 63.56]
mpcvitplus_kd_acc = [76.24, 76.37, 76.87, 77.21, 77.76]
plt.plot(mpcvitplus_kd_lat, mpcvitplus_kd_acc, linewidth=5, linestyle='-', c='crimson', marker='^', markersize=18, markeredgewidth=3, markerfacecolor='crimson', label='MPCViT+')

mpcvitplus_kd_lat = [48.03-5, 52.88-5, 58.66-7, 63.56-3.5]
mpcvitplus_kd_acc = [76.24, 76.37, 76.87, 77.17]  
plt.plot(mpcvitplus_kd_lat, mpcvitplus_kd_acc, linewidth=5, linestyle='--', c='orange', marker='s', markersize=18, markerfacecolor='orange', markeredgewidth=3, label='MPCViT+ w/o Linear Fusion')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
plt.legend(labelspacing=1, prop={'size': 16})
plt.show()
plt.savefig('./mpcvit+_cifar100.png')