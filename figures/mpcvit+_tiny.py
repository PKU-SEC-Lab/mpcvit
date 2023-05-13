import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))


mpcvit_kd_lat = [220.69, 316.77, 621.75]  # 823.74
mpcvit_kd_acc = [62.65, 63.38, 63.45]  # 63.03
plt.plot(mpcvit_kd_lat, mpcvit_kd_acc, linestyle='-', linewidth=5, c='royalblue', marker='o', markersize=18, markerfacecolor='royalblue', markeredgewidth=3, label='MPCViT')

mpcvitplus_kd_lat = [220.69-17, 316.77-17, 621.75-17]  # 823.74-17
mpcvitplus_kd_acc = [62.45, 63.41, 63.53]  # 63.30
plt.plot(mpcvitplus_kd_lat, mpcvitplus_kd_acc, linewidth=5, linestyle='-', c='crimson', marker='^', markersize=18, markerfacecolor='crimson', markeredgewidth=3, label='MPCViT+')

mpcvitplus_kd_lat = [220.69-10, 316.77-10, 621.75-10]  # 823.74-10
mpcvitplus_kd_acc = [62.45, 63.41, 63.53]  # 63.30
plt.plot(mpcvitplus_kd_lat, mpcvitplus_kd_acc, linewidth=5, linestyle='--', c='orange', marker='s', markersize=18, markerfacecolor='orange', label='MPCViT+ w/o Linear Fusion')

plt.xlabel('Latency (s)')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(linestyle='--')
plt.legend(prop={'size': 16})
plt.show()
plt.savefig('./mpcvit+_tiny.png')