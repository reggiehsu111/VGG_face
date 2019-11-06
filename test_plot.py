import matplotlib.pyplot as plt

loss_arr = [(0,1),(1,9),(2,4)]
acc_arr = [(0,4),(1,5),(2,8)]
print(*zip(*loss_arr))
plt.figure(figsize=(5, 5))

plt.plot(*zip(*loss_arr))
plt.savefig('loss.png')
plt.clf()
plt.plot(*zip(*acc_arr))
plt.savefig('acc.png')
