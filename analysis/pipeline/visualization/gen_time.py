import matplotlib.pyplot as plt
import numpy as np

plt.clf()
plt.figure(figsize=(10, 8))

scale = 1e7

x = np.array([1000000, 3000000, 5000000, 7000000, 10000000])/scale
y = [2.623, 8.252, 13.945, 19.692, 27.697] 

plt.plot(x, y, marker='x', color='crimson',markersize=20, markeredgecolor='black')

#plt.xscale('log')
#plt.yscale('log')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.ylabel('Time [s]',fontsize=20)
plt.xlabel('Number of samples (1e7)',fontsize=20)
plt.grid()
#plt.show()
plt.savefig('gen_time.pdf')

