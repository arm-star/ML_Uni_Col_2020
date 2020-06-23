# Visualization and Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset

df = pd.read_csv('fire_nrt_V1_96617/fire_nrt_V1_96617.csv')

# Task 1
x = df.longitude
y = df.latitude
sizes = df.frp

# 1xN subplots
fig, ax = plt.subplots(1, 4, figsize=(10, 3),
                       subplot_kw={'xticks': (), 'yticks': ()})
ax[0].plot(x, y, 'o')
ax[1].scatter(x, y)
ax[2].scatter(x, y, c=x-y, cmap='bwr', edgecolor='k')
ax[3].scatter(x, y, c=x-y, s=sizes, cmap='bwr', edgecolor='k')

# MxN subplots
fig, ax = plt.subplots(2, 2, figsize=(8, 8))

ax[0, 0].plot(x, y)
ax[0, 0].set_ylabel('latitude')
ax[0, 1].plot(x, y, 'o')
ax[0, 1].set_ylabel('latitude')

ax[1, 0].scatter(x, y, c=x-y, cmap='bwr', edgecolor='k')
ax[1, 0].set_xlabel('longitude')
ax[1, 0].set_ylabel('latitude')
ax[1, 1].scatter(x, y, c=x-y, s=sizes, cmap='bwr', edgecolor='k')
ax[1, 1].set_xlabel('longitude')

# Compensate for Overplotting
fig, ax = plt.subplots(1, 3, figsize=(10, 4),
                       subplot_kw={'xlim': (100, 160),
                                   'ylim': (-45, -10)})
ax[0].scatter(x, y)
ax[1].scatter(x, y, alpha=.1)
ax[2].scatter(x, y, alpha=.01)

# hexgrids
plt.figure()
plt.hexbin(x, y, bins='log', extent=(100, 160, -45, -10))
plt.colorbar()
plt.axis("off")

plt.show()


# Task 2
