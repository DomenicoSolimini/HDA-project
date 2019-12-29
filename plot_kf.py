import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from filter_def import range_angle_track, t_max, xs

sns.set()


fig, ax = plt.subplots(2)
fig.suptitle('Range and Angle')
ax[0].plot(range_angle_track[0])
ax[1].plot(range_angle_track[1])
plt.show()

plt.plot(xs.T[0])
