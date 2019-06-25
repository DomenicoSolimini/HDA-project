import matplotlib.pyplot as plt
import h5py
import numpy as np
from PIL import Image

# dopo aver processato il file target3_126 eseguendo da terminale
# python process.py --input <path_to_dataset>/train/target5_001.hdf5

hdf = h5py.File("target3_126.hdf5",'r')
list(hdf.keys()) # ['microdoppler', 'microdoppler_thresholded', 'radar', 'range_doppler']

x =  hdf['microdoppler']
print('shape = ', x.shape, '\nclass = ', type(x), '\n type = ', x.dtype)
# il mio output è
# shape =  (179, 256)
# class =  <class 'numpy.ndarray'>
#  type =  float32

x = np.array(x)
plt.plot(x)
plt.show()
