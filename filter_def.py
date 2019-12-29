# DEFINING EKF
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from numpy import eye, array, asarray
from math import sqrt, atan

# load data from file
range_angle_track = np.load('range_angle_track.npy').T
t_max = range_angle_track.shape[1]

def HJacobian_at(x):
    """ compute Jacobian of H matrix at state x """
    C11 = np.divide(x[0], np.sqrt(x[0]**2+x[3]**2))
    C12 = np.divide(x[3], np.sqrt(x[0]**2+x[3]**2))
    C21 = np.divide(1, x[0]*(1+np.divide(x[3],x[0])**2))
    C22 = -np.divide(x[3], x[0]**2*(1+np.divide(x[3],x[0])**2))

    return array ([[C11*x[0], 0, C12*x[3], 0],
                   [C21*x[0], 0, C22*x[3], 0]])



def hx(x):
    """ compute measurement for range and angle that
        would correspond to state x.
    """
    range = sqrt(x[0]**2 + x[2]**2)
    angle = atan(x[1]/x[3])
    return (range, angle)



dt = 1/15
# define the extended kalman filter object
rk = ExtendedKalmanFilter(dim_x=4, dim_z=2)

# make a starting point guess
components = np.array([np.cos(range_angle_track[0][0]), np.sin(range_angle_track[1][0])])
# starting radial position and velocity are projected on the two axis
pos_0 = range_angle_track[0][0]*components
vel_0 = (range_angle_track[0][1]-range_angle_track[0][0])/dt * components
# initial state vector in the form (x, v_x, y, v_y)
rk.x = np.array([pos_0[0], vel_0[0], pos_0[1], vel_0[1]])

# transition matrix
rk.F = np.array([[1, dt, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, dt],
                 [0, 0, 0, 1]])

# process noise covariance matrix
sigma_Q = 1
rk.Q = np.array([[.25*dt**4, .5*dt**3, 0,          0       ],
                 [.5*dt**2,  dt*2,     0,          0       ],
                 [0,         0,        .25*dt**4,  .5*dt**3],
                 [0,         0,        .5*dt**2,   dt*2    ]]) * sigma_Q

# measurement noise covariance matrix
range_std = (np.zeros(2)+1) * 0.1 # meters
rk.R = np.diag(range_std**2)

rk.P *= .5 # how to tune this value?

xs = [] # stores the predicted values in cartesian space
for i in range(t_max):
    z = range_angle_track.T[i] # i-th measure
    rk.update(z, HJacobian_at, hx)
    xs.append(rk.x)
    rk.predict()

xs = asarray(xs)
