import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits

def affineTransform(point: np.array, Rx: np.array, Ry: np.array, Rz: np.array, translation: np.array):
    R = np.matmul(Rz, Ry)
    R = np.matmul(R, Rx)
    
    return np.matmul(R, point) + translation

def homogeneousTransform(point, Rx, Ry, Rz, translation):
    R = np.matmul(Rz, Ry)
    R = np.matmul(R, Rx)

    mat = np.concatenate((R, translation), axis=1)
    mat = np.concatenate((mat, np.array([[0, 0, 0, 1]])), axis=0)

    point_4d = np.concatenate((point, np.array([[1]])))

    return np.matmul(mat, point_4d)

def radians(angle):
    return angle * np.pi/180



# Initialization
if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    point = np.array([[10,10,10]]).T

    Rz_i = np.array([[np.cos(0),     -np.sin(0),      0           ],
                [np.sin(0),      np.cos(0),      0           ],
                [0,              0,              1           ]])

    Ry_i = np.array([[np.cos(0),      0,              np.sin(0)   ],
                [0,              1,              0           ],
                [-np.sin(0),     0,              np.cos(0)   ]])

    Rx_i = np.array([[1,              0,              0           ],
                [0,              np.cos(0),      -np.sin(0)  ],
                [0,              np.sin(0),      np.cos(0)   ]])

    translation_i = np.array([[0, 0, 0]]).T

    ax.scatter(point[0, 0], point[1, 0], point[2, 0], color='purple')


    # Affine
    # translate 20 along positive Y axis
    translation = translation_i.copy()
    translation[1, 0] = 20
    P = affineTransform(point, Rx_i, Ry_i, Rz_i, translation).T.flatten()
    print("Translate 20 along positive Y axis:\n", P,'\n')
    ax.scatter(P[0], P[1], P[2], color='red')

    # rotate point around Z axis 30 deg
    Rz = Rz_i.copy()
    Rz[0] = [np.cos(radians(30)), -np.sin(radians(30)), 0]
    Rz[1] = [np.sin(radians(30)), np.cos(radians(30)), 0]
    P = affineTransform(point, Rx_i, Ry_i, Rz, translation_i).T.flatten()
    print("Rotate point around Z axis 30 deg:\n", P, '\n')
    ax.scatter(P[0], P[1], P[2], color='black')

    # rotate point around Y axis -10 deg
    Ry = Ry_i.copy()
    Ry[0] = [np.cos(radians(-10)), 0, np.sin(radians(-10))]
    Ry[2] = [-np.sin(radians(-10)), 0, np.cos(radians(-10))]
    P = affineTransform(point, Rx_i, Ry, Rz_i, translation_i).T.flatten()
    print("Rotate point around Y axis -10 deg:\n", P, '\n')
    ax.scatter(P[0], P[1], P[2], color='blue')


    # Homogeneous
    # translate 20 along positive Y axis
    P = homogeneousTransform(point, Rx_i, Ry_i, Rz_i, translation).T.flatten()
    print("Translate 20 along positive Y axis:\n", P,'\n')
    ax.scatter(P[0], P[1], P[2], color='red')

    # rotate point around Z axis 30 deg
    P = homogeneousTransform(point, Rx_i, Ry_i, Rz, translation_i).T.flatten()
    print("Rotate point around Z axis 30 deg:\n", P, '\n')
    ax.scatter(P[0], P[1], P[2], color='black')

    # rotate point around Y axis -10 deg
    P = homogeneousTransform(point, Rx_i, Ry, Rz_i, translation_i).T.flatten()
    print("Rotate point around Y axis -10 deg:\n", P, '\n')
    ax.scatter(P[0], P[1], P[2], color='blue')

    plt.show()