import numpy as np

class SPEEDCamera:
    fwx = 0.0176  # focal length[m]
    fwy = 0.0176  # focal length[m]
    width = 1920  # number of horizontal[pixels]
    height = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fx = fwx / ppx  # horizontal focal length[pixels]
    fy = fwy / ppy  # vertical focal length[pixels]
    K = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    K_inv = np.linalg.inv(K)


class SPARKCamera:
    K = np.array(
        [[1744.92206139719,0,737.272795902663],
         [0,1746.58640701753,528.471960188736],
         [0, 0, 1]]
    )
    k_inv = np.linalg.inv(K)