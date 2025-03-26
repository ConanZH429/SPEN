import numpy as np

class SPEEDCamera:
    def __init__(self, shape):
        self.fwx = 0.0176  # focal length[m]
        self.fwy = 0.0176  # focal length[m]
        self.width = 1920  # number of horizontal[pixels]
        self.height = 1200  # number of vertical[pixels]
        self.shape = shape
        self.scale = self.height / shape[0]
        self.ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
        self.ppy = self.ppx  # vertical pixel pitch[m / pixel]
        self.fx = self.fwx / self.ppx  # horizontal focal length[pixels]
        self.fy = self.fwy / self.ppy  # vertical focal length[pixels]
        self.K_label = np.array([[self.fx, 0, self.width / 2], [0, self.fy, self.height / 2], [0, 0, 1]])
        self.K_label_inv = np.linalg.inv(self.K_label)
        self.K_image = np.array([[self.fx, 0, self.width / 2], [0, self.fy, self.height / 2], [0, 0, 1]])
        self.K_image[:2] = self.K_image[:2] / self.scale
        self.K_image_inv = np.linalg.inv(self.K_image)


class SPARKCamera:
    def __init__(self, shape):
        self.height = 1080
        self.width = 1440
        self.K_label = np.array(
            [[1745.8644618517126, 0, 737.2727957367897],
            [0, 1745.8644618517126, 528.4719595313072],
            [0, 0, 1]]
        )
        self.K_label_inv = np.linalg.inv(self.K_label)
        self.scale = self.height / shape[0]
        self.K_image = np.array(
            [[1745.8644618517126, 0, 737.2727957367897],
            [0, 1745.8644618517126, 528.4719595313072],
            [0, 0, 1]]
        )
        self.K_image[:2] = self.K_image[:2] / self.scale
        self.K_image_inv = np.linalg.inv(self.K_image)