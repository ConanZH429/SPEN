from SPEN.data import SPEEDTrainDataset
from SPEN.vis import display_image
from SPEN.cfg import SPEEDConfig, SPEEDplusConfig
from SPEN.utils import SPEEDCamera
from scipy.io import loadmat
from SPEN.data import SPEEDTrainDataset, SPEEDplusTrainDataset
from torch.utils.data import DataLoader
# config = SPEEDConfig()
# config.image_first_size = (1200, 1920)
# config.image_size = (1200, 1920)
# config.cache = False
# config.ZAxisRotation_p = 0.0
# config.CropAndPadSafe_p = 0.0
# config.AlbumentationAug_p = 0.0
# config.CropAndPaste_p = 0.0
# config.DropBlockSafe_p = 0.0
# dataset = SPEEDTrainDataset(config)
# image_tensor, image, label = dataset[1022]

# camera = SPEEDCamera(config.image_size)

# points_path = "./result_file/tangoPoints.mat"
# points = loadmat(points_path)["tango3Dpoints"].T

# # 68
# # 9592
# display_image("img009592.jpg",
#               points=points,
#               display_label_axis=False,
#               display_point=True,
#               save_path="./result_file/test2.png",)

# display_image("img000001.jpg",
#               dataset_type="SPEED+",
#               image_type="synthetic",
#               display_box=True)

config = SPEEDConfig()
config.cache = False
config.resize_first = False
dataset = SPEEDTrainDataset(config)
train_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)
for batch in train_dataloader:
    image_tensor, image, label = batch
    image = image.squeeze().numpy()
    display_image(image=image,
                  pos=label["pos"].squeeze().numpy(),
                  ori=label["ori"].squeeze().numpy(),
                  box=label["box"].squeeze().numpy(),
                  points=label["points"].squeeze().numpy(),
                  display_points=True,
                  display_box=True)