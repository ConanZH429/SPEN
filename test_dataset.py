from SPEN.data import SPEEDTrainDataset
from SPEN.vis import display_image
from SPEN.cfg import SPEEDConfig
config = SPEEDConfig()
config.image_size = (1200, 1920)

dataset = SPEEDTrainDataset(config)
image_tensor, image, label = dataset[100]

display_image(label["image_name"],
              image=image,
              pos=label["pos"],
              ori=label["ori"])