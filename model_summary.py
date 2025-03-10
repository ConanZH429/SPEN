from torchinfo import summary
from SPEN.model import SPEN
from SPEN.cfg import SPEEDConfig
import torch

config = SPEEDConfig()
config.backbone = "mobilenetv4_conv_medium"
config.neck = "DensAttFPN"
config.neck_args["DensAttFPN"]["att_type"] = "SSIA"
config.avg_size = (3, 3, 3)
config.pos_type = "DiscreteSpher"
config.pos_loss_type = "CE"
config.pos_args["DiscreteSpher"]["r_stride"] = 1
config.pos_args["DiscreteSpher"]["angle_stride"] = 10
config.ori_type = "DiscreteEuler"
config.ori_loss_type = "CE"
config.ori_args["DiscreteEuler"]["stride"] = 5
config.ori_args["DiscreteEuler"]["alpha"] = 0.3
config.ori_args["DiscreteEuler"]["neighbor"] = 1
config.BETA = (0.2, 0.8)
model = SPEN(config=config)
print(model.backbone.feature_info.channels())
batch_size = 1
input_tensor = torch.randn(batch_size, 1, 480, 768)
output = model(input_tensor)
result = summary(model,
                input_size=(batch_size, 1, 480, 768),
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"),
                depth=3)
print()