from torchinfo import summary
from SPEN.model import SPEN
from SPEN.cfg import SPEEDConfig
from ptflops import get_model_complexity_info
import torch

config = SPEEDConfig()
config.neck = "DensAttFPN"
config.neck_args["DensAttFPN"]["att_type"] = "SSIA"
model = SPEN(config=config)
print(model.backbone.feature_info.channels())
batch_size = 1
input_tensor = torch.randn(batch_size, 1, 480, 768)
output = model(input_tensor)
result = summary(model,
                input_size=(batch_size, 1, 480, 768),
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"),
                depth=3)