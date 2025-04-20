from torchinfo import summary
from SPEN.model import SPEN
from SPEN.cfg import SPEEDConfig, SPARKConfig, SPEEDplusConfig
import torch
import time

torch.set_float32_matmul_precision('high')
config = SPEEDplusConfig()
config.backbone = "efficientnet_b3"
config.neck = "BiFPN"
# config.neck_args["DensAttFPN"]["att_type"] = "SSIA"
config.head = "TokenHead"
config.head_args["TokenHead"]["learnable_token_num"] = 6
config.pos_type = "DiscreteSpher"
config.pos_loss = "CE"
config.ori_type = "DiscreteEuler"
config.ori_loss = "CE"
config.pos_args["DiscreteSpher"]["r_stride"] = 1
config.pos_args["DiscreteSpher"]["angle_stride"] = 1
config.ori_type = "DiscreteEuler"
config.ori_loss_type = "CE"
config.ori_args["DiscreteEuler"]["stride"] = 1
config.batch_size = 1
config.image_size = (480, 768)
model = SPEN(config=config)
print(model.backbone.feature_info.channels())
input_tensor = torch.randn(config.batch_size, 1, *config.image_size)
output = model(input_tensor)
result = summary(model,
                input_size=(config.batch_size, 1, *config.image_size),
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"),
                depth=3)
model_metrics = {
    "MACs(G)": result.total_mult_adds / 1e9,
    "Params(M)": result.total_params / 1e6,
}
print(model_metrics)