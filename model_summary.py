from torchinfo import summary
from SPEN.model import SPEN
from SPEN.cfg import SPEEDConfig, SPARKConfig
import torch
import time

torch.set_float32_matmul_precision('high')
config = SPEEDConfig()
config.backbone = "mobilenetv3_large_100"
config.neck = "TaileNeck"
# config.neck_args["DensAttFPN"]["att_type"] = "SSIA"
config.head = "TokenHead"
config.head_args["TokenHead"]["added_tokens_num"] = 6
config.pos_type = "DiscreteSpher"
config.pos_loss = "CE"
config.ori_type = "DiscreteEuler"
config.ori_loss = "CE"
config.pos_args["DiscreteSpher"]["r_stride"] = 1
config.pos_args["DiscreteSpher"]["angle_stride"] = 1
config.ori_type = "DiscreteEuler"
config.ori_loss_type = "CE"
config.ori_args["DiscreteEuler"]["stride"] = 5
model = SPEN(config=config)
print(model.backbone.feature_info.channels())
batch_size = 1
shape = (600, 960)
input_tensor = torch.randn(batch_size, 1, *shape)
output = model(input_tensor)
result = summary(model,
                input_size=(batch_size, 1, *shape),
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"),
                depth=3)
model_metrics = {
    "MACs(G)": result.total_mult_adds / 1e9,
    "Params(M)": result.total_params / 1e6,
}
print(model_metrics)