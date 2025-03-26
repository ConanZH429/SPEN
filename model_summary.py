from torchinfo import summary
from SPEN.model import SPEN
from SPEN.cfg import SPEEDConfig
import torch
import time

torch.set_float32_matmul_precision('high')
config = SPEEDConfig()
config.backbone = "mobilenetv3_large_100"
config.neck = "DensAttFPN"
config.neck_args["DensAttFPN"]["att_type"] = "SSIA"
config.head = "TokenHead"
# config.head_args["MaxPoolHead"]["pool_size"] = (1, )
config.pos_type = "DiscreteSpher"
config.pos_loss = "CE"
config.ori_type = "DiscreteEuler"
config.ori_loss = "CE"
config.pos_args["DiscreteSpher"]["r_stride"] = 1
config.pos_args["DiscreteSpher"]["angle_stride"] = 1
config.ori_type = "DiscreteEuler"
config.ori_loss_type = "CE"
config.ori_args["DiscreteEuler"]["stride"] = 5
config.BETA = (0.2, 0.8)
now = time.time()
model = SPEN(config=config)
print(model.backbone.feature_info.channels())
batch_size = 1
input_tensor = torch.randn(batch_size, 1, 400, 640)
output = model(input_tensor)
result = summary(model,
                input_size=(batch_size, 1, 400, 640),
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"),
                depth=3)
model_metrics = {
    "MACs(G)": result.total_mult_adds / 1e9,
    "Params(M)": result.total_params / 1e6,
}
print(model_metrics)