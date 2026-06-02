import torch
from torch.utils.data import DataLoader

from model import BTCNetV2
from utils import HSIDataset, EntropyCodex


ORIGINAL_CHANNELS: int = 172
COMPRESSED_CHANNELS: int = 27
QUANT_BIT: int = 8
SCALE: int = 4
NUM_FEATURES: int = 32
GAMMA: int = 4
CHECKPOINT: str = "checkpoint/BTCNetV2_8bit.pth"
DATA_DIR: str = "data/AVIRIS/test"
IMAGE_SIZE: int = 256
CROP_HEIGHT: int = 128
CROP_WIDTH: int = 4
DEVICE: str = "cuda:0"

model = BTCNetV2(
    ORIGINAL_CHANNELS, COMPRESSED_CHANNELS, QUANT_BIT, SCALE, NUM_FEATURES, GAMMA
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT), strict=True)
params = 0
params += sum([param.numel() for param in model.encoder.weight]) * QUANT_BIT
params += sum([param.numel() for param in model.encoder.bias]) * QUANT_BIT
params += model.encoder.p_relu.weight.numel() * 32
params += 2 * 2 * 32
print("Encoder total params: ", params)
val_set = HSIDataset(
    root_dir=DATA_DIR,
    img_size=IMAGE_SIZE,
    crop_size=CROP_HEIGHT,
    width=CROP_WIDTH,
    mode="test",
    return_min_max=True,
)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=False)
for ind2, (vx, vfn, min_x, max_x) in enumerate(val_loader):
    vx = vx.view(vx.size()[0] * vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
    vx = vx.to(DEVICE).permute(0, 3, 1, 2).float()
    max_x = max_x.item()
    min_x = min_x.item()
    print(vx.shape)
    coder = EntropyCodex(quantize_bit=QUANT_BIT, device=DEVICE)
    coder.set_model(model)
    restored, bpp, psnr, sam, rmse = coder.run(vx, norm_max=max_x, norm_min=min_x)
    print(f"bpp, psnr, sam, rmse: \n"
          f"{bpp:.12f} \n"
          f"{psnr:.12f} \n"
          f"{sam:.12f} \n"
          f"{rmse:.12f}")