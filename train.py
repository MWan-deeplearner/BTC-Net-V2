import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import BTCNetV2
from utils import HSIDataset, show_title, showTrainInfo, sam, psnr, rmse


ORIGINAL_CHANNELS:   int = 172
COMPRESSED_CHANNELS: int = 27
QUANT_BIT:           int = 6
SCALE:               int = 4
NUM_FEATURES:        int = 32
GAMMA:               int = 4
BATCH_SIZE:          int = 12
DEVICE:              str = 'cuda:0'
MAX_EPOCHS:          int = 5000
VAL_HR:              int = 256
CROP_HEIGHT:         int = 128
INTERVAL:            int = 4
WIDTH:               int = 4
BANDS:               int = 172
MARGINAL:            int = 60
VALID_PERIOD:        int = 1
TRAIN_DATA_DIR:      str = "data/AVIRIS/train"
TEST_DATA_DIR:       str = "data/AVIRIS/test"
CHECKPOINT:          str = "checkpoint/BTCNetV2_8bit.pth"
MODEL_SAVE_PATH:     str = "checkpoint/BTCNetV2_8bit.pth"

train_set = HSIDataset(
    root_dir=TRAIN_DATA_DIR,
    img_size=VAL_HR,
    crop_size=CROP_HEIGHT,
    width=WIDTH,
    mode="train",
    marginal=MARGINAL,
    return_min_max=False
)
test_set = HSIDataset(
    root_dir=TRAIN_DATA_DIR,
    img_size=VAL_HR,
    width=WIDTH,
    mode="test",
    return_min_max=False
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
val_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=False)
model_name = "BTC-Net V2"
net = BTCNetV2(
    original_channels=ORIGINAL_CHANNELS,
    compressed_channels=COMPRESSED_CHANNELS,
    quant_bit=QUANT_BIT,
    scale=SCALE,
    num_features=NUM_FEATURES,
    gamma=GAMMA,
).to(DEVICE)
print(f'Using {model_name} to train.')
total_params = sum([param.numel() for param in net.parameters()])
print(f'Total params of {model_name} is {total_params}')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1, gamma=0.999)
start_epoch = 0
if start_epoch > 0:
    net.load_state_dict(torch.load(CHECKPOINT, strict=False))
    print(f'Load state dict from {CHECKPOINT}.')
max_psnr = float("-inf")
max_epoch = start_epoch
test_epoch = start_epoch
test_loss = max_loss = float("inf")
show_title()
sam_metric, rmse_metric, psnr_metric = 0.0, 0.0, 0.0
for epoch in range(start_epoch + 1, MAX_EPOCHS + start_epoch + 1):
    if epoch == start_epoch + 1:
        showTrainInfo(
            epoch=epoch,
            max_epoch=max_epoch,
            max_psnr=float(max_psnr),
            test_epoch=start_epoch,
            test_loss=test_loss,
            psnr=max_psnr
        )
    epoch_loss = 0.0
    for batch_idx, (x, _) in enumerate(train_loader):
        net.train()
        x = x.view(x.size()[0] * x.size()[1], x.size()[2], x.size()[3], x.size()[4])
        x = x.to(DEVICE).permute(0, 3, 1, 2).float()
        y = net(x)
        loss = torch.nn.functional.l1_loss(y, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % VALID_PERIOD == 0:
        test_epoch = epoch
        test_loss = epoch_loss
        torch.save(net.state_dict(), MODEL_SAVE_PATH)
        with (torch.no_grad()):
            rmses, sams, fnames, psnrs = [], [], [], []
            start_time = time.time()
            for ind2, (vx, vfn) in enumerate(val_loader):
                net.eval()
                vx = vx.view(vx.size()[0] * vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
                vx = vx.to(DEVICE).permute(0, 3, 1, 2).float()
                y = net(vx)
                val_batch_size = len(vfn)
                img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
                y = y.permute(0, 2, 3, 1).cpu().numpy()
                cnt = 0
                for bt in range(val_batch_size):
                    for z in range(0, VAL_HR, INTERVAL):
                        img[bt][:, z:z+WIDTH, :] = y[cnt]
                        cnt += 1
                    # save_path = vfn[bt].split('/')
                    # save_path = save_path[-2] + '-' + save_path[-1]
                    # np.save('Rec/%s.npy' % (save_path), y[bt])
                    GT = train_set._load_mat(vfn[bt]).astype(np.float32)
                    maxv, minv = np.max(GT), np.min(GT)
                    img[bt] = img[bt] * (maxv - minv) + minv ## De-normalization
                    sams.append(sam(GT, img[bt]))
                    rmses.append(rmse(GT, img[bt]))
                    # fnames.append(save_path)
                    psnrs.append(psnr(img[bt], GT))
            sam_metric = np.mean(sams)
            rmse_metric = np.mean(rmses)
            psnr_metric = np.mean(psnrs)
            if psnr_metric > max_psnr:
                max_psnr = psnr_metric
                max_epoch = epoch
                max_loss = epoch_loss
    showTrainInfo(
        epoch=epoch, loss=epoch_loss, test_epoch=test_epoch, sam=float(sam_metric), rmse=float(rmse_metric),
        psnr=float(psnr_metric), max_epoch=max_epoch, max_psnr=float(max_psnr), test_loss=test_loss,
        max_loss=max_loss
    )
