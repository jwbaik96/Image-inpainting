import torch
import yaml
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from model import WrappingModule, Discriminator
from losses import *
from data_process import CelebaDataset, GenerateMask
from tensorboardX import SummaryWriter
from utils import save_parameter, LowPassFilter, HighPassFilter


# yaml load
with open('inpaint.yml', mode='r', encoding='utf8') as file:
    config = yaml.load(file, Loader=yaml.Loader)

# Device setting
if config['use_cuda']:
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_num'])
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# Tensorboard
writer = SummaryWriter()

# Data processing
data_set = CelebaDataset(config)
data_loader = DataLoader(data_set, batch_size=config['batch_size'], num_workers=config['num_workers'],
                         shuffle=config['shuffle'], drop_last=config['drop_last'])
mask_generator = GenerateMask(config)
LPF = LowPassFilter(device)
HPF = HighPassFilter(device)

# Model setting
Gen_net = WrappingModule(config['cnum'], device).to(device)
Dis_net = Discriminator(device).to(device)

# Load parameters if exist
if os.path.exists(config['GEN_PARAM_PATH']):
    Gen_net.load_state_dict(torch.load(config['GEN_PARAM_PATH']))
    print("Generator network parameters loading complete!!!")
if os.path.exists(config['DIS_PARAM_PATH']):
    Dis_net.load_state_dict(torch.load(config['DIS_PARAM_PATH']))
    print("Discriminator network parameters loading complete!!!")

# Loss, Optimizer setting
# LPF loss
LP_L1_loss = L1Loss(weight=config['LP_l1_weight'])
LP_Perceptual_loss = PerceptualLoss(weight=config['LP_Perceptual_weight'], device=device)

# refine loss
L1_loss = L1Loss(weight=config['refine_l1_weight'])
HP_L1_loss = L1LossWithMask(weight=config['HP_l1_weight'])
Gen_loss = SNPatchGenLoss(weight=config['Gen_loss_weight'])
Dis_loss = SNPatchDisLoss(weight=config['Dis_loss_weight'])
VGG_loss = VGGLoss(perceptual_loss_weight=config['Perceptual_loss_weight'], style_loss_weight=config['Style_loss_weight'], device=device)
Feature_loss = FeatureLoss(weight=config['Freature_loss_weight'])
Sobel_loss = SobelEdgeLoss(weight=config['Sobel_loss_weight'], device=device)

# optimizer
Gen_optim = torch.optim.Adam(Gen_net.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
Dis_optim = torch.optim.Adam(Dis_net.parameters(), lr=4*config['learning_rate'], betas=(config['beta1'], config['beta2']))

# lr scheduler
Gen_lr_sch = StepLR(Gen_optim, step_size=int(0.01 * config['num_iter']), gamma=0.97)
Dis_lr_sch = StepLR(Dis_optim, step_size=int(0.01 * config['num_iter']), gamma=0.97)

# -------------------------------------------------------------------------
# Training
iterable_training_loader = iter(data_loader)

LP_L1_loss_saver = 0
L1_loss_saver = 0
Dx_value_saver = 0
DGz_value_saver = 0

for iteration in range(config['check_point'], config['num_iter']):
    try:
        imgs = next(iterable_training_loader).to(device)

    except StopIteration:
        iterable_training_loader = iter(data_loader)
        imgs = next(iterable_training_loader).to(device)

    mask = mask_generator.generate(batch_size=config['batch_size']).to(device)

    # ----------------------------------------------------------
    # Train Discriminator
    Dis_optim.zero_grad()

    coarse_imgs, refine_imgs = Gen_net(imgs, mask)  # Inpainting network
    complete_imgs = refine_imgs * mask + imgs * (1 - mask)

    x = torch.cat([imgs, mask], dim=1)   # real image
    Gz = torch.cat([complete_imgs, mask], dim=1)  # fake image

    x_Gz = torch.cat([x, Gz.detach()], dim=0)

    pred, _ = Dis_net(x_Gz)
    Dx, DGz = torch.chunk(pred, 2, dim=0)

    d_loss = Dis_loss(Dx, DGz)
    d_loss.backward(retain_graph=True)
    Dis_optim.step()

    # ---------------------------------------------------------------
    # Generator train
    Gen_optim.zero_grad()

    # Refine route
    DGz, fake_features = Dis_net(Gz)
    _, real_features = Dis_net(x)
    g_loss = Gen_loss(DGz)
    L1 = L1_loss(imgs, refine_imgs)
    p_loss, s_loss = VGG_loss(imgs, refine_imgs)
    f_loss = Feature_loss(real_features, fake_features)
    sb_loss = Sobel_loss(imgs, refine_imgs)
    refine_imgs = HPF(refine_imgs)
    imgs_HP = HPF(imgs)
    HP_L1 = HP_L1_loss(imgs_HP, refine_imgs, mask)

    # LPF
    coarse_imgs = LPF(coarse_imgs)
    imgs_LP = LPF(imgs)
    LP_L1 = LP_L1_loss(imgs_LP, coarse_imgs)
    LP_p_loss = LP_Perceptual_loss(imgs_LP, coarse_imgs)

    total_loss = g_loss + L1 + p_loss + s_loss + f_loss + sb_loss + LP_L1 + LP_p_loss + HP_L1
    total_loss.backward()
    Gen_optim.step()
    Gen_net.zero_buffer()

    # -----------------------------------------------------------------------------------
    LP_L1_loss_saver += LP_L1.item() / config['print_loss_iter']
    L1_loss_saver += L1.item() / config['print_loss_iter']
    Dx_value_saver += torch.mean(Dx).item() / config['print_loss_iter']
    DGz_value_saver += torch.mean(DGz).item() / config['print_loss_iter']

    Gen_lr_sch.step(epoch=iteration)
    Dis_lr_sch.step(epoch=iteration)

    writer.add_scalars('main_Loss', {'L1_loss': L1.item(), 'LP_L1_loss': LP_L1.item(), 'p_loss': p_loss.item(), 'LP_p_loss': LP_p_loss.item(), 's_loss': s_loss.item(),
                                     'f_loss': f_loss.item(), 'sb_loss': sb_loss.item(), 'HP_L1_loss': HP_L1.item()}, global_step=iteration)
    writer.add_scalars('GAN', {'Dx': torch.mean(Dx).item(), 'DGz': torch.mean(DGz).item()}, global_step=iteration)

    if (iteration + 1) % config['print_loss_iter'] == 0:
        print("iteration: {}/{},  L1_loss: {}, LP_L1_loss: {},  Dx: {},  DGz: {}".format(iteration + 1, config['num_iter'],
                                                                L1_loss_saver, LP_L1_loss_saver, Dx_value_saver, DGz_value_saver))
        LP_L1_loss_saver = 0
        L1_loss_saver = 0
        Dx_value_saver = 0
        DGz_value_saver = 0

    if (iteration + 1) % config['save_parameter_iter'] == 0:
        save_parameter(Gen_net.state_dict(), Dis_net.state_dict(), config, iteration)

print("Training Complete!!!")
save_parameter(Gen_net.state_dict(), Dis_net.state_dict(), config, epoch=None)
writer.close()


