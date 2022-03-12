import torch
import yaml
from model import WrappingModule
from gui import *
import os


# yaml load
with open('inpaint.yml', mode='r', encoding='utf8') as f:
    config = yaml.load(f, Loader=yaml.Loader)


# Device setting
if config['use_cuda']:
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_num'])
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# Model setting
Gen_net = WrappingModule(cnum=config['cnum'], device=device).to(device)
Gen_net.eval()

# Load parameters if exist
if os.path.exists(config['GEN_PARAM_PATH']):
    Gen_net.load_state_dict(torch.load(config['GEN_PARAM_PATH'], map_location=device))
    print("Generator network parameters loading complete!!!")

# GUI
Execute_gui(Gen_net, device)
