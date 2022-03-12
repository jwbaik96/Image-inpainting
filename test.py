import torch
import yaml
import os
from PIL import Image
from torchvision import transforms
from model import WrappingModule
from tensorboardX import SummaryWriter


# yaml load
with open('inpaint.yml', mode='r', encoding='utf8') as f:
    config = yaml.load(f, Loader=yaml.Loader)

# TensorboardX
writer = SummaryWriter()

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

# Data Processing
transform = transforms.ToTensor()
PIL_transform = transforms.ToPILImage('RGB')
test_data_folder = config['TEST_DATA_FOLDER']
test_mask_folder = config['TEST_MASK_FOLDER']
test_data_list = os.listdir(config['TEST_DATA_FOLDER'])
test_mask_list = os.listdir(config['TEST_MASK_FOLDER'])

for index, (img_directory, mask_directory) in enumerate(zip(test_data_list, test_mask_list)):
    img = transform(Image.open(test_data_folder + '/' + img_directory).convert('RGB')).unsqueeze(dim=0).to(device)
    mask = transform(Image.open(test_mask_folder + '/' + mask_directory).convert('L')).unsqueeze(dim=0).to(device)

    coarse_img, refine_img = Gen_net(img, mask)
    Gen_net.zero_buffer()
    masked_img = img * (1.0 - mask) + mask
    complete_img = img * (1.0 - mask) + mask * refine_img
    tensorboard_writing_imgs = torch.cat((img, masked_img, coarse_img, refine_img, complete_img))

    writer.add_images('GT, masked_img, coarse_img, refine_img, complete_img', tensorboard_writing_imgs)

    save_img = PIL_transform(complete_img.squeeze(dim=0).to(torch.device('cpu')))
    save_img.save('./test_result/{}.jpg'.format(img_directory))

writer.close()
