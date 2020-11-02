import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{},".format(0)
print('cuda is available:',torch.cuda.is_available())
print('cuda device count:',torch.cuda.device_count())
print('cuda device 0 name:',torch.cuda.get_device_name(0))