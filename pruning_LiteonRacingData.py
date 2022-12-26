# -*- coding: utf-8 -*-
"""pruning_all_2022_06_24.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hgBe-YxxSncGIFCvIYYgqeK4svg2_Dnw
"""

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
#from PIL import Image
from torchvision import models
import random
#import sklearn
from os import listdir
import cv2
#from collections import Counter
from torch import nn
import time

from torch.utils.data import Dataset
#from torchvision import transforms
#import os
#from torchstat import stat
#import copy
#import math
#import torchvision
#from random import sample

#global variable
dict_L1_norm = {}
mean_L1_norm = {}
std_L1_norm = {}
dict_L2_norm = {}
mean_L2_norm = {}
std_L2_norm = {}
APoZ_dict = {}
mean_APoZ = {}
std_APoZ = {}
apoz_eleCount = {}

model_choose = 3  #1 => resnet50 , 2 => resnet101 , 3 => resnet152
criteria_choose = 2  #1 => L1 , 2 => L2 , 3 => APoZ
regularization_choose = 2 #1 => lasso , 2 => ridge , 3 => group_lasso
path = 'training_record/resnet101_L2_Ridge_0703_3epoch.txt'  # 輸出寫成txt
weight_save_path = '/home/xxx/Model/BGR/20220703/pruned/resnet101_L2_Ridge_3epoch_weight_0703.pt'
model_save_path = '/home/xxx/Model/BGR/20220703/pruned/resnet101_L2_Ridge_3epoch_0703.pt'

model_out_classnum = 8

load_all_seed = int(6)
load_from_gen_seed = int(8)


device = torch.device("cuda")

dict_name_modules = {}
dict_name_parameters = {}

f = open(path, 'w')


def dict_parameters(model):
  global dict_name_parameters
  for name,parameters in model.named_parameters():
    dict_name_parameters[name] = parameters


def dict_modules(model):
  global dict_name_modules
  for name,module in model.named_modules():
    if 'conv' in name:
      dict_name_modules[name] = module

"""# load model"""

def load_model():
  global model_choose
  if model_choose == 1:
    model = models.resnet50(pretrained=False)  #change
    fc_feature = model.fc.in_features    #change
    model.fc = torch.nn.Linear(fc_feature,model_out_classnum)  #change
    model.load_state_dict(torch.load('/home/xxx/Model/BGR/20220703/unpruned/A2B_resnet50_finetune_0703_40epoch.pt')) # ResNet50 nofinetune

  elif model_choose == 2:
    model = models.resnet101(pretrained=False)
    fc_feature = model.fc.in_features    #change
    model.fc = torch.nn.Linear(fc_feature,model_out_classnum)  #change
    model.load_state_dict(torch.load('/home/xxx/Model/BGR/20220703/unpruned/A2B_resnet101_finetune_0703_30epoch.pt'))

  else:
    model = models.resnet152(pretrained=False)
    fc_feature = model.fc.in_features    #change
    model.fc = torch.nn.Linear(fc_feature,model_out_classnum)  #change
    model.load_state_dict(torch.load('/home/xxx/Model/BGR/20220703/unpruned/A2B_resnet152_finetune_0703_40epoch.pt'))

  return model

"""# pruning criteria"""

def criteria_select(mode, model, validation_dataset = None):
  global dict_L1_norm, dict_L2_norm, APoZ_dict, mean_L1_norm, mean_L2_norm, mean_APoZ, std_L1_norm, std_L2_norm, std_APoZ
  if mode == 1:
    dict_L1_norm  = L1_norm_criteria(model)
    mean_L1_norm = mean(mean_L1_norm, dict_L1_norm)
    std_L1_norm = std(std_L1_norm, mean_L1_norm, dict_L1_norm)
    return dict_L1_norm, mean_L1_norm, std_L1_norm
  elif mode == 2:
    dict_L2_norm  = L2_norm_criteria(model)
    mean_L2_norm = mean(mean_L2_norm, dict_L2_norm)
    std_L2_norm = std(std_L2_norm, mean_L2_norm, dict_L2_norm)

    return dict_L2_norm, mean_L2_norm, std_L2_norm
  elif mode == 3:
    APoZ_dict  = APoZ(model)
    mean_APoZ = mean(mean_APoZ, APoZ_dict)
    std_APoZ = std(std_APoZ, mean_APoZ, APoZ_dict)

    return APoZ_dict, mean_APoZ, std_APoZ


  return None, None, None

"""L1_norm"""

def L1_norm_criteria(model):
  global dict_L1_norm
  for name,parameters in model.named_parameters():
    if "conv" in name:
      temp = abs(parameters).sum(dim = 3).sum(dim = 2).sum(dim=1)
      dict_L1_norm[name[:-7]] = torch.div((temp*1000).type(torch.int64),torch.numel(parameters[0])).type(torch.int64)

  return dict_L1_norm

"""L2_norm"""

def L2_norm_criteria(model):
  global dict_L2_norm
  for name,parameters in model.named_parameters():
    if "conv" in name:
      temp = parameters.pow(2.0).sum(dim = 3).sum(dim = 2).sum(dim=1)
      dict_L2_norm[name[:-7]] = torch.div((temp*1000000).type(torch.int64),torch.numel(parameters[0])).type(torch.int64)
  return dict_L2_norm

"""APoZ"""
def _get_module(model, submodule_key):
  tokens = submodule_key.split('.')
  sub_tokens = tokens[:-1]
  cur_mod = model
  for s in sub_tokens:
      cur_mod = getattr(cur_mod, s)
  return getattr(cur_mod, tokens[-1])

def filter_0_percent(temp):
  global device
  temp = temp.squeeze()
  temp = temp.permute(1,0,2,3)
  apoz = torch.zeros(temp.shape[0],dtype = torch.float32).to(device = torch.device('cuda'))
  for i in range(temp.shape[0]):
    temp[i] = F.relu(temp[i])
    compare = torch.zeros((temp.shape[1], temp.shape[2],temp.shape[3]),dtype = torch.float32).to(device = torch.device('cuda'))
    apoz[i] += torch.eq(temp[i],compare).sum(dim=2).sum(dim = 1).sum(dim = 0).item()

  return apoz / (temp.shape[1]*temp.shape[2]*temp.shape[3])

def APoZ(model, input):
  features = []
  def hook(module, input, output):
    features.append(output.clone().detach())
    del output

  global dict_name_modules, APoZ_dict
  APoZ_dict = {}

  handle = []
  for name, modules in dict_name_modules.items():
    handle.append(modules.register_forward_hook(hook))

  model = model.to(device=torch.device('cuda')) # add
  input = input.to(device=torch.device('cuda'))
  y = model(input)

  for i in range(len(handle)):
    handle[i].remove()

  temp = list(dict_name_modules)

  for i in range(len(features)):
    APoZ_dict[temp[i]] = filter_0_percent(features[i])
  
  del features
  del handle

  torch.cuda.empty_cache()

"""mean"""

def mean(mean_dict, criteria_dict):
  for name in criteria_dict.keys():
    a = criteria_dict[name]
    temp = torch.div(torch.bincount(a),torch.sum(torch.bincount(a)))
    for i in range(temp.shape[0]):
      temp[i] = temp[i]*i

    mean_dict[name] = int(torch.sum(temp).item())
  return mean_dict

"""std"""

def std(std_dict, mean_dict, criteria_dict):
  for name in criteria_dict.keys():
    
    for i in range(len(criteria_dict[name])):
      if (i != 0):
        std_dict[name] += (criteria_dict[name][i] - mean_dict[name]) * (criteria_dict[name][i] - mean_dict[name])
      else:
        std_dict[name] = (criteria_dict[name][i] - mean_dict[name]) * (criteria_dict[name][i] - mean_dict[name])
    std_dict[name] = int(pow(std_dict[name]/len(criteria_dict[name]), 0.5).item())
  return std_dict

"""# pruning

pruning module
"""

def _set_module(model, submodule_key, module):
  tokens = submodule_key.split('.')
  sub_tokens = tokens[:-1]
  cur_mod = model
  for s in sub_tokens:
      cur_mod = getattr(cur_mod, s)
  setattr(cur_mod, tokens[-1], module)

"""make mask"""

#True->will remove  -7 => remove.weight
def filter_mask(mode, layerName):
  global dict_L1_norm, mean_L1_norm, std_L1_norm, dict_L2_norm, mean_L2_norm, std_L2_norm, APoZ_dict, mean_APoZ, std_APoZ, device

  if mode == 1:  #L1-norm
    mask = torch.le(dict_L1_norm[layerName[:-7]], mean_L1_norm[layerName[:-7]] - std_L1_norm[layerName[:-7]])
  elif mode == 2: #L2-norm
    mask = torch.le(dict_L2_norm[layerName[:-7]], mean_L2_norm[layerName[:-7]] - std_L2_norm[layerName[:-7]])
  else:       #APoZ
    mask = torch.ge(APoZ_dict[layerName[:-7]], mean_APoZ[layerName[:-7]] - std_APoZ[layerName[:-7]])

  return mask

def fill_value_in_new_layer(model, mask, layerName, dim):   
  global device
  temp = model.state_dict()[layerName] #####

  if "conv" in layerName and "weight" in layerName:

    if dim == 1:
      temp = torch.permute(temp, (1,0,2,3))

    for i in range(temp.shape[0]):
      if mask[i]:
        temp[i] = 0.0

    j = 0      
    for i in range(temp.shape[0]):
      total = torch.eq(temp[j], 0.0).sum(dim=2).sum(dim=1).sum(dim=0)
      
      if total.item() == temp.shape[1]*temp.shape[2]*temp.shape[3]:
        temp = temp[torch.arange(temp.size(0)) != j]
        j -= 1

      j += 1

    if dim == 1:
      temp = torch.permute(temp, (1,0,2,3))

  else:
    j = 0
    for i in range(temp.shape[0]):
      if mask[i]:
        temp = temp[torch.arange(temp.size(0)) != j]
        j -= 1

      j += 1

  return temp

def make_new_layer(model, mode, layerName, layerName_bias, next_layerName, layer_bn_name, layer_bn_name_bias):
  global device, dict_name_modules

  mask = filter_mask(mode, layerName)

  dle_filter_ele_count = mask.sum(dim=0)
  if dle_filter_ele_count< 20:
    return None

  layer_module = dict_name_modules[layerName[:-7]]
  next_layer_module = dict_name_modules[next_layerName[:-7]]

  buffer = []

  buffer.append(fill_value_in_new_layer(model, mask, layerName, 0))
  buffer.append(fill_value_in_new_layer(model, mask, next_layerName, 1))
  buffer.append(fill_value_in_new_layer(model, mask, layer_bn_name, 0))
  buffer.append(fill_value_in_new_layer(model, mask, layer_bn_name_bias, 0))


  #conv
  if layer_module.kernel_size == (3,3):
    _set_module(model, layerName[:-7], nn.Conv2d(in_channels = layer_module.in_channels, out_channels = (layer_module.out_channels - dle_filter_ele_count), kernel_size = layer_module.kernel_size, stride = layer_module.stride, padding = layer_module.padding, bias=False))
  elif layer_module.kernel_size == (1,1):
    _set_module(model, layerName[:-7], nn.Conv2d(in_channels = layer_module.in_channels, out_channels = (layer_module.out_channels - dle_filter_ele_count), kernel_size = layer_module.kernel_size, stride = layer_module.stride, bias=False))

  #next_conv
  if next_layer_module.kernel_size == (3,3):
    _set_module(model, next_layerName[:-7], nn.Conv2d(in_channels = (next_layer_module.in_channels - dle_filter_ele_count), out_channels = next_layer_module.out_channels, kernel_size = next_layer_module.kernel_size, stride = next_layer_module.stride, padding = next_layer_module.padding, bias=False))
  elif next_layer_module.kernel_size == (1,1):
    _set_module(model, next_layerName[:-7], nn.Conv2d(in_channels = (next_layer_module.in_channels - dle_filter_ele_count), out_channels = next_layer_module.out_channels, kernel_size = next_layer_module.kernel_size, stride = next_layer_module.stride, bias=False))

  #bn
  _set_module(model, layer_bn_name[:-7], nn.BatchNorm2d(num_features = (layer_module.out_channels - dle_filter_ele_count), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))


  #------------------
  model.state_dict()[layerName].copy_(buffer[0])

  model.state_dict()[next_layerName].copy_(buffer[1])
  
  model.state_dict()[layer_bn_name].copy_(buffer[2])

  model.state_dict()[layer_bn_name_bias].copy_(buffer[3])

  return model

"""
# model training function"""

def del_tensor_ele(arr,index1,index2, train_len):
  arr1 = arr[0:index1]
  arr2 = arr[index2:train_len]
  return torch.cat((arr1,arr2),dim = 0)

def loss_regularization(r_loss, model, mode):
  regularization_loss = 0
  if mode == 0:   #no regularization
    return r_loss
  elif mode == 1:  #LASSO
    for param in model.parameters():
      regularization_loss += torch.sum(abs(param))

    return r_loss + 0.0001 * regularization_loss
  elif mode == 2:  #Ridge
    for param in model.parameters():
      regularization_loss += torch.sqrt(torch.sum(param.pow(2.0)))

    return r_loss + 0.0001 * regularization_loss
  else:        #Group LASSO
    regularization_lasso = 0
    for param in model.parameters():
      regularization_lasso += torch.sum(abs(param))
    
    regularization_group_filter = 0
    regularization_group_channel = 0
    for name,parameters in model.named_parameters():
      if "conv" in name:
        a = parameters.pow(2.0).sum(dim = 3).sum(dim = 2)
        regularization_group_filter += pow(3,0.5) * a.sum(dim =1).pow(0.5).sum(dim=0)
        regularization_group_channel += pow(2,0.5) * a.pow(0.5).sum(dim =1).sum(dim=0)

    return r_loss + 0.001 * regularization_lasso + 0.0001 * regularization_group_filter + 0.001 * regularization_group_channel

def validate(model, train_dataset, val_dataset, loss_fn):
  train_loader = torch.utils.data.DataLoader(train_dataset , batch_size = 16,shuffle = False,num_workers=1, pin_memory=True, drop_last=True)
  val_loader = torch.utils.data.DataLoader(val_dataset , batch_size = 16,shuffle = False,num_workers=1, pin_memory=True, drop_last=True)

  correct = 0
  total = 0
  with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(device = device)
        labels = labels.to(device = device)
        outputs = model(imgs)
        predicted = torch.max(outputs,dim = 1)
        total += labels.shape[0]
        correct += int((predicted.indices == labels).sum())

  val_correct = 0
  val_total = 0
  val_loss_train = 0.0
  with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device = device)
        labels = labels.to(device = device)
        val_outputs = model(imgs)
        val_loss = loss_fn(val_outputs,labels)
        val_loss_train += val_loss.item()
        val_predicted = torch.max(val_outputs,dim = 1)
        val_total += labels.shape[0]
        val_correct += int((val_predicted.indices == labels).sum())

  return correct/total, val_correct/val_total, val_loss_train/len(val_loader)

def training_loop(n_epochs,optimizer,model,loss_fn,train_dataset,validation_dataset, mode):
  
  for epoch in range(0,n_epochs):
    train_loader = torch.utils.data.DataLoader(train_dataset[epoch%10],batch_size = 16,shuffle= True,num_workers=1, pin_memory=True, drop_last=True)
    loss_train = 0.0
    count = 1
    model.train()  #change
    for imgs,labels in train_loader:
      imgs = imgs.to(device = device) 
      labels = labels.to(device = device) 
      r_loss = loss_fn(outputs,labels)
      
      loss = loss_regularization(r_loss, model, mode)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_train += r_loss.item()
      
    
    model.eval()  #change
    correct, val_correct, val_loss = validate(model, train_dataset[epoch%10], validation_dataset[epoch%10], loss_fn)

    print('Epoch {}, Training loss {}, Training accuracy {}'.format(epoch + 1,loss_train/len(train_loader), correct))
    print('    , Validation loss {}, Validation accuracy {}\n'.format(val_loss,val_correct))
    
    f.write('Epoch '+str(epoch + 1)+' Training loss '+str(loss_train/len(train_loader))+' Training accuracy '+str(correct)+'\n')
    f.write('    , Validation loss '+str(val_loss)+', Validation accuracy '+str(val_correct)+'\n\n')

"""# inference for testing"""

# test
def test(copy_model, test_dataset):
  total_loss = 0.0
  total_accuracy = 0.0

  error_label_counetr = [0]*8
  TN=0
  FN=0
  TP=0
  FP=0

  copy_model.eval()

  test_total = 0
  test_loss_train = 0.
  test_correct = 0
  test_loader = torch.utils.data.DataLoader(test_dataset , batch_size = 1,shuffle = False,num_workers=1, pin_memory=True, drop_last=True)
  loss_fn = torch.nn.CrossEntropyLoss()
  test_start_time = time.time()

  with torch.no_grad():
    for imgs, labels in test_loader:
      imgs = imgs.to(device = device)
      labels = labels.to(device = device)
      test_outputs = copy_model(imgs)  #CHANGE
      test_loss = loss_fn(test_outputs,labels)
      test_loss_train += test_loss.item()
      test_predicted = torch.max(test_outputs,dim = 1) 
      test_total += labels.shape[0]
      test_correct += int((test_predicted.indices == labels).sum())
      
    total_accuracy += test_correct/test_total

  test_end_time = time.time()

  return total_accuracy, (test_end_time -  test_start_time) # add

"""# load data"""

#proccess dataset

class my_Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# load whole B dataset
# 讀入所有資料
def load_all_img(AorB, class_num, p_train_num, p_test_num):
  path = "../xxx/finetune/LiteonRacingData/" + str(AorB) + "/" + str(class_num) + "/" 
  img_train = []
  img_test = []
  f_name = []
  for f in listdir(path):
    if f == "desktop.ini":
      continue
    else:
      f_name.append(f)
  f_name.sort()
  random.seed(load_all_seed)
  random.shuffle(f_name)
  
  count = 0
  for name in f_name:
    if count < p_test_num:
      img_test.append(cv2.resize(cv2.cvtColor(cv2.imread(path + name),cv2.IMREAD_COLOR), (224, 224), interpolation=cv2.INTER_CUBIC))
      count += 1
    elif count < p_train_num + p_test_num:
      img_train.append(cv2.resize(cv2.cvtColor(cv2.imread(path + name),cv2.IMREAD_COLOR), (224, 224), interpolation=cv2.INTER_CUBIC))
      count += 1
    else:
      break
  return img_train, img_test

def open_image_from_Liteon_Charlie_gen(AorB, classNum, folder_ForG, sheet_num=0, load_all=False):
  img_x_train = []
  f_name = []
  path = "/home/xxx/Liteon_Charlie_gen/"+str(AorB)+"/"+ str(classNum) + "_" + str(folder_ForG) + "/"
  for f in listdir(path):
    f_name.append(f)
  f_name.sort()
  random.seed(load_from_gen_seed)
  random.shuffle(f_name)
  
  if load_all == True:
    for name in f_name:
      img_x_train.append(cv2.resize(cv2.cvtColor(cv2.imread(path + name),cv2.IMREAD_COLOR), (224, 224), interpolation=cv2.INTER_CUBIC))
  else:
    if sheet_num <= len(f_name):
      for i in range(sheet_num):
        img_x_train.append(cv2.resize(cv2.cvtColor(cv2.imread(path + f_name[i]),cv2.IMREAD_COLOR), (224, 224), interpolation=cv2.INTER_CUBIC))
    else:
      print("要求超過數量! sheet_num需小於", str(len(f_name)))
      return None
  return img_x_train


if __name__ == '__main__': 
  device = torch.device('cuda')
  

  #-------------------dataset proccessing begin---------------------# 
  B_p_train_num = int(85)
  B_p_test_num = int(5)
  
  # 讀入B所有資料
  img10_x_train_B, img10_x_test_B = load_all_img('B', 0, B_p_train_num, B_p_test_num)
  img11_x_train_B, img11_x_test_B = load_all_img('B', 1, B_p_train_num, B_p_test_num)
  img12_x_train_B, img12_x_test_B = load_all_img('B', 2, B_p_train_num, B_p_test_num)
  img13_x_train_B, img13_x_test_B = load_all_img('B', 3, B_p_train_num, B_p_test_num)
  img14_x_train_B, img14_x_test_B = load_all_img('B', 4, B_p_train_num, B_p_test_num)
  img15_x_train_B, img15_x_test_B = load_all_img('B', 5, B_p_train_num, B_p_test_num)
  img16_x_train_B, img16_x_test_B = load_all_img('B', 6, B_p_train_num, B_p_test_num)
  img17_x_train_B, img17_x_test_B = load_all_img('B', 7, B_p_train_num, B_p_test_num)

  # B 的 1~7類從翻轉圖片補
  img11_x_train_B += open_image_from_Liteon_Charlie_gen('B', 1, "flip", 0, True)
  img12_x_train_B += open_image_from_Liteon_Charlie_gen('B', 2, "flip", 0, True)
  img13_x_train_B += open_image_from_Liteon_Charlie_gen('B', 3, "flip", 0, True)
  img14_x_train_B += open_image_from_Liteon_Charlie_gen('B', 4, "flip", 0, True)
  img15_x_train_B += open_image_from_Liteon_Charlie_gen('B', 5, "flip", 0, True)
  img16_x_train_B += open_image_from_Liteon_Charlie_gen('B', 6, "flip", 0, True)
  img17_x_train_B += open_image_from_Liteon_Charlie_gen('B', 7, "flip", 0, True)

  # B 的 1~7 使用gen圖片補齊資料
  img11_x_train_B += open_image_from_Liteon_Charlie_gen('B', 1, "gen", B_p_train_num-len(img11_x_train_B))
  img12_x_train_B += open_image_from_Liteon_Charlie_gen('B', 2, "gen", B_p_train_num-len(img12_x_train_B))
  img13_x_train_B += open_image_from_Liteon_Charlie_gen('B', 3, "gen", B_p_train_num-len(img13_x_train_B))
  img14_x_train_B += open_image_from_Liteon_Charlie_gen('B', 4, "gen", B_p_train_num-len(img14_x_train_B))
  img15_x_train_B += open_image_from_Liteon_Charlie_gen('B', 5, "gen", B_p_train_num-len(img15_x_train_B))
  img16_x_train_B += open_image_from_Liteon_Charlie_gen('B', 6, "gen", B_p_train_num-len(img16_x_train_B))
  img17_x_train_B += open_image_from_Liteon_Charlie_gen('B', 7, "gen", B_p_train_num-len(img17_x_train_B))

  # 更改 A -> B
  img10_x_train = img10_x_train_B
  img11_x_train = img11_x_train_B
  img12_x_train = img12_x_train_B
  img13_x_train = img13_x_train_B
  img14_x_train = img14_x_train_B
  img15_x_train = img15_x_train_B
  img16_x_train = img16_x_train_B
  img17_x_train = img17_x_train_B

  img10_x_test = img10_x_test_B
  img11_x_test = img11_x_test_B
  img12_x_test = img12_x_test_B
  img13_x_test = img13_x_test_B
  img14_x_test = img14_x_test_B
  img15_x_test = img15_x_test_B
  img16_x_test = img16_x_test_B
  img17_x_test = img17_x_test_B
  
  p_train_num = B_p_train_num
  p_test_num = B_p_test_num

  img10_x_validation = img10_x_train[0:7]
  img11_x_validation = img11_x_train[0:7]
  img12_x_validation = img12_x_train[0:7]
  img13_x_validation = img13_x_train[0:7]
  img14_x_validation = img14_x_train[0:7]
  img15_x_validation = img15_x_train[0:7]
  img16_x_validation = img16_x_train[0:7]
  img17_x_validation = img17_x_train[0:7]
  

  img10_y_train = [0]*p_train_num
  img10_y_test = [0]*p_test_num

  img11_y_train = [1]*p_train_num
  img11_y_test = [1]*p_test_num

  img12_y_train = [2]*p_train_num
  img12_y_test = [2]*p_test_num

  img13_y_train = [3]*p_train_num
  img13_y_test = [3]*p_test_num

  img14_y_train = [4]*p_train_num
  img14_y_test = [4]*p_test_num

  img15_y_train = [5]*p_train_num
  img15_y_test = [5]*p_test_num

  img16_y_train = [6]*p_train_num
  img16_y_test = [6]*p_test_num

  img17_y_train = [7]*p_train_num
  img17_y_test = [7]*p_test_num

  #串接所有類別的訓練資料 (x,y)
  img_x_train = img10_x_train + img11_x_train + img12_x_train + img13_x_train + img14_x_train + img15_x_train + img16_x_train + img17_x_train
  img_y_train = img10_y_train + img11_y_train + img12_y_train + img13_y_train + img14_y_train + img15_y_train + img16_y_train + img17_y_train
  
  print('len(img_x_train):',len(img_x_train))
  print('len(img_y_train):',len(img_y_train))
  f.write('\nlen(img_x_train):'+str(len(img_x_train)))
  f.write('\nlen(img_y_train):'+str(len(img_y_train))+'\n')

  # 打混訓練資料
  index = [i for i in range(len(img_x_train))] 
  random.shuffle(index)
  index = np.array(index)
  img_x_train = np.array(img_x_train)[index]
  img_y_train = np.array(img_y_train)[index]



  #串接所有類別的測試資料 (x,y)
  img_x_test = img10_x_test + img11_x_test + img12_x_test + img13_x_test + img14_x_test + img15_x_test + img16_x_test + img17_x_test
  img_y_test = img10_y_test + img11_y_test + img12_y_test + img13_y_test + img14_y_test + img15_y_test + img16_y_test + img17_y_test
  print('len(img_x_test):',len(img_x_test))
  print('len(img_y_test):',len(img_y_test))
  f.write('\nlen(img_x_test):'+str(len(img_x_test)))
  f.write('\nlen(img_y_test):'+str(len(img_y_test))+'\n')

  img_x_validation = img10_x_validation + img11_x_validation + img12_x_validation + img13_x_validation + img14_x_validation + img15_x_validation + img16_x_validation + img17_x_validation


  img_x_validation = np.array(img_x_validation)
  img_x_validation = img_x_validation.astype("float32")
  for i in range(len(img_x_validation)):
    img_x_validation[i] = img_x_validation[i]/255.0


  img_x_train = img_x_train.astype("float32")
  for i in range(len(img_x_train)):
    img_x_train[i] = img_x_train[i]/255.0

  for i in range(len(img_x_test)):
    img_x_test[i] = img_x_test[i]/255.0

  #訓練pytorch
  x_train = torch.zeros(len(img_x_train),3,224,224,dtype = torch.float32)
  for i in range(len(img_x_train)): 
    #x_train[i] = torch.from_numpy(img_x_train[i]) # for 黑白
    x_train[i] = torch.from_numpy(img_x_train[i]).permute(2,1,0) #for BGR、HSV(cv2預設)

  x_test = torch.zeros(len(img_x_test),3,224,224,dtype = torch.float32)
  for i in range(len(img_x_test)): 
    #x_test[i] = torch.from_numpy(img_x_test[i]) # for 黑白
    x_test[i] = torch.from_numpy(img_x_test[i]).permute(2,1,0)#for BGR、HSV(cv2預設)

  x_validation = torch.zeros(len(img_x_validation),3,224,224,dtype = torch.float32)
  for i in range(len(img_x_validation)): 
    #x_test[i] = torch.from_numpy(img_x_test[i]) # for 黑白
    x_validation[i] = torch.from_numpy(img_x_validation[i]).permute(2,1,0)#for BGR、HSV(cv2預設)

  y_train = torch.from_numpy(np.array(img_y_train))
  y_test = torch.from_numpy(np.array(img_y_test))

  #交叉驗證處理
  x_cross_train = []
  x_cross_val = []
  y_cross_train = []
  y_cross_val = []

  #10折交叉驗證
  len_one_tenth_x_train = int(len(x_train)/10)

  for i in range(10):
    x_buffer_train = x_train
    y_buffer_train = y_train
    x_cross_val.append(x_buffer_train[0+len_one_tenth_x_train*i:len_one_tenth_x_train+len_one_tenth_x_train*i])
    y_cross_val.append(y_buffer_train[0+len_one_tenth_x_train*i:len_one_tenth_x_train+len_one_tenth_x_train*i])
    x_cross_train.append(del_tensor_ele(x_buffer_train,0+len_one_tenth_x_train*i,len_one_tenth_x_train+len_one_tenth_x_train*i, len_one_tenth_x_train*10))
    y_cross_train.append(del_tensor_ele(y_buffer_train,0+len_one_tenth_x_train*i,len_one_tenth_x_train+len_one_tenth_x_train*i, len_one_tenth_x_train*10))

  train_dataset = []
  val_dataset = []
  for i in range(10):
    train_dataset.append(my_Dataset(x_cross_train[i],y_cross_train[i]))
    val_dataset.append(my_Dataset(x_cross_val[i],y_cross_val[i]))

  test_dataset = my_Dataset(x_test, y_test)

  #-------------------dataset proccessing end---------------------# 

  # load_model
  model = load_model().to(device = device)

  loss_fn = torch.nn.CrossEntropyLoss()
  n_epochs = 1
  optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5)

  accuracy = []
  recall = []
  precision = []
  tp_list = []
  tp_fp_list = []
  time_list = []
  layer = []

  dict_parameters(model) 
  dict_modules(model)
  validate_set =[]
  dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)


  start_time = time.time()
  ##compute which layer will start pruned
  strart_find_key = None
  strart_next_key = None
  temp = list(mean_criteria)
  if criteria_choose == 1 or criteria_choose == 2:
    min = 1000.0
    for keys in mean_criteria:
      if not "conv3" in keys:
        if mean_criteria[keys] < min:
          strart_find_key = keys
          strart_next_key = temp[temp.index(keys) + 1]
          min = mean_criteria[keys]
  else:
    max = 0.0
    for keys in mean_criteria:
      if not "conv3" in keys:
        if mean_criteria[keys] > max:
          strart_find_key = keys
          strart_next_key = temp[temp.index(keys) + 1]
          max = mean_criteria[keys]

  print('----------------')
  print('mean:',mean_criteria)
  print('----------------')
  print('which layer will start pruned:',strart_find_key)
  print('----------------')
  
  f.write('\n----------------\n mean : ')
  f.writelines(str(mean_criteria))
  f.write('\n----------------\n which layer will start pruned : ')
  f.writelines(str(strart_find_key))
  f.write('\n----------------\n')
 

  find_key = strart_find_key
  next_key = strart_next_key
  
  
  a, time1 = test(model, test_dataset)
  max_acc  = a

  #first down
  count = 0
  while True: 
    while True:
      if dict_name_modules[find_key].out_channels > 32:
        x = find_key.split('.')
        copy_model = make_new_layer(model, criteria_choose, find_key + ".weight", find_key + '.bias', next_key + '.weight', x[0] + '.' + x[1] + ".bn" + x[2][4] + '.weight', x[0] + '.' + x[1] + ".bn" + x[2][4] + '.bias')
        
        
        if copy_model == None:
          dict_parameters(model) 
          dict_modules(model)
          dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
          break
          

        copy_model = copy_model.to(device = device)
        
        if count % 1 == 0:
          training_loop(n_epochs,optimizer,copy_model,loss_fn,train_dataset,val_dataset, regularization_choose)
          a, time1 = test(copy_model, test_dataset)
          
          print('accuracy', a)
          
          
          if a < (max_acc - 0.00):
            print('max_acc-2.0 :',  max_acc - 0.00)
            dict_parameters(model) 
            dict_modules(model)
            dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
            break
            
            
          if a >= max_acc:
            max_acc = a
            torch.save(copy_model, model_save_path[:-3] + 'best.pt')
              
            
          layer.append(find_key)
          print(layer)
          print(_get_module(copy_model, find_key))
          accuracy.append(a)
          time_list.append(time1)

          model = copy_model
          count += 1
          
        else:
          model = copy_model
          count += 1
          

      else:
        dict_parameters(model) 
        dict_modules(model)
        dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
        break

      dict_parameters(model) 
      dict_modules(model)
      dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)

    if len(temp) - temp.index(find_key) == 2:
      dict_parameters(model) 
      dict_modules(model)
      dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
      break

    if "conv3" in temp[temp.index(find_key) + 1]:
      find_key = temp[temp.index(find_key) + 2]
      next_key = temp[temp.index(next_key) + 2]
    else:
      find_key = temp[temp.index(find_key) + 1]
      next_key = temp[temp.index(next_key) + 1]

  find_key = strart_find_key
  next_key = strart_next_key
    
  #and then up
  while True: 
    while True:
      if dict_name_modules[find_key].out_channels > 32:
        copy_model = make_new_layer(model, criteria_choose, find_key + ".weight", find_key + '.bias', next_key + '.weight', find_key[:8] + ".bn" + find_key[13] + '.weight', find_key[:8] + ".bn" + find_key[13] + '.bias')
        if copy_model == None:
          dict_parameters(model) 
          dict_modules(model)
          dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
          break

        copy_model = copy_model.to(device = device)
        
        if count % 1 == 0:
          training_loop(n_epochs,optimizer,copy_model,loss_fn,train_dataset,val_dataset, regularization_choose)
          a, time1 = test(copy_model, test_dataset)
          print('accuracy', a)
          if a < (max_acc - 0.00):
            print('max_acc-2.0 :',  max_acc - 0.00)
            dict_parameters(model) 
            dict_modules(model)
            dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
            break
  
            
          if a >= max_acc:
            max_acc = a
            torch.save(copy_model, model_save_path[:-3] + 'best.pt')
            
    
          layer.append(find_key)
          print(layer)
          print(_get_module(copy_model, find_key))
          accuracy.append(a)
  
          time_list.append(time1)

          model = copy_model
          count += 1
        else:
          model = copy_model
          count += 1

      else:
        
        dict_parameters(model) 
        dict_modules(model)
        dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
        break

      dict_parameters(model) 
      dict_modules(model)
      dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)


    if temp.index(find_key) == 1:
      dict_parameters(model) 
      dict_modules(model)
      dict_criteria, mean_criteria, std_criteria = criteria_select(criteria_choose, model, validate_set)
      break

    if "conv3" in temp[temp.index(find_key) - 1]:
      find_key = temp[temp.index(find_key) - 2]
      next_key = temp[temp.index(next_key) - 2]
    else:
      find_key = temp[temp.index(find_key) - 1]
      next_key = temp[temp.index(next_key) - 1]

  end_time = time.time()

  print('the counts of pruning: ',count)
  print('----------------')
  print('the time of pruning : ', end_time-start_time)
  print('----------------')
  
  f.write('the counts of pruning : '+str(count)+'\n')
  f.write('----------------\n')
  f.write('the time of pruning : '+str(end_time-start_time)+'\n')
  f.write('----------------\n')
  

  model1 = model.to(device=torch.device('cpu'))

  stat(model1,(3,224,224))
  print('----------------')
  f.write('----------------\n')

  ori_resnet = load_model()

  print('ori_resnet:')
  stat(ori_resnet,(3,224,224))
  print('----------------')
  f.write('----------------\n')

  print('accuracy : ',accuracy)
  print('----------------')
  f.write('accuracy : ')
  f.writelines(str(accuracy))
  f.write('\n----------------\n')

  f.write('time : ')
  f.writelines(str(time_list))
  f.write('\n----------------\n')
  f.write('layer : ')
  f.writelines(str(layer))
  f.close()

  #model = copy_model
  # save weight
  torch.save(model.state_dict(), weight_save_path)
  # save whole model
  torch.save(model, model_save_path)
  
