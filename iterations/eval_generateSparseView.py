import os
import time
import tqdm
import torch
import numpy as np
import shutil
import scipy.io as sco

from ConeBeamLayers.Beijing.BeijingGeometry import BeijingGeometry, BeijingGeometryWithFBP, ForwardProjection, BackProjection
from options import trainPath, validPath, outputPath
import ConeBeamLayers.Beijing.BeijingGeometry as Geometry

parameters = Geometry.parameters

root_path = "/media/nv/c5083702-c81b-474c-a080-1d0316a0c495/fyx/20230531/bodys/zipai/3"
file_name = root_path.split("/")[-1]
if not os.path.exists(root_path):
    os.makedirs(root_path)
matlab_path = os.path.join(root_path, "matlab")
sino_path = os.path.join(root_path, "sino")
input_path = os.path.join(root_path, "input")
label_path = os.path.join(root_path, "label")
os.makedirs(matlab_path)
os.makedirs(sino_path)
os.makedirs(input_path)
os.makedirs(label_path)

projection = np.fromfile("/media/nv/c5083702-c81b-474c-a080-1d0316a0c495/fyx/20230531/bodys/zipai/projection_3.raw", dtype="float32")
label_old_path = "/media/nv/c5083702-c81b-474c-a080-1d0316a0c495/fyx/20230531/bodys/zipai/label_3.raw"
projection = np.reshape(projection, [1,1080*21, 128, 80]) #* 20.0
projection = np.reshape(projection, [1080, 21, 128, 80])
"""
projection_new = np.zeros((24,21,128,80)).astype(np.float32)
interval = 1080/24
for i in range(24):
    projection_new[i] = projection[int(i*interval)]
projection = projection_new
projection = np.reshape(projection, [1, 1, 24*21, 128, 80])
projection = torch.from_numpy(projection).cuda()
"""
parameters = np.reshape(parameters, [1080, 21, 12])

print("projected")
net = BeijingGeometryWithFBP().cuda().eval()

for i in range(15):
    projection_new = np.zeros((72,21,128,80)).astype(np.float32)
    parameters_new = np.zeros((72,21,12)).astype(np.float32)
    for j in range(24):
        projection_new[j*3:j*3+3] = projection[j*45:j*45+3]
        parameters_new[j*3:j*3+3] = parameters[j*45:j*45+3]
    projection_new = np.reshape(projection_new, [1, 1, 72*21, 128, 80])
    projection_new = torch.from_numpy(projection_new).cuda() 
    parameters_new = np.reshape(parameters_new, [72*21, 12])
    parameters_new = torch.from_numpy(parameters_new).cuda()
    Geometry.parameters = parameters_new
    volume = torch.zeros([1,1,64,256,256]).cuda()
    volume = net(volume, projection_new)
    save_dict = {'projection_matrix':parameters_new.detach().cpu().numpy()}
    sco.savemat(os.path.join(matlab_path, file_name + "_" + str(i) + ".mat"), save_dict)
    projection_new.detach().cpu().numpy().tofile( os.path.join(sino_path, file_name + "_" + str(i) + ".raw") )
    volume.detach().cpu().numpy().tofile( os.path.join(input_path, file_name + "_" + str(i) + ".raw") )
    shutil.copy(label_old_path, os.path.join(label_path, file_name + "_" + str(i) + ".raw"))
    #exit()
print("Finish!")
