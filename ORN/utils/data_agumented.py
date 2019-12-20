import torch
import cv2
import math
import numbers
import random
from PIL import Image, ImageOps
from tqdm import tqdm
import os
dataset = 'MNIST'
train_name = 'training.pt'
test_name = 'test.pt'
if dataset == 'MNIST':
    train_data = torch.load(os.path.join(
        'datasets', dataset, 'processed', 'training.pt'))
    test_data = torch.load(os.path.join(
        'datasets', dataset, 'processed', 'test.pt'))
else:
    print('nothing to do')
    exit()

nRotation = 8
diff_angle = 360 / nRotation
w, h = train_data[0][0].shape
center = (w / 2, h / 2)
M = []

def Rotate_Interval(data, M, type='train'):
    rotate_datasets = []
    rotate_labels = []
    datasets = data[0].numpy()
    labels = data[1].numpy()
    for data, label in tqdm(zip(datasets, labels)):
        for i in range(nRotation):
            rotate_mat = cv2.warpAffine(data, M[i], (w, h))

            rotate_datasets.append(torch.tensor(rotate_mat))
            rotate_labels.append(torch.tensor(label))

    state = (rotate_datasets, rotate_labels)
    if type == 'train':
        torch.save(state, 'training-rot8.pt')
    elif type == 'test':
        torch.save(state, 'test-rot8.pt')
    else:
        print('type should be train or test')
        exit()

for i in range(nRotation):
    m = cv2.getRotationMatrix2D(center, i * diff_angle, 1.0)
    M.append(m)

# Rotate_Interval(train_data, M, type='train')
# Rotate_Interval(test_data, M, type='test')


