import torch
from orn import _CUDA
import numpy as np

nBatch = 1
nFeature = 2
nOrientation = 8
H = 1
W = 1
nChannel = nFeature * nOrientation
nthreads = nBatch * nFeature
_FLT_MAX = 3.402823466e+38

feature_data = np.arange(nBatch * nChannel * H * W).\
    reshape(nBatch, nChannel, H, W)
aligned_data = np.zeros((nBatch, nChannel, H, W))
mainDirection_data = np.zeros((nBatch, nFeature))

feature_data = feature_data.flatten()
aligned_data = aligned_data.flatten()
mainDirection_data = mainDirection_data.flatten()

# point python
for n in range(nthreads):
    j = n % nFeature
    i = int(n / nFeature)

    maxVal = -_FLT_MAX
    for l in range(nOrientation):
        val = feature_data[i * (nFeature * nOrientation)
                           + j * (nOrientation)
                           + l]
        if val > maxVal:
            maxVal = val
            mainDirection_data[i * nFeature + j] = l

    for l in range(nOrientation):
        src = feature_data[i * (nFeature * nOrientation)
                           + j * (nOrientation)
                           + l]
        alignedIndex = int(
            ((l - mainDirection_data[i * nFeature + j]) + nOrientation) % nOrientation
        )
        aligned_data[i * (nFeature * nOrientation) \
                    + j * (nOrientation) \
                    + alignedIndex] = src

# print(mainDirection_data.reshape(nBatch, nFeature))
# print(aligned_data.reshape(nBatch, nChannel, H, W))

# accessor torch
feature_data = np.arange(nBatch * nChannel * H * W).\
    reshape(nBatch, nChannel, H, W)
aligned_data = np.zeros((nBatch, nChannel, H, W))
mainDirection_data = np.zeros((nBatch, nFeature))

for n in range(nthreads):
    j = n % nFeature
    i = int(n / nFeature)

    maxVal = -_FLT_MAX
    for l in range(nOrientation):
        val = feature_data[i][j * nOrientation + l][0][0]

        if val > maxVal:
            maxVal = val
            mainDirection_data[i][j] = l

    for l in range(nOrientation):
        src = feature_data[i][j * nOrientation + l][0][0]

        alignedIndex = int(
            (l - mainDirection_data[i][j] + nOrientation) % nOrientation
        )


        aligned_data[i][j * nOrientation + alignedIndex][0][0] = src

a = aligned_data.flatten()

# print(mainDirection_data)
# print(aligned_data)

# cpp+cuda

feature_data = torch.arange(nBatch * nChannel * H * W).\
    reshape(nBatch, nChannel, H, W).float().cuda()

aligned_data, mainDirection_data = _CUDA.rie_alignfeature_forward(feature_data, nOrientation)

b = aligned_data.cpu().numpy().flatten()

# print(mainDirection_data)
# print(aligned_data)

#diffent
diffent = 0
for i in range(nBatch * nChannel * H * W):
    if a[i] != b[i]:
        diffent += 1
# print(diffent)

output_grad_data = torch.arange(nBatch * nChannel * H * W).\
    reshape(nBatch, nChannel, H, W).float().cuda()
mainDirection_data = torch.tensor([[7, 7]]).byte().cuda()

aligned_data = _CUDA.rie_alignfeature_backward(output_grad_data, mainDirection_data, nOrientation)

print(aligned_data)
print(aligned_data[0])
