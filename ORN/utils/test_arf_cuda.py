import torch
from orn import _CUDA
import math

nInputPlane = 20
nOutputPlane = 40
nOrientation = 8
nRotation = 8
kernel_size = (3, 3)

kernel_indices = {
            1: {
                0: (1,),
                45: (1,),
                90: (1,),
                135: (1,),
                180: (1,),
                225: (1,),
                270: (1,),
                315: (1,)
            },
            3: {
                0: (1,2,3,4,5,6,7,8,9),
                45: (2,3,6,1,5,9,4,7,8),
                90: (3,6,9,2,5,8,1,4,7),
                135: (6,9,8,3,5,7,2,1,4),
                180: (9,8,7,6,5,4,3,2,1),
                225: (8,7,4,9,5,1,6,3,2),
                270: (7,4,1,8,5,2,9,6,3),
                315: (4,1,2,7,5,3,8,9,6)
            }
        }
delta_orientation = 360 / nOrientation
delta_rotation = 360 / nRotation
kH, kW = kernel_size
indices = torch.ByteTensor(nOrientation * kH * kW, nRotation)
for i in range(0, nOrientation):
    for j in range(0, kH * kW):
        for k in range(0, nRotation):
            angle = delta_rotation * k
            layer = (i + math.floor(angle / delta_orientation)) % nOrientation
            kernel = kernel_indices[kW][angle][j]
            indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
indices = indices.view(nOrientation, kH, kW, nRotation)

# arf_cuda_forward
weights = torch.arange(nInputPlane * nOutputPlane * nOrientation * kW * kH).reshape(nOutputPlane, nInputPlane, nOrientation, kW, kH)
output = _CUDA.arf_mappingrotate_forward(weights.float().cuda(), indices.cuda())#.reshape(nOrientation * kH * kW, nRotation).cuda())

# nOrientation = 1
grad_output = torch.arange(nOutputPlane * nRotation * nInputPlane * nOrientation * kW * kH).\
    reshape(nOutputPlane * nRotation, nInputPlane * nOrientation, kW, kH)

grad_input = _CUDA.arf_mappingrotate_backward(grad_output.float().cuda(), indices.cuda())

print(grad_input)
# output_data1 = output.cpu().numpy()
# a = output_data1.flatten()
# # for i in range(nRotation * nOutputPlane):
# #     print('One bank')
# #     if (i % nOutputPlane) == 0:
# #         print('One input')
# #     for j in range(nOrientation * nInputPlane):
# #         print(output_data[i][j])
#
# nEntry = nOrientation * kW * kH
# nthreads = nOutputPlane * nInputPlane * nEntry
# output_data = np.zeros(nOutputPlane * nRotation * nInputPlane * nOrientation * kW * kH, dtype=np.float).reshape(
#    nOutputPlane * nRotation, nInputPlane * nOrientation, kW, kH)
# weight_data = np.arange(nInputPlane * nOutputPlane * nOrientation * kW * kH).reshape(
#     nOutputPlane, nInputPlane, nOrientation, kW, kH)
# indices_data = indices.reshape(nEntry, nRotation)
#
# start_time = time.time()
# for n in range(nthreads):
#     w = n % kW
#     h = int(n / kW) % kH
#     e = int(n / (kW * kH)) % nRotation
#     j = int(n / nEntry) % nInputPlane
#     i = int(int(n / nEntry) / int(nInputPlane))
#     l = n % nEntry
#
#     val = weight_data[i][j][e][h][w]
#     for k in range(nRotation):
#
#         index = indices_data[l][k] - 1
#         _o = int(index / (kW * kH))
#         _w = index % kW
#         _h = int(index / kW) % kH
#
#         output_data[i * nRotation + k][j * nOrientation + _o][_h][_w] = val
#
# spend_time = (time.time() - start_time) * 1000
#
# b = output_data.flatten()
#
# different = 0
# for i in range(nthreads * nRotation):
#     if a[i] != b[i]:
#         print(i, a[i], b[i])
#         different += 1
# print(different)



