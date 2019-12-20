import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from orn import _CUDA


class _ARF_Cuda_ForwardBackward(Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(input)
        ctx.indices = indices
        output = _CUDA.arf_mappingrotate_forward(input.cuda(), indices)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        indices = ctx.indices
        grad_input = _CUDA.arf_mappingrotate_backward(grad_output, indices)
        return grad_input, None

class MappingRotate(nn.Module):
    def __init__(self):
        super(MappingRotate, self).__init__()

    def forward(self, input, indices):
            return _ARF_Cuda_ForwardBackward.apply(input, indices)





