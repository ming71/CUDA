#pragma once
#include "cuda/vision.h"

torch::Tensor ARF_MappingRotate_forward(
    const torch::Tensor weight,
    const torch::Tensor indices){

    return ARF_MappingRotate_forward_cuda(weight, indices);
}

torch::Tensor ARF_MappingRotate_backward(
    const torch::Tensor weight,
    const torch::Tensor indices){

  return ARF_MappingRotate_backward_cuda(weight, indices);
}
