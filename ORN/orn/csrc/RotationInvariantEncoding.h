#pragma once
#include "cuda/vision.h"

std::vector<torch::Tensor> RIE_AlignFeature_forward(
    const torch::Tensor feature,
    const uint8 nOrientation){

    return RIE_AlignFeature_forward_cuda(feature, nOrientation);
}

torch::Tensor RIE_AlignFeature_backward(
    const torch::Tensor feature,
    const torch::Tensor mainDirection,
    const uint8 nOrientation){

  return RIE_AlignFeature_backward_cuda(feature, mainDirection, nOrientation);
}
