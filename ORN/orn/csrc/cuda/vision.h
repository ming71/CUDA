#pragma once
#include <torch/extension.h>

typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

torch::Tensor ARF_MappingRotate_forward_cuda(
    const torch::Tensor weight,
    const torch::Tensor indices);

torch::Tensor ARF_MappingRotate_backward_cuda(
    const torch::Tensor weight,
    const torch::Tensor indices);

std::vector<torch::Tensor> RIE_AlignFeature_forward_cuda(
    const torch::Tensor feature,
    const uint8 nOrientation);

torch::Tensor RIE_AlignFeature_backward_cuda(
    const torch::Tensor feature,
    const torch::Tensor mainDirection,
    const uint8 nOrientation);
