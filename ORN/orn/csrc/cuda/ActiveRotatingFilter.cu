#include <torch/types.h>
#include <stdio.h>
#include <cuda.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void MappingRotateKernel(
    const uint32 nthreads,
    const torch::PackedTensorAccessor<Dtype, 5, torch::RestrictPtrTraits, size_t> weight_data,
    const torch::PackedTensorAccessor<uint8, 2, torch::RestrictPtrTraits, size_t> indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    torch::PackedTensorAccessor<Dtype, 4, torch::RestrictPtrTraits, size_t> output_data)
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        
    	const uint8 kW = 3;
        const uint8 kH = 3;
		uint8 k;
        
    	uint16 w = n % kW;
    	uint16 h = n / kW % kH;
    	uint16 e = n / (kW * kH) % nOrientation;
		uint16 j = n / nEntry % nInputPlane;
		uint16 i = n / nEntry / nInputPlane;
		uint16 l = n % nEntry;

		Dtype val = weight_data[i][j][e][h][w];
        for (k = 0; k < nRotation; k++) {
        	uint8 index = indices_data[l][k] - 1;
            uint16 _o = index / (kH * kW);
        	uint8 _w = index % kW;
        	uint8 _h = index / kW % kH;
            
        	output_data[i * nRotation + k][j * nOrientation + _o][_h][_w] = val;
        }
    }
}


torch::Tensor ARF_MappingRotate_forward_cuda(
    const torch::Tensor weight,		//weight is the conv kernel paras
    const torch::Tensor indices){

    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(indices.type().is_cuda(), "indices must be a CUDA tensor");

    const uint16 nOutputPlane = weight.size(0);
    const uint16 nInputPlane = weight.size(1);
    const uint8 nOrientation = weight.size(2);
    const uint8 kH = weight.size(3);
    const uint8 kW = weight.size(4);
    const uint8 nRotation = indices.size(3);
    const uint16 nEntry = nOrientation * kH * kW;
    const uint32 count = nOutputPlane * nInputPlane * nEntry;

    const auto weight_data = weight.cuda();
    const auto indices_data = indices.reshape({nOrientation * kH * kW, nRotation});
    auto output_data = torch::zeros({nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW}, weight.options().dtype(at::kFloat).device(at::kCUDA));
    //output_data is the conv paras kernel rotated by the code
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "arf_cuda_forward", [&] {
        MappingRotateKernel<scalar_t> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>(
            count,
            weight_data.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
            indices_data.packed_accessor<uint8, 2, torch::RestrictPtrTraits, size_t>(),
            nInputPlane,
            nOutputPlane,
            nOrientation,
            nRotation,
            nEntry,
            output_data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>());
    });
    
    THCudaCheck(cudaGetLastError());
    return output_data;
}

template <typename Dtype>
__global__ void MappingAlignKernel(
    const uint32 nthreads,
    const torch::PackedTensorAccessor<Dtype, 4, torch::RestrictPtrTraits, size_t> weight_data,
    const torch::PackedTensorAccessor<uint8, 2, torch::RestrictPtrTraits, size_t> indices_data,
    const uint16 nInputPlane,
    const uint16 nOutputPlane,
    const uint8 nOrientation,
    const uint8 nRotation,
    const uint16 nEntry,
    torch::PackedTensorAccessor<Dtype, 5, torch::RestrictPtrTraits, size_t> input_data)
{
    CUDA_KERNEL_LOOP(n, nthreads) {
        const uint8 kW = 3;
        const uint8 kH = 3;
		uint8 k;
        
    	uint16 w = n % kW;
    	uint16 h = n / kW % kH;
    	uint16 e = n / (kW * kH) % nOrientation;
		uint16 j = n / nEntry % nInputPlane;
		uint16 i = n / nEntry / nInputPlane;
		uint16 l = n % nEntry;

		//input_data[i][j][e][h][w] = 0.0;
        for (k = 0; k < nRotation; k++) {
        	const uint8 index = indices_data[l][k] - 1;
            const uint16 _o = index / (kH * kW);
        	const uint8 _w = index % kW;
        	const uint8 _h = index / kW % kH;
        	input_data[i][j][e][h][w] += weight_data[i * nRotation + k][j * nOrientation + _o][_h][_w];
        }
    }
}

torch::Tensor ARF_MappingRotate_backward_cuda(
	const torch::Tensor weight,		//weight is the conv output grad paras
	const torch::Tensor indices){
	
	AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
	AT_ASSERTM(indices.type().is_cuda(), "indices must be a CUDA tensor");
	
	const uint8 nOrientation = indices.size(0);
	const uint8 kH = indices.size(1);
	const uint8 kW = indices.size(2);
	const uint8 nRotation = indices.size(3);
	const uint16 nOutputPlane = weight.size(0) / nRotation;
	const uint16 nInputPlane = weight.size(1) / nOrientation;
	const uint16 nEntry = nOrientation * kH * kW;
	const uint32 count = nOutputPlane * nInputPlane * nEntry;	

 	const auto weight_data = weight;
	const auto indices_data = indices.reshape({nOrientation * kH * kW, nRotation});
	auto input_data = torch::zeros({nOutputPlane, nInputPlane, nOrientation, kH, kW}, weight.options().dtype(at::kFloat).device(at::kCUDA));
	//input_data is the conv input grad paras aligned by codes
	
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
	AT_DISPATCH_FLOATING_TYPES(weight.type(), "arf_cuda_backward", [&] {
		MappingAlignKernel<scalar_t> <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream >>>(
				count,
				weight_data.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
				indices_data.packed_accessor<uint8, 2, torch::RestrictPtrTraits, size_t>(),
				nInputPlane,
				nOutputPlane,
				nOrientation,
				nRotation,
				nEntry,
				input_data.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>());
	});
	
	THCudaCheck(cudaGetLastError());
	return input_data;
}


