#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifdef __cplusplus  //c++编译器可能会改变函数名，这里声明编译以下代码时使用c编译器
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_align_kernel.h"

// 当前开辟的所有线程数是blockDim.x * gridDim.x，当需要并行的任务总数超过了当前开辟的所有线程数时，可以让线程循环的完成任务。一种常见的用法。
// 比如，一共开辟了5*2共十个线程，一共有30个任务，0号线程在干完任务0后，可以继续干任务0+10，之后可以继续干任务0+10+10。同理1号线程可以按顺序去做任务1,11,21。
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)             


	__global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) {
		/*
		nthreads：pooling后的featuremap像素点总数量，即num_rois * aligned_height * aligned_width * channels，num_rois表示当前batch里所有的roi数量，
		aligned_height，aligned_width分别表示pooling后的h和pooling后的w，channels表示通道数（pooling前后不变）。每个线程负责一个pooling结果，所以这个数值也是线程总数量
		bottom_data：需要进行roialign的featuremap的首地址，注意这个特征图由python里的(bs, c, h, w)4维矩阵变成了c语言里的(bs*c*h*w)一维数组。
		spatial_scale：原图和特诊图之间的比例。原图的height/特征图的height
		height：特征图的height
		width：特征图的width
		channels：特征图的channels
		aligned_height：pooling后的h,一般为7
		aligned_width：pooling后的w，一般为7
		bottom_rois：存储rois的首地址，在python里是2维的(num_rois, 5)，[[batch_index,x1,y1,x2,y2],...]，这里变成了c语言里的(num_rois * 5)一维数组。
		top_data：pooling结果的首地址，最后的结果存储在这里。它的形状是(num_rois * aligned_height * aligned_width * channels)一维数组，每一个都和index对应
		*/
		CUDA_1D_KERNEL_LOOP(index, nthreads) {  // 用函数宏定义中的内容代替，即index代替for循环中的i,nthreads代替for循环中的n

			// (n, c, ph, pw) is an element in the aligned output
			/*
			根据index（线程号）判断，当前线程应该计算top_data的哪个位置，
			当前计算的就是第n个roi中的第c个通道上的ph（取值范围:[0, aligned_height)）,pw（取值范围:[0, aligned_width)）块
			*/

			int pw = index % aligned_width;
			int ph = (index / aligned_width) % aligned_height;
			int c = (index / aligned_width / aligned_height) % channels;
			int n = index / aligned_width / aligned_height / channels;

			// bottom_rois += n * 5;
			float roi_batch_ind = bottom_rois[n * 5 + 0];     // bottom_rois以5位单位，0位置放当前roi属于当前batch中的第几张图片(从0开始排序)，也就是batch_index
			float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;  // 1-4位置放当前roi左上角，右下角坐标
			float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale; //这些坐标是在featuremap上的坐标，通过spatial_scale转换过来，注意是float类型，无损失!!!
			float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
			float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

			// Force malformed ROIs to be 1x1
			float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.); // roi区域宽度，注意是float类型，无损失!!!
			float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.); //roi区域高度，注意是float类型，无损失!!!
			float bin_size_h = roi_height / (aligned_height - 1.); //这个地方是这份代码的特别之处，和原版的roialign有个小不同。他把roi区域分成了(aligned_height - 1.)*(aligned_width - 1.)个块，
			float bin_size_w = roi_width / (aligned_width - 1.); //那么在height方向可以产生aligned_height个交点，在width方向可以产生aligned_width个交点，后面就是用双线性插值求交点处的值
																
			float h = (float)(ph)* bin_size_h + roi_start_h; //当前所求的块（交点）处的h坐标，注意是float类型，无损失!!!
			float w = (float)(pw)* bin_size_w + roi_start_w; //当前所求的块（交点）处的w坐标，注意是float类型，无损失!!!

			int hstart = fminf(floor(h), height - 2); // 获得双线性插值采样点（交点）周围四个坐标中的左上角坐标。注意是int类型，准备双插!!!
			int wstart = fminf(floor(w), width - 2);  //之所以和width-2比较取较小值，是因为现在求的是左上角，要给右下角留下位置，不能让右下角超出featuremap范围   

			int img_start = roi_batch_ind * channels * height * width; //当前处理featuremap在bottom_data中的起始位置。bottom_data是一维的，所以每一个featuremap占据channels * height * width位置

			// bilinear interpolation
			if (h < 0 || h >= height || w < 0 || w >= width) { //超出featuremap范围的交点直接置0
				top_data[index] = 0.;
			}
			else {
				float h_ratio = h - (float)(hstart);
				float w_ratio = w - (float)(wstart);
				int upleft = img_start + (c * height + hstart) * width + wstart;  //把左上角左边从3维度变成一维度。因为bottom_data是一维度的
				int upright = upleft + 1;
				int downleft = upleft + width;   //左下角坐标和左上角坐标在一维度上相差width
				int downright = downleft + 1;

				top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
					+ bottom_data[upright] * (1. - h_ratio) * w_ratio
					+ bottom_data[downleft] * h_ratio * (1. - w_ratio)
					+ bottom_data[downright] * h_ratio * w_ratio;       //双线性插值公式 f(i+u,j+v) = (1-u)(1-v)f(i,j)+ u(1-v)f(i+1,j) + (1-u)vf(i,j+1) + uvf(i+1,j+1)
			}
		}
	}


	int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, cudaStream_t stream) {
		const int kThreadsPerBlock = 1024;   //每个线程块（block）设置1024个线程
		const int output_size = num_rois * aligned_height * aligned_width * channels;   //要处理的总任务数量，即pooling完之后featuremap的大小
		cudaError_t err;

		// 设置(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock 线程块（block）
		ROIAlignForward << <(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >> >(   
			output_size, bottom_data, spatial_scale, height, width, channels,
			aligned_height, aligned_width, bottom_rois, top_data); //开始cuda

		err = cudaGetLastError();
		if (cudaSuccess != err) {
			fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
			exit(-1);
		}

		return 1;
	}


	__global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) {
		/*
		roialign的反向很简单，就是正向传播时，只有参与过双插的点才会有梯度，其他的点没有梯度。
		所以只要把正向传播再进行一遍，找到做双插的点，每个做双插的点，在双插时前面的系数乘以梯度就是该点最终的梯度。
		
		nthreads ：任务数，和正向传播时数量一样。正向时，每个线程通过交点周围4个点的值计算交点处的值；反向时，每个线程计算一个交点周围4个点的梯度。
		top_diff ：pooling后每个点的梯度。这是存储数组的首地址。
		bottom_diff ：pooling前整个featuremap上每个点的梯度。这也是首地址，是我们想要的结果。
		*/

		CUDA_1D_KERNEL_LOOP(index, nthreads) {

			// (n, c, ph, pw) is an element in the aligned output
			int pw = index % aligned_width;
			int ph = (index / aligned_width) % aligned_height;
			int c = (index / aligned_width / aligned_height) % channels;
			int n = index / aligned_width / aligned_height / channels;

			float roi_batch_ind = bottom_rois[n * 5 + 0];
			float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
			float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
			float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
			float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;
			/* int roi_start_w = round(bottom_rois[1] * spatial_scale); */
			/* int roi_start_h = round(bottom_rois[2] * spatial_scale); */
			/* int roi_end_w = round(bottom_rois[3] * spatial_scale); */
			/* int roi_end_h = round(bottom_rois[4] * spatial_scale); */

			// Force malformed ROIs to be 1x1
			float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
			float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
			float bin_size_h = roi_height / (aligned_height - 1.);
			float bin_size_w = roi_width / (aligned_width - 1.);

			float h = (float)(ph)* bin_size_h + roi_start_h;
			float w = (float)(pw)* bin_size_w + roi_start_w;

			int hstart = fminf(floor(h), height - 2);
			int wstart = fminf(floor(w), width - 2);

			int img_start = roi_batch_ind * channels * height * width;

			// bilinear interpolation
			if (!(h < 0 || h >= height || w < 0 || w >= width)) {
				float h_ratio = h - (float)(hstart);
				float w_ratio = w - (float)(wstart);
				int upleft = img_start + (c * height + hstart) * width + wstart;
				int upright = upleft + 1;
				int downleft = upleft + width;
				int downright = downleft + 1;

				//以上都和前向传播一样，下面是计算4个点的梯度，双插系数*梯度
				atomicAdd(bottom_diff + upleft, top_diff[index] * (1. - h_ratio) * (1 - w_ratio));
				atomicAdd(bottom_diff + upright, top_diff[index] * (1. - h_ratio) * w_ratio);
				atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1 - w_ratio));
				atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);
			}
		}
	}

	int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width,
		const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, cudaStream_t stream) {
		const int kThreadsPerBlock = 1024;
		const int output_size = num_rois * aligned_height * aligned_width * channels;
		cudaError_t err;

		ROIAlignBackward << <(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >> >(
			output_size, top_diff, spatial_scale, height, width, channels,
			aligned_height, aligned_width, bottom_diff, bottom_rois);

		err = cudaGetLastError();
		if (cudaSuccess != err) {
			fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
			exit(-1);
		}

		return 1;
	}


#ifdef __cplusplus
}
#endif