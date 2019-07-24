#include <cfloat>
#include <cmath>
//#include <cstdio>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;

/*
1. 顺时针为正
2. spatial_scale大于1
*/

namespace caffe {

template <typename Dtype>
__device__ Dtype bilinear_interpolate(const Dtype *bottom_data,
                                         const int height, const int width,
                                         Dtype y, Dtype x) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;
  
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (Dtype)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (Dtype)x_low;
  } else {
    x_high = x_low + 1;
  }

  Dtype ly = y - y_low;   
  Dtype lx = x - x_low;  
  Dtype hy = 1. - ly;    
  Dtype hx = 1. - lx;    
  // do bilinear interpolation 
  Dtype lt = bottom_data[y_low * width + x_low];
  Dtype rt = bottom_data[y_low * width + x_high];
  Dtype lb = bottom_data[y_high * width + x_low];
  Dtype rb = bottom_data[y_high * width + x_high];
  Dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  Dtype val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename Dtype>
__global__ void RotateROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data,const Dtype* info) {
    
    int sample_num_h = 2 ;   // 新增设置--采样点：行列各采样两个点共四个采样点
    int sample_num_w = 2 ;

  CUDA_KERNEL_LOOP(index, nthreads) {

    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
// -------------施工区---------------
    const Dtype *offset_bottom_rois = bottom_rois + n * 6;   
    int roi_batch_ind = offset_bottom_rois[0];    // 指针指向的地址内容进行取出
    Dtype cx = offset_bottom_rois[1] / spatial_scale;   // 将roi从原图坐标缩放到特征图区域
    Dtype cy = offset_bottom_rois[2] / spatial_scale;   
    Dtype h  = offset_bottom_rois[3] / spatial_scale;
    Dtype w  = offset_bottom_rois[4] / spatial_scale;
    Dtype angle = offset_bottom_rois[5] /180.0*3.1415926535;

    // 定义旋转矩阵（不用缩放）
    Dtype Alpha = cos(angle);
    Dtype Beta = sin(angle);

    Dtype M[2][3]; 
    M[0][0] = Alpha;
    M[0][1] = Beta;    // 定义顺时针为正
    M[0][2] = cx-cx*Alpha-cy*Beta;
    M[1][0] = -Beta;
    M[1][1] = Alpha;
    M[1][2] = cy-cx*Beta-cy*Alpha;

  // 正框而言
    Dtype roi_start_w = cx - w / 2  ;
    Dtype roi_start_h = cy - h / 2  ;
    Dtype roi_end_w   = cx + w / 2  ;
    Dtype roi_end_h   = cy + h / 2  ;
// -------------------------------------------
    // Force malformed ROIs to be 1x1
    Dtype roi_width = fmaxf((Dtype)roi_end_w - roi_start_w, 0.);  // float
    Dtype roi_height = fmaxf((Dtype)roi_end_h - roi_start_h, 0.); 

    Dtype bin_size_h = roi_height / pooled_height; // 7*7 个bin尺寸
    Dtype bin_size_w = roi_width / pooled_width; 

    // bottom_data指向当前bsc下2D特征图左上角地址
    const Dtype *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;  

    Dtype output_val = 0;
    for (int iy = 0; iy < sample_num_h; iy++) {
      Dtype y = roi_start_h + ph * bin_size_h +
                          (Dtype)(iy + Dtype(.5f)) * bin_size_h /
                              (Dtype)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        Dtype x = roi_start_w + pw * bin_size_w +
                            (Dtype)(ix + Dtype(.5f)) * bin_size_w /
                                (Dtype)(sample_num_w);
// -------------施工区---------------
        // 计算旋转的插值坐标
        x = M[0][0] * x + M[0][1] * y + M[0][2] ;
        y = M[1][0] * x + M[1][1] * y + M[1][2] ;
        // xy的约束在双线性插值中定义
// -------------------------------------------
        Dtype val = bilinear_interpolate<Dtype>(offset_bottom_data,
                                                      height, width, y, x);  
        output_val += val;
      }
    }
    output_val /= (sample_num_h * sample_num_w);    // 这里的align采样点均值池化
    top_data[index] = output_val;
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  const Dtype* image_info = bottom[2]->gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data,image_info);
  CUDA_POST_KERNEL_CHECK;
}


/*
梯度插值是前向传播计算的子集，只完成插值的部分参数计算
实际上这个函数就是计算了lagrange插值的四个权重w1-w4系数（就够了，因为这次每项的函数值为特征图梯度，这部分在外面计算）
*/
template <typename Dtype>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              Dtype y, Dtype x,
                                              Dtype &w1, Dtype &w2,
                                              Dtype &w3, Dtype &w4,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (Dtype)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (Dtype)x_low;
  } else {
    x_high = x_low + 1;
  }
  Dtype ly = y - y_low;
  Dtype lx = x - x_low;
  Dtype hy = 1. - ly;
  Dtype hx = 1. - lx;
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename Dtype>
__global__ void RotateROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
//----------施工区----------------------
    int sample_num_h = 2;    
    int sample_num_w = 2;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype *offset_bottom_rois = bottom_rois + n * 6;

    int roi_batch_ind = offset_bottom_rois[0];    // 指针指向的地址内容进行取出
    Dtype cx = offset_bottom_rois[1] / spatial_scale;   
    Dtype cy = offset_bottom_rois[2] / spatial_scale;   
    Dtype h  = offset_bottom_rois[3] / spatial_scale;
    Dtype w  = offset_bottom_rois[4] / spatial_scale;
    Dtype angle = offset_bottom_rois[5] /180.0*3.1415926535;

    // 定义旋转矩阵（不用缩放）
    Dtype Alpha = cos(angle);
    Dtype Beta = sin(angle);

    Dtype M[2][3]; 
    M[0][0] = Alpha;
    M[0][1] = Beta;    // 定义顺时针为正
    M[0][2] = cx-cx*Alpha-cy*Beta;
    M[1][0] = -Beta;
    M[1][1] = Alpha;
    M[1][2] = cy-cx*Beta-cy*Alpha;

  // 正框而言
    Dtype roi_start_w = cx - w / 2  ;
    Dtype roi_start_h = cy - h / 2  ;
    Dtype roi_end_w   = cx + w / 2  ;
    Dtype roi_end_h   = cy + h / 2  ;
// -------------------------------------------------
    // Force malformed ROIs to be 1x1
    Dtype roi_width = fmaxf((Dtype)roi_end_w - roi_start_w, 0.);
    Dtype roi_height = fmaxf((Dtype)roi_end_h - roi_start_h, 0.);

    Dtype bin_size_h = roi_height / pooled_height;
    Dtype bin_size_w = roi_width / pooled_width;

    Dtype *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;
    int offset_top = (n * channels + c) * pooled_height * pooled_width +
                    ph * pooled_width + pw;
    Dtype offset_top_diff = top_diff[offset_top];

    const Dtype count = (Dtype)(sample_num_h * sample_num_w); // 统计采样点个数，均值池化反向传播时梯度平均

    for (int iy = 0; iy < sample_num_h; iy++) {
        Dtype y =
          roi_start_h + ph * bin_size_h +
          (Dtype)(iy + .5f) * bin_size_h / (Dtype)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        Dtype x =
            roi_start_w + pw * bin_size_w +
            (Dtype)(ix + .5f) * bin_size_w / (Dtype)(sample_num_w);
            // 计算旋转的插值坐标
            x = M[0][0] * x + M[0][1] * y + M[0][2] ;
            y = M[1][0] * x + M[1][1] * y + M[1][2] ;
        Dtype w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<Dtype>(
            height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high);
        Dtype g1 = offset_top_diff * w1 / count;
        Dtype g2 = offset_top_diff * w2 / count;
        Dtype g3 = offset_top_diff * w3 / count;
        Dtype g4 = offset_top_diff * w4 / count;
        // atomicAdd原子加操作：输入是地址和值，将addr上的值和输入值相加，并存储到addr地址上
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        }
      }
    }
  }
}

template <typename Dtype>
void RotateROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  int counter = top[0]->count();
  //NOLINT_NEXT_LINE(whitespace/operators)
  RotateROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(counter), CAFFE_CUDA_NUM_THREADS>>>(
     count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
  //std::cout<<top_diff[0]<<std::endl;
}

INSTANTIATE_LAYER_GPU_FUNCS(RotateROIPoolingLayer);

}  // namespace caffe
