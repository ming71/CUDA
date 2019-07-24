#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// 每个block开辟的线程数1024
#define THREADS_PER_BLOCK 1024

// 设置(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock 线程块（block）
inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num); 
}

/*
多处用到了C++的template：
这里的scalar_t是个形参，是啥都行，随便取名
*/
template <typename scalar_t>
// __device__：声明了插值计算是在GPU调用且GPU计算的；这行的scalar_t说明返回类型是scalar_t
// scalar_t：  是一个宏，特化的时候会传入具体的类型。
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {

  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  // 找出目标插值坐标附四点坐标
  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;
  
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  // lagrange插值多项式系数
  scalar_t ly = y - y_low; 
  scalar_t lx = x - x_low; 
  scalar_t hy = 1. - ly;    
  scalar_t hx = 1. - lx;    
  // do bilinear interpolation 
  //  bottom_data 是一维向量，按width将原来的二维特征图向量化，所以都要乘width以便索引到index
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}


template <typename scalar_t>
// __global__：该函数是CPU接口调用，在GPU运行；返回值必须为void
__global__ void ROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
  // 用函数宏定义中的内容代替，即index代替for循环中的i,nthreads代替for循环中的n
  // 表示会线程数大于当前grid开启上限时，一直在block中循环线程计算直到完成任务
  // 具体：pooling后的所有RoI像素点总数量进行同步/循环的计算，每各单独计算核单次求取一个点的坐标
  CUDA_1D_KERNEL_LOOP(index, nthreads) {  
    // (n, c, ph, pw) is an element in the aligned output
    /*  
    根据index（线程号）判断，当前线程应该计算top_data（输出ROI池化图）的哪个位置
    */
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // -------------施工区---------------
    // 注意：定义的传入bottom_rois是xywhxita的形式 ; spatial_scale < 1 ; 顺时针为正
    const Dtype *offset_bottom_rois = bottom_rois + n * 6;   
    int roi_batch_ind = offset_bottom_rois[0];    // 指针指向的地址内容进行取出
    Dtype cx = offset_bottom_rois[1] * spatial_scale;   // 将roi从原图坐标缩放到特征图区域
    Dtype cy = offset_bottom_rois[2] * spatial_scale;   
    Dtype h  = offset_bottom_rois[3] * spatial_scale;
    Dtype w  = offset_bottom_rois[4] * spatial_scale;
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

    // -------------------------------------


    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);  
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.); 

    scalar_t bin_size_h = roi_height / pooled_height; 
    scalar_t bin_size_w = roi_width / pooled_width;

 
    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;  

    int sample_num_h = (sample_num > 0)   
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =    
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);  


    scalar_t output_val = 0;
    for (int iy = 0; iy < sample_num_h; iy++) {
      // 计算采样点的y坐标：roi的h + bin的位置（如：7*7的第几个bin）+ bin内的偏移（bin宽高除以采样点个数）
      // 以w方向采样三个点为例，会把bin_w分为三份，每份取中点，就是插值点
      const scalar_t y = roi_start_h + ph * bin_size_h +
                         (scalar_t)(iy + scalar_t(.5f)) * bin_size_h /
                             (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        // 计算采样点x坐标，原理相同不赘述
        const scalar_t x = roi_start_w + pw * bin_size_w +
                           (scalar_t)(ix + scalar_t(.5f)) * bin_size_w /
                               (scalar_t)(sample_num_w);
        x = M[0][0] * x + M[0][1] * y + M[0][2] ;
        y = M[1][0] * x + M[1][1] * y + M[1][2] ;
        scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data,
                                                      height, width, y, x);  // 双线性插值得到结果
        output_val += val;
      }
    }
    output_val /= (sample_num_h * sample_num_w);    // 这里的align取值方式是均值
    top_data[index] = output_val;
  }
}

int ROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           at::Tensor output) {
  // 总线程数即要处理的总任务数量，pooling完之后所有featuremap的像素数
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "ROIAlignLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();
        // 指定grid和thread数；开始cuda计算
        ROIAlignForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_height,
                pooled_width, top_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}


template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
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
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }
  // hx四项是偏移，参见双线性lagrange插值公式
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // w1-w4分别是双线性插值公式四项的权重
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}



template <typename scalar_t>
__global__ void ROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t *bottom_diff) {

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype *offset_bottom_rois = bottom_rois + n * 6;

    int roi_batch_ind = offset_bottom_rois[0];    // 指针指向的地址内容进行取出
    Dtype cx = offset_bottom_rois[1] * spatial_scale;   // 将roi从原图坐标缩放到特征图区域
    Dtype cy = offset_bottom_rois[2] * spatial_scale;   
    Dtype h  = offset_bottom_rois[3] * spatial_scale;
    Dtype w  = offset_bottom_rois[4] * spatial_scale;
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

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;
    int offset_top = (n * channels + c) * pooled_height * pooled_width +
                     ph * pooled_width + pw;
    scalar_t offset_top_diff = top_diff[offset_top];

    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);
    
    const scalar_t count = (scalar_t)(sample_num_h * sample_num_w); // 统计采样点个数


    for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y =
          roi_start_h + ph * bin_size_h +
          (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x =
            roi_start_w + pw * bin_size_w +
            (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);
        // 遍历到每个插值点的xy坐标，进行梯度插值计算
        x = M[0][0] * x + M[0][1] * y + M[0][2] ;
        y = M[1][0] * x + M[1][1] * y + M[1][2] ;
        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(
            height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high);
        //以上都和前向传播一样，下面是计算4个点的梯度，双插系数*梯度
        // 注意：前向传播是均值池化，反向传播时特征图除以pooled_size = count均分给之前的位置
        scalar_t g1 = offset_top_diff * w1 / count;
        scalar_t g2 = offset_top_diff * w2 / count;
        scalar_t g3 = offset_top_diff * w3 / count;
        scalar_t g4 = offset_top_diff * w4 / count;
        /*
        atomicAdd原子加操作：输入是地址和值，将addr上的值和输入值相加，并存储到addr地址上
        */
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

int ROIAlignBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sample_num,
                channels, height, width, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}