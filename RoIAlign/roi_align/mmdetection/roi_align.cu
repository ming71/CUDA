#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

// 这里宏定义函数CUDA_1D_KERNEL_LOOP(i, n)，表示线程数大于当前grid开启上限时，一直在block中循环线程计算直到完成任务。后面会传入参数实例化
/*
当前开辟的所有线程数是blockDim.x * gridDim.x ；
当需要并行的任务总数超过了当前开辟的所有线程数时，可以让线程循环的完成任务。一种常见的用法；
比如，一共开辟了5*2共十个线程，一共有30个任务，0号线程在干完任务0后，可以继续干任务0+10，之后可以继续干任务0+10+10；
同理1号线程可以按顺序去做任务1,11,21。
*/
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
（但为了好看，往往写形参时和你真正后面传入的参数是一样的，比如后面调用时传入的函数类别是scalar_t，这里写的也是scalar_t）
*/
template <typename scalar_t>
// __device__：声明了插值计算是在GPU调用且GPU计算的；这行的scalar_t说明返回类型是scalar_t
// scalar_t：  是一个宏，特化的时候会传入具体的类型。
//             scalar_t被定义为张量在该上下文中实际处于运行时的类型。因此，如果有一个模板函数，用这个scalar_t别名实例化它，然后正确的函数就会被调用
//             后面可以看到，都是用scalar_t来实例化参数的
/*            
-----形参-----
bottom_data：需要进行roialign的featuremap的首地址指针（depth=1），注意特征图是(h*w)的一维数组。（关于指针和数组调用关系，参加forward函数开始的注释）
height/width：特征图的高宽
xy ： 要差值的点的坐标

注意：[区分采样点和像素点]这个函数只是对一个期望的坐标点，取周边四个像素进行插值；
如果设置每个bin取四个采样点，每个点都会同样取其自身周边四个像素点进行这样的bilinear_interpolate；
即：采样点不限制个数，像素点都是四个！
*/
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  // 这四项是找出目标插值坐标附近像素的左上角坐标，各个方向+1可以得到四个周边点坐标
  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;
  // 避免越界
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

  // w1-w4分别是双线性插值公式四项的权重
  // hx四项是偏移，参见双线性lagrange插值公式
  // lt等四项是周边四个点的像素值
  // val是四项相加得到插值点的像素
  scalar_t ly = y - y_low;  // 采样点到下边距离
  scalar_t lx = x - x_low;  // 采样点到左边距离
  scalar_t hy = 1. - ly;    // 采样点到上边距离
  scalar_t hx = 1. - lx;    // 采样点到右边距离
  // do bilinear interpolation 
  // 由最近的4个点插值得到
  // 这里注意 bottom_data 是一维向量，按width将原来的二维特征图向量化，所以都要乘width以便索引到index
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
/* 
---------参数--------
nthreads：     线程总数。pooling后的featuremap像素点总数量，即num_rois * aligned_height * aligned_width * channels，num_rois表示当前batch里所有的roi数量，
bottom_data：  同上，需要进行roialign的featuremap的！首地址！，注意这个特征图由python里的(bs, c, h, w)4维矩阵变成了c语言里的(bs*c*h*w)一维数组。
bottom_rois：  存储rois的首地址，在python里是2维的(num_rois, 5)，[[batch_index,x1,y1,x2,y2],...]，这里变成了c语言里的(num_rois * 5)一维数组。
spatial_scale：特征图和原图之间的比例。特征图的height/原图的height
sample_num：   采样点数
height/width： 特征图尺寸
pooled_height/pooled_width： 一般是7
top_data：pooling结果的首地址，最后的结果存储在这里。是(num_rois * pooled_height * pooled_width * channels)一维数组，每一个都和index对应
命名：上面的参数中，top是计算后的结果也就是存储roi_align输出的7*7地址；bottom是计算前，也就是特征图的地址
--------关于指针运算-------
上面的bottom_data和bottom_rois定义的都是指针，所以传递的都是首地址！
1. 实际也是数组，数组的变量名就是首地址
2. 进行加减运算时，是指针（地址）的运算，内容不变
3. 当指针带上[]时，就是从当前指针地址开始的数组了，取出的是内容
如：bottom_rois表示所有rois的首地址，a = bottom_rois+ 100*5 表示第100个roi信息的首地址 a[0]~a[4]分别是第100个roi的bs x1y1x2y2等具体信息（不再是地址了）
*/
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
    根据index（线程号）判断，当前线程应该计算top_data（输出ROI池化图）的哪个位置，
    当前计算的就是第n个roi中的第c个通道上的ph（取值范围:[0, pooled_height)）,pw（取值范围:[0, pooled_width)） 
    */
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    /*
    offset_bottom_bottom_rois以5位单位
    0位置放当前roi属于当前batch中的第几张图片(从0开始排序)，也就是batch_index
    注意缩放：1-4位置放当前roi左上角，右下角坐标，针对 真实图像大小而言的，所以需要通过spatial_scale  缩放！！;
             spatial_scale乘子将roi坐标缩放到featuremap后，是float型，无量化损失!!!
    */
    // 再次注意：这里面没有二维数组，所有数据都是 一维向量 的形式进行索引
    /* 
    第一行关于指针：
    数组的变量名是指向 首地址 的指针，因此可以直接用bottom_rois + n * 5赋值指针；
    定义一个指向bottom_rois + n * 5位置的指针，指向第n个roi的首地址，其有五个参数bs x1 y1 x2 y2；
    可以直接将指针作为新的数组索引，0从当前所指的位置开始
    */
    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;   
    int roi_batch_ind = offset_bottom_rois[0];    // 指针指向的地址内容进行取出
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;   // 这个xyxy是ROI的坐标是要align的区域，float型，将roi从原图坐标缩放到特征图区域上！
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;   // 分别是左上点和右下点的坐标（不是像素）
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;  // 这里+1应该纯粹为了避免重叠
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);  // roi区域宽度，float，无损失（与0比较以取正值）
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.); // roi区域高度

    scalar_t bin_size_h = roi_height / pooled_height; // 划分成多个bin，每个bin的高和宽
    scalar_t bin_size_w = roi_width / pooled_width;

    /*
    和上面类似：bottom_data指向特征图存储的 ！首地址！
    具体来说是定位到当前bs当前channel（depth=1）的这张2D特征图左上角地址
    */
    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;  

    int sample_num_h = (sample_num > 0)   // 三目运算，设置了>0的sample_num，那么x方向取这么多个点
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =    
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);  // y方向同理，总共2*2=4个采样点

    // 下面四行代码更本没用，就是抄人加的挪过来忘了删掉，它实现的方式插值都形式不一样，这四个变量完全没必要
    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h; // h/w是bin中心点在特征图的坐标（也就是bottom）
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;
    int hstart = fminf(floor(h), height - 2); //和width-2比较取较小值，是因为现在求的是左上角，要给右下角留下位置，不能让右下角超出featuremap范围
    int wstart = fminf(floor(w), width - 2);

    scalar_t output_val = 0;
    // y方向遍历
    for (int iy = 0; iy < sample_num_h; iy++) {
      // 计算采样点的y坐标：roi的h + bin的位置（如：7*7的第几个bin）+ bin内的偏移（bin宽高除以采样点个数）
      // 但是这个iy+.5f（就是+0.5）说明不是按照严格的等距离采样的
      // 以w方向采样三个点为例，会把bin_w分为三份，每份取中点，就是插值点
      const scalar_t y = roi_start_h + ph * bin_size_h +
                         (scalar_t)(iy + scalar_t(.5f)) * bin_size_h /
                             (scalar_t)(sample_num_h);
      // x方向遍历采样
      for (int ix = 0; ix < sample_num_w; ix++) {
        // 计算采样点x坐标，原理相同不赘述
        const scalar_t x = roi_start_w + pw * bin_size_w +
                           (scalar_t)(ix + scalar_t(.5f)) * bin_size_w /
                               (scalar_t)(sample_num_w);
        scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data,
                                                      height, width, y, x);  // 双线性插值得到结果
        output_val += val;
      }
    }
    output_val /= (sample_num_h * sample_num_w);    // 这里的align取值方式是均值
    // 最终Roi上的这个点插值计算完毕赋值即可
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

/*
梯度插值不同于前向传播会返回插值的数值，这里没有返回值，而是完成传入参数的赋值
至于参数计算，实际是前向传播计算的子集，只完成插值的部分参数计算
实际上这个函数就是计算了lagrange插值的四个权重w1-w4系数（就够了，因为这次每项的函数值为特征图梯度，这部分在外面计算）
*/
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


/*
roialign的反向很简单，就是正向传播时，只有参与过双插的点才会有梯度，其他的点没有梯度。
所以只要把正向传播再进行一遍，找到做双插的点，每个做双插的点，在双插时前面的系数乘以梯度就是该点最终的梯度。

nthreads ：线程数，和正向传播时数量一样。
            正向时，每个线程通过交点周围4个点的值计算交点处的值；反向时，每个线程计算一个交点周围4个点的梯度。
top_diff ：pooling后每个点的梯度。这是存储数组的首地址。
bottom_diff ：pooling前整个featuremap上每个点的梯度，也是首地址，同样可以指针运算和数组取值，这里存储的是想要的梯度结果。
命名：
  上面的参数中，top是计算后的结果也就是输出的7*7梯度图的首地址，存储L对y的偏导数；
  bottom是计算前，也就是特征图梯度的地址，，存储L对x的偏导数，这是要求的东西
*/
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

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    scalar_t roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;
    // -----------相比forward下面为修改的代码--------------
    // 将forward的offset_bottom_data指针换成了offset_bottom_diff；新加了offset_top
    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;
    int offset_top = (n * channels + c) * pooled_height * pooled_width +
                     ph * pooled_width + pw;
    scalar_t offset_top_diff = top_diff[offset_top];
    // ----------------------------------------------------------------------
    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_height / pooled_height);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);
    
    const scalar_t count = (scalar_t)(sample_num_h * sample_num_w); // 统计采样点个数

    // 这四行依然是多余的
    scalar_t h = (scalar_t)(ph + 0.5) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw + 0.5) * bin_size_w + roi_start_w;
    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y =
          roi_start_h + ph * bin_size_h +
          (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x =
            roi_start_w + pw * bin_size_w +
            (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);
        // 遍历到每个插值点的xy坐标，进行梯度插值计算
        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        // w1-w4分别是双线性插值公式四项的权重
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