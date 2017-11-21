#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void TransformMapping(const int nthreads,
  const int num, const int channels, const int height, const int width,
  const Dtype* rotate_angle_n, const Dtype* translation_w_n, const Dtype* translation_h_n, const Dtype* scale_w_n, const Dtype* scale_h_n,
  const Dtype* elastic_distortion_mtx,
  const Dtype* bottom_data, Dtype* top_data)  {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n_top = index / ( channels * height * width);
    int c_top = index / ( height * width ) % channels;
    int h_top = index / width % height;
    int w_top = index % width;

    Dtype rotate_angle = rotate_angle_n[n_top];
    Dtype translation_w = translation_w_n[n_top];
    Dtype translation_h = translation_h_n[n_top];
    Dtype scale_w = scale_w_n[n_top];
    Dtype scale_h = scale_h_n[n_top];
    

    //Dtype w_bottom = w_top * cosf(rotate_angle) / scale_w + h_top * sinf(rotate_angle) / scale_h - translation_w;
    //Dtype h_bottom = w_top * ( -sinf(rotate_angle) / scale_w ) + h_top * cosf(rotate_angle) / scale_h - translation_h;

    Dtype w_bottom = cosf(rotate_angle) / scale_w * w_top + sinf(rotate_angle) / scale_h * h_top 
            - translation_w * cosf(rotate_angle) / scale_w - translation_h * sinf(rotate_angle) / scale_h
            - width * cosf(rotate_angle) / (2 * scale_w) - height * sinf(rotate_angle) / ( 2 * scale_h) + width / 2.0;
    Dtype h_bottom = -sinf(rotate_angle) /scale_w * w_top + cosf(rotate_angle) / scale_h * h_top
            + translation_w * sinf(rotate_angle) / scale_h - translation_h * cosf(rotate_angle) / scale_h
            + width * sinf(rotate_angle) / (2 * scale_w) - height * cosf(rotate_angle) / (2 * scale_h) + height / 2.0;

    w_bottom += elastic_distortion_mtx[n_top * (2 * height * width) + 0 * height * width + h_top * width + w_top];
    h_bottom += elastic_distortion_mtx[n_top * (2 * height * width) + 1 * height * width + h_top * width + w_top];

    int c_bottom = c_top;
    int n_bootom = n_top;

    if(w_bottom >=0 && w_bottom < width && h_bottom >=0 && h_bottom < height){
      if(false){
        top_data[index] = bottom_data[n_bootom * channels * height * width + c_bottom * height * width + (int)h_bottom * width + (int)w_bottom];
        int q11_w = fmaxf(floor(w_bottom), 0);
        int q11_h = fminf(ceil(h_bottom), height-1);
        int q12_w = fmaxf(floor(w_bottom), 0);
        int q12_h = fmaxf(floor(h_bottom), 0);
        int q21_w = fminf(ceil(w_bottom), width-1);
        int q21_h = fminf(ceil(h_bottom), height-1);
        int q22_w = fminf(ceil(w_bottom), width-1);
        int q22_h = fmaxf(floor(h_bottom), 0);
        Dtype q11_data = 0, q12_data = 0, q21_data = 0, q22_data = 0;
        q11_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q11_h * width + q11_w];
        top_data[index] = q11_data;
      }

      else{
      // Here need bilinear interpolation
      // ref to http://www.cnblogs.com/xpvincent/archive/2013/03/15/2961448.html
      // but the left-top point is zero point
      int q11_w = fmaxf(floor(w_bottom), 0);
      int q11_h = fminf(ceil(h_bottom), height-1);
      int q12_w = fmaxf(floor(w_bottom), 0);
      int q12_h = fmaxf(floor(h_bottom), 0);
      int q21_w = fminf(ceil(w_bottom), width-1);
      int q21_h = fminf(ceil(h_bottom), height-1);
      int q22_w = fminf(ceil(w_bottom), width-1);
      int q22_h = fmaxf(floor(h_bottom), 0);
      Dtype q11_data = 0, q12_data = 0, q21_data = 0, q22_data = 0;

      if(q11_h == q22_h && q11_w == q22_w){
        q11_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q11_h * width + q11_w];
        top_data[index] = q11_data;
      }
      else if(q11_h == q22_h && q11_w != q22_w){
        q11_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q11_h * width + q11_w];
        q21_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q21_h * width + q21_w];
        top_data[index] = fabsf( (q21_w - w_bottom) / (q21_w - q11_w) ) * q11_data + fabsf( (w_bottom - q11_w) / (q21_w - q11_w) ) * q21_data;
      }
      else if(q11_h != q22_h && q11_w == q22_w){
        q11_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q11_h * width + q11_w];
        q12_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q12_h * width + q12_w];
        top_data[index] = fabs( (h_bottom - q12_h) / (q11_h - q12_h) ) * q11_data + fabs( (q11_h - h_bottom) / (q11_h - q12_h) ) * q12_data;
      }
      else{
        q11_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q11_h * width + q11_w];
        q12_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q12_h * width + q12_w];
        q21_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q21_h * width + q21_w];
        q22_data = bottom_data[n_bootom * (channels * width * height) + c_bottom * (height * width) + q22_h * width + q22_w];

        Dtype w2 = q22_w;
        Dtype w1 = q11_w;
        Dtype h1 = q11_h;
        Dtype h2 = q22_h;
        Dtype area = fabsf( (w2 - w1) * (h2 - h1) );

        top_data[index] = q11_data * fabsf( (w2 - w_bottom) * (h2 - h_bottom) / area ) 
                        + q12_data * fabsf( (w2 - w_bottom) * (h1 - h_bottom) / area )
                        + q21_data * fabsf( (w1 - w_bottom) * (h2 - h_bottom) / area )
                        + q22_data * fabsf( (w1 - w_bottom) * (h1 - h_bottom) / area );
        /*
        top_data[index] = q11_data * fabsf( (h_bottom - q12_h) / (q11_h - q12_h) * (q21_w - w_bottom) / (q21_w - q11_w) ) +
                      q12_data * fabsf( (q11_h - h_bottom) / (q11_h - q12_h) * (q22_w - w_bottom) / (q22_w - q12_w) ) +
                      q22_data * fabsf( (q21_h - h_bottom) / (q21_h - q22_h) * (w_bottom - q12_w) / (q22_w - q12_w) ) +
                      q21_data * fabsf( (h_bottom - q22_h) / (q21_h - q22_h) * (w_bottom - q11_w) / (q21_w - q11_w) );
        */
      }
      }

      //top_data[index] = bottom[n_bootom * (channels * width * height) + c_bottom * (height * width) + h_bottom * width + w_bottom];
    }
    else {
      top_data[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void ColorDistortion(const int nthreads,
  const int num, const int channels, const int height, const int width,
  const Dtype* colordistortion_delta1, const Dtype* colordistortion_delta2, const Dtype* colordistortion_delta3, const Dtype* colordistortion_delta4,
  const Dtype* bottom_data, Dtype* top_data)  {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / ( channels * height * width);
    int c = index / ( height * width ) % channels;
    int h = index / width % height;
    int w = index % width;

    top_data[index] = bottom_data[index] + // ori_value
                      colordistortion_delta1[n * channels + c] + // absolute distortion
                      colordistortion_delta2[n * channels + c] * bottom_data[index] + // relative distortion
                      colordistortion_delta3[n * channels + c] * (h - (Dtype)height / 2) + // position-wise(height) distortion
                      colordistortion_delta4[n * channels + c] * (w - (Dtype)width / 2);  // position-wise(width) distortion

  }
}


template <typename Dtype>
__global__ void HorizontalFlip(const int nthreads,
  const int num, const int channels, const int height, const int width,
  const Dtype* bottom_data, const Dtype* h_flip_indicator, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / ( channels * height * width);
    int c = index / ( height * width ) % channels;
    int top_h = index / width % height;
    int top_w = index % width;

    //if(h_flip_indicator[n] > (Dtype)0.5){
    int bottom_h = top_h;
    int bottom_w = h_flip_indicator[n] > (Dtype)0.5 ? width - 1 - top_w : top_w;
    int bottom_index = n * (channels * height * width) + c * (height * width) + bottom_h * width + bottom_w;
    top_data[index] = bottom_data[bottom_index];
    //}
  }
}

/*
void GenTransformInstanceParameter(Dtype rotate_angle_scope,
      Dtype translation_w_scope, Dtype translation_h_scope,
      Dtype scale_w_scope, Dtype scale_h_scope,
      Dtype* rotate_angle, Dtype* translation_w, Dtype* translation_h, Dtype* scale_w, Dtype* scale_h){
  Dtype rotate_angle_min = -rotate_angle_scope;
  Dtype rotate_angle_max = rotate_angle_scope;
  Dtype translation_w_min = -translation_w_scope;
  Dtype translation_w_max = translation_w_scope;
  Dtype translation_h_min = -translation_h_scope;
  Dtype translation_h_max = translation_h_scope;
  Dtype scale_w_min = -scale_w_scope;
  Dtype scale_w_max = scale_w_scope;
  Dtype scale_h_min = -scale_h_scope;
  Dtype scale_h_max = scale_h_scope;
  caffe_gpu_rng_uniform
}
*/


template <typename Dtype>
__global__ void Channelwise_Convolution(const int nthreads,
  const int num, const int channels, const int height, const int width,
  const int kernel_height, const int kernel_width,
  const Dtype* input_data, const Dtype* kernel_data, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / ( channels * height * width);
    int c = index / ( height * width ) % channels;
    int h = index / width % height;
    int w = index % width;

    int h_start = h - kernel_height / 2;
    int h_end = h + kernel_height / 2;
    int w_start = w - kernel_width / 2;
    int w_end = w + kernel_width / 2;

    output_data[index] = 0;
    for(int h_ = h_start; h_ < h_end; h_++){
      for(int w_ = w_start; w_ < w_end; w_++){
        int ker_h_ = h_ - h_start;
        int ker_w_ = w_ - w_start;
        if(h_ >= 0 && h_ <= height -1 && w_ >= 0 && w_ <= width-1){
          output_data[index] += input_data[n * channels * height * width + c * height * width + h_ * width + w_] * kernel_data[ker_h_ * kernel_width + ker_w_];
        }
      }
    }
  }
}

template <typename Dtype>
void DataTransformerLayer<Dtype>::Initialize_Elastic_Mtx(Blob<Dtype>& elastic_distortion_mtx_0, Blob<Dtype>& gaussian_kernel_mtx,Blob<Dtype>& elastic_distortion_mtx_1, Dtype amplitude, Dtype radius){
  if(!this->_elastic_transform){
    return;
  }

  caffe_gpu_rng_gaussian<Dtype>(elastic_distortion_mtx_0.count(), (Dtype)0, amplitude * radius, elastic_distortion_mtx_0.mutable_gpu_data());

  //caffe_gpu_set()
  int num = elastic_distortion_mtx_0.num();
  int channels = elastic_distortion_mtx_0.channels(); // should be 2
  int height = elastic_distortion_mtx_0.height();
  int width = elastic_distortion_mtx_0.width();

  int k_height = gaussian_kernel_mtx.height();
  int k_width  =gaussian_kernel_mtx.width();
  CHECK_EQ(k_height % 2, 1) << " kernel height must be odd";
  CHECK_EQ(k_width % 2, 1) << " kernel width must be odd";
  const int count = elastic_distortion_mtx_0.count();
  Channelwise_Convolution<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, num, channels, height, width,
                  k_height, k_width,
                  elastic_distortion_mtx_0.gpu_data(), gaussian_kernel_mtx.gpu_data(), elastic_distortion_mtx_1.mutable_gpu_data());
}


template <typename Dtype>
__global__ void RemappingBottom2Canvas(const int nthreads,
  const int num_bottom, const int channels_bottom, const int height_bottom, const int width_bottom,
  const int height_top, const int width_top,
  const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n_bottom = index / ( channels_bottom * height_bottom * width_bottom);
    int c_bottom = index / ( height_bottom * width_bottom ) % channels_bottom;
    int h_bottom = index / width_bottom % height_bottom;
    int w_bottom = index % width_bottom;


    int num_top = num_bottom;
    int channels_top = channels_bottom;
    int n_top = n_bottom;
    int c_top = c_bottom;
    int h_top = (int)(h_bottom - height_bottom / 2.0 + height_top / 2.0);
    int w_top = (int)(w_bottom - width_bottom / 2.0 + width_top / 2.0);
    if(h_top >= 0 && h_top < height_top && w_top >= 0 && w_top < width_top){
      int top_index = n_top * (channels_top * height_top * width_top) + c_top * (height_top * width_top) + h_top * width_top + w_top;
      top_data[top_index] = bottom_data[index];
    }
  }
}



template <typename Dtype>
void DataTransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 0 END!";
  const int num_bottom = bottom[0]->num();
  const int channels_bottom = bottom[0]->channels();
  const int height_bottom = bottom[0]->height();
  const int width_bottom = bottom[0]->width();
  const int count_bottom = bottom[0]->count();

  const int num_top = top[0]->num();
  const int channels_top = top[0]->channels();
  const int height_top = top[0]->height();
  const int width_top = top[0]->width();
  const int count_top = top[0]->count();

  caffe_gpu_set<Dtype>(_new_canvas.count(), (Dtype)0, _new_canvas.mutable_gpu_data());
  RemappingBottom2Canvas<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(count_bottom, num_bottom, channels_bottom, height_bottom, width_bottom,
            height_top, width_top,
            bottom[0]->gpu_data(), this->_new_canvas.mutable_gpu_data());


  /*
    caffe_gpu_rng_uniform(this->_rotate_angle_info.count(), -this->_rotate_angle_scope, this->_rotate_angle_scope, this->_rotate_angle_info.mutable_gpu_data());
      LOG(INFO)<<"INITIAL_ELASTIC_MTX 1_1 END!";
      LOG(INFO)<< this->_colordistortion_delta1.count() << "   "<<this->_colordistortion_delta1_sigma;
      LOG(INFO)<< this->_colordistortion_delta2.count() << "   "<<this->_colordistortion_delta2_sigma;
      LOG(INFO)<< this->_colordistortion_delta3.count() << "   "<<this->_colordistortion_delta3_sigma;
      LOG(INFO)<< this->_colordistortion_delta4.count() << "   "<<this->_colordistortion_delta4_sigma;
  */
  /* 1.仿射变换
  2.通道抖动
  */
  // Color distortion
  //注意这块，好像batchsize太小了会出现问题NNDX
  caffe_gpu_rng_gaussian(this->_colordistortion_delta1.count(), (Dtype)0, this->_colordistortion_delta1_sigma, this->_colordistortion_delta1.mutable_gpu_data());
  caffe_gpu_rng_gaussian(this->_colordistortion_delta2.count(), (Dtype)0, this->_colordistortion_delta2_sigma, this->_colordistortion_delta2.mutable_gpu_data());
  caffe_gpu_rng_gaussian(this->_colordistortion_delta3.count(), (Dtype)0, this->_colordistortion_delta3_sigma, this->_colordistortion_delta3.mutable_gpu_data());
  caffe_gpu_rng_gaussian(this->_colordistortion_delta4.count(), (Dtype)0, this->_colordistortion_delta4_sigma, this->_colordistortion_delta4.mutable_gpu_data());
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 1 END!";
  ColorDistortion<Dtype><<<CAFFE_GET_BLOCKS(count_top), CAFFE_CUDA_NUM_THREADS>>>(count_top, num_top, channels_top, height_top, width_top,
                  this->_colordistortion_delta1.gpu_data(), this->_colordistortion_delta2.gpu_data(), this->_colordistortion_delta3.gpu_data(), this->_colordistortion_delta4.gpu_data(),
                  this->_new_canvas.gpu_data(), this->_temp_mtx_0.mutable_gpu_data());


  // Affine tranformer Parameters
  // bottom[0] -> Scale -> Rotate -> Translation -> top[0]
  // Material http://www.jianshu.com/p/5e1f776a33f9
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 2 END!";
  caffe_gpu_rng_uniform(this->_rotate_angle_info.count(), -this->_rotate_angle_scope, this->_rotate_angle_scope, this->_rotate_angle_info.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->_translation_w_info.count(), -this->_translation_w_scope, this->_translation_w_scope, this->_translation_w_info.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->_translation_h_info.count(), -this->_translation_h_scope, this->_translation_h_scope, this->_translation_h_info.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->_scale_w_info.count(), (Dtype)1/this->_scale_w_scope, this->_scale_w_scope, this->_scale_w_info.mutable_gpu_data());
  caffe_gpu_rng_uniform(this->_scale_h_info.count(), (Dtype)1/this->_scale_h_scope, this->_scale_h_scope, this->_scale_h_info.mutable_gpu_data());
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 3 END!";
  Initialize_Elastic_Mtx(this->_elastic_distortion_mtx_0, this->_gaussian_kernel_mtx,this->_elastic_distortion_mtx_1, this->_amplitude, this->_radius);
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 4 END!";
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 0 END!";
  TransformMapping<Dtype><<<CAFFE_GET_BLOCKS(count_top), CAFFE_CUDA_NUM_THREADS>>>(count_top,
      num_top, channels_top, height_top, width_top,
      this->_rotate_angle_info.gpu_data(), this->_translation_w_info.gpu_data(), this->_translation_h_info.gpu_data(), this->_scale_w_info.gpu_data(), this->_scale_h_info.gpu_data(),
      this->_elastic_distortion_mtx_1.gpu_data(),
      this->_temp_mtx_0.gpu_data(), this->_temp_mtx_1.mutable_gpu_data());
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 1 END!";
  //caffe_gpu_memcpy(top[0]->count(), top[0]->gpu_data(), this->_temp_mtx_0.mutable_gpu_data());
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 5 END!";
  //Horizontal Flip
  if(this->_h_flip){
    caffe_gpu_rng_uniform(this->_h_flip_indicator.count(),(Dtype)0, (Dtype)1, this->_h_flip_indicator.mutable_gpu_data());
  }
    HorizontalFlip<Dtype><<<CAFFE_GET_BLOCKS(count_top), CAFFE_CUDA_NUM_THREADS>>>(count_top,
      num_top, channels_top, height_top, width_top, this->_temp_mtx_1.gpu_data(), this->_h_flip_indicator.gpu_data(), top[0]->mutable_gpu_data());
  /*
  }
  else{
    caffe_gpu_memcpy(this->_temp_mtx_1.count(), this->_temp_mtx_1.gpu_data(), top[0]->mutable_gpu_data());
  }
  */
  //LOG(INFO)<<"INITIAL_ELASTIC_MTX 3 END!";
  
}

template <typename Dtype>
void DataTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Nothing to do here.
  return;
  
}

INSTANTIATE_LAYER_GPU_FUNCS(DataTransformerLayer);

}  // namespace caffe
