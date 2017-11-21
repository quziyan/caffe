#define _USE_MATH_DEFINES
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layers/data_transformer_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h>
namespace caffe {

template <typename Dtype>
void DataTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  this->_canvas_w = this->layer_param_.data_transformer_l_param().canvas_w();
  this->_canvas_h = this->layer_param_.data_transformer_l_param().canvas_h();
  this->_canvas_w = (this->_canvas_w <=0) ? bottom[0]->width() : this->_canvas_w;
  this->_canvas_h = (this->_canvas_h <=0) ? bottom[0]->height() : this->_canvas_h;

  this->_colordistortion_delta1.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  this->_colordistortion_delta2.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  this->_colordistortion_delta3.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  this->_colordistortion_delta4.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

  this->_colordistortion_delta1_sigma = this->layer_param_.data_transformer_l_param().delta1_sigma();
  this->_colordistortion_delta2_sigma = this->layer_param_.data_transformer_l_param().delta2_sigma();
  this->_colordistortion_delta3_sigma = this->layer_param_.data_transformer_l_param().delta3_sigma();
  this->_colordistortion_delta4_sigma = this->layer_param_.data_transformer_l_param().delta4_sigma();

  this->_rotate_angle_scope = this->layer_param_.data_transformer_l_param().rotate_angle_scope();
  this->_translation_w_scope = this->layer_param_.data_transformer_l_param().translation_w_scope();
  this->_translation_h_scope = this->layer_param_.data_transformer_l_param().translation_h_scope();
  this->_scale_w_scope = this->layer_param_.data_transformer_l_param().scale_w_scope();
  this->_scale_h_scope = this->layer_param_.data_transformer_l_param().scale_h_scope();

  this->_rotate_angle_info.Reshape(bottom[0]->num(),1,1,1);
  this->_translation_w_info.Reshape(bottom[0]->num(),1,1,1);
  this->_translation_h_info.Reshape(bottom[0]->num(),1,1,1);
  this->_scale_w_info.Reshape(bottom[0]->num(),1,1,1);
  this->_scale_h_info.Reshape(bottom[0]->num(),1,1,1);
  
  this->_h_flip = this->layer_param_.data_transformer_l_param().h_flip();
  this->_h_flip_indicator.Reshape(bottom[0]->num(), 1, 1, 1);
  caffe_gpu_set(this->_h_flip_indicator.count(), (Dtype)0, this->_h_flip_indicator.mutable_gpu_data());

  this->_elastic_transform = this->layer_param_.data_transformer_l_param().elastic_transform();
  this->_amplitude = this->layer_param_.data_transformer_l_param().amplitude();
  this->_radius = this->layer_param_.data_transformer_l_param().radius();
  
  /*
  new version : add customized canvas width and height
  */
  this->_elastic_distortion_mtx_0.Reshape(bottom[0]->num(), 2, this->_canvas_h, this->_canvas_w);
  caffe_gpu_set(this->_elastic_distortion_mtx_0.count(), (Dtype)0, this->_elastic_distortion_mtx_0.mutable_gpu_data());
  this->_elastic_distortion_mtx_1.Reshape(bottom[0]->num(), 2, this->_canvas_h, this->_canvas_w);
  caffe_gpu_set(this->_elastic_distortion_mtx_1.count(), (Dtype)0, this->_elastic_distortion_mtx_1.mutable_gpu_data());

  int gaussian_ker_w_h = ceil(this->_radius * 3 * 2 + 1);
  if (gaussian_ker_w_h % 2 == 0){
    gaussian_ker_w_h += 1;
  }

  this->_gaussian_kernel_mtx.Reshape(1, 1, gaussian_ker_w_h, gaussian_ker_w_h);
  FillBlob_Gassian_Filter(this->_gaussian_kernel_mtx, this->_radius);

  this->_temp_mtx_0.Reshape(bottom[0]->num(), bottom[0]->channels(), this->_canvas_h, this->_canvas_w);
  this->_temp_mtx_1.Reshape(bottom[0]->num(), bottom[0]->channels(), this->_canvas_h, this->_canvas_w);

  this->_new_canvas.Reshape(bottom[0]->num(), bottom[0]->channels(), this->_canvas_h, this->_canvas_w);

  /* old version
  this->_elastic_distortion_mtx_0.Reshape(bottom[0]->num(), 2, bottom[0]->height(), bottom[0]->width());
  caffe_gpu_set(this->_elastic_distortion_mtx_0.count(), (Dtype)0, this->_elastic_distortion_mtx_0.mutable_gpu_data());
  this->_elastic_distortion_mtx_1.Reshape(bottom[0]->num(), 2, bottom[0]->height(), bottom[0]->width());
  caffe_gpu_set(this->_elastic_distortion_mtx_1.count(), (Dtype)0, this->_elastic_distortion_mtx_1.mutable_gpu_data());

  int gaussian_ker_w_h = ceil(this->_radius * 3 * 2 + 1);
  if (gaussian_ker_w_h % 2 == 0){
    gaussian_ker_w_h += 1;
  }

  this->_gaussian_kernel_mtx.Reshape(1, 1, gaussian_ker_w_h, gaussian_ker_w_h);
  FillBlob_Gassian_Filter(this->_gaussian_kernel_mtx, this->_radius);

  this->_temp_mtx_0.ReshapeLike(*(bottom[0]));
  this->_temp_mtx_1.ReshapeLike(*(bottom[0]));
  */



}


template <typename Dtype>
void DataTransformerLayer<Dtype>::FillBlob_Gassian_Filter(Blob<Dtype>& gaussian_kernel_mtx, Dtype radius){
  int k_height = gaussian_kernel_mtx.height();
  int k_width  =gaussian_kernel_mtx.width();
  CHECK_EQ(k_height % 2, 1) << " kernel height must be odd";
  CHECK_EQ(k_width % 2, 1) << " kernel width must be odd";
  int midpos_h = k_height / 2;
  int midpos_w = k_width / 2;
  for(int h_ = 0; h_<k_height; h_++){
    for(int w_ = 0; w_< k_width; w_++){
      int h_axis = h_ - midpos_h;
      int w_axis = w_ - midpos_w;
      gaussian_kernel_mtx.mutable_cpu_data()[h_ * k_width + w_] = 1/(2*M_PI*pow(radius,2)) * exp(  -( h_axis*h_axis + w_axis*w_axis )/(2*pow(radius,2))  );
    }
  }
}


template <typename Dtype>
void DataTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), this->_canvas_h, this->_canvas_w);
  //top[0]->ReshapeLike(*(bottom[0]));
}


template <typename Dtype>
void DataTransformerLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void DataTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}

#ifdef CPU_ONLY
STUB_GPU(DataTransformerLayer);
#endif

INSTANTIATE_CLASS(DataTransformerLayer);
REGISTER_LAYER_CLASS(DataTransformer);

}  // namespace caffe
