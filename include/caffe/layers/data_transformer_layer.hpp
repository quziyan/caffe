#ifndef DATA_TRANSFORMER_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class DataMixedSynthesisLayer : public Layer<Dtype> {
 public:
  explicit DataMixedSynthesisLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "DataMixedSynthesis"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype RandomBeta(Dtype alpha, Dtype beta);
  void RandomBetas(const int N, Dtype alpha, Dtype beta, Dtype* betas);
  int _gen_data_num;
  //parameter of beta distribution, which lambda is under distribution BETA(alpha, alpha)
  int _initial_alpha;
  Blob<Dtype> _lambdas;
  Blob<Dtype> _index_1, _index_2;

};
}
#endif //# DATA_TRANSFORMER_LAYER_HPP_
