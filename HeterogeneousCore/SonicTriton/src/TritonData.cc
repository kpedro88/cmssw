#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

template <typename IO>
TritonData<IO>::TritonData(std::shared_ptr<IO> data) : data_(data) {
  //convert google::protobuf::RepeatedField to vector
  const auto& dimsTmp = data_->Dims();
  dims_.assign(dimsTmp.begin(), dimsTmp.end());

  //check if variable dimensions
  variable_dims_ = any_neg(dims_);
  if (variable_dims_)
    product_dims_ = -1;
  else
    product_dims_ = dim_product(dims_);
}

template <>
void TritonInputData::reset() {
  vec_.clear();
  shape_.clear();
  data_->Reset();
}

template <>
void TritonOutputData::reset() {
  vec_.clear();
  shape_.clear();
}

//explicit template instantiation declarations
template class TritonData<nvidia::inferenceserver::client::InferContext::Input>;
template class TritonData<nvidia::inferenceserver::client::InferContext::Output>;
