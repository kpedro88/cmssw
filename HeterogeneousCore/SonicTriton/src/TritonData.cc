#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "model_config.pb.h"

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

namespace nvidia {
  namespace inferenceserver {
    //in librequest.so, but corresponding header src/core/model_config.h not available
    size_t GetDataTypeByteSize(const DataType dtype);
  }  // namespace inferenceserver
}  // namespace nvidia

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

  //get byte size for input conversion
  byte_size_ = ni::GetDataTypeByteSize(data_->DType());
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
template class TritonData<nic::InferContext::Input>;
template class TritonData<nic::InferContext::Output>;
