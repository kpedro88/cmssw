#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonUtils.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "model_config.pb.h"

#include <cstring>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

namespace nvidia {
  namespace inferenceserver {
    //in librequest.so, but corresponding header src/core/model_config.h not available
    size_t GetDataTypeByteSize(const DataType dtype);
  }  // namespace inferenceserver
}  // namespace nvidia

template <typename IO>
TritonData<IO>::TritonData(const std::string& name, std::shared_ptr<IO> data)
    : name_(name), data_(data), batch_size_(0) {
  //convert google::protobuf::RepeatedField to vector
  const auto& dimsTmp = data_->Dims();
  dims_.assign(dimsTmp.begin(), dimsTmp.end());

  //check if variable dimensions
  variable_dims_ = any_neg(dims_);
  if (variable_dims_)
    product_dims_ = -1;
  else
    product_dims_ = dim_product(dims_);

  dtype_ = data_->DType();
  dname_ = ni::DataType_Name(dtype_);
  //get byte size for input conversion
  byte_size_ = ni::GetDataTypeByteSize(dtype_);
}

//io accessors
template <typename IO>
template <typename DT>
void TritonData<IO>::to_server(std::shared_ptr<std::vector<DT>> ptr) {}

template <typename IO>
template <typename DT>
void TritonData<IO>::from_server(std::vector<DT>& data_in) const {}

template <>
template <typename DT>
void TritonInputData::to_server(std::shared_ptr<std::vector<DT>> ptr) {
  const auto& data_in = *(ptr.get());

  //shape must be specified for variable dims
  if (variable_dims_) {
    if (shape_.size() != dims_.size()) {
      throw cms::Exception("TritonDataError")
          << name_ << " input(): incorrect or missing shape (" << triton_utils::print_vec(shape_)
          << ") for model with variable dimensions (" << triton_utils::print_vec(dims_) << ")";
    } else {
      triton_utils::wrap(data_->SetShape(shape_), name_ + " input(): unable to set input shape");
    }
  }

  if (byte_size_ != sizeof(DT))
    throw cms::Exception("TritonDataError") << name_ << " input(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byte_size_ << " for " << dname_ << ")";

  int64_t nInput = size_shape();
  for (unsigned i0 = 0; i0 < batch_size_; ++i0) {
    const DT* arr = &(data_in[i0 * nInput]);
    triton_utils::wrap(data_->SetRaw(reinterpret_cast<const uint8_t*>(arr), nInput * byte_size_),
                       name_ + " input(): unable to set data for batch entry " + std::to_string(i0));
  }

  //keep input data in scope
  callback_ = [ptr]() { return; };
}

template <>
template <typename DT>
void TritonOutputData::from_server(std::vector<DT>& data_out) const {
  if (!result_) {
    throw cms::Exception("TritonDataError") << name_ << " output(): missing result";
  }

  //shape must be specified for variable dims
  if (variable_dims_) {
    if (shape_.size() != dims_.size()) {
      throw cms::Exception("TritonDataError")
          << name_ << " output(): incorrect or missing shape (" << triton_utils::print_vec(shape_)
          << ") for model with variable dimensions (" << triton_utils::print_vec(dims_) << ")";
    }
  }

  if (byte_size_ != sizeof(DT)) {
    throw cms::Exception("TritonDataError") << name_ << " output(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byte_size_ << " for " << dname_ << ")";
  }

  int64_t nOutput = size_shape();
  data_out.resize(nOutput * batch_size_, 0);
  for (unsigned i0 = 0; i0 < batch_size_; ++i0) {
    const uint8_t* r0;
    size_t content_byte_size;
    triton_utils::wrap(result_->GetRaw(i0, &r0, &content_byte_size),
                       "output(): unable to get raw for entry " + std::to_string(i0));
    std::memcpy(&(data_out[i0 * nOutput]), r0, nOutput * byte_size_);
  }
}

template <>
void TritonInputData::reset() {
  shape_.clear();
  data_->Reset();
  if(callback_) callback_();
}

template <>
void TritonOutputData::reset() {
  shape_.clear();
  result_.reset();
}

//explicit template instantiation declarations
template class TritonData<nic::InferContext::Input>;
template class TritonData<nic::InferContext::Output>;

template void TritonInputData::to_server(std::shared_ptr<std::vector<float>> data_in);
template void TritonInputData::to_server(std::shared_ptr<std::vector<int64_t>> data_in);

template void TritonOutputData::from_server(std::vector<float>& data_out) const;
