#ifndef HeterogeneousCore_SonicTriton_TritonData
#define HeterogeneousCore_SonicTriton_TritonData

#include <vector>
#include <string>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <algorithm>

#include "request_grpc.h"

//store all the info needed for triton input and output
template <typename IO>
class TritonData {
public:
  //constructor
  TritonData(std::shared_ptr<IO> data);

  //accessors

  //dims is fixed by the model info on server, immutable
  const std::vector<int64_t>& dims() const { return dims_; }

  //some members can be modified
  std::shared_ptr<IO>& data() { return data_; }
  std::vector<float>& vec() { return vec_; }
  std::vector<int64_t>& shape() { return shape_; }
  void reset();

  //const accessors
  const std::shared_ptr<IO>& data() const { return data_; }
  const std::vector<float>& vec() const { return vec_; }
  const std::vector<int64_t>& shape() const { return shape_; }

  //utilities
  bool variable_dims() const { return variable_dims_; }
  int64_t size_dims() const { return product_dims_; }
  //default to dims if shape isn't filled
  int64_t size_shape() const { return shape_.empty() ? size_dims() : dim_product(shape_); }

private:
  //helpers
  bool any_neg(const std::vector<int64_t>& vec) const {
    return std::any_of(vec.begin(), vec.end(), [](int64_t i) { return i < 0; });
  }
  int64_t dim_product(const std::vector<int64_t>& vec) const {
    return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
  }

  //members
  std::shared_ptr<IO> data_;
  std::vector<int64_t> dims_;
  bool variable_dims_;
  int64_t product_dims_;
  std::vector<float> vec_;
  std::vector<int64_t> shape_;
};

using TritonInputData = TritonData<nvidia::inferenceserver::client::InferContext::Input>;
using TritonInputMap = std::unordered_map<std::string, TritonInputData>;
using TritonOutputData = TritonData<nvidia::inferenceserver::client::InferContext::Output>;
using TritonOutputMap = std::unordered_map<std::string, TritonOutputData>;

//avoid "explicit specialization after instantiation" error
template <>
void TritonInputData::reset();
template <>
void TritonOutputData::reset();

//explicit template instantiation declarations
extern template class TritonData<nvidia::inferenceserver::client::InferContext::Input>;
extern template class TritonData<nvidia::inferenceserver::client::InferContext::Output>;

#endif
