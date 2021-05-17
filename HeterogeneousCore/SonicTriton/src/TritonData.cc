#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "model_config.pb.h"

#include "cuda_runtime_api.h"

#include <cstring>
#include <sstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

namespace nvidia {
  namespace inferenceserver {
    //in libgrpcclient.so, but corresponding header src/core/model_config.h not available
    size_t GetDataTypeByteSize(const inference::DataType dtype);
    inference::DataType ProtocolStringToDataType(const std::string& dtype);
  }  // namespace inferenceserver
}  // namespace nvidia

template <typename IO>
TritonMemResource<IO>::TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow) : data_(data), name_(name), size_(size), addr_(nullptr), status_(true) {}

template <typename IO>
bool TritonMemResource<IO>::set(bool canThrow){
  return triton_utils::warnOrThrowIfError(data_->data_->SetSharedMemory(name_, data_->totalByteSize_, 0), "unable to set shared memory ("+name_+")", canThrow);
}

template <typename IO>
TritonHeapResource<IO>::TritonHeapResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow) : TritonMemResource<IO>(data, name, size, canThrow) {}

template <>
void TritonInputHeapResource::copy(const void* values, size_t offset) {
  triton_utils::throwIfError(data_->data_->AppendRaw(reinterpret_cast<const uint8_t*>(values), data_->byteSizePerBatch_), data_->name_ + " toServer(): unable to set data for batch entry "+std::to_string(offset/data_->byteSizePerBatch_));
}

template <>
void TritonOutputHeapResource::copy(void** values) {
  size_t contentByteSize;
  triton_utils::throwIfError(data_->result_->RawData(data_->name_, const_cast<const uint8_t**>(reinterpret_cast<uint8_t**>(values)), &contentByteSize), data_->name_+" fromServer(): unable to get raw");
  if (contentByteSize != data_->totalByteSize_) {
    throw cms::Exception("TritonDataError") << data_->name_ << " fromServer(): unexpected content byte size " << contentByteSize
                                            << " (expected " << data_->totalByteSize_ << ")";
  }
}

//shared memory helpers based on:
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc (cpu)
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/simple_grpc_cudashm_client.cc (gpu)

template <typename IO>
TritonCpuShmResource<IO>::TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow) : TritonMemResource<IO>(data, name, size, canThrow) {
  //get shared memory region descriptor
  int shm_fd = shm_open(this->name_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (shm_fd == -1){
    triton_utils::warnOrThrow("unable to get shared memory descriptor for key: " + this->name_, canThrow);
    this->status_ &= false;
  }

  //extend shared memory object
  int res = ftruncate(shm_fd, this->size_);
  if (res == -1){
    triton_utils::warnOrThrow("unable to initialize shared memory key " + this->name_ + " to requested size: " + std::to_string(this->size_), canThrow);
    this->status_ &= false;
  }

  //map to process address space
  constexpr size_t offset(0);
  this->addr_ = (uint8_t*)mmap(NULL, this->size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if(this->addr_ == MAP_FAILED){
    triton_utils::warnOrThrow("unable to map to process address space for shared memory key: " + this->name_, canThrow);
    this->status_ &= false;
  }

  //close descriptor
  if(::close(shm_fd) == -1){
    triton_utils::warnOrThrow("unable to close descriptor for shared memory key: " + this->name_, canThrow);
    this->status_ &= false;
  }

  this->status_ &= triton_utils::warnOrThrowIfError(this->data_->client()->RegisterSystemSharedMemory(this->name_, this->name_, this->size_), "unable to register shared memory region: "+this->name_, canThrow);
}

template <typename IO>
TritonCpuShmResource<IO>::~TritonCpuShmResource<IO>() {
  triton_utils::warnIfError(this->data_->client()->UnregisterSystemSharedMemory(this->name_), "unable to unregister shared memory region: "+this->name_);

  //unmap
  int tmp_fd = munmap(this->addr_, this->size_);
  if (tmp_fd == -1)
    edm::LogWarning("TritonWarning") << "unable to munmap for shared memory key: " << this->name_;

  //unlink
  int shm_fd = shm_unlink(this->name_.c_str());
  if (shm_fd == -1)
    edm::LogWarning("TritonWarning") << "unable to unlink for shared memory key: " << this->name_;
}

template <>
void TritonInputCpuShmResource::copy(const void* values, size_t offset) {
  std::memcpy(addr_ + offset, values, data_->byteSizePerBatch_);
}

template <>
void TritonOutputCpuShmResource::copy(void** values) {
  *values = addr_;
}

template <typename IO>
TritonGpuShmResource<IO>::TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow) : TritonMemResource<IO>(data, name, size, canThrow), deviceId_(0), handle_(std::make_shared<cudaIpcMemHandle_t>()) {
  this->status_ &= triton_utils::cudaCheck(cudaMalloc((void**)&this->addr_, this->size_), "unable to allocate GPU memory for key: "+this->name_, canThrow);
  //todo: get server device id somehow?
  this->status_ &= triton_utils::cudaCheck(cudaSetDevice(deviceId_), "unable to set device ID to "+std::to_string(deviceId_), canThrow);
  this->status_ &= triton_utils::cudaCheck(cudaIpcGetMemHandle(handle_.get(), this->addr_), "unable to get IPC handle for key: "+this->name_, canThrow);
  this->status_ &= triton_utils::warnOrThrowIfError(this->data_->client()->RegisterCudaSharedMemory(this->name_, *handle_, deviceId_, this->size_), "unable to register CUDA shared memory region: "+this->name_, canThrow);
}

template <typename IO>
TritonGpuShmResource<IO>::~TritonGpuShmResource<IO>() {
  triton_utils::warnIfError(this->data_->client()->UnregisterCudaSharedMemory(this->name_), "unable to unregister CUDA shared memory region: "+this->name_);
  triton_utils::cudaCheck(cudaFree(this->addr_), "unable to free GPU memory for key: "+this->name_, false);
}

template <>
void TritonInputGpuShmResource::copy(const void* values, size_t offset) {
  triton_utils::cudaCheck(cudaMemcpy((void*)(addr_ + offset), values, data_->byteSizePerBatch_, cudaMemcpyHostToDevice), data_->name_ + " toServer(): unable to memcpy "+std::to_string(data_->byteSizePerBatch_)+" bytes to GPU", true);
}

template <>
void TritonOutputGpuShmResource::copy(void** values) {
  //copy back from gpu, keep in scope
  auto ptr = std::make_shared<std::vector<uint8_t>>(data_->totalByteSize_);
  triton_utils::cudaCheck(cudaMemcpy((void*)(ptr->data()), (void*)(addr_), data_->totalByteSize_, cudaMemcpyDeviceToHost), data_->name_ + " fromServer(): unable to memcpy "+std::to_string(data_->totalByteSize_)+" bytes from GPU", true);
  *values = ptr->data();
  data_->holder_ = ptr;
}

//dims: kept constant, represents config.pbtxt parameters of model (converted from google::protobuf::RepeatedField to vector)
//fullShape: if batching is enabled, first entry is batch size; values can be modified
//shape: view into fullShape, excluding batch size entry
template <typename IO>
TritonData<IO>::TritonData(const std::string& name, const TritonData<IO>::TensorMetadata& model_info, TritonClient* client, const std::string& pid)
    : name_(name),
      client_(client),
      useShm_(client_->useSharedMemory()),
      //ensure unique name for shared memory region
      shmName_(useShm_ ? pid+"_"+xput()+std::to_string(uid()) : ""),
      dims_(model_info.shape().begin(), model_info.shape().end()),
      noBatch_(client_->noBatch()),
      batchSize_(0),
      fullShape_(dims_),
      shape_(fullShape_.begin() + (noBatch_ ? 0 : 1), fullShape_.end()),
      variableDims_(anyNeg(shape_)),
      productDims_(variableDims_ ? -1 : dimProduct(shape_)),
      dname_(model_info.datatype()),
      dtype_(ni::ProtocolStringToDataType(dname_)),
      byteSize_(ni::GetDataTypeByteSize(dtype_)),
      totalByteSize_(0) {
  //create input or output object
  IO* iotmp;
  createObject(&iotmp);
  data_.reset(iotmp);
}

template <typename IO>
TritonData<IO>::~TritonData() {
  memResource_.reset();
}

template <>
void TritonInputData::createObject(nic::InferInput** ioptr) {
  nic::InferInput::Create(ioptr, name_, fullShape_, dname_);
}

template <>
void TritonOutputData::createObject(nic::InferRequestedOutput** ioptr) {
  nic::InferRequestedOutput::Create(ioptr, name_);
  //another specialization for output: can't use shared memory if output size is not known
  useShm_ &= !variableDims_;
}

template <>
std::string TritonInputData::xput() const { return "input"; }

template <>
std::string TritonOutputData::xput() const { return "output"; }

template <typename IO>
auto TritonData<IO>::client() { return client_->client(); }

//setters
template <typename IO>
bool TritonData<IO>::setShape(const TritonData<IO>::ShapeType& newShape, bool canThrow) {
  bool result = true;
  for (unsigned i = 0; i < newShape.size(); ++i) {
    result &= setShape(i, newShape[i], canThrow);
  }
  return result;
}

template <typename IO>
bool TritonData<IO>::setShape(unsigned loc, int64_t val, bool canThrow) {
  std::stringstream msg;
  unsigned locFull = fullLoc(loc);

  //check boundary
  if (locFull >= fullShape_.size()) {
    msg << name_ << " setShape(): dimension " << locFull << " out of bounds (" << fullShape_.size() << ")";
    triton_utils::warnOrThrow(msg.str(), canThrow);
    return false;
  }

  if (val != fullShape_[locFull]) {
    if (dims_[locFull] == -1) {
      fullShape_[locFull] = val;
      return true;
    } else {
      msg << name_ << " setShape(): attempt to change value of non-variable shape dimension " << loc;
      triton_utils::warnOrThrow(msg.str(), canThrow);
      return false;
    }
  }

  return true;
}

template <typename IO>
void TritonData<IO>::setBatchSize(unsigned bsize) {
  batchSize_ = bsize;
  if (!noBatch_)
    fullShape_[0] = batchSize_;
}

template <typename IO>
void TritonData<IO>::computeSizes() {
  sizeShape_ = sizeShape();
  byteSizePerBatch_ = byteSize_*sizeShape_;
  totalByteSize_ = byteSizePerBatch_*batchSize_;
}
template <typename IO>
void TritonData<IO>::resetSizes() {
  sizeShape_ = 0;
  byteSizePerBatch_ = 0;
  totalByteSize_ = 0;
}

//create a memory resource if none exists;
//otherwise, reuse the memory resource, resizing it if necessary
template <typename IO>
bool TritonData<IO>::updateMem(size_t size, bool canThrow) {
  bool status = true;
  if(!memResource_ or size > memResource_->size()) {
    if(useShm_ and client_->serverType()==TritonServerType::LocalCPU){
      memResource_.reset();
      memResource_ = std::make_shared<TritonCpuShmResource<IO>>(this, shmName_, size, canThrow);
    }
    else if(useShm_ and client_->serverType()==TritonServerType::LocalGPU){
      memResource_.reset();
      memResource_ = std::make_shared<TritonGpuShmResource<IO>>(this, shmName_, size, canThrow);
    }
    //for remote/heap, size increases don't matter
    else if(!memResource_)
      memResource_ = std::make_shared<TritonHeapResource<IO>>(this, shmName_, size, canThrow);

    status &= memResource_->status();
  }

  return status;
}

//io accessors
template <>
template <typename DT>
TritonInputContainer<DT> TritonInputData::allocate(bool reserve) {
  //automatically creates a vector for each batch entry (if batch size known)
  auto ptr = std::make_shared<TritonInput<DT>>(batchSize_);
  if(reserve and !anyNeg(shape_)){
    computeSizes();
    for(auto& vec: *ptr){
      vec.reserve(sizeShape_);
    }
  }
  return ptr;
}

template <>
template <typename DT>
void TritonInputData::toServer(TritonInputContainer<DT> ptr) {
  const auto& data_in = *ptr;

  //check batch size
  if (data_in.size() != batchSize_) {
    throw cms::Exception("TritonDataError") << name_ << " toServer(): input vector has size " << data_in.size()
                                            << " but specified batch size is " << batchSize_;
  }

  //shape must be specified for variable dims or if batch size changes
  data_->SetShape(fullShape_);

  if (byteSize_ != sizeof(DT))
    throw cms::Exception("TritonDataError") << name_ << " toServer(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";

  computeSizes();
  updateMem(totalByteSize_, true);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    memResource_->copy(data_in[i0].data(), i0*byteSizePerBatch_);
  }
  memResource_->set(true);

  //keep input data in scope
  holder_ = ptr;
}

//sets up shared memory for outputs, if possible
template <>
bool TritonOutputData::prepare() {
  computeSizes();

  bool status = updateMem(totalByteSize_, false) && memResource_->set(false);

  return status;
}

template <>
template <typename DT>
TritonOutput<DT> TritonOutputData::fromServer() const {
  if (!result_) {
    throw cms::Exception("TritonDataError") << name_ << " fromServer(): missing result";
  }

  if (byteSize_ != sizeof(DT)) {
    throw cms::Exception("TritonDataError") << name_ << " fromServer(): inconsistent byte size " << sizeof(DT)
                                            << " (should be " << byteSize_ << " for " << dname_ << ")";
  }

  uint8_t* r0;
  memResource_->copy((void**)&r0);
  const DT* r1 = reinterpret_cast<const DT*>(r0);

  TritonOutput<DT> dataOut;
  dataOut.reserve(batchSize_);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    auto offset = i0 * sizeShape_;
    dataOut.emplace_back(r1 + offset, r1 + offset + sizeShape_);
  }

  return dataOut;
}

template <>
void TritonInputData::reset() {
  holder_.reset();
  data_->Reset();
  //reset shape
  if(variableDims_){
    for(unsigned i = 0; i < shape_.size(); ++i){
      unsigned locFull = fullLoc(i);
      fullShape_[locFull] = dims_[locFull];
    }
  }
  resetSizes();
}

template <>
void TritonOutputData::reset() {
  result_.reset();
  holder_.reset();
  resetSizes();
}

//explicit template instantiation declarations
template class TritonData<nic::InferInput>;
template class TritonData<nic::InferRequestedOutput>;

template TritonInputContainer<float> TritonInputData::allocate(bool reserve);
template TritonInputContainer<int64_t> TritonInputData::allocate(bool reserve);

template void TritonInputData::toServer(TritonInputContainer<float> data_in);
template void TritonInputData::toServer(TritonInputContainer<int64_t> data_in);

template TritonOutput<float> TritonOutputData::fromServer() const;
