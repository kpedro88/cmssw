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

//shared memory helper, based on:
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc (cpu)
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/simple_grpc_cudashm_client.cc (gpu)
TritonShmResource::TritonShmResource(bool cpu, const std::string& name, size_t size, bool canThrow) : cpu_(cpu), name_(name), size_(size), addr_(nullptr), deviceId_(0), handle_(std::make_shared<cudaIpcMemHandle_t>()) {
  if(cpu_){
    //get shared memory region descriptor
    int shm_fd = shm_open(name_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (shm_fd == -1)
      triton_utils::warnOrThrow("unable to get shared memory descriptor for key: " + name_, canThrow);

    //extend shared memory object
    int res = ftruncate(shm_fd, size_);
    if (res == -1)
      triton_utils::warnOrThrow("unable to initialize shared memory key " + name_ + " to requested size: " + std::to_string(size_), canThrow);

    //map to process address space
    constexpr size_t offset(0);
    addr_ = (uint8_t*)mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
    if(addr_ == MAP_FAILED)
      triton_utils::warnOrThrow("unable to map to process address space for shared memory key: " + name_, canThrow);

    //close descriptor
    if(::close(shm_fd) == -1)
      triton_utils::warnOrThrow("unable to close descriptor for shared memory key: " + name_, canThrow);
  }
  else {
    triton_utils::cudaCheck(cudaMalloc((void**)&addr_, size_), "unable to allocate GPU memory for key: "+name_, canThrow);
    //todo: get server device id somehow?
    triton_utils::cudaCheck(cudaSetDevice(deviceId_), "unable to set device ID to "+std::to_string(deviceId_), canThrow);
    triton_utils::cudaCheck(cudaIpcGetMemHandle(handle_.get(), addr_), "unable to get IPC handle for key: "+name_, canThrow);
  }
}

bool TritonShmResource::close(bool canThrow) {
  if(cpu_){
    //unmap
    int tmp_fd = munmap(addr_, size_);
    if (tmp_fd == -1){
      triton_utils::warnOrThrow("unable to munmap for shared memory key: " + name_, canThrow);
      return false;
    }

    //unlink
    int shm_fd = shm_unlink(name_.c_str());
    if (shm_fd == -1){
      triton_utils::warnOrThrow("unable to unlink for shared memory key: " + name_, canThrow);
      return false;
    }
  }
  else {
    triton_utils::cudaCheck(cudaFree(addr_), "unable to free GPU memory for key: "+name_, canThrow);
  }
  return true;
}

TritonShmResource::~TritonShmResource() {
  //avoid throwing in destructor
  close(false);
}

//dims: kept constant, represents config.pbtxt parameters of model (converted from google::protobuf::RepeatedField to vector)
//fullShape: if batching is enabled, first entry is batch size; values can be modified
//shape: view into fullShape, excluding batch size entry
template <typename IO>
TritonData<IO>::TritonData(const std::string& name, const TritonData<IO>::TensorMetadata& model_info, TritonClient* client, const std::string& pid)
    : name_(name),
      client_(client),
      useShm_(client_->useSharedMemory() and client_->serverType()!=TritonServerType::Remote),
      //ensure unique name for shared memory region
      shmName_(useShm_ ? pid+"_"+xput()+std::to_string(uid()) : ""),
      cpu_(client_->serverType()==TritonServerType::LocalCPU),
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
  if(memResource_){
    if(cpu_)
      triton_utils::warnIfError(client_->client()->UnregisterSystemSharedMemory(shmName_), name_ + " destructor: unable to unregister shared memory region");
    else
      triton_utils::warnIfError(client_->client()->UnregisterCudaSharedMemory(shmName_), name_ + " destructor: unable to unregister CUDA shared memory region");
    memResource_.reset();
  }
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
bool TritonData<IO>::updateShm(size_t size, bool canThrow) {
  bool sizeIncreased = !memResource_ or size > memResource_->size();
  bool status = true;
  if(sizeIncreased) {
    if(memResource_) {
      if(cpu_)
        status &= triton_utils::warnOrThrowIfError(client_->client()->UnregisterSystemSharedMemory(shmName_), name_ + " updateShm(): unable to unregister shared memory region", canThrow);
      else
        status &= triton_utils::warnOrThrowIfError(client_->client()->UnregisterCudaSharedMemory(shmName_), name_ + " updateShm(): unable to unregister CUDA shared memory region", canThrow);
      memResource_.reset();
    }
    memResource_ = std::make_shared<TritonShmResource>(cpu_, shmName_, size, canThrow);
    if(cpu_)
      status &= triton_utils::warnOrThrowIfError(client_->client()->RegisterSystemSharedMemory(shmName_, shmName_, memResource_->size()), name_ + " updateShm(): unable to register shared memory region", canThrow);
    else
      status &= triton_utils::warnOrThrowIfError(client_->clientCuda()->RegisterCudaSharedMemory(shmName_, *(memResource_->handle()), memResource_->deviceId(), memResource_->size()), name_ + " updateShm(): unable to register CUDA shared memory region", canThrow);
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
  //decide between shared memory or gRPC call
  if (useShm_) {
    LogDebug(client_->fullDebugName()) << name_ << " toServer(): using " << (cpu_ ? "CPU" : "GPU") << " shared memory";
    updateShm(totalByteSize_, true);
    //copy into shared memory region
    for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
      if(cpu_)
        std::memcpy(memResource_->addr() + i0*byteSizePerBatch_, data_in[i0].data(), byteSizePerBatch_);
      else
        triton_utils::cudaCheck(cudaMemcpy((void*)(memResource_->addr() + i0*byteSizePerBatch_), (void*)(data_in[i0].data()), byteSizePerBatch_, cudaMemcpyHostToDevice), name_ + " toServer(): unable to memcpy "+std::to_string(byteSizePerBatch_)+" bytes to GPU", true);
    }
    constexpr size_t offset(0);
    triton_utils::throwIfError(data_->SetSharedMemory(shmName_, totalByteSize_, offset), name_ + " toServer(): unable to set shared memory");
  }
  else {
    for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
      triton_utils::throwIfError(data_->AppendRaw(reinterpret_cast<const uint8_t*>(data_in[i0].data()), byteSizePerBatch_),
                                 name_ + " toServer(): unable to set data for batch entry " + std::to_string(i0));
    }
  }
  //keep input data in scope
  holder_ = ptr;
}

//sets up shared memory for outputs, if possible
template <>
bool TritonOutputData::prepare() {
  if (!useShm_) return true;
  computeSizes();

  LogDebug(client_->fullDebugName()) << name_ << " prepare(): using " << (cpu_ ? "CPU" : "GPU") << " shared memory";
  bool status = true;
  status &= updateShm(totalByteSize_, false);
  status &= triton_utils::warnIfError(data_->SetSharedMemory(shmName_, totalByteSize_, 0), name_ + " prepare(): unable to set shared memory");

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

  TritonOutput<DT> dataOut;
  const uint8_t* r0;
  if (useShm_) {
    LogDebug(client_->fullDebugName()) << name_ << " fromServer(): using " << (cpu_ ? "CPU" : "GPU") << " shared memory";
    //outputs already loaded into ptr
    if(cpu_)
      r0 = memResource_->addr();
    else {
      //copy back from gpu, keep in scope
      auto ptr = std::make_shared<std::vector<uint8_t>>(totalByteSize_);
      triton_utils::cudaCheck(cudaMemcpy((void*)(ptr->data()), (void*)(memResource_->addr()), totalByteSize_, cudaMemcpyDeviceToHost), name_ + " fromServer(): unable to memcpy "+std::to_string(totalByteSize_)+" bytes from GPU", true);
      r0 = ptr->data();
      holder_ = ptr;
    }
  }
  else {
    size_t contentByteSize;
    triton_utils::throwIfError(result_->RawData(name_, &r0, &contentByteSize), name_ + " fromServer(): unable to get raw");
    if (contentByteSize != totalByteSize_) {
      throw cms::Exception("TritonDataError") << name_ << " fromServer(): unexpected content byte size " << contentByteSize
                                              << " (expected " << totalByteSize_ << ")";
    }
  }

  const DT* r1 = reinterpret_cast<const DT*>(r0);
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
