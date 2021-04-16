#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "model_config.pb.h"

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

//shared memory helper functions
//simplified from https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc
namespace {
  void createSharedMemoryRegion(const std::string& shm_key, size_t byte_size, void** shm_addr) {
    //get shared memory region descriptor
    int shm_fd = shm_open(shm_key.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (shm_fd == -1)
      throw cms::Exception("TritonSharedMemoryError") << "unable to get shared memory descriptor for key: " << shm_key;

    //extend shared memory object
    int res = ftruncate(shm_fd, byte_size);
    if (res == -1)
      throw cms::Exception("TritonSharedMemoryError") << "unable to initialize shared memory key " << shm_key << " to requested size: " << byte_size;

    //map to process address space
    constexpr size_t offset(0);
    *shm_addr = mmap(NULL, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
    if(*shm_addr == MAP_FAILED)
      throw cms::Exception("TritonSharedMemoryError") << "unable to map to process address space for shared memory key: " << shm_key;

    //close descriptor
    if(close(shm_fd) == -1)
      throw cms::Exception("TritonSharedMemoryError") << "unable to close descriptor for shared memory key: " << shm_key;
  }
  void destroySharedMemoryRegion(const std::string& shm_key, size_t byte_size, void* shm_addr) {
    //unmap
    int tmp_fd = munmap(shm_addr, byte_size);
    if (tmp_fd == -1)
      throw cms::Exception("TritonSharedMemoryError") << "unable to munmap for shared memory key: " << shm_key;

    //unlink
    int shm_fd = shm_unlink(shm_key.c_str());
    if (shm_fd == -1)
      throw cms::Exception("TritonSharedMemoryError") << "unable to unlink for shared memory key: " << shm_key;
  }
}

//dims: kept constant, represents config.pbtxt parameters of model (converted from google::protobuf::RepeatedField to vector)
//fullShape: if batching is enabled, first entry is batch size; values can be modified
//shape: view into fullShape, excluding batch size entry
template <typename IO>
TritonData<IO>::TritonData(const std::string& name, const TritonData<IO>::TensorMetadata& model_info, TritonClient* client, const std::string& pid)
    : name_(name),
      client_(client),
      //ensure unique name for shared memory region
      shmName_(client_->serverType()!=TritonServerType::Remote ? pid+"_"+xput()+std::to_string(uid()) : ""),
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
      totalByteSize_(0),
      holderShm_(nullptr) {
  //create input or output object
  IO* iotmp;
  createObject(&iotmp);
  data_.reset(iotmp);
}

template <>
void TritonInputData::createObject(nic::InferInput** ioptr) const {
  nic::InferInput::Create(ioptr, name_, fullShape_, dname_);
}

template <>
void TritonOutputData::createObject(nic::InferRequestedOutput** ioptr) const {
  nic::InferRequestedOutput::Create(ioptr, name_);
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
  unsigned full_loc = loc + (noBatch_ ? 0 : 1);

  //check boundary
  if (full_loc >= fullShape_.size()) {
    msg << name_ << " setShape(): dimension " << full_loc << " out of bounds (" << fullShape_.size() << ")";
    if (canThrow)
      throw cms::Exception("TritonDataError") << msg.str();
    else {
      edm::LogWarning("TritonDataWarning") << msg.str();
      return false;
    }
  }

  if (val != fullShape_[full_loc]) {
    if (dims_[full_loc] == -1) {
      fullShape_[full_loc] = val;
      return true;
    } else {
      msg << name_ << " setShape(): attempt to change value of non-variable shape dimension " << loc;
      if (canThrow)
        throw cms::Exception("TritonDataError") << msg.str();
      else {
        edm::LogWarning("TritonDataError") << msg.str();
        return false;
      }
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

//io accessors
template <>
template <typename DT>
TritonInputContainer<DT> TritonInputData::allocate(bool reserve) {
  //automatically creates a vector for each batch entry
  auto ptr = std::make_shared<TritonInput<DT>>(batchSize_);
  if(reserve){
    auto size = sizeShape();
    if(size>0){
      for(auto& vec: *ptr){
        vec.reserve(size);
      }
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

  int64_t nInput = sizeShape();
  //decide between shared memory or gRPC call
  if (client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " toServer(): using CPU shared memory";
    size_t byteSizePerBatch = byteSize_*nInput;
    totalByteSize_ = byteSizePerBatch*batchSize_;
    DT* ptrShm;
    createSharedMemoryRegion(shmName_, totalByteSize_, (void**)&ptrShm);
    //copy into shared memory region
    for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
      std::memcpy(ptrShm + i0*nInput, data_in[i0].data(), byteSizePerBatch);
    }
    triton_utils::throwIfError(client_->client()->RegisterSystemSharedMemory(shmName_, shmName_, totalByteSize_), name_ + " toServer(): unable to register shared memory region");
    triton_utils::throwIfError(data_->SetSharedMemory(shmName_, totalByteSize_, 0), name_ + " toServer(): unable to set shared memory");
    //keep shm ptr
    holderShm_ = ptrShm;
    //possible future enhancement:
    //modify allocate() so input is written directly into shared memory, w/o memcpy
    //to do this at runtime would require a custom allocator that normally behaves as std::allocator,
    //but behaves as allocator<T,managed_shared_memory::segment_manager> (using boost::interprocess) in the LocalCPU case
    //todo: determine if this would work even if batch size and concrete shape not known before calling allocate(), and if it would actually be faster
  }
  else if (client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
  else {
    for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
      const DT* arr = data_in[i0].data();
      triton_utils::throwIfError(data_->AppendRaw(reinterpret_cast<const uint8_t*>(arr), nInput * byteSize_),
                                 name_ + " input(): unable to set data for batch entry " + std::to_string(i0));
    }
    //keep input data in scope
    holder_ = std::move(ptr);
  }
}

//sets up shared memory for outputs, if possible
template <>
void TritonOutputData::prepare() {
  //can't use shared memory if output size is not known
  if (client_->serverType()==TritonServerType::Remote or variableDims_) return;

  uint64_t nOutput = sizeShape();
  totalByteSize_ = byteSize_*nOutput*batchSize_;
  if (client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " prepare(): using CPU shared memory";
    //type-agnostic: just use char
    uint8_t* ptrShm;
    createSharedMemoryRegion(shmName_, totalByteSize_, (void**)&ptrShm);
    triton_utils::throwIfError(client_->client()->RegisterSystemSharedMemory(shmName_, shmName_, totalByteSize_), name_ + " prepare(): unable to register shared memory region");
    triton_utils::throwIfError(data_->SetSharedMemory(shmName_, totalByteSize_, 0), name_ + " prepare(): unable to set shared memory");
    //keep shm ptr
    holderShm_ = ptrShm;
  }
  else if (client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
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

  uint64_t nOutput = sizeShape();
  TritonOutput<DT> dataOut;
  const uint8_t* r0;
  if (!variableDims_ and client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " fromServer(): using CPU shared memory";
    //outputs already loaded into ptr
    r0 = (uint8_t*)holderShm_;
  }
  else if (!variableDims_ and client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
  else {
    size_t contentByteSize;
    size_t expectedContentByteSize = nOutput * byteSize_ * batchSize_;
    triton_utils::throwIfError(result_->RawData(name_, &r0, &contentByteSize), name_ + " fromServer(): unable to get raw");
    if (contentByteSize != expectedContentByteSize) {
      throw cms::Exception("TritonDataError") << name_ << " fromServer(): unexpected content byte size " << contentByteSize
                                              << " (expected " << expectedContentByteSize << ")";
    }
  }

  const DT* r1 = reinterpret_cast<const DT*>(r0);
  dataOut.reserve(batchSize_);
  for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
    auto offset = i0 * nOutput;
    dataOut.emplace_back(r1 + offset, r1 + offset + nOutput);
  }

  return dataOut;
}

template <>
void TritonInputData::reset() {
  if (client_->serverType()==TritonServerType::LocalCPU)
    resetShm();
  else if (client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
  else
    holder_.reset();
  data_->Reset();
}

template <>
void TritonOutputData::reset() {
  if (!variableDims_){
    if (client_->serverType()==TritonServerType::LocalCPU)
      resetShm();
    else if (client_->serverType()==TritonServerType::LocalGPU) {
      //todo
    }
  }
  result_.reset();
}

template <typename IO>
void TritonData<IO>::resetShm() {
  triton_utils::throwIfError(client_->client()->UnregisterSystemSharedMemory(shmName_), name_ + " reset(): unable to unregister shared memory region");
  destroySharedMemoryRegion(shmName_, totalByteSize_, holderShm_);
  totalByteSize_ = 0;
  holderShm_ = nullptr;
}

//explicit template instantiation declarations
template class TritonData<nic::InferInput>;
template class TritonData<nic::InferRequestedOutput>;

template TritonInputContainer<float> TritonInputData::allocate(bool reserve);
template TritonInputContainer<int64_t> TritonInputData::allocate(bool reserve);

template void TritonInputData::toServer(TritonInputContainer<float> data_in);
template void TritonInputData::toServer(TritonInputContainer<int64_t> data_in);

template TritonOutput<float> TritonOutputData::fromServer() const;
