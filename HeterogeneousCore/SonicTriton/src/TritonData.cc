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
#include <new>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

namespace nvidia {
  namespace inferenceserver {
    //in libgrpcclient.so, but corresponding header src/core/model_config.h not available
    size_t GetDataTypeByteSize(const inference::DataType dtype);
    inference::DataType ProtocolStringToDataType(const std::string& dtype);
  }  // namespace inferenceserver
}  // namespace nvidia

//shared memory helper
//based on https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc
//very simplified allocator:
//the shared memory region is created when the memory resource is initialized
//allocate() and deallocate() just increment and decrement a counter that keeps track of position in shm region
//region is actually destroyed in destructor
template <typename DT>
TritonShmResource<DT>::TritonShmResource(std::string name, size_t size) : name_(name), size_(size), counter_(0), addr_(nullptr) {
  //get shared memory region descriptor
  int shm_fd = shm_open(name_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (shm_fd == -1)
    throw cms::Exception("TritonSharedMemoryError") << "unable to get shared memory descriptor for key: " << name_;

  //extend shared memory object
  int res = ftruncate(shm_fd, size_);
  if (res == -1)
    throw cms::Exception("TritonSharedMemoryError") << "unable to initialize shared memory key " << name_ << " to requested size: " << size_;

  //map to process address space
  constexpr size_t offset(0);
  addr_ = (DT*)mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if(addr_ == MAP_FAILED)
    throw cms::Exception("TritonSharedMemoryError") << "unable to map to process address space for shared memory key: " << name_;

  //close descriptor
  if(::close(shm_fd) == -1)
    throw cms::Exception("TritonSharedMemoryError") << "unable to close descriptor for shared memory key: " << name_;
}

template <typename DT>
void TritonShmResource<DT>::close() {
  //unmap
  int tmp_fd = munmap(addr_, size_);
  if (tmp_fd == -1)
    throw cms::Exception("TritonSharedMemoryError") << "unable to munmap for shared memory key: " << name_;

  //unlink
  int shm_fd = shm_unlink(name_.c_str());
  if (shm_fd == -1)
    throw cms::Exception("TritonSharedMemoryError") << "unable to unlink for shared memory key: " << name_;
}

template <typename DT>
TritonShmResource<DT>::~TritonShmResource() {
  //avoid throwing in destructor
  try {
    close();
  }
  catch (...) {}
}

template <typename DT>
void* TritonShmResource<DT>::do_allocate(std::size_t bytes, std::size_t alignment) {
  size_t old_counter = counter_;
  counter_ += bytes;
  if(counter_>size_)
    throw std::runtime_error("Attempt to allocate "+std::to_string(bytes)+" bytes in region with only "+std::to_string(size_-old_counter)+" bytes free");
  void* result = addr_ + old_counter;
  std::cout << "TritonShmResource::allocate() : " << bytes << " bytes, " << size_-counter_ << " remaining (" << result << ")" << std::endl;
  return result;
}

template <typename DT>
void TritonShmResource<DT>::do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
  if(bytes>counter_)
    throw std::runtime_error("Attempt to deallocate "+std::to_string(bytes)+" bytes in region with only "+std::to_string(counter_)+" bytes used");
  counter_ -= bytes;
  std::cout << "TritonShmResource::deallocate() : " << bytes << " bytes, " << counter_ << " remaining" << std::endl;
}

template <typename DT>
bool TritonShmResource<DT>::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return dynamic_cast<const TritonShmResource*>(&other) != nullptr;
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
      totalByteSize_(0) {
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
  auto size = sizeShape();
  size_t byteSizePerBatch = byteSize_*size;
  totalByteSize_ = byteSizePerBatch*batchSize_;
  std::cout << "totalByteSize = size * byteSize * batchSize = " << size << " * " << byteSize_ << " * " << batchSize_ << " = " << totalByteSize_ << std::endl;
  //choose allocator: shared memory or default (heap)
  if (client_->useSharedMemory() and client_->serverType()==TritonServerType::LocalCPU) {
    edm::LogInfo(client_->fullDebugName()) << name_ << " toServer(): using CPU shared memory";
    //allocated bytes include space for vector overhead
    memResource_ = std::make_shared<TritonShmResource<DT>>(shmName_, (batchSize_+1)*sizeof(std::pmr::vector<DT>) + totalByteSize_);
  }
  //automatically creates a vector for each batch entry
  auto ptr = std::make_shared<TritonInput<DT>>(batchSize_, memResource_ ? memResource_.get() : std::pmr::get_default_resource());
  if(reserve and !anyNeg(shape_)){
    for(auto& vec: *ptr){
      vec.reserve(size);
    }
  }
  return ptr;
}

template <>
template <typename DT>
void TritonInputData::toServer(TritonInputContainer<DT> ptr) {
  const auto& data_in = *ptr;
  for(unsigned i = 0; i < batchSize_; ++i){
    edm::LogInfo(client_->fullDebugName()) << name_ << "[" << i << "] = " << triton_utils::printColl(data_in[i], ",");
  }

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
  if (client_->useSharedMemory() and client_->serverType()==TritonServerType::LocalCPU) {
    size_t byteSizePerBatch = byteSize_*nInput;
    totalByteSize_ = byteSizePerBatch*batchSize_;
/*    edm::LogInfo(client_->fullDebugName()) << name_ << " toServer(): using CPU shared memory";
    memResource_ = std::make_shared<TritonShmResource<DT>>(shmName_, totalByteSize_);
    //todo: just use allocate again here
    //copy into shared memory region
    for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
      std::memcpy(ptrShm + i0*nInput, data_in[i0].data(), byteSizePerBatch);
    }*/
    triton_utils::throwIfError(client_->client()->RegisterSystemSharedMemory(shmName_, shmName_, totalByteSize_), name_ + " toServer(): unable to register shared memory region");
    //offset calculated to exclude vector overhead at beginning of shm region
    size_t offset = (data_in[0].data() - std::static_pointer_cast<TritonShmResource<DT>>(memResource_)->addr());
    std::cout << shmName_ << " offset = " << offset << std::endl;
    triton_utils::throwIfError(data_->SetSharedMemory(shmName_, totalByteSize_, offset), name_ + " toServer(): unable to set shared memory");
    //possible future enhancement:
    //modify allocate() so input is written directly into shared memory, w/o memcpy
    //to do this at runtime would require a custom allocator that normally behaves as std::allocator,
    //but behaves as allocator<T,managed_shared_memory::segment_manager> (using boost::interprocess) in the LocalCPU case
    //todo: determine if this would work even if batch size and concrete shape not known before calling allocate(), and if it would actually be faster
  }
  else if (client_->useSharedMemory() and client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
  else {
    for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
      const DT* arr = data_in[i0].data();
      triton_utils::throwIfError(data_->AppendRaw(reinterpret_cast<const uint8_t*>(arr), nInput * byteSize_),
                                 name_ + " input(): unable to set data for batch entry " + std::to_string(i0));
    }
  }
  //keep input data in scope
  holder_ = std::move(ptr);
}

//sets up shared memory for outputs, if possible
template <>
bool TritonOutputData::prepare() {
  //can't use shared memory if output size is not known
  if (!client_->useSharedMemory() or client_->serverType()==TritonServerType::Remote or variableDims_) return true;

  bool status = true;
  uint64_t nOutput = sizeShape();
  totalByteSize_ = byteSize_*nOutput*batchSize_;
  if (client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " prepare(): using CPU shared memory";
    //type-agnostic: just use char
    memResource_ = std::make_shared<TritonShmResource<uint8_t>>(shmName_, totalByteSize_);
    status &= triton_utils::warnIfError(client_->client()->RegisterSystemSharedMemory(shmName_, shmName_, totalByteSize_), name_ + " prepare(): unable to register shared memory region");
    status &= triton_utils::warnIfError(data_->SetSharedMemory(shmName_, totalByteSize_, 0), name_ + " prepare(): unable to set shared memory");
  }
  else if (client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
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

  uint64_t nOutput = sizeShape();
  TritonOutput<DT> dataOut;
  const uint8_t* r0;
  if (client_->useSharedMemory() and !variableDims_ and client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " fromServer(): using CPU shared memory";
    //outputs already loaded into ptr
    r0 = std::static_pointer_cast<TritonShmResource<uint8_t>>(memResource_)->addr();
  }
  else if (client_->useSharedMemory() and !variableDims_ and client_->serverType()==TritonServerType::LocalGPU) {
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
  if (client_->useSharedMemory() and client_->serverType()==TritonServerType::LocalCPU)
    resetShm();
  else if (client_->useSharedMemory() and client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
  else
    holder_.reset();
  data_->Reset();
}

template <>
void TritonOutputData::reset() {
  if (client_->useSharedMemory() and !variableDims_){
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
  memResource_.reset();
  totalByteSize_ = 0;
}

//explicit template instantiation declarations
template class TritonData<nic::InferInput>;
template class TritonData<nic::InferRequestedOutput>;

template TritonInputContainer<float> TritonInputData::allocate(bool reserve);
template TritonInputContainer<int64_t> TritonInputData::allocate(bool reserve);

template void TritonInputData::toServer(TritonInputContainer<float> data_in);
template void TritonInputData::toServer(TritonInputContainer<int64_t> data_in);

template TritonOutput<float> TritonOutputData::fromServer() const;
