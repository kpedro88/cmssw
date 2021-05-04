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
#include <cerrno>

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
TritonShmResource::TritonShmResource(std::string name, size_t size, bool canThrow) : name_(name), size_(size), counter_(0), addr_(nullptr) {
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

bool TritonShmResource::remap(size_t newSize, bool canThrow) {
  void* new_addr = mremap(addr_, size_, newSize, MREMAP_MAYMOVE);
  if(new_addr == (void*)-1){
    triton_utils::warnOrThrow("unable to remap shared memory key " + name_ + " from " + std::to_string(size_) + " to " + std::to_string(newSize) + " bytes (errno "+std::to_string(errno)+")", canThrow);
    return false;
  }
  addr_ = (uint8_t*)new_addr;
  return true;
}

void TritonShmResource::close(bool canThrow) {
  //unmap
  int tmp_fd = munmap(addr_, size_);
  if (tmp_fd == -1)
    triton_utils::warnOrThrow("unable to munmap for shared memory key: " + name_, canThrow);

  //unlink
  int shm_fd = shm_unlink(name_.c_str());
  if (shm_fd == -1)
    triton_utils::warnOrThrow("unable to unlink for shared memory key: " + name_, canThrow);
}

TritonShmResource::~TritonShmResource() {
  //avoid throwing in destructor
  close(false);
}

//todo: make this resizeable with mremap
//requires a better shm interface from Triton: need to be able to specify non-contiguous chunks
//currently, all inner vectors, including overhead, are in one contiguous chunk
//this assumes all push_back() calls are in order, which might not be true for resizeable usage
//each inner vector should have its own shm chunk in resizeable case (overhead on heap)
void* TritonShmResource::do_allocate(std::size_t bytes, std::size_t alignment) {
  size_t old_counter = counter_;
  counter_ += bytes;
  if(counter_>size_)
    throw std::runtime_error("Attempt to allocate "+std::to_string(bytes)+" bytes in region with only "+std::to_string(size_-old_counter)+" bytes free");
  void* result = addr_ + old_counter;
  return result;
}

void TritonShmResource::do_deallocate(void* p, std::size_t bytes, std::size_t alignment) {
  if(bytes>counter_)
    throw std::runtime_error("Attempt to deallocate "+std::to_string(bytes)+" bytes in region with only "+std::to_string(counter_)+" bytes used");
  counter_ -= bytes;
}

bool TritonShmResource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
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
      lockShape_(false),
      concreteShape_(false),
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
    triton_utils::warnIfError(client_->client()->UnregisterSystemSharedMemory(shmName_), name_ + " destructor: unable to unregister shared memory region");
    memResource_.reset();
  }
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
void TritonData<IO>::checkLockShape() const {
  if(lockShape_)
    throw cms::Exception("TritonDataError") << name_ << " setShape(): disabled because allocate() was already called with a concrete shape and shared memory";
}

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

//create a memory resource if none exists;
//otherwise, reuse the memory resource, resizing it if necessary
template <typename IO>
bool TritonData<IO>::updateShm(size_t size, bool canThrow) {
  bool sizeIncreased = false;
  bool status = true;
  if(!memResource_) {
    memResource_ = std::make_shared<TritonShmResource>(shmName_, size, canThrow);
    sizeIncreased = true;
  }
  else {
    sizeIncreased = size > memResource_->size();
    if(sizeIncreased)
      status &= triton_utils::warnOrThrowIfError(client_->client()->UnregisterSystemSharedMemory(shmName_), name_ + " updateShm(): unable to unregister shared memory region", canThrow);
    if(size != memResource_->size())
      status &= memResource_->remap(size, canThrow);
  }

  //only need to update the server if size increased
  if(sizeIncreased)
    status &= triton_utils::warnOrThrowIfError(client_->client()->RegisterSystemSharedMemory(shmName_, shmName_, memResource_->size()), name_ + " updateShm(): unable to register shared memory region", canThrow);

  return status;
}

//io accessors
template <>
template <typename DT>
TritonInputContainer<DT> TritonInputData::allocate(bool reserve) {
  auto size = sizeShape();
  size_t byteSizePerBatch = byteSize_*size;
  totalByteSize_ = byteSizePerBatch*batchSize_;
  //choose allocator: shared memory or default (heap)
  //shared memory can only be used at this stage if the batch size and shape are set
  concreteShape_ = batchSize_>0 and !anyNeg(shape_);
  if (client_->useSharedMemory() and concreteShape_ and client_->serverType()==TritonServerType::LocalCPU) {
    edm::LogInfo(client_->fullDebugName()) << name_ << " allocate(): using CPU shared memory";
    //allocated bytes include space for vector overhead
    updateShm((batchSize_)*sizeof(std::pmr::vector<DT>) + totalByteSize_, true);
    //prevent changing batch size or shape
    lockShape_ = true;
    client_->lockBatch();
  }
  //automatically creates a vector for each batch entry (if batch size known)
  auto ptr = std::make_shared<TritonInput<DT>>(batchSize_, (concreteShape_ and memResource_) ? memResource_.get() : std::pmr::get_default_resource());
  //can only reserve if batch size and shape are set
  if(reserve and concreteShape_){
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
    size_t offset = 0;
    //check if more-efficient shm approach could not be used
    if(!concreteShape_) {
      edm::LogInfo(client_->fullDebugName()) << name_ << " toServer(): using CPU shared memory";
      updateShm(totalByteSize_, true);
      //copy into shared memory region
      for (unsigned i0 = 0; i0 < batchSize_; ++i0) {
        std::memcpy(memResource_->addr() + i0*byteSizePerBatch, data_in[i0].data(), byteSizePerBatch);
      }
    }
    else {
      //offset calculated to exclude vector overhead at beginning of shm region
      offset = ((uint8_t*)data_in[0].data() - memResource_->addr());
    }
    triton_utils::throwIfError(data_->SetSharedMemory(shmName_, totalByteSize_, offset), name_ + " toServer(): unable to set shared memory");
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
  holder_ = ptr;
}

//sets up shared memory for outputs, if possible
template <>
bool TritonOutputData::prepare() {
  //can't use shared memory if output size is not known
  if (!client_->useSharedMemory() or variableDims_ or client_->serverType()==TritonServerType::Remote) return true;

  bool status = true;
  uint64_t nOutput = sizeShape();
  totalByteSize_ = byteSize_*nOutput*batchSize_;
  if (client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " prepare(): using CPU shared memory";
    status &= updateShm(totalByteSize_, false);
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
  bool canUseShm = client_->useSharedMemory() and !variableDims_;
  if (canUseShm and client_->serverType()==TritonServerType::LocalCPU) {
    LogDebug(client_->fullDebugName()) << name_ << " fromServer(): using CPU shared memory";
    //outputs already loaded into ptr
    r0 = memResource_->addr();
  }
  else if (canUseShm and client_->serverType()==TritonServerType::LocalGPU) {
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
  holder_.reset();
  if (client_->useSharedMemory() and client_->serverType()==TritonServerType::LocalGPU) {
    //todo
  }
  data_->Reset();
  //reset batch and shape
  if(!noBatch_) batchSize_ = 0;
  if(variableDims_){
    for(unsigned i = 0; i < shape_.size(); ++i){
      unsigned locFull = fullLoc(i);
      fullShape_[locFull] = dims_[locFull];
    }
  }
  lockShape_ = false;
  concreteShape_ = false;
  totalByteSize_ = 0;
}

template <>
void TritonOutputData::reset() {
  if (client_->useSharedMemory() and !variableDims_){
    if (client_->serverType()==TritonServerType::LocalGPU) {
      //todo
    }
  }
  result_.reset();
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
