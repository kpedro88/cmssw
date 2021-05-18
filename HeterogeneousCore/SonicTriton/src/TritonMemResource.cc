#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonMemResource.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "HeterogeneousCore/SonicTriton/interface/grpc_client_gpu.h"

#include "cuda_runtime_api.h"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

template <typename IO>
TritonMemResource<IO>::TritonMemResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow)
    : data_(data), name_(name), size_(size), addr_(nullptr), status_(true) {}

template <typename IO>
bool TritonMemResource<IO>::set(bool canThrow) {
  return triton_utils::warnOrThrowIfError(data_->data_->SetSharedMemory(name_, data_->totalByteSize_, 0),
                                          "unable to set shared memory (" + name_ + ")",
                                          canThrow);
}

template <typename IO>
TritonHeapResource<IO>::TritonHeapResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow)
    : TritonMemResource<IO>(data, name, size, canThrow) {}

template <>
void TritonInputHeapResource::copy(const void* values, size_t offset) {
  triton_utils::throwIfError(
      data_->data_->AppendRaw(reinterpret_cast<const uint8_t*>(values), data_->byteSizePerBatch_),
      data_->name_ + " toServer(): unable to set data for batch entry " +
          std::to_string(offset / data_->byteSizePerBatch_));
}

template <>
void TritonOutputHeapResource::copy(const uint8_t** values) {
  size_t contentByteSize;
  triton_utils::throwIfError(data_->result_->RawData(data_->name_, values, &contentByteSize),
                             data_->name_ + " fromServer(): unable to get raw");
  if (contentByteSize != data_->totalByteSize_) {
    throw cms::Exception("TritonDataError") << data_->name_ << " fromServer(): unexpected content byte size "
                                            << contentByteSize << " (expected " << data_->totalByteSize_ << ")";
  }
}

//shared memory helpers based on:
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/shm_utils.cc (cpu)
// https://github.com/triton-inference-server/server/blob/v2.3.0/src/clients/c++/examples/simple_grpc_cudashm_client.cc (gpu)

template <typename IO>
TritonCpuShmResource<IO>::TritonCpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow)
    : TritonMemResource<IO>(data, name, size, canThrow) {
  //get shared memory region descriptor
  int shm_fd = shm_open(this->name_.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
  if (shm_fd == -1) {
    triton_utils::warnOrThrow("unable to get shared memory descriptor for key: " + this->name_, canThrow);
    this->status_ &= false;
  }

  //extend shared memory object
  int res = ftruncate(shm_fd, this->size_);
  if (res == -1) {
    triton_utils::warnOrThrow(
        "unable to initialize shared memory key " + this->name_ + " to requested size: " + std::to_string(this->size_),
        canThrow);
    this->status_ &= false;
  }

  //map to process address space
  constexpr size_t offset(0);
  this->addr_ = (uint8_t*)mmap(nullptr, this->size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, offset);
  if (this->addr_ == MAP_FAILED) {
    triton_utils::warnOrThrow("unable to map to process address space for shared memory key: " + this->name_, canThrow);
    this->status_ &= false;
  }

  //close descriptor
  if (::close(shm_fd) == -1) {
    triton_utils::warnOrThrow("unable to close descriptor for shared memory key: " + this->name_, canThrow);
    this->status_ &= false;
  }

  this->status_ &= triton_utils::warnOrThrowIfError(
      this->data_->client()->RegisterSystemSharedMemory(this->name_, this->name_, this->size_),
      "unable to register shared memory region: " + this->name_,
      canThrow);
}

template <typename IO>
TritonCpuShmResource<IO>::~TritonCpuShmResource<IO>() {
  triton_utils::warnIfError(this->data_->client()->UnregisterSystemSharedMemory(this->name_),
                            "unable to unregister shared memory region: " + this->name_);

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
void TritonOutputCpuShmResource::copy(const uint8_t** values) {
  *values = addr_;
}

template <typename IO>
TritonGpuShmResource<IO>::TritonGpuShmResource(TritonData<IO>* data, const std::string& name, size_t size, bool canThrow)
    : TritonMemResource<IO>(data, name, size, canThrow), deviceId_(0), handle_(std::make_shared<cudaIpcMemHandle_t>()) {
  this->status_ &= triton_utils::cudaCheck(
      cudaMalloc((void**)&this->addr_, this->size_), "unable to allocate GPU memory for key: " + this->name_, canThrow);
  //todo: get server device id somehow?
  this->status_ &= triton_utils::cudaCheck(
      cudaSetDevice(deviceId_), "unable to set device ID to " + std::to_string(deviceId_), canThrow);
  this->status_ &= triton_utils::cudaCheck(
      cudaIpcGetMemHandle(handle_.get(), this->addr_), "unable to get IPC handle for key: " + this->name_, canThrow);
  this->status_ &= triton_utils::warnOrThrowIfError(
      this->data_->client()->RegisterCudaSharedMemory(this->name_, *handle_, deviceId_, this->size_),
      "unable to register CUDA shared memory region: " + this->name_,
      canThrow);
}

template <typename IO>
TritonGpuShmResource<IO>::~TritonGpuShmResource<IO>() {
  triton_utils::warnIfError(this->data_->client()->UnregisterCudaSharedMemory(this->name_),
                            "unable to unregister CUDA shared memory region: " + this->name_);
  triton_utils::cudaCheck(cudaFree(this->addr_), "unable to free GPU memory for key: " + this->name_, false);
}

template <>
void TritonInputGpuShmResource::copy(const void* values, size_t offset) {
  triton_utils::cudaCheck(
      cudaMemcpy((void*)(addr_ + offset), values, data_->byteSizePerBatch_, cudaMemcpyHostToDevice),
      data_->name_ + " toServer(): unable to memcpy " + std::to_string(data_->byteSizePerBatch_) + " bytes to GPU",
      true);
}

template <>
void TritonOutputGpuShmResource::copy(const uint8_t** values) {
  //copy back from gpu, keep in scope
  auto ptr = std::make_shared<std::vector<uint8_t>>(data_->totalByteSize_);
  triton_utils::cudaCheck(
      cudaMemcpy((void*)(ptr->data()), (void*)(addr_), data_->totalByteSize_, cudaMemcpyDeviceToHost),
      data_->name_ + " fromServer(): unable to memcpy " + std::to_string(data_->totalByteSize_) + " bytes from GPU",
      true);
  *values = ptr->data();
  data_->holder_ = ptr;
}

template class TritonHeapResource<nic::InferInput>;
template class TritonCpuShmResource<nic::InferInput>;
template class TritonGpuShmResource<nic::InferInput>;
template class TritonHeapResource<nic::InferRequestedOutput>;
template class TritonCpuShmResource<nic::InferRequestedOutput>;
template class TritonGpuShmResource<nic::InferRequestedOutput>;
