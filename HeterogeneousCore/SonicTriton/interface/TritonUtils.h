#ifndef HeterogeneousCore_SonicTriton_TritonUtils
#define HeterogeneousCore_SonicTriton_TritonUtils

#include <string>
#include <vector>

#include "request_grpc.h"

namespace triton_utils {

using Error = nvidia::inferenceserver::client::Error;

template <typename T>
std::string print_vec(const std::vector<T>& vec, const std::string& delim=", ");

//helper to turn triton error into exception
void wrap(const Error& err, const std::string& msg);

//helper to turn triton error into warning
bool warn(const Error& err, const std::string& msg);

}

extern template std::string triton_utils::print_vec(const std::vector<int64_t>& vec, const std::string& delim);

#endif
