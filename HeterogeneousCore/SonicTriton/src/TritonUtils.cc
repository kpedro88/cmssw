#include "HeterogeneousCore/SonicTriton/interface/TritonUtils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>
#include <iterator>

namespace triton_utils {

template <typename T>
std::string print_vec(const std::vector<T>& vec, const std::string& delim) {
  if (vec.empty())
    return "";
  std::stringstream msg;
  //avoid trailing delim
  std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<int64_t>(msg, delim.c_str()));
  //last element
  msg << vec.back();
  return msg.str();
}

void wrap(const Error& err, const std::string& msg) {
  if (!err.IsOk())
    throw cms::Exception("TritonServerFailure") << msg << ": " << err;
}

bool warn(const Error& err, const std::string& msg) {
  if (!err.IsOk())
    edm::LogWarning("TritonServerWarning") << msg << ": " << err;
  return err.IsOk();
}

}

template std::string triton_utils::print_vec(const std::vector<int64_t>& vec, const std::string& delim);

