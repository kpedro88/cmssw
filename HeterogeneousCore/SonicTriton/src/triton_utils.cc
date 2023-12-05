#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"
#include "Utilities/OpenSSL/interface/openssl_init.h"

#include <sstream>
#include <experimental/iterator>

namespace triton_utils {

  template <typename C>
  std::string printColl(const C& coll, const std::string& delim) {
    if (coll.empty())
      return "";
    std::stringstream msg;
    //avoid trailing delim
    std::copy(std::begin(coll), std::end(coll), std::experimental::make_ostream_joiner(msg, delim));
    return msg.str();
  }

  //mostly copied from GeneratorInterface/SherpaInterface/src/SherpackUtilities.cc
  std::string md5_file(const std::string& filename) {
    const unsigned buflen(4096);
    char buffer[buflen];
    cms::openssl_init();
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    const EVP_MD *md = EVP_get_digestbyname("MD5");
    EVP_DigestInit_ex(mdctx, md, nullptr);

    //Open File
    int fd = open(filename.c_str(), O_RDONLY);
    int nb_read;
    while ((nb_read = read(fd, buffer, buflen - 1))) {
      EVP_DigestUpdate(mdctx, buffer, nb_read);
      memset(buffer, 0, buflen);
    }

    unsigned int md_len = 0;
    unsigned char tmp[EVP_MAX_MD_SIZE];
    EVP_DigestFinal_ex(mdctx, tmp, &md_len);
    EVP_MD_CTX_free(mdctx);

    //Convert the result
    char* result;
    for (unsigned int k = 0; k < md_len; ++k) {
      sprintf(result + k * 2, "%02x", tmp[k]);
    }

    return std::string(result);
  }

}  // namespace triton_utils

template std::string triton_utils::printColl(const edm::Span<std::vector<int64_t>::const_iterator>& coll,
                                             const std::string& delim);
template std::string triton_utils::printColl(const std::vector<uint8_t>& coll, const std::string& delim);
template std::string triton_utils::printColl(const std::vector<float>& coll, const std::string& delim);
template std::string triton_utils::printColl(const std::unordered_set<std::string>& coll, const std::string& delim);
