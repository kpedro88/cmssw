#ifndef DataFormats_SiPixelRecHits_interface_SiPixelRecHitsSoA_h
#define DataFormats_SiPixelRecHits_interface_SiPixelRecHitsSoA_h

#include <cstddef>
#include <cstdint>
#include <vector>


class SiPixelRecHitsSoA {
 public:
  SiPixelRecHitsSoA() = default;
  explicit SiPixelRecHitsSoA(size_t nhits, const uint32_t* hits, const float* pos);
  ~SiPixelRecHitsSoA() = default;
  auto size() const { return hits_.size(); }

  uint32_t hits(size_t i) const { return hits_[i]; }
  float    pos(size_t i) const { return pos_[i]; }

  const std::vector<uint32_t>& hitsVector() const { return hits_; }
  const std::vector<float>&    posVector() const  { return pos_; }

 private:
  std::vector<uint32_t> hits_;
  std::vector<float>    pos_;

};

#endif
