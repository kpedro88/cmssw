#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitsSoA.h"

#include <cassert>

SiPixelRecHitsSoA::SiPixelRecHitsSoA(size_t nhits, const uint32_t *hits, const float *pos)
  : hits_(hits, hits + 2000),
    pos_(pos,   pos + 3*nhits){}
