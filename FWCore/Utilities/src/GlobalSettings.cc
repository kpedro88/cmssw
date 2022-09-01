#include "FWCore/Utilities/interface/GlobalSettings.h"

namespace cms {
  GlobalSettings::GlobalSettings() : trace_(false) {}

  GlobalSettings& GlobalSettings::get_() {
    [[cms::thread_safe]] static GlobalSettings gs_instance;
    return gs_instance;
  }

  const GlobalSettings& GlobalSettings::get() {
    return get_();
  }
}
