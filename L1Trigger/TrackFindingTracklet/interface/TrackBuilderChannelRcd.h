#ifndef L1Trigger_TrackFindingTracklet_LayerEncodingRcd_h
#define L1Trigger_TrackFindingTracklet_LayerEncodingRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "L1Trigger/TrackerDTC/interface/SetupRcd.h"

namespace trackFindingTracklet {

  typedef edm::mpl::Vector<trackerDTC::SetupRcd> RcdsTrackBuilderChannel;

  class TrackBuilderChannelRcd : public edm::eventsetup::DependentRecordImplementation<TrackBuilderChannelRcd, RcdsTrackBuilderChannel> {};

}  // namespace trackFindingTracklet

#endif