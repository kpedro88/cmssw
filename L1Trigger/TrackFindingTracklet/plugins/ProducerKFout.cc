#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"

#include <string>
#include <numeric>

using namespace std;
using namespace edm;
using namespace trackerDTC;
using namespace trackerTFP;

namespace trackFindingTracklet {

  /*! \class  trackFindingTracklet::ProducerKFout
   *  \brief  Converts KF output into TTTracks
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class ProducerKFout : public stream::EDProducer<> {
  public:
    explicit ProducerKFout(const ParameterSet&);
    ~ProducerKFout() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}

    // ED input token of kf stubs
    EDGetTokenT<TTDTC::Streams> edGetTokenStubs_;
    // ED input token of kf tracks
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for TTTracks
    EDPutTokenT<TTTracks> edPutToken_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to extract structured data from TTDTC::Frames
    const DataFormats* dataFormats_;
    // used data formats
    const DataFormat* zT_;
    const DataFormat* cot_;
    const DataFormat* phi_;
    const DataFormat* z_;
  };

  ProducerKFout::ProducerKFout(const ParameterSet& iConfig) :
    iConfig_(iConfig)
  {
    const string& label = iConfig.getParameter<string>("LabelKF");
    const string& branchStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<TTDTC::Streams>(InputTag(label, branchStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchTracks));
    edPutToken_ = produces<TTTracks>(branchTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
    dataFormats_ = nullptr;
    // used data formats
    zT_ = nullptr;
    cot_ = nullptr;
    phi_ = nullptr;
    z_ = nullptr;
  }

  void ProducerKFout::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from TTDTC::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // used data formats
    zT_ = &dataFormats_->format(Variable::zT, Process::kfin);
    cot_ = &dataFormats_->format(Variable::cot, Process::kfin);
    phi_ = &dataFormats_->format(Variable::phi, Process::sf);
    z_ = &dataFormats_->format(Variable::z, Process::sf);
  }

  void ProducerKFout::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty KFout product
    TTTracks ttTracks;
    // read in KF Product and produce KFout product
    if (setup_->configurationSupported()) {
      Handle<TTDTC::Streams> handleStubs;
      iEvent.getByToken<TTDTC::Streams>(edGetTokenStubs_, handleStubs);
      const TTDTC::Streams& streamsStubs = *handleStubs.product();
      Handle<StreamsTrack> handleTracks;
      iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
      const StreamsTrack& streamsTracks = *handleTracks.product();
      // count number of kf tracks
      int nTracks(0);
      for (const StreamTrack& stream : streamsTracks)
        nTracks += accumulate(stream.begin(), stream.end(), 0, [](int& sum, const FrameTrack& frame){ return sum += frame.first.isNonnull() ? 1 : 0; });
      ttTracks.reserve(nTracks);
      // convert kf track frames per region and stub frames per region and layer to TTTracks
      for (int region = 0; region < setup_->numRegions(); region++) {
        const int offset = region * setup_->numLayers();
        int iTrk(0);
        for (const FrameTrack& frameTrack : streamsTracks[region]) {
          if (frameTrack.first.isNull())
            continue;
          // convert stub frames to kf stubs
          vector<StubKF> stubs;
          stubs.reserve(setup_->numLayers());
          for (int layer = 0; layer < setup_->numLayers(); layer++) {
            const TTDTC::Frame& frameStub = streamsStubs[offset + layer][iTrk];
            if (frameStub.first.isNonnull())
              stubs.emplace_back(frameStub, dataFormats_, layer);
          }
          // convert track frame to kf track
          TrackKF track(frameTrack, dataFormats_);
          // convert kf track and kf stubs to TTTrack
          ttTracks.emplace_back(track.ttTrack(stubs));
          iTrk++;
        }
      }
    }
    // store products
    iEvent.emplace(edPutToken_, move(ttTracks));
  }

} // namespace trackFindingTracklet

DEFINE_FWK_MODULE(trackFindingTracklet::ProducerKFout);