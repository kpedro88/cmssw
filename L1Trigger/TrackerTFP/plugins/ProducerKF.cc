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
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncoding.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilter.h"

#include <string>

using namespace std;
using namespace edm;
using namespace trackerDTC;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerKF
   *  \brief  L1TrackTrigger Kamlan Filter emulator
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerKF : public stream::EDProducer<> {
  public:
    explicit ProducerKF(const ParameterSet&);
    ~ProducerKF() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endStream() { /*kalmanFilterFormats_->endJob();*/ }

    // ED input token of sf stubs and tracks
    EDGetTokenT<TTDTC::Streams> edGetTokenStubs_;
    EDGetTokenT<TTDTC::Streams> edGetTokenLost_;
    EDGetTokenT<StreamsTrack> edGetTokenTracks_;
    // ED output token for accepted stubs and tracks
    EDPutTokenT<TTDTC::Streams> edPutTokenAcceptedStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenAcceptedTracks_;
    // ED output token for lost stubs and tracks
    EDPutTokenT<TTDTC::Streams> edPutTokenLostStubs_;
    EDPutTokenT<StreamsTrack> edPutTokenLostTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // KalmanFilterFormats token
    ESGetToken<KalmanFilterFormats, KalmanFilterFormatsRcd> esGetTokenKalmanFilterFormats_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to extract structured data from TTDTC::Frames
    const DataFormats* dataFormats_;
    // helper class to
    KalmanFilterFormats* kalmanFilterFormats_;
  };

  ProducerKF::ProducerKF(const ParameterSet& iConfig) : iConfig_(iConfig)
  {
    const string& label = iConfig.getParameter<string>("LabelKFin");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLostStubs = iConfig.getParameter<string>("BranchLostStubs");
    const string& branchLostTracks = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenStubs_ = consumes<TTDTC::Streams>(InputTag(label, branchAcceptedStubs));
    edGetTokenLost_ = consumes<TTDTC::Streams>(InputTag(label, branchLostStubs));
    edGetTokenTracks_ = consumes<StreamsTrack>(InputTag(label, branchAcceptedTracks));
    edPutTokenAcceptedStubs_ = produces<TTDTC::Streams>(branchAcceptedStubs);
    edPutTokenAcceptedTracks_ = produces<StreamsTrack>(branchAcceptedTracks);
    edPutTokenLostStubs_ = produces<TTDTC::Streams>(branchLostStubs);
    edPutTokenLostTracks_ = produces<StreamsTrack>(branchLostTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenKalmanFilterFormats_ = esConsumes<KalmanFilterFormats, KalmanFilterFormatsRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
    dataFormats_ = nullptr;
    kalmanFilterFormats_ = nullptr;
  }

  void ProducerKF::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from TTDTC::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to
    kalmanFilterFormats_ = const_cast<KalmanFilterFormats*>(&iSetup.getData(esGetTokenKalmanFilterFormats_));
  }

  void ProducerKF::produce(Event& iEvent, const EventSetup& iSetup) {
    const int numStreamsTracks = setup_->numRegions();
    const int numStreamsStubs = numStreamsTracks * setup_->numLayers();
    // empty KF products
    TTDTC::Streams acceptedStubs(numStreamsStubs);
    StreamsTrack acceptedTracks(numStreamsTracks);
    TTDTC::Streams lostStubs(numStreamsStubs);
    StreamsTrack lostTracks(numStreamsTracks);
    // read in SF Product and produce KF product
    if (setup_->configurationSupported()) {
      Handle<TTDTC::Streams> handleStubs;
      iEvent.getByToken<TTDTC::Streams>(edGetTokenStubs_, handleStubs);
      Handle<TTDTC::Streams> handleLost;
      iEvent.getByToken<TTDTC::Streams>(edGetTokenLost_, handleLost);
      Handle<StreamsTrack> handleTracks;
      iEvent.getByToken<StreamsTrack>(edGetTokenTracks_, handleTracks);
      const int numChannel = handleTracks->size() / setup_->numRegions();
      for (int region = 0; region < setup_->numRegions(); region++) {
        // object to fit tracks in a processing region
        KalmanFilter kf(iConfig_, setup_, dataFormats_, kalmanFilterFormats_, region, numChannel);
        // read in and organize input stubs
        kf.consume(*handleStubs, *handleLost);
        // read in and organize input tracks
        kf.consume(*handleTracks);
        // fill output products
        kf.produce(acceptedStubs, acceptedTracks, lostStubs, lostTracks);
      }
    }
    // store products
    iEvent.emplace(edPutTokenAcceptedStubs_, move(acceptedStubs));
    iEvent.emplace(edPutTokenAcceptedTracks_, move(acceptedTracks));
    iEvent.emplace(edPutTokenLostStubs_, move(lostStubs));
    iEvent.emplace(edPutTokenLostTracks_, move(lostTracks));
  }

} // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerKF);