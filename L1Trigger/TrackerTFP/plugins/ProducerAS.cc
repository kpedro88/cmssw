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

#include <string>

using namespace std;
using namespace edm;
using namespace trackerDTC;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerAS
   *  \brief  Associate fitted TTTracks with found TTTracks
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class ProducerAS : public stream::EDProducer<> {
  public:
    explicit ProducerAS(const ParameterSet&);
    ~ProducerAS() override {}

  private:
    void beginRun(const Run&, const EventSetup&) override;
    void produce(Event&, const EventSetup&) override;
    void endJob() {}

    // ED input token of kf tracks
    EDGetTokenT<StreamsTrack> edGetTokenKF_;
    // ED input token of kf TTtracks
    EDGetTokenT<TTTracks> edGetTokenTT_;
    // ED output token for TTTrackMap
    EDPutTokenT<TTTrackMap> edPutToken_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
  };

  ProducerAS::ProducerAS(const ParameterSet& iConfig) :
    iConfig_(iConfig)
  {
    const string& labelKF = iConfig.getParameter<string>("LabelKF");
    const string& labelTT = iConfig.getParameter<string>("LabelTT");
    const string& branch = iConfig.getParameter<string>("BranchAcceptedTracks");
    // book in- and output ED products
    edGetTokenKF_ = consumes<StreamsTrack>(InputTag(labelKF, branch));
    edGetTokenTT_ = consumes<TTTracks>(InputTag(labelTT, branch));
    edPutToken_ = produces<TTTrackMap>(branch);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
  }

  void ProducerAS::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
  }

  void ProducerAS::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty KFTTTrack product
    TTTrackMap ttTrackMap;
    // read in KFin and KF Product and produce AssociatorKF product
    if (setup_->configurationSupported()) {
      Handle<StreamsTrack> handleKF;
      iEvent.getByToken<StreamsTrack>(edGetTokenKF_, handleKF);
      const StreamsTrack& streams = *handleKF.product();
      Handle<TTTracks> handleTT;
      iEvent.getByToken<TTTracks>(edGetTokenTT_, handleTT);
      int i(0);
      for (const StreamTrack& stream : streams)
        for (const FrameTrack& frame : stream)
          if (frame.first.isNonnull())
            ttTrackMap.emplace(TTTrackRef(handleTT, i++), frame.first);
    }
    // store products
    iEvent.emplace(edPutToken_, move(ttTrackMap));
  }

} // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::ProducerAS);