#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"

#include <sstream>

using namespace std;
using namespace edm;
using namespace trackerDTC;

namespace trackerTFP {

  /*! \class  trackerTFP::AnalyzerDemonstrator
   *  \brief  Class to demontrate correctness of track trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class AnalyzerDemonstrator : public one::EDAnalyzer<one::WatchRuns> {
  public:
    AnalyzerDemonstrator(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    //
    void convert(const Event& iEvent, const EDGetTokenT<StreamsTrack>& tokenTracks, const EDGetTokenT<TTDTC::Streams>& tokenStubs, vector<vector<TTDTC::BV>>& bits) const;
    //
    template<typename T>
    void convert(const T& collection, vector<vector<TTDTC::BV>>& bits) const;
    // ED input token of Tracks
    EDGetTokenT<TTDTC::Streams> edGetTokenStubsIn_;
    EDGetTokenT<TTDTC::Streams> edGetTokenStubsOut_;
    // ED input token of Stubs
    EDGetTokenT<StreamsTrack> edGetTokenTracksIn_;
    EDGetTokenT<StreamsTrack> edGetTokenTracksOut_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // Demonstrator token
    ESGetToken<Demonstrator, DemonstratorRcd> esGetTokenDemonstrator_;
    //
    const Setup* setup_;
    //
    const Demonstrator* demonstrator_;
  };

  AnalyzerDemonstrator::AnalyzerDemonstrator(const ParameterSet& iConfig) {
    // book in- and output ED products
    const string& labelIn = iConfig.getParameter<string>("LabelIn");
    const string& labelOut = iConfig.getParameter<string>("LabelOut");
    const string& branchStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    edGetTokenStubsIn_ = consumes<TTDTC::Streams>(InputTag(labelIn, branchStubs));
    edGetTokenStubsOut_ = consumes<TTDTC::Streams>(InputTag(labelOut, branchStubs));
    if (labelIn == "TrackerTFPProducerKFin" || labelIn == "TrackerTFPProducerKF")
      edGetTokenTracksIn_ = consumes<StreamsTrack>(InputTag(labelIn, branchTracks));
    if (labelOut == "TrackerTFPProducerKF" || labelOut == "TrackerTFPProducerDR")
      edGetTokenTracksOut_ = consumes<StreamsTrack>(InputTag(labelOut, branchTracks));
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDemonstrator_ = esConsumes<Demonstrator, DemonstratorRcd, Transition::BeginRun>();
    // initial ES product
    setup_ = nullptr;
    demonstrator_ = nullptr;
  }

  void AnalyzerDemonstrator::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    //
    setup_ = &iSetup.getData(esGetTokenSetup_);
    //
    demonstrator_ = &iSetup.getData(esGetTokenDemonstrator_);
  }

  void AnalyzerDemonstrator::analyze(const Event& iEvent, const EventSetup& iSetup) {
    vector<vector<TTDTC::BV>> input;
    vector<vector<TTDTC::BV>> output;
    convert(iEvent, edGetTokenTracksIn_, edGetTokenStubsIn_, input);
    convert(iEvent, edGetTokenTracksOut_, edGetTokenStubsOut_, output);
    demonstrator_->analyze(input, output);
  }

  //
  void AnalyzerDemonstrator::convert(const Event& iEvent, const EDGetTokenT<StreamsTrack>& tokenTracks, const EDGetTokenT<TTDTC::Streams>& tokenStubs, vector<vector<TTDTC::BV>>& bits) const {
    const bool tracks = !tokenTracks.isUninitialized();
    Handle<TTDTC::Streams> handleStubs;
    Handle<StreamsTrack> handleTracks;
    iEvent.getByToken<TTDTC::Streams>(tokenStubs, handleStubs);
    int numChannelStubs = handleStubs->size();
    int numChannelTracks(0);
    if (tracks) {
      iEvent.getByToken<StreamsTrack>(tokenTracks, handleTracks);
      numChannelTracks = handleTracks->size();
    }
    bits.reserve(numChannelTracks + numChannelStubs);
    numChannelTracks /= setup_->numRegions();
    numChannelStubs /= (setup_->numRegions() * (tracks ? numChannelTracks : 1));
    for (int region = 0; region < setup_->numRegions(); region++) {
      const int offsetTracks = region * numChannelTracks;
      for (int channelTracks = 0; channelTracks < numChannelTracks; channelTracks++) {
        const int offsetStubs = (region * numChannelTracks + channelTracks) * numChannelStubs;
        if (tracks)
          convert(handleTracks->at(offsetTracks + channelTracks), bits);
        for (int channelStubs = 0; channelStubs < numChannelStubs; channelStubs++)
          convert(handleStubs->at(offsetStubs + channelStubs), bits);
      }
    }
  }

  //
  template<typename T>
  void AnalyzerDemonstrator::convert(const T& collection, vector<vector<TTDTC::BV>>& bits) const {
    bits.emplace_back();
    vector<TTDTC::BV>& bvs = bits.back();
    bvs.reserve(collection.size());
    transform(collection.begin(), collection.end(), back_inserter(bvs), [](const auto& frame){ return frame.second; });
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::AnalyzerDemonstrator);