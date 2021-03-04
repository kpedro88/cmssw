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
#include "L1Trigger/TrackFindingTracklet/interface/TrackBuilderChannel.h"

#include <string>
#include <vector>
#include <deque>
#include <iterator>
#include <cmath>
#include <numeric>

using namespace std;
using namespace edm;
using namespace trackerDTC;
using namespace trackerTFP;

namespace trackFindingTracklet {

  /*! \class  trackFindingTracklet::ProducerKFin
   *  \brief  transforms hybrid TTTracks into KF input
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class ProducerKFin : public stream::EDProducer<> {
  public:
    explicit ProducerKFin(const ParameterSet&);
    ~ProducerKFin() override {}

  private:
    virtual void beginRun(const Run&, const EventSetup&) override;
    virtual void produce(Event&, const EventSetup&) override;
    virtual void endJob() {}

    // ED input token of TTTracks
    EDGetTokenT<TTTracks> edGetTokenTTTracks_;
    // ED output token for stubs
    EDPutTokenT<TTDTC::Streams> edPutTokenAcceptedStubs_;
    EDPutTokenT<TTDTC::Streams> edPutTokenLostStubs_;
    // ED output token for tracks
    EDPutTokenT<StreamsTrack> edPutTokenAcceptedTracks_;
    EDPutTokenT<StreamsTrack> edPutTokenLostTracks_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // DataFormats token
    ESGetToken<DataFormats, DataFormatsRcd> esGetTokenDataFormats_;
    // LayerEncoding token
    ESGetToken<LayerEncoding, LayerEncodingRcd> esGetTokenLayerEncoding_;
    // TrackBuilderChannel token
    ESGetToken<TrackBuilderChannel, TrackBuilderChannelRcd> esGetTokenTrackBuilderChannel_;
    // configuration
    ParameterSet iConfig_;
    // helper class to store configurations
    const Setup* setup_;
    // helper class to extract structured data from TTDTC::Frames
    const DataFormats* dataFormats_;
    // helper class to encode layer
    const LayerEncoding* layerEncoding_;
    // helper class to assign tracks to channel
    TrackBuilderChannel* trackBuilderChannel_;
    //
    bool enableTruncation_;
  };

  ProducerKFin::ProducerKFin(const ParameterSet& iConfig) :
    iConfig_(iConfig)
  {
    const InputTag& inputTag = iConfig.getParameter<InputTag>("InputTag");
    const string& branchAcceptedStubs = iConfig.getParameter<string>("BranchAcceptedStubs");
    const string& branchAcceptedTracks = iConfig.getParameter<string>("BranchAcceptedTracks");
    const string& branchLostStubs = iConfig.getParameter<string>("BranchLostStubs");
    const string& branchLostTracks = iConfig.getParameter<string>("BranchLostTracks");
    // book in- and output ED products
    edGetTokenTTTracks_ = consumes<TTTracks>(inputTag);
    edPutTokenAcceptedStubs_ = produces<TTDTC::Streams>(branchAcceptedStubs);
    edPutTokenAcceptedTracks_ = produces<StreamsTrack>(branchAcceptedTracks);
    edPutTokenLostStubs_ = produces<TTDTC::Streams>(branchLostStubs);
    edPutTokenLostTracks_ = produces<StreamsTrack>(branchLostTracks);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    esGetTokenDataFormats_ = esConsumes<DataFormats, DataFormatsRcd, Transition::BeginRun>();
    esGetTokenLayerEncoding_ = esConsumes<LayerEncoding, LayerEncodingRcd, Transition::BeginRun>();
    esGetTokenTrackBuilderChannel_ = esConsumes<TrackBuilderChannel, TrackBuilderChannelRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
    dataFormats_ = nullptr;
    layerEncoding_ = nullptr;
    trackBuilderChannel_ = nullptr;
    //
    enableTruncation_ = iConfig.getParameter<bool>("EnableTruncation");
  }

  void ProducerKFin::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (!setup_->configurationSupported())
      return;
    // check process history if desired
    if (iConfig_.getParameter<bool>("CheckHistory"))
      setup_->checkHistory(iRun.processHistory());
    // helper class to extract structured data from TTDTC::Frames
    dataFormats_ = &iSetup.getData(esGetTokenDataFormats_);
    // helper class to encode layer
    layerEncoding_ = &iSetup.getData(esGetTokenLayerEncoding_);
    // helper class to assign tracks to channel
    trackBuilderChannel_ = const_cast<TrackBuilderChannel*>(&iSetup.getData(esGetTokenTrackBuilderChannel_));
  }

  void ProducerKFin::produce(Event& iEvent, const EventSetup& iSetup) {
    auto toFrameStub = [](StubKFin* stub) { return stub->frame(); };
    auto toFrameTrack = [](const TrackKFin& track){ return track.frame(); };
    // dataformat used for track cotTheta wrt eta sector centre
    const DataFormat& dfCot = dataFormats_->format(Variable::cot, Process::sf);
    // dataformat used for track z at raiud chosenRofZ wrt eta sector centre
    const DataFormat& dfZT = dataFormats_->format(Variable::zT, Process::sf);
    // dataformat used for track qOverPt in 1 / cm
    const DataFormat& dfQoverPt = dataFormats_->format(Variable::qOverPt, Process::sf);
    // dataformat used for track phi at radius schoenRofPhi wrt phi sector centre
    const DataFormat& dfPhiT = dataFormats_->format(Variable::phiT, Process::sf);
    // dataformat used for stub phi residual wrt track
    const DataFormat& dfPhi = dataFormats_->format(Variable::phi, Process::sf);
    // dataformat used for stub z residual wrt track
    const DataFormat& dfZ = dataFormats_->format(Variable::z, Process::sf);
    const int numStreamsTracks = setup_->numRegions() * trackBuilderChannel_->numChannels();
    const int numStreamsStubs = numStreamsTracks * setup_->numLayers();
    // empty KFin products
    TTDTC::Streams streamAcceptedStubs(numStreamsStubs);
    StreamsTrack streamAcceptedTracks(numStreamsTracks);
    TTDTC::Streams streamLostStubs(numStreamsStubs);
    StreamsTrack streamLostTracks(numStreamsTracks);
    // read in hybrid track finding product and produce KFin product
    if (setup_->configurationSupported()) {
      Handle<TTTracks> handleTTTracks;
      iEvent.getByToken<TTTracks>(edGetTokenTTTracks_, handleTTTracks);
      const TTTracks& ttTracks = *handleTTTracks.product();
      // Assign input tracks to channels according to TrackBuilder step.
      vector<TTTrackRefs> ttTrackRefsStreams(numStreamsTracks);
      vector<int> nTTTracksStreams(numStreamsTracks, 0);
      int channelId;
      for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks)
        if (trackBuilderChannel_->channelId(ttTrack, channelId))
          nTTTracksStreams[channelId]++;
      channelId = 0;
      for (int nTTTracksStream : nTTTracksStreams)
        ttTrackRefsStreams[channelId++].reserve(nTTTracksStream);
      int i(0);
      for (const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack : ttTracks)
        if (trackBuilderChannel_->channelId(ttTrack, channelId))
          ttTrackRefsStreams[channelId].emplace_back(TTTrackRef(handleTTTracks, i++));
      for (channelId = 0; channelId < numStreamsTracks; channelId++) {
        // Create vector of stubs/tracks in KF format from TTTracks
        const TTTrackRefs& ttTrackRefs = ttTrackRefsStreams[channelId];
        vector<TrackKFin> tracks;
        tracks.reserve(ttTrackRefs.size());
        vector<StubKFin> stubs;
        const int nStubs = accumulate(ttTrackRefs.begin(), ttTrackRefs.end(), 0, [](int& sum, const TTTrackRef& ttTrackRef){ return sum += ttTrackRef->getStubRefs().size(); });
        stubs.reserve(nStubs);
        int trackId(0);
        vector<int> numLayerStubs(setup_->numLayers(), 0);
        for (const TTTrackRef& ttTrackRef : ttTrackRefs) {
          // prevent more than 256 tracks per channel
          if (trackId >= dataFormats_->format(Variable::trackId, Process::kfin).range())
            continue;
          // get rz parameter
          double cot = ttTrackRef->tanL();
          double zT = ttTrackRef->z0() + setup_->chosenRofZ() * cot;
          int binEta(-1);
          for (; binEta < setup_->numSectorsEta(); binEta++)
            if (zT < sinh(setup_->boundarieEta(binEta + 1)) * setup_->chosenRofZ())
              break;
          // cut on outer eta sector boundaries
          if (binEta == -1 || binEta == setup_->numSectorsEta())
            continue;
          cot -= setup_->sectorCot(binEta);
          zT -= setup_->sectorCot(binEta) * setup_->chosenRofZ();
          // cut on eta and |z0| < 15 cm
          if (!dfZT.inRange(zT) || !dfCot.inRange(cot))
            continue;
          const int binZT = dfZT.toUnsigned(dfZT.integer(zT));
          const int binCot = dfCot.toUnsigned(dfCot.integer(cot));
          // get kf layer encoding for this rough r-z track parameter
          const vector<int>& layerEncoding = layerEncoding_->layerEncoding(binEta, binZT, binCot);
          // get rphi parameter
          double qOverPt = ttTrackRef->rInv() / 2.;
          // calculcate track phi at radius hybridChosenRofPhi with respect to phi sector centre
          double phiT = deltaPhi(ttTrackRef->phi() - setup_->hybridChosenRofPhi() * qOverPt - ttTrackRef->phiSector() * setup_->baseRegion());
          const int sectorPhi = phiT < 0. ? 0 : 1; // dirty hack
          phiT -= (sectorPhi - .5) * setup_->baseSector();
          // cut on nonant size and pt
          if (!dfPhiT.inRange(phiT) || !dfQoverPt.inRange(qOverPt))
            continue;
          // loop over stubs
          TTBV hitPattern(0, setup_->numLayers());
          vector<int> layerMap(setup_->numLayers(), 0);
          for (const TTStubRef& ttStubRef : ttTrackRef->getStubRefs()) {
            const GlobalPoint& gp = setup_->stubPos(ttStubRef);
            const double r = gp.perp() - setup_->hybridChosenRofPhi();
            const double phi = deltaPhi(gp.phi() - (ttTrackRef->phi() - qOverPt * gp.perp()));
            const double z = gp.z() - (ttTrackRef->z0() + ttTrackRef->tanL() * gp.perp());
            // layers consitent with rough r-z track parameters are counted from 0 onwards
            int layer = distance(layerEncoding.begin(), find(layerEncoding.begin(), layerEncoding.end(), setup_->layerId(ttStubRef)));
            // put stubs from layer 7 to layer 6 since layer 7 almost never has stubs
            if (layer >= setup_->numLayers())
              layer = setup_->numLayers() - 1;
            // cut on phi and z residuals
            if (!dfPhi.inRange(phi) || !dfZ.inRange(z))
              continue;
            hitPattern.set(layer);
            int& nLayerStubs = layerMap[layer];
            // cut on max 4 stubs per layer
            if (nLayerStubs == setup_->kfMaxStubsPerLayer())
              continue;
            nLayerStubs++;
            numLayerStubs[layer]++;
            stubs.emplace_back(ttStubRef, dataFormats_, r, phi, z, trackId, layer);
          }
          const TTBV& maybePattern = layerEncoding_->maybePattern(binEta, binZT, binCot);
          tracks.emplace_back(ttTrackRef, dataFormats_, hitPattern, setup_->layerMap(hitPattern, layerMap), maybePattern, phiT, qOverPt, zT, cot, sectorPhi, binEta, trackId++);
        }
        // truncate and transform stubs
        vector<vector<StubKFin*>> layerStubs(setup_->numLayers());
        for (int layer = 0; layer < setup_->numLayers(); layer++)
          layerStubs[layer].reserve(numLayerStubs[layer]);
        for (StubKFin& stub : stubs)
          layerStubs[stub.layer()].push_back(&stub);
        const int offset = channelId * setup_->numLayers();
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          TTDTC::Stream& acceptedStubs = streamAcceptedStubs[offset + layer];
          TTDTC::Stream& lostStubs = streamLostStubs[offset + layer];
          vector<StubKFin*>& stubsLayer = layerStubs[layer];
          if (enableTruncation_ && (int)stubsLayer.size() > setup_->numFrames()) {
            const auto it = next(stubsLayer.begin(), setup_->numFrames());
            lostStubs.reserve(setup_->numFrames() - (int)stubsLayer.size());
            transform(it, stubsLayer.end(), back_inserter(lostStubs), toFrameStub);
            stubsLayer.erase(it, stubsLayer.end());
          }
          transform(stubsLayer.begin(), stubsLayer.end(), back_inserter(acceptedStubs), toFrameStub);
        }
        // truncate and transform tracks
        StreamTrack& acceptedTracks = streamAcceptedTracks[channelId];
        StreamTrack& lostTracks = streamLostTracks[channelId];
        if (enableTruncation_ && (int)tracks.size() > setup_->numFrames()) {
          const auto it = next(tracks.begin(), setup_->numFrames());
          lostTracks.reserve(setup_->numFrames() - (int)tracks.size());
          transform(it, tracks.end(), back_inserter(lostTracks), toFrameTrack);
          tracks.erase(it, tracks.end());
        }
        transform(tracks.begin(), tracks.end(), back_inserter(acceptedTracks), toFrameTrack);
      }
    }
    // store products
    iEvent.emplace(edPutTokenAcceptedStubs_, move(streamAcceptedStubs));
    iEvent.emplace(edPutTokenAcceptedTracks_, move(streamAcceptedTracks));
    iEvent.emplace(edPutTokenLostStubs_, move(streamLostStubs));
    iEvent.emplace(edPutTokenLostTracks_, move(streamLostTracks));
  }

} // namespace trackFindingTracklet

DEFINE_FWK_MODULE(trackFindingTracklet::ProducerKFin);