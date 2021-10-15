
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/Vertex/interface/ZVertexHeterogeneous.h"

#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"
#include "EventFilter/SiPixelRawToDigi/interface/ErrorChecker.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitsSoA.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

class PatatrackSonicProducer : public TritonEDProducer<> {
public:
  explicit PatatrackSonicProducer(const edm::ParameterSet &);
  void acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) override;
  void produce(edm::Event &iEvent, edm::EventSetup const &iSetup, Output const &iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:

  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher_;
  std::vector<unsigned int> fedIds_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;

  PixelDataFormatter::Errors errors_;
  edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
  edm::EDPutTokenT<SiPixelRecHitsSoA> hitsSOA_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  edm::EDPutTokenT<ZVertexHeterogeneous> vertexSOA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> trackSOA_;

  
  //uint32_t pdigi_[200000];
  //uint32_t rawIdArr_[200000];
  //uint16_t adc_ [200000];
  //int32_t  clus_[200000];
  //uint32_t hits_[4000];
  //float    pos_ [40000];
  bool debug_ = false;

  std::shared_ptr<uint32_t[]> pdigi_;
  std::shared_ptr<uint32_t[]> rawIdArr_;
  std::shared_ptr<uint16_t[]>  adc_;
  std::shared_ptr<int32_t[]>  clus_;
  std::shared_ptr<uint32_t[]> hits_;
  std::shared_ptr<float[]>    pos_;
};

PatatrackSonicProducer::PatatrackSonicProducer(const edm::ParameterSet &iConfig)
    : TritonEDProducer<>(iConfig, "PatatrackProducer"),
      cablingMapToken_(esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd>(
										 edm::ESInputTag("", iConfig.getParameter<std::string>("CablingMapLabel")))),
      rawGetToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("InputLabel"))),
      hitsSOA_(produces<SiPixelRecHitsSoA>()),
      digiPutToken_(produces<SiPixelDigisSoA>()),
      vertexSOA_(produces<ZVertexHeterogeneous>()),
      trackSOA_(produces<PixelTrackHeterogeneous>()),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)) {

  for(unsigned int i0 = 0; i0 < 139; i0++) 
    if(i0 != 10  && i0 != 11  && i0 != 22  && i0 != 23  && i0 != 34  && i0 != 35  && i0 != 46  && i0 != 47  && i0 != 58  && i0 != 59  && 
       i0 != 70  && i0 != 71  && i0 != 82  && i0 != 83  && i0 != 94  && i0 != 95  && i0 != 103 && i0 != 104 && i0 != 105 && i0 != 106 && 
       i0 != 107 && i0 != 115 && i0 != 116 && i0 != 117 && i0 != 118 && i0 != 119 && i0 != 127 && i0 != 128 && i0 != 129 && i0 != 130 && i0 != 131)
      fedIds_.push_back(i0+1200);

  hits_.reset(new uint32_t[2000]);
  pos_.reset(new float   [40000]);
  pdigi_.reset(new uint32_t[200000]);
  rawIdArr_.reset(new uint32_t[200000]);
  adc_.reset(new uint16_t[200000]);
  clus_.reset(new int32_t[200000]);
}

void PatatrackSonicProducer::acquire(edm::Event const &iEvent, edm::EventSetup const &iSetup, Input &iInput) {
  const auto& buffers = iEvent.get(rawGetToken_);
  // initialize cabling map or update if necessary
  //if (recordWatcher_.check(iSetup)) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    //auto cablingMap = iSetup.getTransientHandle(cablingMapToken_);
    //fedIds_ = cablingMap->fedIds();
    //}
  //Note Fed data quality  checks ae curently done on the CPU server, and could be moved here
  auto& input = iInput.at("input");
  auto  feds  = input.allocate<uint32_t>();
  auto& vin   = (*feds)[0];
  unsigned int pSize = 0; 
  vin.push_back(fedIds_.size()); pSize++;
  ErrorChecker errorcheck;
  bool errorsInEvent = false;
  errors_.clear();
  for (unsigned int fedId : fedIds_) {
    vin.push_back(fedId); pSize++;
    const FEDRawData& rawData = buffers.FEDData(fedId);
    int nWords = rawData.size() / sizeof(uint64_t);
    if (nWords == 0) {
      std::cout << " !!!! Continuing " << std::endl;
      continue;
    }
    const cms_uint64_t* trailer = reinterpret_cast<const cms_uint64_t*>(rawData.data()) + (nWords - 1);
    if (not errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors_)) {
      continue;
    }
    unsigned int rawsize=rawData.size()/4;
    vin.push_back(rawsize); pSize++;
    vin.resize(vin.size()+rawsize);
    std::memcpy(&vin[vin.size() - rawsize],rawData.data(), rawData.size());
    pSize += rawsize;
  }
  input.toServer(feds);
  std::cout << " ---> sent ";
} 

void PatatrackSonicProducer::produce(edm::Event &iEvent,
				     const edm::EventSetup &iSetup,
				     Output const &iOutput) {

  std::cout << " --> received " << std::endl;
  //pdigi_.reset(new uint32_t[200000]);
  //rawIdArr_.reset(new uint32_t[200000]);
  //adc_.reset(new uint16_t[200000]);
  //clus_.reset(new int32_t[200000]);
  //hits_.reset(new uint32_t[4000]);
  //pos_.reset(new float   [40000]);
  //std::unique_ptr<uint32_t[]>  pdigi_(new uint32_t[200000]);
  //std::unique_ptr<uint32_t[]>  rawIdArr_(new uint32_t[200000]);
  //std::unique_ptr<uint16_t[]>  adc_(new uint16_t[200000]);
  //std::unique_ptr<int32_t[]>   clus_(new int32_t[200000]);
  //std::unique_ptr<uint32_t[]>  hits_(new uint32_t[4000]);
  //std::unique_ptr<float[]>     pos_(new float   [40000]);
  auto hits   = std::make_unique<SiPixelRecHitsSoA>();
  auto tracks = std::make_unique<pixelTrack::TrackSoA>();

  //PixelTrackHeterogeneous tracks;
  const auto &output1 = iOutput.begin()->second;
  const auto &outputs_from_server = output1.fromServer<uint8_t>();
  auto output = (outputs_from_server[0]);  
  unsigned int pCount = 0;
  uint32_t nHits      = 0; //output[pCount]; pCount++;
  std::memcpy(&nHits,&(output.front())+pCount,sizeof(uint32_t)); pCount += 4;
  static const unsigned nMax = 2000; 
  //if(nHits_ < 2000) nMax = nHits_;
  std::memcpy(hits_.get(),&(output.front())+pCount,nMax*sizeof(uint32_t));    pCount += 4*nMax;
  std::memcpy(pos_.get(),&(output.front())+pCount,3*nHits*sizeof(float));     pCount += 4*3*nHits;

  uint32_t nDigis    = 0; //output[pCount]; pCount++;
  std::memcpy(&nDigis,&(output.front())+pCount,sizeof(uint32_t)); pCount += 4;
  std::memcpy(pdigi_.get(),   &(output.front())+pCount,nDigis*sizeof(uint32_t)); pCount += 4*nDigis;
  std::memcpy(rawIdArr_.get(),&(output.front())+pCount,nDigis*sizeof(uint32_t)); pCount += 4*nDigis;
  std::memcpy(adc_.get(),     &(output.front())+pCount,nDigis*sizeof(uint16_t)); pCount += 2*nDigis;
  std::memcpy(clus_.get(),    &(output.front())+pCount,nDigis*sizeof(int32_t));  pCount += 4*nDigis;
  /*
  std::memcpy(pdigi_,   &(output.front())+pCount,nDigis*sizeof(uint32_t)); pCount += nDigis;
  std::memcpy(rawIdArr_,&(output.front())+pCount,nDigis*sizeof(uint32_t)); pCount += nDigis;
  std::memcpy(adc_,     &(output.front())+pCount,nDigis*sizeof(uint16_t)); pCount += nDigis;
  std::memcpy(clus_,    &(output.front())+pCount,nDigis*sizeof(int32_t)); pCount += nDigis;
  */

  unsigned int nTracks = 0;//output[pCount]; pCount++;
  std::memcpy(&nTracks,&(output.front())+pCount,sizeof(uint32_t)); pCount += 4;
  std::memcpy((*tracks).chi2.data(),      &(output.front())+pCount,nTracks*sizeof(float));                 pCount+=4*nTracks;
  std::memcpy((*tracks).qualityData(),    &(output.front())+pCount,nTracks*sizeof(uint8_t));               pCount+=1*nTracks;
  std::memcpy((*tracks).eta.data(),       &(output.front())+pCount,nTracks*sizeof(float));                 pCount+=4*nTracks;
  std::memcpy((*tracks).pt.data(),        &(output.front())+pCount,nTracks*sizeof(float));                 pCount+=4*nTracks;
  std::memcpy((*tracks).stateAtBS.state(0).data(),     &(output.front())+pCount,nTracks*sizeof(float)*5);  pCount+=4*(nTracks*5);
  std::memcpy((*tracks).stateAtBS.covariance(0).data(),&(output.front())+pCount,nTracks*sizeof(float)*15); pCount+=4*(nTracks*15);
  std::memcpy((void*)(*tracks).hitIndices.content.data(),&(output.front())+pCount,nTracks*sizeof(uint32_t)*5); pCount+=4*(nTracks*5);
  std::memcpy((*tracks).hitIndices.off.data(),           &(output.front())+pCount,(nTracks+1)*sizeof(int32_t));    pCount+=4*(nTracks+1);
  std::memcpy((void*)(*tracks).detIndices.content.data(),&(output.front())+pCount,nTracks*sizeof(uint32_t)*5);     pCount+=4*(nTracks*5);
  std::memcpy((*tracks).detIndices.off.data(),           &(output.front())+pCount,(nTracks+1)*sizeof(int32_t));    pCount+=4*(nTracks+1);


  auto vertices = std::make_unique<ZVertexSoA>();
  //ZVertexHeterogeneous vertices;
  //vertices->nvFinal = output[pCount]; pCount++;
  static constexpr uint32_t MAXTRACKS = 32 * 1024;
  static constexpr uint32_t MAXVTX = 1024;
  std::memcpy(&(vertices->nvFinal),&(output.front())+pCount,sizeof(uint32_t)); pCount += 4;
  std::memcpy((vertices)->idv    , &(output.front())+pCount,MAXTRACKS*sizeof(int16_t));   pCount+=2*MAXTRACKS;
  std::memcpy((vertices)->zv     , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
  std::memcpy((vertices)->wv     , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
  std::memcpy((vertices)->chi2   , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
  std::memcpy((vertices)->ptv2   , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=4*MAXVTX;
  std::memcpy((vertices)->ndof   , &(output.front())+pCount,MAXVTX*sizeof(int32_t));      pCount+=4*MAXVTX;
  std::memcpy((vertices)->sortInd, &(output.front())+pCount,MAXVTX*sizeof(uint16_t));     pCount+=4*MAXVTX;

  iEvent.emplace(hitsSOA_,      nHits, hits_.get(), pos_.get()); 
  //iEvent.emplace(digiPutToken_, nDigis, pdigi_, rawIdArr_, adc_, clus_);
  iEvent.emplace(digiPutToken_, nDigis, pdigi_.get(), rawIdArr_.get(), adc_.get(), clus_.get());
  iEvent.emplace(trackSOA_,  PixelTrackHeterogeneous(std::move(tracks)));
  iEvent.emplace(vertexSOA_, ZVertexHeterogeneous(std::move(vertices)));
}

void PatatrackSonicProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  desc.add<edm::InputTag>("InputLabel");
  desc.add<std::string>("CablingMapLabel");
  desc.addOptionalUntracked<bool>("debugMode", false);
  descriptions.add("deepMETSonicProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PatatrackSonicProducer);
