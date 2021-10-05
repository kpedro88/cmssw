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

  //std::unique_ptr<SiPixelFedCablingTree> cabling_;
  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher_;
  std::vector<unsigned int> fedIds_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;

  edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
  edm::EDPutTokenT<SiPixelDigisSoA> digiPutToken_;

  edm::EDPutTokenT<ZVertexHeterogeneous> vertexSOA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> trackSOA_;

  uint32_t pdigi_[200000];
  uint32_t rawIdArr_[200000];
  uint16_t adc_[200000];
  int32_t  clus_[200000];
  int nDigis_;
  bool debug_ = false;
};

PatatrackSonicProducer::PatatrackSonicProducer(const edm::ParameterSet &iConfig)
    : TritonEDProducer<>(iConfig, "PatatrackProducer"),
      cablingMapToken_(esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd>(
										 edm::ESInputTag("", iConfig.getParameter<std::string>("CablingMapLabel")))),
      rawGetToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("InputLabel"))),
      digiPutToken_(produces<SiPixelDigisSoA>()),
      vertexSOA_(produces<ZVertexHeterogeneous>()),
      trackSOA_(produces<PixelTrackHeterogeneous>()),
      debug_(iConfig.getUntrackedParameter<bool>("debugMode", false)) {
  //fedIds_.push_back(1244);
  for(unsigned int i0 = 0; i0 < 139; i0++) 
    if(i0 != 10  && i0 != 11  && i0 != 22  && i0 != 23  && i0 != 34  && i0 != 35  && i0 != 46  && i0 != 47  && i0 != 58  && i0 != 59  && 
       i0 != 70  && i0 != 71  && i0 != 82  && i0 != 83  && i0 != 94  && i0 != 95  && i0 != 103 && i0 != 104 && i0 != 105 && i0 != 106 && 
       i0 != 107 && i0 != 115 && i0 != 116 && i0 != 117 && i0 != 118 && i0 != 119 && i0 != 127 && i0 != 128 && i0 != 129 && i0 != 130 && i0 != 131)
      fedIds_.push_back(i0+1200);
}

//PatatrackSonicProducer::~PatatrackSonicProducer() {}
/* Deal with this stuff later
void PatatrackSonicProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
}
*/

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
  for (unsigned int fedId : fedIds_) {
    vin.push_back(fedId); pSize++;
    const FEDRawData& rawData = buffers.FEDData(fedId);
    unsigned int rawsize=rawData.size()/4;
    vin.push_back(rawsize); pSize++;
    //std::cout << "---> " << fedId << " -- size " << rawsize <<  std::endl;//" -- data --  " << rawData.data() << std::endl; 
    vin.resize(vin.size()+rawsize);
    std::memcpy(&vin[vin.size() - rawsize],rawData.data(), rawData.size());
    pSize += rawsize;
  }
  //std::cout << "-Size--> " << pSize << " -- " << vin.size() << std::endl;
  input.toServer(feds);
} 

void PatatrackSonicProducer::produce(edm::Event &iEvent,
				     const edm::EventSetup &iSetup,
				     Output const &iOutput) {
  
  std::cout << "--> Produce " << std::endl;
  //PixelTrackHeterogeneous pTrack;
  
  auto tracks = std::make_unique<pixelTrack::TrackSoA>();

  //PixelTrackHeterogeneous tracks;
  const auto &output1 = iOutput.begin()->second;
  const auto &outputs_from_server = output1.fromServer<uint32_t>();
  auto output = (outputs_from_server[0]);  
  unsigned int nTracks = output[0];
  //unsiged int output_[0] = output[0nTracks = output[0];]; pCount++;
  unsigned int pCount = 1;
  std::memcpy((*tracks).chi2.data(),      &(output.front())+pCount,nTracks*sizeof(float));                 pCount+=nTracks;
  std::memcpy((*tracks).qualityData(), &(output.front())+pCount,nTracks*sizeof(uint8_t));               pCount+=nTracks;
  std::memcpy((*tracks).eta.data(),       &(output.front())+pCount,nTracks*sizeof(float));                 pCount+=nTracks;
  std::memcpy((*tracks).pt.data(),        &(output.front())+pCount,nTracks*sizeof(float));                 pCount+=nTracks;
  std::memcpy((*tracks).stateAtBS.state(0).data(),     &(output.front())+pCount,nTracks*sizeof(float)*5);  pCount+=(nTracks*5);
  std::memcpy((*tracks).stateAtBS.covariance(0).data(),&(output.front())+pCount,nTracks*sizeof(float)*15); pCount+=(nTracks*15);

  auto vertices = std::make_unique<ZVertexSoA>();
  //ZVertexHeterogeneous vertices;
  vertices->nvFinal = output[pCount]; pCount++;
  static constexpr uint32_t MAXTRACKS = 32 * 1024;
  static constexpr uint32_t MAXVTX = 1024;
  /*
  std::cout << "--> Produce 2 " << output[pCount] <<" -- " <<pCount << std::endl;
  std::cout << "-1--> " << (vertices)->idv << std::endl;
  std::cout << "-2--> " << (&(output.front())+pCount) << std::endl;
  */
  std::memcpy((vertices)->idv    , &(output.front())+pCount,MAXTRACKS*sizeof(int16_t));   pCount+=MAXTRACKS;
  std::memcpy((vertices)->zv     , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=MAXVTX;
  std::memcpy((vertices)->wv     , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=MAXVTX;
  std::memcpy((vertices)->chi2   , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=MAXVTX;
  std::memcpy((vertices)->ptv2   , &(output.front())+pCount,MAXVTX*sizeof(float));        pCount+=MAXVTX;
  std::memcpy((vertices)->ndof   , &(output.front())+pCount,MAXVTX*sizeof(int32_t));      pCount+=MAXVTX;
  std::memcpy((vertices)->sortInd, &(output.front())+pCount,MAXVTX*sizeof(uint16_t));     pCount+=MAXVTX;
  /*
  uint32_t nDigis_    = 150000;//outputs_from_server[0][0];
  auto digis = (outputs_from_server[0]);
  std::memcpy(rawIdArr_,&(digis.front())+1          ,nDigis_*sizeof(uint32_t));
  std::memcpy(pdigi_,   &(digis.front())+1+nDigis_  ,nDigis_*sizeof(uint32_t));
  std::memcpy(adc_,     &(digis.front())+1+2*nDigis_,nDigis_*sizeof(uint16_t));
  std::memcpy(clus_,    &(digis.front())+1+3*nDigis_,nDigis_*sizeof(uint32_t));
  iEvent.emplace(digiPutToken_, nDigis_, pdigi_, rawIdArr_, adc_, clus_);
  */
  //iEvent.emplace(trackSOA_,  std::move(tracks.get()));//ZVertexHeterogeneous(std::move(m_soa)));
  //iEvent.put(std::move(tracks));
  iEvent.emplace(trackSOA_,  PixelTrackHeterogeneous(std::move(tracks)));
  iEvent.emplace(vertexSOA_, ZVertexHeterogeneous(std::move(vertices)));
  //iEvent.put(std::move(vertices));
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
