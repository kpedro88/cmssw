#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimCalorimetry/HGCalSimProducers/plugins/HGCDigiProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset,
                                 edm::ProducesCollector producesCollector,
                                 edm::ConsumesCollector& iC)
    : HGCDigiProducer(pset, iC) {
  premixStage1_ = pset.getParameter<bool>("premixStage1");
  if (premixStage1_) {
    producesCollector.produces<PHGCSimAccumulator>(theDigitizer_.digiCollection());
  } else {
    producesCollector.produces<HGCalDigiCollection>(theDigitizer_.digiCollection());
  }
}

HGCDigiProducer::HGCDigiProducer(edm::ParameterSet const& pset, edm::ConsumesCollector& iC)
    : DigiAccumulatorMixMod(), theDigitizer_(pset, iC) {}

//
void HGCDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& es) {
  //std::cout<<"HGCDigiProducer::initializeEvent started"<<std::endl;
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());
  //std::cout<<"HGCDigiProducer::initializeEvent middle"<<std::endl;
  theDigitizer_.initializeEvent(event, es);
  //std::cout<<"HGCDigiProducer::initializeEvent ended"<<std::endl;
}

//
void HGCDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& es) {
  theDigitizer_.finalizeEvent(event, es, randomEngine_);
  randomEngine_ = nullptr;  // to precent access outside event
  //std::cout<<"HGCDigiProducer finalizeEvent ended"<<std::endl;
}

//
void HGCDigiProducer::accumulate(edm::Event const& event, edm::EventSetup const& es) {
  if (premixStage1_) {
    theDigitizer_.accumulate_forPreMix(event, es, randomEngine_);
  }

  else {
    theDigitizer_.accumulate(event, es, randomEngine_);
  }
}

void HGCDigiProducer::accumulate(PileUpEventPrincipal const& event,
                                 edm::EventSetup const& es,
                                 edm::StreamID const& streamID) {
  if (premixStage1_) {
    theDigitizer_.accumulate_forPreMix(event, es, randomEngine_);
  } else {
    theDigitizer_.accumulate(event, es, randomEngine_);
  }
}

//
void HGCDigiProducer::beginRun(edm::Run const&, edm::EventSetup const& es) { theDigitizer_.beginRun(es); }

//
void HGCDigiProducer::endRun(edm::Run const&, edm::EventSetup const&) { theDigitizer_.endRun(); }

DEFINE_DIGI_ACCUMULATOR(HGCDigiProducer);
