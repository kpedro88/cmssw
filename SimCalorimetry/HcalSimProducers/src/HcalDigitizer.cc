#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalTestHitGenerator.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseGenerator.h"
#include <boost/foreach.hpp>
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"

//#define DebugLog

HcalDigitizer::HcalDigitizer(const edm::ParameterSet& ps, edm::ConsumesCollector& iC) :
  theGeometry(0),
  theRecNumber(0),
  theParameterMap(new HcalSimParameterMap(ps)),
  theShapes(new HcalShapes()),
  theHBHEResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHBHESiPMResponse(new HcalSiPMHitResponse(theParameterMap, theShapes)),
  theHOResponse(new CaloHitResponse(theParameterMap, theShapes)),   
  theHOSiPMResponse(new HcalSiPMHitResponse(theParameterMap, theShapes)),
  theHFResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHFQIE10Response(new CaloHitResponse(theParameterMap, theShapes)),
  theZDCResponse(new CaloHitResponse(theParameterMap, theShapes)),
  theHBHEAmplifier(0),
  theHFAmplifier(0),
  theHOAmplifier(0),
  theZDCAmplifier(0),
  theHFQIE10Amplifier(0),
  theHBHEQIE11Amplifier(0),
  theIonFeedback(0),
  theCoderFactory(0),
  theUpgradeCoderFactory(0),
  theHBHEElectronicsSim(0),
  theHFElectronicsSim(0),
  theHOElectronicsSim(0),
  theZDCElectronicsSim(0),
  theUpgradeHBHEElectronicsSim(0),
  theUpgradeHFElectronicsSim(0),
  theHFQIE10ElectronicsSim(0),
  theHBHEQIE11ElectronicsSim(0),
  theHBHEHitFilter(),
  theHBHEQIE11HitFilter(),
  theHFHitFilter(ps.getParameter<bool>("doHFWindow")),
  theHFQIE10HitFilter(ps.getParameter<bool>("doHFWindow")),
  theHOHitFilter(),
  theHOSiPMHitFilter(),
  theZDCHitFilter(),
  theHitCorrection(0),
  theNoiseGenerator(0),
  theNoiseHitGenerator(0),
  theHBHEDigitizer(0),
  theHODigitizer(0),
  theHOSiPMDigitizer(0),
  theHFDigitizer(0),
  theZDCDigitizer(0),
  theHBHEUpgradeDigitizer(0),
  theHFUpgradeDigitizer(0),
  theHFQIE10Digitizer(0),
  theHBHEQIE11Digitizer(0),
  theRelabeller(0),
  isZDC(true),
  isHCAL(true),
  zdcgeo(true),
  hbhegeo(true),
  hogeo(true),
  hfgeo(true),
  hitsProducer_(ps.getParameter<std::string>("hitsProducer")),
  theHOSiPMCode(ps.getParameter<edm::ParameterSet>("ho").getParameter<int>("siPMCode")),
  deliveredLumi(0.),
  m_HEDarkening(0),
  m_HFRecalibration(0)
{
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag(hitsProducer_, "ZDCHITS"));
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag(hitsProducer_, "HcalHits"));

  bool doNoise = ps.getParameter<bool>("doNoise");
  bool PreMix1 = ps.getParameter<bool>("HcalPreMixStage1");  // special threshold/pedestal treatment
  bool PreMix2 = ps.getParameter<bool>("HcalPreMixStage2");  // special threshold/pedestal treatment
  bool doEmpty = ps.getParameter<bool>("doEmpty");
  bool doHBHEUpgrade = ps.getParameter<bool>("HBHEUpgradeQIE");
  bool doHFUpgrade   = ps.getParameter<bool>("HFUpgradeQIE");
  deliveredLumi     = ps.getParameter<double>("DelivLuminosity");
  bool agingFlagHE = ps.getParameter<bool>("HEDarkening");
  bool agingFlagHF = ps.getParameter<bool>("HFDarkening");
  double minFCToDelay= ps.getParameter<double>("minFCToDelay");

  if(PreMix1 && PreMix2) {
     throw cms::Exception("Configuration")
      << "HcalDigitizer cannot operate in PreMixing digitization and PreMixing\n"
         "digi combination modes at the same time.  Please set one mode to False\n"
         "in the configuration file.";
  }

  // need to make copies, because they might get different noise generators
  theHBHEAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHFAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHOAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theZDCAmplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHFQIE10Amplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);
  theHBHEQIE11Amplifier = new HcalAmplifier(theParameterMap, doNoise, PreMix1, PreMix2);

  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);
  theUpgradeCoderFactory = new HcalCoderFactory(HcalCoderFactory::UPGRADE);

  theHBHEElectronicsSim = new HcalElectronicsSim(theHBHEAmplifier, theCoderFactory, PreMix1);
  theHFElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theCoderFactory, PreMix1);
  theHOElectronicsSim = new HcalElectronicsSim(theHOAmplifier, theCoderFactory, PreMix1);
  theZDCElectronicsSim = new HcalElectronicsSim(theZDCAmplifier, theCoderFactory, PreMix1);
  theUpgradeHBHEElectronicsSim = new HcalElectronicsSim(theHBHEAmplifier, theUpgradeCoderFactory, PreMix1);
  theUpgradeHFElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theUpgradeCoderFactory, PreMix1);
  theHFQIE10ElectronicsSim = new HcalElectronicsSim(theHFQIE10Amplifier, theUpgradeCoderFactory, PreMix1); //should this use a different coder factory?
  theHBHEQIE11ElectronicsSim = new HcalElectronicsSim(theHBHEQIE11Amplifier, theUpgradeCoderFactory, PreMix1); //should this use a different coder factory?

  bool doHOHPD = (theHOSiPMCode != 1);
  bool doHOSiPM = (theHOSiPMCode != 0);
  if(doHOHPD) {
    theHOResponse = new CaloHitResponse(theParameterMap, theShapes);
	theHOHitFilter.setSubdets({HcalOuter});
    theHOResponse->setHitFilter(&theHOHitFilter);
    theHODigitizer = new HODigitizer(theHOResponse, theHOElectronicsSim, doEmpty);
  }
  if(doHOSiPM) {
    theHOSiPMResponse = new HcalSiPMHitResponse(theParameterMap, theShapes);
	theHOSiPMHitFilter.setSubdets({HcalOuter});
    theHOSiPMResponse->setHitFilter(&theHOSiPMHitFilter);
    theHOSiPMDigitizer = new HODigitizer(theHOSiPMResponse, theHOElectronicsSim, doEmpty);
  }
  
  theHBHEResponse->initHBHEScale();
  edm::LogInfo("HcalDigitizer") <<"Set scale for HB towers";
  theHBHEHitFilter.setSubdets({HcalBarrel,HcalEndcap});
  theHBHEResponse->setHitFilter(&theHBHEHitFilter);
  bool    changeResponse = ps.getParameter<bool>("ChangeResponse");
  edm::FileInPath fname  = ps.getParameter<edm::FileInPath>("CorrFactorFile");
  if (changeResponse) {
    std::string corrFileName = fname.fullPath();
    edm::LogInfo("HcalDigitizer") << "Set scale for HB towers from " << corrFileName;
    theHBHEResponse->setHBHEScale(corrFileName); //GMA
  }
  theHBHEQIE11HitFilter.setSubdets({HcalBarrel,HcalEndcap});
  theHBHESiPMResponse->setHitFilter(&theHBHEQIE11HitFilter);
  
  if(doHBHEUpgrade){
    theHBHEUpgradeDigitizer = new UpgradeDigitizer(theHBHESiPMResponse, theUpgradeHBHEElectronicsSim, doEmpty);
  }
  else { //QIE8 and QIE11 can coexist in HBHE
    theHBHEQIE11Digitizer = new QIE11Digitizer(theHBHESiPMResponse, theHBHEQIE11ElectronicsSim, doEmpty);
    theHBHEDigitizer = new HBHEDigitizer(theHBHEResponse, theHBHEElectronicsSim, doEmpty);
  }

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  //initialize: they won't be called later if flag is set
  theTimeSlewSim = 0;
  if(doTimeSlew) {
    // no time slewing for HF
    theTimeSlewSim = new HcalTimeSlewSim(theParameterMap,minFCToDelay);
    theHBHEAmplifier->setTimeSlewSim(theTimeSlewSim);
    theHBHEQIE11Amplifier->setTimeSlewSim(theTimeSlewSim);
    theHOAmplifier->setTimeSlewSim(theTimeSlewSim);
    theZDCAmplifier->setTimeSlewSim(theTimeSlewSim);
  }

  theHFResponse->setHitFilter(&theHFHitFilter);
  theHFQIE10Response->setHitFilter(&theHFQIE10HitFilter);
  theZDCResponse->setHitFilter(&theZDCHitFilter);
  
  if(doHFUpgrade){
    theHFUpgradeDigitizer = new UpgradeDigitizer(theHFResponse, theUpgradeHFElectronicsSim, doEmpty);
  }
  else { //QIE8 and QIE10 can coexist in HF
    theHFQIE10Digitizer = new QIE10Digitizer(theHFQIE10Response, theHFQIE10ElectronicsSim, doEmpty);
	theHFDigitizer = new HFDigitizer(theHFResponse, theHFElectronicsSim, doEmpty);
  }
  theZDCDigitizer = new ZDCDigitizer(theZDCResponse, theZDCElectronicsSim, doEmpty);

  testNumbering_ = ps.getParameter<bool>("TestNumbering");
//  std::cout << "Flag to see if Hit Relabeller to be initiated " << testNumbering_ << std::endl;
  if (testNumbering_) theRelabeller=new HcalHitRelabeller(ps);

  bool doHPDNoise = ps.getParameter<bool>("doHPDNoise");
  if(doHPDNoise) {
    theNoiseGenerator = new HPDNoiseGenerator(ps); 
    if(theHBHEDigitizer) theHBHEDigitizer->setNoiseSignalGenerator(theNoiseGenerator);
    if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->setNoiseSignalGenerator(theNoiseGenerator);
  }

  if(ps.getParameter<bool>("doIonFeedback") && theHBHEResponse) {
    theIonFeedback = new HPDIonFeedbackSim(ps, theShapes);
    theHBHEResponse->setPECorrection(theIonFeedback);
    if(ps.getParameter<bool>("doThermalNoise")) {
      theHBHEAmplifier->setIonFeedbackSim(theIonFeedback);
    }
  }

  if(ps.getParameter<bool>("injectTestHits") ) {
    theNoiseHitGenerator = new HcalTestHitGenerator(ps);
    if(theHBHEDigitizer) theHBHEDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHBHEUpgradeDigitizer) theHBHEUpgradeDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHODigitizer) theHODigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHOSiPMDigitizer) theHOSiPMDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHFDigitizer) theHFDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
	if(theHFQIE10Digitizer) theHFQIE10Digitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    theZDCDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
  }

  if(agingFlagHE) m_HEDarkening = new HEDarkening();
  if(agingFlagHF) m_HFRecalibration = new HFRecalibration(ps.getParameter<edm::ParameterSet>("HFRecalParameterBlock"));
}


HcalDigitizer::~HcalDigitizer() {
  if(theHBHEDigitizer)         delete theHBHEDigitizer;
  if(theHBHEQIE11Digitizer)    delete theHBHEQIE11Digitizer;
  if(theHODigitizer)           delete theHODigitizer;
  if(theHOSiPMDigitizer)       delete theHOSiPMDigitizer;
  if(theHFDigitizer)           delete theHFDigitizer;
  if(theHFQIE10Digitizer)      delete theHFQIE10Digitizer;
  delete theZDCDigitizer;
  if(theHBHEUpgradeDigitizer)  delete theHBHEUpgradeDigitizer;
  if(theHFUpgradeDigitizer)    delete theHFUpgradeDigitizer;
  delete theParameterMap;
  delete theHBHEResponse;
  delete theHBHESiPMResponse;
  delete theHOResponse;
  delete theHOSiPMResponse;
  delete theHFResponse;
  delete theHFQIE10Response;
  delete theZDCResponse;
  delete theHBHEElectronicsSim;
  delete theHFElectronicsSim;
  delete theHOElectronicsSim;
  delete theZDCElectronicsSim;
  delete theUpgradeHBHEElectronicsSim;
  delete theUpgradeHFElectronicsSim;
  delete theHFQIE10ElectronicsSim;
  delete theHBHEQIE11ElectronicsSim;
  delete theHBHEAmplifier;
  delete theHFAmplifier;
  delete theHOAmplifier;
  delete theZDCAmplifier;
  delete theHFQIE10Amplifier;
  delete theHBHEQIE11Amplifier;
  delete theCoderFactory;
  delete theUpgradeCoderFactory;
  delete theHitCorrection;
  delete theNoiseGenerator;
  if (theRelabeller)           delete theRelabeller;
}


void HcalDigitizer::setHBHENoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEElectronicsSim);
  if (theHBHEDigitizer) theHBHEDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEQIE11ElectronicsSim);
  if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEQIE11Amplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHFNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHFElectronicsSim);
  if(theHFDigitizer) theHFDigitizer->setNoiseSignalGenerator(noiseGenerator);
  if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHFQIE10ElectronicsSim);
  if(theHFQIE10Digitizer) theHFQIE10Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFQIE10Amplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHONoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theHOElectronicsSim);
  if(theHODigitizer) theHODigitizer->setNoiseSignalGenerator(noiseGenerator);
  if(theHOSiPMDigitizer) theHOSiPMDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHOAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setZDCNoiseSignalGenerator(HcalBaseSignalGenerator * noiseGenerator) {
  noiseGenerator->setParameterMap(theParameterMap);
  noiseGenerator->setElectronicsSim(theZDCElectronicsSim);
  theZDCDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theZDCAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  
  theHBHEAmplifier->setDbService(conditions.product());
  theHFAmplifier->setDbService(conditions.product());
  theHOAmplifier->setDbService(conditions.product());
  theZDCAmplifier->setDbService(conditions.product());
  theHFQIE10Amplifier->setDbService(conditions.product());
  theHBHEQIE11Amplifier->setDbService(conditions.product());
  
  theUpgradeHBHEElectronicsSim->setDbService(conditions.product());
  theUpgradeHFElectronicsSim->setDbService(conditions.product());
  theHFQIE10ElectronicsSim->setDbService(conditions.product());
  theHBHEQIE11ElectronicsSim->setDbService(conditions.product());

  theCoderFactory->setDbService(conditions.product());
  theUpgradeCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

  if(theHitCorrection != 0) {
    theHitCorrection->clear();
  }

  //initialize hits
  if(theHBHEDigitizer) theHBHEDigitizer->initializeHits();
  if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->initializeHits();
  if(theHODigitizer) theHODigitizer->initializeHits();
  if(theHOSiPMDigitizer) theHOSiPMDigitizer->initializeHits();
  if(theHBHEUpgradeDigitizer) theHBHEUpgradeDigitizer->initializeHits();
  if(theHFQIE10Digitizer) theHFQIE10Digitizer->initializeHits();
  if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->initializeHits();
  if(theHFDigitizer) theHFDigitizer->initializeHits();
  theZDCDigitizer->initializeHits();

}

void HcalDigitizer::accumulateCaloHits(edm::Handle<std::vector<PCaloHit> > const& hcalHandle, edm::Handle<std::vector<PCaloHit> > const& zdcHandle, int bunchCrossing, CLHEP::HepRandomEngine* engine, const HcalTopology *htopoP) {

  // Step A: pass in inputs, and accumulate digis
  if(isHCAL) {
    std::vector<PCaloHit> hcalHitsOrig = *hcalHandle.product();
    std::vector<PCaloHit> hcalHits;
    hcalHits.reserve(hcalHitsOrig.size());

    //evaluate darkening before relabeling
    if (testNumbering_) {
      if(m_HEDarkening || m_HFRecalibration){
	darkening(hcalHitsOrig);
      }
      // Relabel PCaloHits if necessary
      edm::LogInfo("HcalDigitizer") << "Calling Relabeller";
      theRelabeller->process(hcalHitsOrig);
    }
    
    //eliminate bad hits
    for (unsigned int i=0; i< hcalHitsOrig.size(); i++) {
      DetId id(hcalHitsOrig[i].id());
      HcalDetId hid(id);
      if (!htopoP->validHcal(hid)) {
        edm::LogError("HcalDigitizer") << "bad hcal id found in digitizer. Skipping " << id.rawId() << " " << hid << std::endl;
      } else {
#ifdef DebugLog
        std::cout << "HcalDigitizer format " << hid.oldFormat() << " for " << hid << std::endl;
#endif
        DetId newid = DetId(hid.newForm());
#ifdef DebugLog
        std::cout << "Hit " << i << " out of " << hcalHits.size() << " " << std::hex << id.rawId() << " --> " << newid.rawId() << std::dec << " " << HcalDetId(newid.rawId()) << '\n';
#endif
        hcalHitsOrig[i].setID(newid.rawId());
        hcalHits.push_back(hcalHitsOrig[i]);
      }
    }

    if(theHitCorrection != 0) {
      theHitCorrection->fillChargeSums(hcalHits);
    }
    if(hbhegeo) {
      if(theHBHEDigitizer) theHBHEDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHBHEQIE11Digitizer) theHBHEQIE11Digitizer->add(hcalHits, bunchCrossing, engine);
      if(theHBHEUpgradeDigitizer) theHBHEUpgradeDigitizer->add(hcalHits, bunchCrossing, engine);
    }

    if(hogeo) {
      if(theHODigitizer) theHODigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHOSiPMDigitizer) theHOSiPMDigitizer->add(hcalHits, bunchCrossing, engine);
    }

    if(hfgeo) {
      if(theHFDigitizer) theHFDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->add(hcalHits, bunchCrossing, engine);
      if(theHFQIE10Digitizer) theHFQIE10Digitizer->add(hcalHits, bunchCrossing, engine);
    } 
  } else {
    edm::LogInfo("HcalDigitizer") << "We don't have HCAL hit collection available ";
  }

  if(isZDC) {
    if(zdcgeo) {
      theZDCDigitizer->add(*zdcHandle.product(), bunchCrossing, engine);
    } 
  } else {
    edm::LogInfo("HcalDigitizer") << "We don't have ZDC hit collection available ";
  }
}

void HcalDigitizer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* engine) {
  // Step A: Get Inputs
  edm::InputTag zdcTag(hitsProducer_, "ZDCHITS");
  edm::Handle<std::vector<PCaloHit> > zdcHandle;
  e.getByLabel(zdcTag, zdcHandle);
  isZDC = zdcHandle.isValid();

  edm::InputTag hcalTag(hitsProducer_, "HcalHits");
  edm::Handle<std::vector<PCaloHit> > hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);
  isHCAL = hcalHandle.isValid();

  edm::ESHandle<HcalTopology> htopo;
  eventSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology *htopoP=htopo.product();

  accumulateCaloHits(hcalHandle, zdcHandle, 0, engine, htopoP);
}

void HcalDigitizer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup, CLHEP::HepRandomEngine* engine) {
  // Step A: Get Inputs
  edm::InputTag zdcTag(hitsProducer_, "ZDCHITS");
  edm::Handle<std::vector<PCaloHit> > zdcHandle;
  e.getByLabel(zdcTag, zdcHandle);
  isZDC = zdcHandle.isValid();

  edm::InputTag hcalTag(hitsProducer_, "HcalHits");
  edm::Handle<std::vector<PCaloHit> > hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);
  isHCAL = hcalHandle.isValid();

  edm::ESHandle<HcalTopology> htopo;
  eventSetup.get<HcalRecNumberingRecord>().get(htopo);
  const HcalTopology *htopoP=htopo.product();

  accumulateCaloHits(hcalHandle, zdcHandle, e.bunchCrossing(), engine, htopoP);
}

void HcalDigitizer::finalizeEvent(edm::Event& e, const edm::EventSetup& eventSetup, CLHEP::HepRandomEngine* engine) {

  // Step B: Create empty output
  std::unique_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::unique_ptr<HODigiCollection> hoResult(new HODigiCollection());
  std::unique_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
  std::unique_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());
  std::unique_ptr<HBHEUpgradeDigiCollection> hbheupgradeResult(new HBHEUpgradeDigiCollection());
  std::unique_ptr<HFUpgradeDigiCollection> hfupgradeResult(new HFUpgradeDigiCollection());
  std::unique_ptr<QIE10DigiCollection> hfQIE10Result(new QIE10DigiCollection());
  std::unique_ptr<QIE11DigiCollection> hbheQIE11Result(new QIE11DigiCollection());

  // Step C: Invoke the algorithm, getting back outputs.
  if(isHCAL&&hbhegeo){
    if(theHBHEDigitizer)        theHBHEDigitizer->run(*hbheResult, engine);
    if(theHBHEQIE11Digitizer)    theHBHEQIE11Digitizer->run(*hbheQIE11Result, engine);
    if(theHBHEUpgradeDigitizer) theHBHEUpgradeDigitizer->run(*hbheupgradeResult, engine);
  }
  if(isHCAL&&hogeo) {
    if(theHODigitizer) theHODigitizer->run(*hoResult, engine);
    if(theHOSiPMDigitizer) theHOSiPMDigitizer->run(*hoResult, engine);
  }
  if(isHCAL&&hfgeo) {
    if(theHFDigitizer) theHFDigitizer->run(*hfResult, engine);
    if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->run(*hfupgradeResult, engine);
    if(theHFQIE10Digitizer) theHFQIE10Digitizer->run(*hfQIE10Result, engine);
  }
  if(isZDC&&zdcgeo) {
    theZDCDigitizer->run(*zdcResult, engine);
  }
  
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE digis : " << hbheResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HO digis   : " << hoResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF digis   : " << hfResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL ZDC digis  : " << zdcResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE upgrade digis : " << hbheupgradeResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF upgrade digis : " << hfupgradeResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF QIE10 digis : " << hfQIE10Result->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE QIE11 digis : " << hbheQIE11Result->size();

#ifdef DebugLog
  std::cout << std::endl;
  std::cout << "HCAL HBHE digis : " << hbheResult->size() << std::endl;
  std::cout << "HCAL HO   digis : " << hoResult->size() << std::endl;
  std::cout << "HCAL HF   digis : " << hfResult->size() << std::endl;
  std::cout << "HCAL ZDC  digis : " << zdcResult->size() << std::endl;
  std::cout << "HCAL HBHE upgrade digis : " << hbheupgradeResult->size() << std::endl;
  std::cout << "HCAL HF   upgrade digis : " << hfupgradeResult->size() << std::endl;
  std::cout << "HCAL HF QIE10 digis : " << hfQIE10Result->size() << std::endl;
  std::cout << "HCAL HBHE QIE11 digis : " << hbheQIE11Result->size() << std::endl;
#endif

  // Step D: Put outputs into event
  e.put(std::move(hbheResult));
  e.put(std::move(hoResult));
  e.put(std::move(hfResult));
  e.put(std::move(zdcResult));
  e.put(std::move(hbheupgradeResult),"HBHEUpgradeDigiCollection");
  e.put(std::move(hfupgradeResult), "HFUpgradeDigiCollection");
  e.put(std::move(hfQIE10Result), "HFQIE10DigiCollection");
  e.put(std::move(hbheQIE11Result), "HBHEQIE11DigiCollection");

#ifdef DebugLog
  std::cout << std::endl << "========>  HcalDigitizer e.put " << std::endl <<  std::endl;
#endif

  if(theHitCorrection) {
    theHitCorrection->clear();
  }
}


void HcalDigitizer::beginRun(const edm::EventSetup & es) {
  checkGeometry(es);
  theShapes->beginRun(es);
}


void HcalDigitizer::endRun() {
  theShapes->endRun();
}


void HcalDigitizer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<CaloGeometryRecord>().get(geometry);
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  eventSetup.get<HcalRecNumberingRecord>().get(pHRNDC);

  // See if it's been updated
  if (&*geometry != theGeometry) {
    theGeometry = &*geometry;
    theRecNumber= &*pHRNDC;
    updateGeometry(eventSetup);
  }
}


void  HcalDigitizer::updateGeometry(const edm::EventSetup & eventSetup) {
  if(theHBHEResponse) theHBHEResponse->setGeometry(theGeometry);
  if(theHBHESiPMResponse) theHBHESiPMResponse->setGeometry(theGeometry);
  if(theHOResponse) theHOResponse->setGeometry(theGeometry);
  if(theHOSiPMResponse) theHOSiPMResponse->setGeometry(theGeometry);
  theHFResponse->setGeometry(theGeometry);
  theHFQIE10Response->setGeometry(theGeometry);
  theZDCResponse->setGeometry(theGeometry);
  if(theRelabeller) theRelabeller->setGeometry(theGeometry,theRecNumber);

  const std::vector<DetId>& hbCells = theGeometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& heCells = theGeometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  const std::vector<DetId>& hoCells = theGeometry->getValidDetIds(DetId::Hcal, HcalOuter);
  const std::vector<DetId>& hfCells = theGeometry->getValidDetIds(DetId::Hcal, HcalForward);
  const std::vector<DetId>& zdcCells = theGeometry->getValidDetIds(DetId::Calo, HcalZDCDetId::SubdetectorId);
  //const std::vector<DetId>& hcalTrigCells = geometry->getValidDetIds(DetId::Hcal, HcalTriggerTower);
  //const std::vector<DetId>& hcalCalib = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);
//  std::cout<<"HcalDigitizer::CheckGeometry number of cells: "<<zdcCells.size()<<std::endl;
  if(zdcCells.empty()) zdcgeo = false;
  if(hbCells.empty() && heCells.empty()) hbhegeo = false;
  if(hoCells.empty()) hogeo = false;
  if(hfCells.empty()) hfgeo = false;
  // combine HB & HE

  hbheCells = hbCells;
  hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());
  //handle mixed QIE8/11 scenario in HBHE
  if(theHBHEUpgradeDigitizer) theHBHEUpgradeDigitizer->setDetIds(hbheCells);
  else buildHBHEQIECells(hbheCells,eventSetup);
  
  buildHOSiPMCells(hoCells, eventSetup);
  
  //handle mixed QIE8/10 scenario in HF
  if(theHFUpgradeDigitizer) theHFUpgradeDigitizer->setDetIds(hfCells);
  else buildHFQIECells(hfCells,eventSetup);
  
  theZDCDigitizer->setDetIds(zdcCells);
}

void HcalDigitizer::buildHFQIECells(const std::vector<DetId>& allCells, const edm::EventSetup & eventSetup) {
	//if results are already cached, no need to look again
	if(theHFQIE8DetIds.size()>0 || theHFQIE10DetIds.size()>0) return;
	
	//get the QIETypes
	edm::ESHandle<HcalQIETypes> q;
    eventSetup.get<HcalQIETypesRcd>().get(q);
	edm::ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
   
    HcalQIETypes qieTypes(*q.product());
    if (qieTypes.topo()==0) {
      qieTypes.setTopo(htopo.product());
    }
	
	for(std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      HcalQIENum qieType = HcalQIENum(qieTypes.getValues(*detItr)->getValue());
      if(qieType == QIE8) {
        theHFQIE8DetIds.push_back(*detItr);
      } else if(qieType == QIE10) {
        theHFQIE10DetIds.push_back(*detItr);
      } else { //default is QIE8
        theHFQIE8DetIds.push_back(*detItr);
      }
    }
	
	if(theHFQIE8DetIds.size()>0) theHFDigitizer->setDetIds(theHFQIE8DetIds);
	else {
		delete theHFDigitizer;
		theHFDigitizer = NULL;
	}
	
	if(theHFQIE10DetIds.size()>0) theHFQIE10Digitizer->setDetIds(theHFQIE10DetIds);
	else {
		delete theHFQIE10Digitizer;
		theHFQIE10Digitizer = NULL;
	}
}

void HcalDigitizer::buildHBHEQIECells(const std::vector<DetId>& allCells, const edm::EventSetup & eventSetup) {
	//if results are already cached, no need to look again
	if(theHBHEQIE8DetIds.size()>0 || theHBHEQIE11DetIds.size()>0) return;
	
	//get the QIETypes
	edm::ESHandle<HcalQIETypes> q;
    eventSetup.get<HcalQIETypesRcd>().get(q);
	edm::ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
   
    HcalQIETypes qieTypes(*q.product());
    if (qieTypes.topo()==0) {
      qieTypes.setTopo(htopo.product());
    }
	
	for(std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      HcalQIENum qieType = HcalQIENum(qieTypes.getValues(*detItr)->getValue());
      if(qieType == QIE8) {
        theHBHEQIE8DetIds.push_back(*detItr);
      }
      else if(qieType == QIE11) {
        theHBHEQIE11DetIds.push_back(*detItr);
      }
      else { //default is QIE8
        theHBHEQIE8DetIds.push_back(*detItr);
      }
    }
	
	if(theHBHEQIE8DetIds.size()>0) theHBHEDigitizer->setDetIds(theHBHEQIE8DetIds);
	else {
		delete theHBHEDigitizer;
		theHBHEDigitizer = NULL;
	}
	
	if(theHBHEQIE11DetIds.size()>0) theHBHEQIE11Digitizer->setDetIds(theHBHEQIE11DetIds);
	else {
		delete theHBHEQIE11Digitizer;
		theHBHEQIE11Digitizer = NULL;
	}
	
	if(theHBHEQIE8DetIds.size()>0 && theHBHEQIE11DetIds.size()>0){
		theHBHEHitFilter.setDetIds(theHBHEQIE8DetIds);
		theHBHEQIE11HitFilter.setDetIds(theHBHEQIE11DetIds);
	}
}

void HcalDigitizer::buildHOSiPMCells(const std::vector<DetId>& allCells, const edm::EventSetup & eventSetup) {
  // all HPD

  if(theHOSiPMCode == 0) {
    theHODigitizer->setDetIds(allCells);
  } else if(theHOSiPMCode == 1) {
    theHOSiPMDigitizer->setDetIds(allCells);
    // FIXME pick Zecotek or hamamatsu?
  } else if(theHOSiPMCode == 2) {
    std::vector<HcalDetId> zecotekDetIds, hamamatsuDetIds;
    edm::ESHandle<HcalMCParams> p;
    eventSetup.get<HcalMCParamsRcd>().get(p);
    edm::ESHandle<HcalTopology> htopo;
    eventSetup.get<HcalRecNumberingRecord>().get(htopo);
   
    HcalMCParams mcParams(*p.product());
    if (mcParams.topo()==0) {
      mcParams.setTopo(htopo.product());
    }

    for(std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      int shapeType = mcParams.getValues(*detItr)->signalShape();
      if(shapeType == HcalShapes::ZECOTEK) {
        zecotekDetIds.emplace_back(*detItr);
        theHOSiPMDetIds.push_back(*detItr);
      } else if(shapeType == HcalShapes::HAMAMATSU) {
        hamamatsuDetIds.emplace_back(*detItr);
        theHOSiPMDetIds.push_back(*detItr);
      } else {
        theHOHPDDetIds.push_back(*detItr);
      }
    }

    if(theHOHPDDetIds.size()>0) theHODigitizer->setDetIds(theHOHPDDetIds);
    else {
      delete theHODigitizer;
      theHODigitizer = NULL;
    }
	
    if(theHOSiPMDetIds.size()>0) theHOSiPMDigitizer->setDetIds(theHOSiPMDetIds);
    else {
      delete theHOSiPMDigitizer;
      theHOSiPMDigitizer = NULL;
    }
	
	if(theHOHPDDetIds.size()>0 && theHOSiPMDetIds.size()>0){
      theHOSiPMHitFilter.setDetIds(theHOSiPMDetIds);
      theHOHitFilter.setDetIds(theHOHPDDetIds);
    }
	
    theParameterMap->setHOZecotekDetIds(zecotekDetIds);
    theParameterMap->setHOHamamatsuDetIds(hamamatsuDetIds);

    // make sure we don't got through this exercise again
    theHOSiPMCode = -2;
  }
}

void HcalDigitizer::darkening(std::vector<PCaloHit>& hcalHits) {

  for (unsigned int ii=0; ii<hcalHits.size(); ++ii) {
    uint32_t tmpId = hcalHits[ii].id();
    int det, z, depth, ieta, phi, lay;
    HcalTestNumbering::unpackHcalIndex(tmpId,det,z,depth,ieta,phi,lay);
	
    bool darkened = false;
    float dweight = 1.;
	
    if(det==int(HcalEndcap) && m_HEDarkening){
      //HE darkening
      dweight = m_HEDarkening->degradation(deliveredLumi,ieta,lay-2);//NB:diff. layer count
      darkened = true;
    } else if(det==int(HcalForward) && m_HFRecalibration){
      //HF darkening - approximate: invert recalibration factor
      dweight = 1.0/m_HFRecalibration->getCorr(ieta,depth,deliveredLumi);
      darkened = true;
    }
	
    //create new hit with darkened energy
    //if(darkened) hcalHits[ii] = PCaloHit(hcalHits[ii].energyEM()*dweight,hcalHits[ii].energyHad()*dweight,hcalHits[ii].time(),hcalHits[ii].geantTrackId(),hcalHits[ii].id());
	
    //reset hit energy
    if(darkened) hcalHits[ii].setEnergy(hcalHits[ii].energy()*dweight);	
  }
  
}
    

