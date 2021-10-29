#ifndef PhysicsTools_PatUtils_ShiftedParticleProducer_h
#define PhysicsTools_PatUtils_ShiftedParticleProducer_h

/** \class ShiftedParticleProducer
 *
 * Vary energy of electrons/muons/tau-jets by +/- 1 standard deviation,
 * in order to estimate resulting uncertainty on MET
 *
 * NOTE: energy scale uncertainties need to be specified in python config
 *
 * \author Matthieu marionneau ETH
 * 
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include <string>
#include <vector>

class ShiftedParticleProducer : public edm::stream::EDProducer<>
{
  typedef edm::View<reco::Candidate> CandidateView;

 public:

  explicit ShiftedParticleProducer(const edm::ParameterSet& cfg);
  ~ShiftedParticleProducer() override;
  
 private:

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  double getUncShift(const CandidateView::const_iterator& originalParticle);


  std::string moduleLabel_;

  edm::EDGetTokenT<CandidateView> srcToken_;

  struct binningEntryType
  {
  binningEntryType(const std::string& uncertainty, const std::string& moduleLabel)
      : binSelection_(nullptr),
      binUncertainty_(uncertainty),
      energyDep_(false)
    {
      binUncFormula_ = std::make_unique<reco::FormulaEvaluator>(binUncertainty_);
    }
  binningEntryType(const edm::ParameterSet& cfg, const std::string& moduleLabel)
  : binSelection_(new StringCutObjectSelector<reco::Candidate>(cfg.getParameter<std::string>("binSelection"))),
      binUncertainty_(cfg.getParameter<std::string>("binUncertainty")),
      energyDep_(false)
    {
      binUncFormula_ = std::make_unique<reco::FormulaEvaluator>(binUncertainty_);
      if(cfg.exists("energyDependency") ) {energyDep_=cfg.getParameter<bool>("energyDependency");
      }
    }
    ~binningEntryType()
    {
    }
    std::unique_ptr<StringCutObjectSelector<reco::Candidate> > binSelection_;
    //double binUncertainty_;
    std::string binUncertainty_;
    std::unique_ptr<reco::FormulaEvaluator> binUncFormula_;
    bool energyDep_;
  };
  std::vector<binningEntryType*> binning_;

  double shiftBy_; // set to +1.0/-1.0 for up/down variation of energy scale
};

#endif
