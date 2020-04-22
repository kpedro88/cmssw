#ifndef GeneratorInterface_Pythia8Interface_SuepHook_h
#define GeneratorInterface_Pythia8Interface_SuepHook_h

#include "Pythia8/Pythia.h"

#include "GeneratorInterface/Pythia8Interface/interface/suep_shower.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class SuepHook : public Pythia8::UserHooks {
public:
  SuepHook(const edm::ParameterSet& iConfig);
  ~SuepHook() {}

  bool initAfterBeams() override;

  bool canVetoProcessLevel() override { return true; }
  bool doVetoProcessLevel(Pythia8::Event& event) override;

protected:
  int idMediator_, idDark_;
  float temperature_, mMediator_, mDark_;
  std::unique_ptr<Suep_shower> suep_shower_;
};

#endif
