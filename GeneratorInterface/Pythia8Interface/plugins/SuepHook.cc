#include "SuepHook.h"

SuepHook::SuepHook(const edm::ParameterSet& iConfig) :
  idMediator_(iConfig.getParameter<int>("idMediator")),
  idDark_(iConfig.getParameter<int>("idDark")),
  temperature_(iConfig.getParameter<double>("temperature"))
{ }

bool SuepHook::initAfterBeams() {
  mMediator_ = particleDataPtr->m0(idMediator_);
  mDark_ = particleDataPtr->m0(idDark_);
  bool medDecay = particleDataPtr->mayDecay(idMediator_);
  if(!medDecay){
    infoPtr->errorMsg("Error in SuepHook::initAfterBeams: mediator decay should be enabled");
    return false;
  }

  //construct the shower helper
  suep_shower_ = std::make_unique<Suep_shower>(mDark_, temperature_, mMediator_, rndmPtr);

  return true;
}

//based on https://gitlab.com/simonknapen/suep_generator/-/blob/master/suep_main.cc:AttachSuepShower
bool SuepHook::doVetoProcessLevel(Pythia8::Event& event) {
    Pythia8::Vec4 pMediator, pDark;

    // Generate the shower, output are 4 vectors in the rest frame of the shower
    const auto& suep_shower4momenta = suep_shower_->generate_shower();

    // Find the mediator in the event
    for (int i = 0; i < event.size(); ++i){ 
        if (event[i].id()==idMediator_ && event[i].isFinal()) {
          pMediator = event[i].p();

          // undo mediator decay
          event[i].undoDecay();

          // Loop over hidden sector mesons and append to the event
          for (const auto& vDark : suep_shower4momenta){
            //construct pythia 4vector
            pDark.p(vDark[1],vDark[2],vDark[3],vDark[0]);
            
            // boost to the lab frame
            pDark.bst(pMediator.px()/pMediator.e(),pMediator.py()/pMediator.e(), pMediator.pz()/pMediator.e());
            
            //append particle to the event w/ hidden meson pdg code
            event.append(idDark_, 91, i, 0, 0, 0, 0, 0, pDark.px(), pDark.py(), pDark.pz(), pDark.e(), mDark_);
          }
          
          // Change the status code of the mediator to reflect that it has decayed.
          event[i].statusNeg();
          
          //set daughters of the mediator (Not all mesons are recognized as daughters, hopefully doesn't result in issues.)
          event[i].daughters(event.size()-2, event.size()-1); 
          break;
        }
    }

  //allow event to continue
  return false;
}
