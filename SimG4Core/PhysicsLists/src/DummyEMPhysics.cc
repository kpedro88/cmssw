#include "SimG4Core/PhysicsLists/interface/DummyEMPhysics.h"
#include "G4EmParameters.hh"
#include "G4ParticleTable.hh"

#include "G4ParticleDefinition.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"

#include "G4eMultipleScattering.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4LeptonConstructor.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"

#include "G4SystemOfUnits.hh"

DummyEMPhysics::DummyEMPhysics(G4int ver) :
  G4VPhysicsConstructor("CMSEmGeantV"), verbose(ver) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);
  // inactivate energy loss fluctuations (we don't have model for it in GV)
  param->SetLossFluctuations(false);
  // inactivate to use cuts as final range
  param->SetUseCutAsFinalRange(false);
  //
  // MSC options:
  param->SetMscStepLimitType(fUseSafety); 
  //  param->SetMscSkin(3);
  param->SetMscRangeFactor(0.2);  // default EM-opt1 value
  SetPhysicsType(bElectromagnetic);
}

void DummyEMPhysics::ConstructParticle() {
  // gamma
  G4Gamma::Gamma();

  // leptons
  G4Electron::Electron();
  G4Positron::Positron();

  G4LeptonConstructor pLeptonConstructor;
  pLeptonConstructor.ConstructParticle();
}

void DummyEMPhysics::ConstructProcess() {

  if(verbose > 0) {
    G4cout << "### " << GetPhysicsName() << " Construct Processes " << G4endl;
  }

  // This EM builder takes GeantV variant of physics

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  G4ParticleDefinition* particle = G4Gamma::Gamma();

  //      ph->RegisterProcess(new G4PhotoElectricEffect, particle);
  ph->RegisterProcess(new G4ComptonScattering(), particle);
  ph->RegisterProcess(new G4GammaConversion(), particle);
  G4double LivermoreLowEnergyLimit  = 1.*eV;
  G4double LivermoreHighEnergyLimit = 1.*TeV;
  G4PhotoElectricEffect* thePhotoElectricEffect = new G4PhotoElectricEffect();
  G4LivermorePhotoElectricModel* theLivermorePhotoElectricModel = new G4LivermorePhotoElectricModel();
  theLivermorePhotoElectricModel->SetLowEnergyLimit(LivermoreLowEnergyLimit);
  theLivermorePhotoElectricModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
  thePhotoElectricEffect->AddEmModel(0, theLivermorePhotoElectricModel);
  ph->RegisterProcess(thePhotoElectricEffect, particle);

  particle = G4Electron::Electron();

  G4eMultipleScattering* msce = new G4eMultipleScattering;
  G4GoudsmitSaundersonMscModel* msce1 = new G4GoudsmitSaundersonMscModel();
  msce->AddEmModel(0, msce1);
  ph->RegisterProcess(msce,particle);
  ph->RegisterProcess(new G4eIonisation(), particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);

  particle = G4Positron::Positron();

  G4eMultipleScattering* mscp = new G4eMultipleScattering;
  G4GoudsmitSaundersonMscModel* mscp1 = new G4GoudsmitSaundersonMscModel();
  mscp->AddEmModel(0, mscp1);
  ph->RegisterProcess(mscp,particle);
  ph->RegisterProcess(new G4eIonisation(), particle);
  ph->RegisterProcess(new G4eBremsstrahlung(), particle);
  ph->RegisterProcess(new G4eplusAnnihilation(), particle);
}
