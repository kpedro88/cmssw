#include "CondFormats/EgammaObjects/interface/GBRForest.h"

//_______________________________________________________________________
GBRForest::GBRForest() : 
  fInitialResponse(0.)
{}

void GBRForest::initNewTrees() {
  if(fTrees.empty()) return;

  fTreesNew.reserve(fTrees.size());
  for(const auto& tree : fTrees){
     fTreesNew.emplace_back(tree);
  }
}
