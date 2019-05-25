#include "CondFormats/EgammaObjects/interface/GBRTreeNew.h"

//_______________________________________________________________________
GBRTreeNew::GBRTreeNew() {}

//_______________________________________________________________________
GBRTreeNew::GBRTreeNew(int nIntermediate, int nTerminal)
{

  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;

  fNodes.reserve(nIntermediate);
  fResponses.reserve(nTerminal);

}

GBRTreeNew::GBRTreeNew(const GBRTree& tree) : fResponses(tree.Responses()) {
  fNodes.reserve(tree.CutIndices().size());
  for(unsigned i = 0; i < tree.CutIndices().size(); ++i){
    fNodes.emplace_back(tree.CutIndices()[i],tree.CutVals()[i],tree.LeftIndices()[i],tree.RightIndices()[i]);
  }
}
