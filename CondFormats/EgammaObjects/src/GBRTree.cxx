#include "CondFormats/EgammaObjects/interface/GBRTree.h"

//_______________________________________________________________________
GBRTree::GBRTree() {}

//_______________________________________________________________________
GBRTree::GBRTree(int nIntermediate, int nTerminal)
{

  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;
  
  fNodes.reserve(nIntermediate);
  fResponses.reserve(nTerminal);

}
