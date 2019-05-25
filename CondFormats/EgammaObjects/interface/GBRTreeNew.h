
#ifndef EGAMMAOBJECTS_GBRTreeNew
#define EGAMMAOBJECTS_GBRTreeNew

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GBRForest                                                            //
//                                                                      //
// A fast minimal implementation of Gradient-Boosted Regression Trees   //
// which has been especially optimized for size on disk and in memory.  //                                                                  
//                                                                      //
// Designed to be built from TMVA-trained trees, but could also be      //
// generalized to otherwise-trained trees, classification,              //
//  or other boosting methods in the future                             //
//                                                                      //
//  Josh Bendavid - MIT                                                 //
//////////////////////////////////////////////////////////////////////////

// The decision tree is implemented here as a set of two arrays, one for
// intermediate nodes, containing the variable index and cut value, as well
// as the indices of the 'left' and 'right' daughter nodes.  Positive indices
// indicate further intermediate nodes, whereas negative indices indicate
// terminal nodes, which are stored simply as a vector of regression responses

#include "GBRTree.h"
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

  class GBRTreeNew {

    public:
       struct GBRNode {
         GBRNode() : fCutIndex(0), fCutVal(0), fLeftIndex(0), fRightIndex(0) {}
         GBRNode(unsigned cutIndex, float cutVal, int leftIndex, int rightIndex) :
           fCutIndex(cutIndex), fCutVal(cutVal), fLeftIndex(leftIndex), fRightIndex(rightIndex) {}

         unsigned fCutIndex;
         float fCutVal;
         int fLeftIndex;
         int fRightIndex;

         COND_SERIALIZABLE;
       };

       GBRTreeNew();
       GBRTreeNew(const GBRTree& tree);
       explicit GBRTreeNew(int nIntermediate, int nTerminal);
       
       double GetResponse(const float* vector) const;

       std::vector<GBRNode> &Nodes() { return fNodes; }
       const std::vector<GBRNode> &Nodes() const { return fNodes; }
       
       std::vector<float> &Responses() { return fResponses; }       
       const std::vector<float> &Responses() const { return fResponses; }
       
    protected:      

       std::vector<float> fResponses;  
       std::vector<GBRNode> fNodes;        
  
  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline double GBRTreeNew::GetResponse(const float* vector) const
{
   int index = 0;
   do {
      auto r = fNodes[index].fRightIndex;
      auto l = fNodes[index].fLeftIndex;
     index =  vector[fNodes[index].fCutIndex] > fNodes[index].fCutVal ? r : l;
   } while (index>0);
   return fResponses[-index];
}

#endif
