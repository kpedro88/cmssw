#ifndef L1Trigger_TrackerTFP_State_h
#define L1Trigger_TrackerTFP_State_h

#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <vector>


namespace trackerTFP {

  // 
  class State {
  public:
    //
    State(State* state);
    // proto state constructor
    State(const DataFormats* dataFormats, TrackKFin* track);
    // combinatoric state constructor
    State(State* state, StubKFin* stub);
    // updated state constructor
    State(State* state, const std::vector<double>& doubles);
    ~State(){}

    void finish();
    double chi2() const { return chi2_; }
    int numSkippedLayers() const { return numSkippedLayers_; }
    int numConsistentLayers() const { return numConsistentLayers_; }
    TrackKFin* track() const { return track_; }
    State* parent() const { return parent_; }
    StubKFin*  stub() const { return stub_; }
    double r() const { return stub_->r(); }
    double phi() const { return stub_->phi(); }
    double z() const { return stub_->z(); }
    int sectorPhi() const { return track_->sectorPhi(); }
    int sectorEta() const { return track_->sectorEta(); }
    const TTBV& hitPattern() const { return hitPattern_; }
    int trackId() const { return track_->trackId(); }
    const std::vector<int>& layerMap() const { return layerMap_; }
    bool barrel() const { return setup_->barrel(stub_->ttStubRef()); }
    bool psModule() const { return setup_->psModule(stub_->ttStubRef()); }
    int layer() const { return stub_->layer(); }
    void x0(double d) { x0_ = d; }
    void x1(double d) { x1_ = d; }
    void x2(double d) { x2_ = d; }
    void x3(double d) { x3_ = d; }
    double x0() const { return x0_; }
    double x1() const { return x1_; }
    double x2() const { return x2_; }
    double x3() const { return x3_; }
    double C00() const { return C00_; }
    double C01() const { return C01_; }
    double C11() const { return C11_; }
    double C22() const { return C22_; }
    double C23() const { return C23_; }
    double C33() const { return C33_; }
    double H12() const { return r() + setup_->chosenRofPhi() - setup_->chosenRofZ(); }
    double H00() const { return -r(); }
    double m0() const { return stub_->phi(); }
    double m1() const { return stub_->z(); }
    double v0() const { return setup_->v0(stub_->ttStubRef(), track_->qOverPt()); }
    double v1() const { return setup_->v1(stub_->ttStubRef(), track_->cotGlobal()); }
    int nPS() const { return nPS_; }
    FrameTrack frame() const;
    std::vector<StubKF> stubs() const;

  private:
    //
    const DataFormats* dataFormats_;
    //
    const trackerDTC::Setup* setup_;
    // found mht track
    TrackKFin* track_;
    // previous state, nullptr for first states
    State* parent_;
    // stub to add
    StubKFin* stub_;
    // shows which stub on each layer has been added so far
    std::vector<int> layerMap_;
    // shows which layer has been added so far
    TTBV hitPattern_;
    double x0_;
    double x1_;
    double x2_;
    double x3_;
    double C00_;
    double C01_;
    double C11_;
    double C22_;
    double C23_;
    double C33_;
    std::vector<double> chi20_;
    std::vector<double> chi21_;
    double chi2_;
    int nPS_;
    int numSkippedLayers_;
    int numConsistentLayers_;
  };

}

#endif