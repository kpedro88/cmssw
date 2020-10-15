#include "L1Trigger/TrackerTFP/interface/State.h"

using namespace std;
using namespace trackerDTC;

namespace trackerTFP {

  //
  State::State(State* state) :
    dataFormats_(state->dataFormats_),
    setup_(state->setup_),
    track_(state->track_),
    parent_(state->parent_),
    stub_(state->stub_),
    layerMap_(state->layerMap_),
    hitPattern_(state->hitPattern_),
    x0_(state->x0_),
    x1_(state->x1_),
    x2_(state->x2_),
    x3_(state->x3_),
    C00_(state->C00_),
    C01_(state->C01_),
    C11_(state->C11_),
    C22_(state->C22_),
    C23_(state->C23_),
    C33_(state->C33_),
    chi20_(state->chi20_),
    chi21_(state->chi21_),
    chi2_(state->chi2_),
    nPS_(state->nPS_),
    numSkippedLayers_(state->numSkippedLayers_),
    numConsistentLayers_(state->numConsistentLayers_)
  {}

  // proto state constructor
  State::State(const DataFormats* dataFormats, TrackKFin* track) :
    dataFormats_(dataFormats),
    setup_(dataFormats->setup()),
    track_(track),
    parent_(nullptr),
    stub_(nullptr),
    layerMap_(setup_->numLayers()),
    hitPattern_(0, setup_->numLayers()),
    chi2_(0.),
    nPS_(0),
    numSkippedLayers_(0),
    numConsistentLayers_(0)
  {
    // initial track parameter residuals w.r.t. found track
    x0_ = 0.;
    x1_ = 0.;
    x2_ = 0.;
    x3_ = 0.;
    // initial uncertainties
    C00_  = pow(dataFormats_->base(Variable::qOverPt, Process::sf), 2);
    C11_  = pow(dataFormats_->base(Variable::phiT, Process::sf), 2);
    C22_  = pow(dataFormats_->base(Variable::cot, Process::sf), 2);
    C33_  = pow(dataFormats_->base(Variable::zT, Process::sf), 2);
    C01_  = 0.;
    C23_  = 0.;
    // first stub
    if (track->stubs()[track->hitPattern().plEncode()].empty()) {
      cout << track->trackId() << endl;
      cout << track->hitPattern() << " ";
      for (int n : track->layerMap())
        cout << n;
      cout << " ";
      for (const vector<StubKFin*>& layer : track->stubs())
        cout << layer.size();
      cout << endl;
      throw cms::Exception("...");
    }
    stub_ = track->layerStub(track->hitPattern().plEncode());
    //
    chi20_.reserve(setup_->kfMaxLayers());
    chi21_.reserve(setup_->kfMaxLayers());
  }

  // combinatoric state constructor
  State::State(State* state, StubKFin* stub) : State(state)
  {
    parent_ = state->parent();
    stub_ = stub;
  }

  // updated state constructor
  State::State(State* state, const std::vector<double>& doubles) : State(state)
  {
    parent_ = state;
    // updated track parameter and uncertainties
    x0_ = doubles[0];
    x1_ = doubles[1];
    x2_ = doubles[2];
    x3_ = doubles[3];
    C00_ = doubles[4];
    C11_ = doubles[5];
    C22_ = doubles[6];
    C33_ = doubles[7];
    C01_ = doubles[8];
    C23_ = doubles[9];
    // update maps
    const int layer = stub_->layer();
    hitPattern_.set(layer);
    const vector<StubKFin*>& stubs = track_->layerStubs(layer);
    layerMap_[layer] = distance(stubs.begin(), find(stubs.begin(), stubs.end(), stub_));
    nPS_ += setup_->psModule(stub_->ttStubRef()) ? 1 : 0;
    // pick next stub
    stub_ = nullptr;
    if (hitPattern_.count() != setup_->kfMaxLayers()) {
      for (int nextLayer = layer + 1; nextLayer < setup_->numLayers(); nextLayer++) {
        if (track_->hitPattern(nextLayer)) {
          stub_ = track_->layerStub(nextLayer);
          break;
        }
      }
    }
  }

  FrameTrack State::frame() const {
    TrackKF track(*track_, x1_, x0_, x3_, x2_);
    return track.frame();
  }

  vector<StubKF> State::stubs() const {
    vector<StubKF> stubs;
    stubs.reserve(hitPattern_.count());
    State* s = parent_;
    while (s) {
      stubs.emplace_back(*(s->stub()), x0_, x1_, x2_, x3_);
      s = s->parent();
    }
    return stubs;
  }

  //
  void State::finish() {
    State* p = parent_;
    while (p) {
      const TTStubRef& ttStubRef = p->stub()->ttStubRef();
      const double r0 = p->m0() - (x1_ + x0_ * p->H00());
      const double r1 = p->m1() - (x3_ + x2_ * p->H12());
      const double d0 = setup_->dPhi(ttStubRef, track_->qOverPt());
      const double d1 = setup_->dZ(ttStubRef, track_->cotGlobal());
      const double v0 = setup_->v0(ttStubRef, track_->qOverPt());
      const double v1 = setup_->v1(ttStubRef, track_->cotGlobal());
      if (abs(r0) < d0 / 2. && abs(r1) < d1 / 2.)
        numConsistentLayers_++;
      const double chi20 = pow(r0, 2) / v0;
      const double chi21 = pow(r1, 2) / v1;
      chi2_ += chi20 + chi21;
      chi20_.push_back(chi20);
      chi21_.push_back(chi21);
      p = p->parent();
    }
    numSkippedLayers_ = hitPattern_.count(0, hitPattern_.pmEncode(), false);
  }

} // namespace trackerTFP