#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iterator>
#include <algorithm>

using namespace std;
using namespace edm;
using namespace trackerDTC;

namespace trackerTFP {

  // default constructor, trying to needs space as proper constructed object
  DataFormats::DataFormats() :
    numDataFormats_(0),
    formats_(+Variable::end, std::vector<DataFormat*>(+Process::end, nullptr)),
    numUnusedBitsStubs_(+Process::end, TTBV::S),
    numUnusedBitsTracks_(+Process::end, TTBV::S),
    numChannel_(+Process::end, 0)
  {
    setup_ = nullptr;
    countFormats();
    dataFormats_.reserve(numDataFormats_);
    numStreams_.reserve(+Process::end);
  }

  // method to count number of unique data formats
  template<Variable v = Variable::begin, Process p = Process::begin>
  void DataFormats::countFormats() {
    if constexpr(config_[+v][+p] == p)
      numDataFormats_++;
    if constexpr(++p != Process::end)
      countFormats<v, ++p>();
    else if constexpr(++v != Variable::end)
      countFormats<++v>();
  }

  // proper constructor
  DataFormats::DataFormats(const ParameterSet& iConfig, const Setup* setup) : DataFormats() {
    iConfig_ = iConfig;
    setup_ = setup;
    fillDataFormats();
    for (const Process p : Processes)
      for (const Variable v : stubs_[+p])
        numUnusedBitsStubs_[+p] -= formats_[+v][+p] ? formats_[+v][+p]->width() : 0;
    for (const Process p : Processes)
      for (const Variable v : tracks_[+p])
        numUnusedBitsTracks_[+p] -= formats_[+v][+p] ? formats_[+v][+p]->width() : 0;
    numChannel_[+Process::dtc] = setup_->numDTCsPerRegion();
    numChannel_[+Process::pp] = setup_->numDTCsPerTFP();
    numChannel_[+Process::gp] = setup_->numSectors();
    numChannel_[+Process::ht] = setup_->htNumBinsQoverPt();
    numChannel_[+Process::mht] = setup_->htNumBinsQoverPt();
    numChannel_[+Process::sf] = setup_->htNumBinsQoverPt();
    numChannel_[+Process::kfin] = setup_->kfNumWorker() * setup_->numLayers();
    numChannel_[+Process::kf] = setup_->kfNumWorker();
    transform(numChannel_.begin(), numChannel_.end(), back_inserter(numStreams_), [this](int channel){ return channel * setup_->numRegions(); });
  }

  // constructs data formats of all unique used variables and flavours
  template<Variable v = Variable::begin, Process p = Process::begin>
  void DataFormats::fillDataFormats() {
    if constexpr(config_[+v][+p] == p) {
      dataFormats_.emplace_back(Format<v, p>(iConfig_, setup_));
      fillFormats<v, p>();
    }
    if constexpr(++p != Process::end)
      fillDataFormats<v, ++p>();
    else if constexpr(++v != Variable::end)
      fillDataFormats<++v>();
  }

  // helper (loop) data formats of all unique used variables and flavours
  template<Variable v, Process p, Process it = Process::begin>
  void DataFormats::fillFormats() {
    if (config_[+v][+it] == p) {
      formats_[+v][+it] = &dataFormats_.back();
    }
    if constexpr(++it != Process::end)
      fillFormats<v, p, ++it>();
  }

  // converts bits to ntuple of variables
  template<typename ...Ts>
  void DataFormats::convertStub(const TTDTC::BV& bv, tuple<Ts...>& data, Process p) const {
    TTBV ttBV(bv);
    extractStub(ttBV, data, p);
  }

  // helper (loop) to convert bits to ntuple of variables
  template<int it = 0, typename ...Ts>
  void DataFormats::extractStub(TTBV& ttBV, std::tuple<Ts...>& data, Process p) const {
    Variable v = *next(stubs_[+p].begin(), sizeof...(Ts) - 1 - it);
    formats_[+v][+p]->extract(ttBV, get<sizeof...(Ts) - 1 - it>(data));
    if constexpr(it + 1 != sizeof...(Ts))
      extractStub<it + 1>(ttBV, data, p);
  }

  // converts ntuple of variables to bits
  template<typename... Ts>
  void DataFormats::convertStub(const std::tuple<Ts...>& data, TTDTC::BV& bv, Process p) const {
    TTBV ttBV(1, numUnusedBitsStubs_[+p]);
    attachStub(data, ttBV, p);
    bv = ttBV.bs();
  }

  // helper (loop) to convert ntuple of variables to bits
  template<int it = 0, typename... Ts>
  void DataFormats::attachStub(const tuple<Ts...>& data, TTBV& ttBV, Process p) const {
    Variable v = *next(stubs_[+p].begin(), it);
    formats_[+v][+p]->attach(get<it>(data), ttBV);
    if constexpr(it + 1 != sizeof...(Ts))
      attachStub<it + 1>(data, ttBV, p);
  }

  // converts bits to ntuple of variables
  template<typename ...Ts>
  void DataFormats::convertTrack(const TTDTC::BV& bv, tuple<Ts...>& data, Process p) const {
    TTBV ttBV(bv);
    extractTrack(ttBV, data, p);
  }

  // helper (loop) to convert bits to ntuple of variables
  template<int it = 0, typename ...Ts>
  void DataFormats::extractTrack(TTBV& ttBV, std::tuple<Ts...>& data, Process p) const {
    Variable v = *next(tracks_[+p].begin(), sizeof...(Ts) - 1 - it);
    formats_[+v][+p]->extract(ttBV, get<sizeof...(Ts) - 1 - it>(data));
    if constexpr(it + 1 != sizeof...(Ts))
      extractTrack<it + 1>(ttBV, data, p);
  }

  // converts ntuple of variables to bits
  template<typename... Ts>
  void DataFormats::convertTrack(const std::tuple<Ts...>& data, TTDTC::BV& bv, Process p) const {
    TTBV ttBV(1, numUnusedBitsTracks_[+p]);
    attachTrack(data, ttBV, p);
    bv = ttBV.bs();
  }

  // helper (loop) to convert ntuple of variables to bits
  template<int it = 0, typename... Ts>
  void DataFormats::attachTrack(const tuple<Ts...>& data, TTBV& ttBV, Process p) const {
    Variable v = *next(tracks_[+p].begin(), it);
    formats_[+v][+p]->attach(get<it>(data), ttBV);
    if constexpr(it + 1 != sizeof...(Ts))
      attachTrack<it + 1>(data, ttBV, p);
  }

  // construct Stub from Frame
  template<typename ...Ts>
  Stub<Ts...>::Stub(const TTDTC::Frame& frame, const DataFormats* dataFormats, Process p) :
    dataFormats_(dataFormats),
    p_(p),
    frame_(frame),
    trackId_(0)
  {
    dataFormats_->convertStub(frame.second, data_, p_);
  }

  // construct Stub from other Stub
  template<typename ...Ts>
  template<typename ...Others>
  Stub<Ts...>::Stub(const Stub<Others...>& stub, Ts... data) :
    dataFormats_(stub.dataFormats()),
    p_(++stub.p()),
    frame_(stub.frame().first, TTDTC::BV()),
    data_(data...),
    trackId_(0)
  {
  }

  // construct Stub from TTStubRef
  template<typename ...Ts>
  Stub<Ts...>::Stub(const TTStubRef& ttStubRef, const DataFormats* dataFormats, Process p, Ts... data) :
    dataFormats_(dataFormats),
    p_(p),
    frame_(ttStubRef, TTDTC::BV()),
    data_(data...),
    trackId_(0)
  {
  }

  // construct StubPP from Frame
  StubPP::StubPP(const TTDTC::Frame& frame, const DataFormats* formats) :
    Stub(frame, formats, Process::pp)
  {
    for(int sectorEta = sectorEtaMin(); sectorEta <= sectorEtaMax(); sectorEta++)
      for(int sectorPhi = 0; sectorPhi < width(Variable::sectorsPhi); sectorPhi++)
        sectors_[sectorEta * width(Variable::sectorsPhi) + sectorPhi] = sectorsPhi()[sectorPhi];
  }

  // construct StubGP from Frame
  StubGP::StubGP(const TTDTC::Frame& frame, const DataFormats* formats, int sectorPhi, int sectorEta) :
    Stub(frame, formats, Process::gp), sectorPhi_(sectorPhi), sectorEta_(sectorEta)
  {
    const Setup* setup = dataFormats_->setup();
    qOverPtBins_ = TTBV(0, setup->htNumBinsQoverPt());
    for (int qOverPt = qOverPtMin(); qOverPt <= qOverPtMax(); qOverPt++)
      qOverPtBins_.set(qOverPt + qOverPtBins_.size() / 2);
  }

  // construct StubGP from StubPP
  StubGP::StubGP(const StubPP& stub, int sectorPhi, int sectorEta) :
    Stub(stub, stub.r(), stub.phi(), stub.z(), stub.layer(), stub.qOverPtMin(), stub.qOverPtMax()),
    sectorPhi_(sectorPhi),
    sectorEta_(sectorEta)
  {
    const Setup* setup = dataFormats_->setup();
    get<1>(data_) -= (sectorPhi_ - .5) * setup->baseSector();
    get<2>(data_) -= (r() + dataFormats_->chosenRofPhi()) * setup->sectorCot(sectorEta_);
    dataFormats_->convertStub(data_, frame_.second, p_);
  }

  // construct StubHT from Frame
  StubHT::StubHT(const TTDTC::Frame& frame, const DataFormats* formats, int qOverPt) :
    Stub(frame, formats, Process::ht), qOverPt_(qOverPt)
  {
    fillTrackId();
  }

  // construct StubHT from StubGP
  StubHT::StubHT(const StubGP& stub, int phiT, int qOverPt) :
    Stub(stub, stub.r(), stub.phi(), stub.z(), stub.layer(), stub.sectorPhi(), stub.sectorEta(), phiT),
    qOverPt_(qOverPt)
  {
    get<1>(data_) += format(Variable::qOverPt).floating(this->qOverPt()) * r() - format(Variable::phiT).floating(this->phiT());
    fillTrackId();
    dataFormats_->convertStub(data_, frame_.second, p_);
  }

  void StubHT::fillTrackId() {
    TTBV ttBV(bv());
    trackId_ = ttBV.extract(width(Variable::sectorPhi) + width(Variable::sectorEta) + width(Variable::phiT));
  }

  // construct StubMHT from Frame
  StubMHT::StubMHT(const TTDTC::Frame& frame, const DataFormats* formats) :
    Stub(frame, formats, Process::mht)
  {
    fillTrackId();
  }

  // construct StubMHT from StubHT
  StubMHT::StubMHT(const StubHT& stub, int phiT, int qOverPt) :
    Stub(stub, stub.r(), stub.phi(), stub.z(), stub.layer(), stub.sectorPhi(), stub.sectorEta(), stub.phiT(), stub.qOverPt())
  {
    const Setup* setup = dataFormats_->setup();
    get<6>(data_) = this->phiT() * setup->mhtNumBinsPhiT() + phiT;
    get<7>(data_) = this->qOverPt() * setup->mhtNumBinsQoverPt() + qOverPt;
    get<1>(data_) += base(Variable::qOverPt) * (qOverPt - .5) * r() - base(Variable::phiT) * (phiT - .5);
    dataFormats_->convertStub(data_, frame_.second, p_);
    fillTrackId();
  }

  // fills track id
  void StubMHT::fillTrackId() {
    TTBV ttBV(bv());
    trackId_ = ttBV.extract(width(Variable::sectorPhi) + width(Variable::sectorEta) + width(Variable::phiT) + width(Variable::qOverPt));
  }

  // construct StubSF from Frame
  StubSF::StubSF(const TTDTC::Frame& frame, const DataFormats* formats) :
    Stub(frame, formats, Process::sf), chi2_(0.)
  {
    fillTrackId();
  }

  // construct StubSF from StubMHT
  StubSF::StubSF(const StubMHT& stub, int layer, int zT, int cot, double chi2) :
    Stub(stub, stub.r(), stub.phi(), 0., layer, stub.sectorPhi(), stub.sectorEta(), stub.phiT(), stub.qOverPt(), zT, cot), chi2_(chi2)
  {
    const double zTD = format(Variable::zT).floating(zT);
    const double cotD = format(Variable::cot).floating(cot);
    const Setup* setup = dataFormats_->setup();
    get<2>(data_) = stub.z() - (zTD + cotD * (stub.r() + dataFormats_->chosenRofPhi() - setup->chosenRofZ()));
    dataFormats_->convertStub(data_, frame_.second, p_);
    fillTrackId();
  }

  // fills track id
  void StubSF::fillTrackId() {
    TTBV ttBV(bv());
    trackId_ = ttBV.extract(width(Variable::sectorPhi) + width(Variable::sectorEta) + width(Variable::phiT) + width(Variable::qOverPt) + width(Variable::zT) + width(Variable::cot));
  }

  // construct StubKFin from Frame
  StubKFin::StubKFin(const TTDTC::Frame& frame, const DataFormats* formats, int layer) :
    Stub(frame, formats, Process::kfin),
    layer_(layer)
  {
    trackId_ = trackId();
  }

  // construct StubKFin from StubSF
  StubKFin::StubKFin(const StubSF& stub, int layer, int trackId) :
    Stub(stub, stub.r(), stub.phi(), stub.z(), trackId),
    layer_(layer)
  {
    dataFormats_->convertStub(data_, frame_.second, p_);
    trackId_ = trackId;
  }

  // construct StubKFin from TTStubRef
  StubKFin::StubKFin(const TTStubRef& ttStubRef, const DataFormats* dataFormats, double r, double phi, double z, int trackId, int layer) :
    Stub(ttStubRef, dataFormats, Process::kfin, r, phi, z, trackId),
    layer_(layer)
  {
    dataFormats_->convertStub(data_, frame_.second, p_);
    trackId_ = trackId;
  }

  // construct StubKF from Frame
  StubKF::StubKF(const TTDTC::Frame& frame, const DataFormats* formats, int layer) :
    Stub(frame, formats, Process::kf), layer_(layer) {}

  // construct StubKF from StubKFin
  StubKF::StubKF(const StubKFin& stub, double qOverPt, double phiT, double cot, double zT) :
    Stub(stub, stub.r(), stub.phi(), stub.z()),
    layer_(stub.layer())
  {
    const Setup* setup = dataFormats_->setup();
    get<1>(data_) -= phiT - this->r() * qOverPt;
    get<2>(data_) -= zT + (this->r() + setup->chosenRofPhi() - setup->chosenRofZ()) * cot;
    dataFormats_->convertStub(data_, frame_.second, p_);
  }

  // construct Track from Frame
  template<typename ...Ts>
  Track<Ts...>::Track(const FrameTrack& frame, const DataFormats* dataFormats, Process p) :
    dataFormats_(dataFormats),
    p_(p),
    frame_(frame)
  {
    dataFormats_->convertTrack(frame.second, data_, p_);
  }

  // construct Track from other Track
  template<typename ...Ts>
  template<typename ...Others>
  Track<Ts...>::Track(const Track<Others...>& track, Ts... data) :
    dataFormats_(track.dataFormats()),
    p_(++track.p()),
    frame_(track.frame().first, TTDTC::BV()),
    data_(data...) {}

  // construct Track from Stub
  template<typename ...Ts>
  template<typename ...Others>
  Track<Ts...>::Track(const Stub<Others...>& stub, const TTTrackRef& ttTrackRef, Ts... data) :
    dataFormats_(stub.dataFormats()),
    p_(++stub.p()),
    frame_(ttTrackRef, TTDTC::BV()),
    data_(data...) {}

  // construct Track from TTTrackRef
  template<typename ...Ts>
  Track<Ts...>::Track(const TTTrackRef& ttTrackRef, const DataFormats* dataFormats, Process p, Ts... data) :
    dataFormats_(dataFormats),
    p_(p),
    frame_(ttTrackRef, TTDTC::BV()),
    data_(data...) {}

  // construct TrackKFin from Frame
  TrackKFin::TrackKFin(const FrameTrack& frame, const DataFormats* dataFormats, const vector<StubKFin*>& stubs) :
    Track(frame, dataFormats, Process::kfin),
    stubs_(setup()->numLayers())
  {
    deque<StubKFin*> myStubs;
    for (StubKFin* stub : stubs)
      if (stub->trackId() == trackId())
        myStubs.push_back(stub);
    vector<int> nStubs(stubs_.size(), 0);
    for (StubKFin* stub : myStubs)
      nStubs[stub->layer()]++;
    for (int layer = 0; layer < dataFormats->setup()->numLayers(); layer++)
      stubs_[layer].reserve(nStubs[layer]);
    for (StubKFin* stub : myStubs)
      stubs_[stub->layer()].push_back(stub);
  }

  // construct TrackKFin from StubSF
  TrackKFin::TrackKFin(const StubSF& stub, const TTTrackRef& ttTrackRef, const TTBV& hitPattern, const TTBV& layerMap, const TTBV& maybePattern) :
    Track(stub, ttTrackRef, hitPattern, layerMap, maybePattern, 0., 0., 0., 0., stub.sectorPhi(), stub.sectorEta(), ttTrackRef->hitPattern()),
    stubs_(setup()->numLayers())
  {
    get<3>(data_) = format(Variable::phiT, Process::mht).floating(stub.phiT());
    get<4>(data_) = format(Variable::qOverPt, Process::mht).floating(stub.qOverPt());
    get<5>(data_) = format(Variable::zT, Process::sf).floating(stub.zT());
    get<6>(data_) = format(Variable::cot, Process::sf).floating(stub.cot());
    dataFormats_->convertTrack(data_, frame_.second, p_);
  }

  // construct TrackKFin from TTTrackRef
  TrackKFin::TrackKFin(const TTTrackRef& ttTrackRef, const DataFormats* dataFormats, const TTBV& hitPattern, const TTBV& layerMap, const TTBV& maybePattern, double phiT, double qOverPt, double zT, double cot, int sectorPhi, int sectorEta, int trackId) :
    Track(ttTrackRef, dataFormats, Process::kfin, hitPattern, layerMap, maybePattern, phiT, qOverPt, zT, cot, sectorPhi, sectorEta, trackId),
    stubs_(setup()->numLayers())
  {
    dataFormats_->convertTrack(data_, frame_.second, p_);
  }

  vector<TTStubRef> TrackKFin::ttStubRefs(const TTBV& hitPattern, const vector<int>& layerMap) const {
    vector<TTStubRef> stubs;
    stubs.reserve(hitPattern.count());
    for (int layer = 0; layer < setup()->numLayers(); layer++)
      if (hitPattern[layer])
        stubs.push_back(stubs_[layer][layerMap[layer]]->ttStubRef());
    return stubs;
  }

  // construct TrackKF from Frame
  TrackKF::TrackKF(const FrameTrack& frame, const DataFormats* dataFormats) :
    Track(frame, dataFormats, Process::kf) {}

  // construct TrackKF from TrackKfin
  TrackKF::TrackKF(const TrackKFin& track, double phiT, double qOverPt, double zT, double cot) :
    Track(track, track.phiT(), track.qOverPt(), track.zT(), track.cot(), track.sectorPhi(), track.sectorEta(), false)
  {
    get<0>(data_) += phiT;
    get<1>(data_) += qOverPt;
    get<2>(data_) += zT;
    get<3>(data_) += cot;
    get<6>(data_) = abs(qOverPt) < dataFormats_->format(Variable::qOverPt, Process::sf).base() / 2. && abs(phiT) < dataFormats_->format(Variable::phiT, Process::sf).base() / 2.;
    dataFormats_->convertTrack(data_, frame_.second, p_);
  }

  // conversion to TTTrack with given stubs
  TTTrack<Ref_Phase2TrackerDigi_> TrackKF::ttTrack(const vector<StubKF>& stubs) const {
    TTBV hitVector(0, setup()->numLayers());
    double chi2phi(0.);
    double chi2z(0.);
    vector<TTStubRef> ttStubRefs;
    const int nLayer = stubs.size();
    ttStubRefs.reserve(nLayer);
    for (const StubKF& stub : stubs) {
      hitVector.set(stub.layer());
      const TTStubRef& ttStubRef = stub.ttStubRef();
      chi2phi += pow(stub.phi(), 2) / setup()->v0(ttStubRef, this->qOverPt());
      chi2z += pow(stub.z(), 2) / setup()->v1(ttStubRef, this->qOverPt());
      ttStubRefs.push_back(ttStubRef);
    }
    static constexpr int nParPhi = 2;
    static constexpr int nParZ = 2;
    static constexpr int nPar = nParPhi + nParZ;
    const int dofPhi = nLayer - nParPhi;
    const int dofZ = nLayer - nParZ;
    chi2phi /= dofPhi;
    chi2z /= dofZ;
    const int sectorPhi = frame_.first->phiSector();
    const int sectorEta = frame_.first->etaSector();
    const double qOverPt = this->qOverPt() / setup()->invPtToDphi();
    const double phi0 = deltaPhi(this->phiT() + this->qOverPt() * dataFormats_->chosenRofPhi() + dataFormats_->format(Variable::phiT, Process::ht).range() * (sectorPhi - .5));
    const double cot = this->cot() + setup()->sectorCot(this->sectorEta());
    const double z0 = this->zT() - this->cot() * setup()->chosenRofZ();
    static constexpr double trkMVA1 = 0.;
    static constexpr double trkMVA2 = 0.;
    static constexpr double trkMVA3 = 0.;
    const int hitPattern = hitVector.val();
    const double bField = setup()->bField();
    TTTrack<Ref_Phase2TrackerDigi_> ttTrack(qOverPt, phi0, cot, z0, chi2phi, chi2z, trkMVA1, trkMVA2, trkMVA3, hitPattern, nPar, bField);
    ttTrack.setStubRefs(ttStubRefs);
    ttTrack.setPhiSector(sectorPhi);
    ttTrack.setEtaSector(sectorEta);
    return ttTrack;
  }

  // construct TrackDR from Frame
  TrackDR::TrackDR(const FrameTrack& frame, const DataFormats* dataFormats) :
    Track(frame, dataFormats, Process::dr) {}

  // construct TrackDR from TrackKF
  TrackDR::TrackDR(const TrackKF& track) :
    Track(track, 0., 0., 0., 0.)
  {
    get<0>(data_) = track.phiT() + track.qOverPt() * dataFormats_->chosenRofPhi() + dataFormats_->format(Variable::phi, Process::gp).range() * (track.sectorPhi() - .5);
    get<1>(data_) = track.qOverPt();
    get<2>(data_) = track.zT() - track.cot() * setup()->chosenRofZ();
    get<3>(data_) = track.cot() + setup()->sectorCot(track.sectorEta());
    dataFormats_->convertTrack(data_, frame_.second, p_);
  }

  // conversion to TTTrack
  TTTrack<Ref_Phase2TrackerDigi_> TrackDR::ttTrack() const {
    const double qOverPt = this->qOverPt();
    const double phi0 = this->phi0();
    const double cot = this->cot();
    const double z0 = this->z0();
    const double chi2phi = 0.;
    const double chi2z = 0;
    const double trkMVA1 = 0.;
    const double trkMVA2 = 0.;
    const double trkMVA3 = 0.;
    const int hitPattern = 0.;
    const int nPar = 4;
    const double bField = 0.;
    const int sectorPhi = frame_.first->phiSector();
    const int sectorEta = frame_.first->etaSector();
    TTTrack<Ref_Phase2TrackerDigi_> ttTrack(qOverPt, phi0, cot, z0, chi2phi, chi2z, trkMVA1, trkMVA2, trkMVA3, hitPattern, nPar, bField);
    ttTrack.setPhiSector(sectorPhi);
    ttTrack.setEtaSector(sectorEta);
    return ttTrack;
  }

  template<>
  Format<Variable::phiT, Process::ht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    range_ = 2. * M_PI / (double)(setup->numRegions() * setup->numSectorsPhi());
    base_ = range_ / (double)setup->htNumBinsPhiT();
    width_ = ceil(log2(setup->htNumBinsPhiT()));
  }

  template<>
  Format<Variable::phiT, Process::mht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::ht> ht(iConfig, setup);
    range_ = ht.range();
    base_ = ht.base() / setup->mhtNumBinsPhiT();
    width_ = ceil(log2(setup->htNumBinsPhiT() * setup->mhtNumBinsPhiT()));
  }

  template<>
  Format<Variable::qOverPt, Process::ht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const double mintPt = iConfig.getParameter<bool>("UseHybrid") ? setup->hybridMinPt() : setup->minPt();
    range_ = 2. * setup->invPtToDphi() / mintPt;
    base_ = range_ / (double)setup->htNumBinsQoverPt();
    width_ = ceil(log2(setup->htNumBinsQoverPt()));
  }

  template<>
  Format<Variable::qOverPt, Process::mht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const Format<Variable::qOverPt, Process::ht> ht(iConfig, setup);
    range_ = ht.range();
    base_ = ht.base() / setup->mhtNumBinsQoverPt();
    width_ = ceil(log2(setup->htNumBinsQoverPt() * setup->mhtNumBinsQoverPt()));
  }

  template<>
  Format<Variable::r, Process::ht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const double chosenRofPhi = iConfig.getParameter<bool>("UseHybrid") ? setup->hybridChosenRofPhi() : setup->chosenRofPhi();
    width_ = setup->widthR();
    range_ = 2. * max(abs(setup->outerRadius() - chosenRofPhi), abs(setup->innerRadius() - chosenRofPhi));
    const Format<Variable::phiT, Process::ht> phiT(iConfig, setup);
    const Format<Variable::qOverPt, Process::ht> qOverPt(iConfig, setup);
    base_ = phiT.base() / qOverPt.base();
    const int shift = ceil(log2(range_ / base_ / pow(2., width_)));
    base_ *= pow(2., shift);
  }

  template<>
  Format<Variable::phi, Process::gp>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    width_ = setup->widthPhi();
    const Format<Variable::phiT, Process::ht> phiT(iConfig, setup);
    const Format<Variable::qOverPt, Process::ht> qOverPt(iConfig, setup);
    const Format<Variable::r, Process::ht> r(iConfig, setup);
    range_ = phiT.range() + qOverPt.range() * r.base() * pow(2., r.width()) / 4.;
    const int shift = ceil(log2(range_ / phiT.base() / pow(2., width_)));
    base_ = phiT.base() * pow(2., shift);
  }

  template<>
  Format<Variable::phi, Process::dtc>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::ht> phiT(iConfig, setup);
    const Format<Variable::qOverPt, Process::ht> qOverPt(iConfig, setup);
    const Format<Variable::r, Process::ht> r(iConfig, setup);
    range_ = 2. * M_PI / (double)setup->numRegions() + qOverPt.range() * r.base() * pow(2., r.width()) / 4.;
    const Format<Variable::phi, Process::gp> gp(iConfig, setup);
    base_ = gp.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::phi, Process::ht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::ht> phiT(iConfig, setup);
    range_ = 2. * phiT.base();
    const Format<Variable::phi, Process::gp> gp(iConfig, setup);
    base_ = gp.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::phi, Process::mht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const Format<Variable::phiT, Process::mht> phiT(iConfig, setup);
    range_ = 2. * phiT.base();
    const Format<Variable::phi, Process::ht> ht(iConfig, setup);
    base_ = ht.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::z, Process::dtc>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    width_ = setup->widthZ();
    range_ = 2. * setup->halfLength();
    const Format<Variable::r, Process::ht> r(iConfig, setup);
    const int shift = ceil(log2(range_ / r.base() / pow(2., width_)));
    base_ = r.base() * pow(2., shift);
  }

  template<>
  Format<Variable::z, Process::gp>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    range_ = setup->neededRangeChiZ();
    const Format<Variable::z, Process::dtc> dtc(iConfig, setup);
    base_ = dtc.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::zT, Process::sf>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    width_ = iConfig.getParameter<ParameterSet>("SeedFilter").getParameter<int>("WidthZT");
    const Format<Variable::z, Process::dtc> z(iConfig, setup);
    range_ = -1.;
    for (int eta = 0; eta < setup->numSectorsEta(); eta++)
      range_ = max(range_, (sinh(setup->boundarieEta(eta + 1)) - sinh(setup->boundarieEta(eta))));
    range_ *= setup->chosenRofZ();
    const int shift = ceil(log2(range_ / z.base() / pow(2., width_)));
    base_ = z.base() * pow(2., shift);
  }

  template<>
  Format<Variable::cot, Process::sf>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    width_ = iConfig.getParameter<ParameterSet>("SeedFilter").getParameter<int>("WidthCot");
    const Format<Variable::zT, Process::sf> zT(iConfig, setup);
    range_ = (zT.range() + 2. * setup->beamWindowZ()) / setup->chosenRofZ();
    const int shift = ceil(log2(range_)) - width_;
    base_ = pow(2., shift);
  }

  template<>
  Format<Variable::z, Process::sf>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(true) {
    const Format<Variable::zT, Process::sf> zT(iConfig, setup);
    const Format<Variable::cot, Process::sf> cot(iConfig, setup);
    range_ = zT.base() + cot.range() * setup->chosenRofZ() + 2. * setup->maxLength();
    const Format<Variable::z, Process::dtc> dtc(iConfig, setup);
    base_ = dtc.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::layer, Process::ht>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(false) {
    range_ = setup->numLayers();
    width_ = ceil(log2(range_));
  }

  template<>
  Format<Variable::sectorEta, Process::gp>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(false) {
    range_ = setup->numSectorsEta();
    width_ = ceil(log2(range_));
  }

  template<>
  Format<Variable::sectorPhi, Process::gp>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(false) {
    range_ = setup->numSectorsPhi();
    width_ = ceil(log2(range_));
  }

  template<>
  Format<Variable::sectorsPhi, Process::dtc>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(false) {
    range_ = setup->numSectorsPhi();
    width_ = setup->numSectorsPhi();
  }

  template<>
  Format<Variable::trackId, Process::kfin>::Format(const ParameterSet& iConfig, const Setup* setup) : DataFormat(false) {
    range_ = setup->htNumBinsQoverPt() * setup->sfMaxTracks();
    width_ = ceil(log2(range_));
  }

  template<>
  Format<Variable::match, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(false) {
    width_ = 1;
    range_ = 1.;
  }

  template<>
  Format<Variable::hitPattern, Process::kfin>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(false) {
    width_ = setup->numLayers();
  }

  template<>
  Format<Variable::layerMap, Process::kfin>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(false) {
    width_ = setup->numLayers() * ceil(log2(setup->kfMaxStubsPerLayer()));
  }

  template<>
  Format<Variable::phi0, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::qOverPt, Process::ht> qOverPt(iConfig, setup);
    const Format<Variable::phiT, Process::ht> phiT(iConfig, setup);
    const double chosenRofPhi = iConfig.getParameter<bool>("UseHybrid") ? setup->hybridChosenRofPhi() : setup->chosenRofPhi();
    width_ = iConfig.getParameter<ParameterSet>("DuplicateRemoval").getParameter<int>("WidthPhi0");
    range_ = 2. * M_PI / (double)setup->numRegions() + qOverPt.range() * chosenRofPhi;
    base_ = phiT.base();
    const int shift = ceil(log2(range_ / base_ / pow(2., width_)));
    base_ *= pow(2., shift);
  }

  template<>
  Format<Variable::qOverPt, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::qOverPt, Process::ht> qOverPt(iConfig, setup);
    width_ = iConfig.getParameter<ParameterSet>("DuplicateRemoval").getParameter<int>("WidthQoverPt");
    range_ = qOverPt.range();
    base_ = qOverPt.base();
    const int shift = ceil(log2(range_ / base_ / pow(2., width_)));
    base_ *= pow(2., shift);
  }

  template<>
  Format<Variable::z0, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::zT, Process::sf> zT(iConfig, setup);
    width_ = iConfig.getParameter<ParameterSet>("DuplicateRemoval").getParameter<int>("WidthZ0");
    range_ = 2. * setup->beamWindowZ();
    base_ = zT.base();
    const int shift = ceil(log2(range_ / base_ / pow(2., width_)));
    base_ *= pow(2., shift);
  }

  template<>
  Format<Variable::cot, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::cot, Process::sf> cot(iConfig, setup);
    width_ = iConfig.getParameter<ParameterSet>("DuplicateRemoval").getParameter<int>("WidthCot");
    range_ = 2. * setup->maxCot();
    base_ = cot.base();
    const int shift = ceil(log2(range_ / base_ / pow(2., width_)));
    base_ *= pow(2., shift);
  }

  template<>
  Format<Variable::phiT, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::phi0, Process::dr> phi0(iConfig, setup);
    const Format<Variable::phiT, Process::ht> phiT(iConfig, setup);
    range_ = phiT.range();
    base_ = phi0.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::zT, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::z0, Process::dr> z0(iConfig, setup);
    const Format<Variable::zT, Process::sf> zT(iConfig, setup);
    range_ = zT.range();
    base_ = z0.base();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  Format<Variable::cot, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup) : DataFormat(true) {
    const Format<Variable::cot, Process::dr> dr(iConfig, setup);
    const Format<Variable::cot, Process::sf> sf(iConfig, setup);
    range_ = sf.range();
    base_ = dr.base();
    width_ = ceil(log2(range_ / base_));
  }

} // namespace trackerTFP