#ifndef L1Trigger_TrackerTFP_DataFormats_h
#define L1Trigger_TrackerTFP_DataFormats_h

/*----------------------------------------------------------------------
Classes to calculate and provide dataformats used by Track Trigger emulator
enabling automated conversions from frames to stubs/tracks and vice versa
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/TrackerTFP/interface/DataFormatsRcd.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <vector>
#include <cmath>
#include <initializer_list>
#include <tuple>
#include <iostream>

namespace trackerTFP {

  // track trigger processes
  enum class Process { begin, fe = begin, dtc, pp, gp, ht, mht, sf, kfin, kf, dr, end, x };
  // track trigger variables
  enum class Variable { begin, r = begin, phi, z, layer, sectorsPhi, sectorEta, sectorPhi, phiT, qOverPt, zT, cot, trackId, match, hitPattern, layerMap, phi0, z0, end, x };
  // track trigger process order
  constexpr std::initializer_list<Process> Processes = {Process::fe, Process::dtc, Process::pp, Process::gp, Process::ht, Process::mht, Process::sf, Process::kfin, Process::kf, Process::dr};
  // conversion: Process to int
  inline constexpr int operator+(Process p) { return static_cast<int>(p); }
  // conversion: Variable to int
  inline constexpr int operator+(Variable v) { return static_cast<int>(v); }
  // increment of Process
  inline constexpr Process operator++(Process p) { return Process(+p + 1); }
  // increment of Variable
  inline constexpr Variable operator++(Variable v) { return Variable(+v + 1); }

  //Base class representing format of a variable
  class DataFormat {
  public:
    DataFormat(bool twos) : twos_(twos), width_(0), base_(1.), range_(0.) {}
    ~DataFormat() {}
    // converts int to bitvector
    TTBV ttBV(int i) const { return TTBV(i, width_, twos_); }
    // converts double to bitvector
    TTBV ttBV(double d) const { return TTBV(d, base_, width_, twos_); }
    // extracts int from bitvector, removing these bits from bitvector
    void extract(TTBV& in, int& out) const { out = in.extract(width_, twos_); }
    // extracts double from bitvector, removing these bits from bitvector
    void extract(TTBV& in, double& out) const { out = in.extract(base_, width_, twos_); }
    // extracts double from bitvector, removing these bits from bitvector
    void extract(TTBV& in, TTBV& out) const { out = in.slice(width_, twos_); }
    // attaches integer to bitvector
    void attach(const int i, TTBV& ttBV) const { ttBV += TTBV(i, width_, twos_); }
    // attaches double to bitvector
    void attach(const double d, TTBV& ttBV) const { ttBV += TTBV(d, base_, width_, twos_); }
    // attaches bitvector to bitvector
    void attach(const TTBV bv, TTBV& ttBV) const { ttBV += bv; }
    // converts int to double
    double floating(int i) const { return (i + .5) * base_; }
    // converts double to int
    int integer(double d) const { return std::floor(d / base_); }
    // converts double to int and back to double
    double digi(double d) const { return floating(integer(d)); }
    // converts binary integer value to twos complement integer value
    int toSigned(int i) const { return i - std::pow(2, width_) / 2; }
    // converts twos complement integer value to binary integer value
    int toUnsigned(int i) const { return i + std::pow(2, width_) / 2; }
    // returns false if data format would oferflow for this double value
    bool inRange(double d) const { return d >= -range_ / 2. && d < range_ / 2.; }
    // returns false if data format would oferflow for this int value
    bool inRange(int i) const { return inRange(floating(i)); }
    // true if twos'complement or false if binary representation is chosen
    bool twos() const { return twos_; }
    // number of used bits
    int width() const { return width_; }
    // precision
    double base() const { return base_; }
    // covered range
    double range() const { return range_; }
  protected:
    // true if twos'complement or false if binary representation is chosen
    bool twos_;
    // number of used bits
    int width_;
    // precision
    double base_;
    // covered range
    double range_;
  };

  // class representing format of a specific variable
  template<Variable v, Process p>
  class Format : public DataFormat {
  public:
    Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
    ~Format() {}
  };

  template<> Format<Variable::phiT, Process::ht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phiT, Process::mht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::qOverPt, Process::ht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::qOverPt, Process::mht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::r, Process::ht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phi, Process::ht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phi, Process::mht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phi, Process::gp>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phi, Process::dtc>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::z, Process::dtc>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::z, Process::gp>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::z, Process::sf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::zT, Process::sf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::cot, Process::sf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::layer, Process::ht>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::sectorEta, Process::gp>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::sectorPhi, Process::gp>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::sectorsPhi, Process::gp>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::trackId, Process::kfin>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::match, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::hitPattern, Process::kfin>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::layerMap, Process::kfin>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phi0, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::qOverPt, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::z0, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::cot, Process::dr>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::phiT, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::zT, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
  template<> Format<Variable::cot, Process::kf>::Format(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);

  /*! \class  trackerTFP::DataFormats
   *  \brief  Class to calculate and provide dataformats used by Track Trigger emulator
   *  \author Thomas Schuh
   *  \date   2020, June
   */
  class DataFormats {
  private:
    // variable flavour mapping, decalers which processing step is using which variable definition
    static constexpr std::array<std::array<Process, +Process::end>, +Variable::end> config_ = {{
    //  Process::fe  Process::dtc  Process::pp   Process::gp  Process::ht  Process::mht  Process::sf   Process::kfin  Process::kf   Process::dr
      {{Process::x,  Process::ht,  Process::ht,  Process::ht, Process::ht, Process::ht,  Process::ht,  Process::ht,   Process::ht,  Process::x }}, // Variable::r
      {{Process::x,  Process::dtc, Process::dtc, Process::gp, Process::ht, Process::mht, Process::mht, Process::mht,  Process::mht, Process::x }}, // Variable::phi
      {{Process::x,  Process::dtc, Process::dtc, Process::gp, Process::gp, Process::gp,  Process::sf,  Process::sf,   Process::sf,  Process::x }}, // Variable::z
      {{Process::x,  Process::ht,  Process::ht,  Process::ht, Process::ht, Process::ht,  Process::ht,  Process::x,    Process::x,   Process::x }}, // Variable::layer
      {{Process::x,  Process::dtc, Process::dtc, Process::x,  Process::x,  Process::x,   Process::x,   Process::x,    Process::x,   Process::x }}, // Variable::sectorsPhi
      {{Process::x,  Process::gp,  Process::gp,  Process::gp, Process::gp, Process::gp,  Process::gp,  Process::gp,   Process::gp,  Process::x }}, // Variable::sectorEta
      {{Process::x,  Process::x,   Process::x,   Process::gp, Process::gp, Process::gp,  Process::gp,  Process::gp,   Process::gp,  Process::x }}, // Variable::sectorPhi
      {{Process::x,  Process::ht,  Process::ht,  Process::ht, Process::ht, Process::mht, Process::mht, Process::mht,  Process::kf,  Process::x }}, // Variable::phiT
      {{Process::x,  Process::ht,  Process::ht,  Process::ht, Process::ht, Process::mht, Process::mht, Process::mht,  Process::dr,  Process::dr}}, // Variable::qOverPt
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::sf,  Process::sf,   Process::kf,  Process::x }}, // Variable::zT
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::sf,  Process::sf,   Process::kf,  Process::dr}}, // Variable::cot
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::x,   Process::kfin, Process::x,   Process::x }}, // Variable::trackId
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::x,   Process::x,    Process::kf,  Process::x }}, // Variable::match
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::x,   Process::kfin, Process::x,   Process::x }}, // Variable::hitPattern
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::x,   Process::kfin, Process::x,   Process::x }}, // Variable::layerMap
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::x,   Process::x,    Process::x,   Process::dr}}, // Variable::phi0
      {{Process::x,  Process::x,   Process::x,   Process::x,  Process::x,  Process::x,   Process::x,   Process::x,    Process::x,   Process::dr}}  // Variable::z0
    }};
    // stub word assembly, shows which stub variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> stubs_ = {{
      {},                                                                                                                                                                   // Process::fe
      {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::sectorsPhi, Variable::sectorEta, Variable::sectorEta, Variable::qOverPt, Variable::qOverPt},     // Process::dtc
      {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::sectorsPhi, Variable::sectorEta, Variable::sectorEta, Variable::qOverPt, Variable::qOverPt},     // Process::pp
      {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::qOverPt, Variable::qOverPt},                                                                     // Process::gp
      {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::sectorPhi, Variable::sectorEta, Variable::phiT},                                                 // Process::ht
      {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::sectorPhi, Variable::sectorEta, Variable::phiT, Variable::qOverPt},                              // Process::mht
      {Variable::r, Variable::phi, Variable::z, Variable::layer, Variable::sectorPhi, Variable::sectorEta, Variable::phiT, Variable::qOverPt, Variable::zT, Variable::cot}, // Process::sf
      {Variable::r, Variable::phi, Variable::z, Variable::trackId},                                                                                                         // Process::kfin
      {Variable::r, Variable::phi, Variable::z},                                                                                                                            // Process::kf
      {}                                                                                                                                                                    // Process::dr
    }};
    // track word assembly, shows which track variables are used by each process
    static constexpr std::array<std::initializer_list<Variable>, +Process::end> tracks_ = {{
      {},                                                                                                                                                                                            // Process::fe
      {},                                                                                                                                                                                            // Process::dtc
      {},                                                                                                                                                                                            // Process::pp
      {},                                                                                                                                                                                            // Process::gp
      {},                                                                                                                                                                                            // Process::ht
      {},                                                                                                                                                                                            // Process::mht
      {},                                                                                                                                                                                            // Process::sf
      {Variable::hitPattern, Variable::layerMap, Variable::hitPattern, Variable::phiT, Variable::qOverPt, Variable::zT, Variable::cot, Variable::sectorPhi, Variable::sectorEta, Variable::trackId}, // Process::kfin
      {Variable::phiT, Variable::qOverPt, Variable::zT, Variable::cot, Variable::sectorPhi, Variable::sectorEta, Variable::match},                                                                   // Process::kf
      {Variable::phi0, Variable::qOverPt, Variable::z0, Variable::cot}                                                                                                                               // Process::dr
    }};
  public:
    DataFormats();
    DataFormats(const edm::ParameterSet& iConfig, const trackerDTC::Setup* setup);
    ~DataFormats(){}
    // converts bits to ntuple of variables
    template<typename ...Ts>
    void convertStub(const TTDTC::BV& bv, std::tuple<Ts...>& data, Process p) const;
    // converts ntuple of variables to bits
    template<typename... Ts>
    void convertStub(const std::tuple<Ts...>& data, TTDTC::BV& bv, Process p) const;
    // converts bits to ntuple of variables
    template<typename ...Ts>
    void convertTrack(const TTDTC::BV& bv, std::tuple<Ts...>& data, Process p) const;
    // converts ntuple of variables to bits
    template<typename... Ts>
    void convertTrack(const std::tuple<Ts...>& data, TTDTC::BV& bv, Process p) const;
    // access to run-time constants
    const trackerDTC::Setup* setup() const { return setup_; }
    // number of bits being used for specific variable flavour
    int width(Variable v, Process p) const { return formats_[+v][+p]->width(); }
    // precision being used for specific variable flavour
    double base(Variable v, Process p) const { return formats_[+v][+p]->base(); }
    // number of unused frame bits for a given Stub flavour
    int numUnusedBitsStubs(Process p) const { return numUnusedBitsStubs_[+p]; }
    // number of unused frame bits for a given Track flavour
    int numUnusedBitsTracks(Process p) const { return numUnusedBitsTracks_[+p]; }
    // number of channels of a given process on a TFP
    int numChannel(Process p) const { return numChannel_[+p]; }
    // number of channels of a given process for whole system
    int numStreams(Process p) const { return numStreams_[+p]; }
    // access to spedific format
    const DataFormat& format(Variable v, Process p) const { return *formats_[+v][+p]; }
    // critical radius defining region overlap shape in cm
    double chosenRofPhi() const { return hybrid() ? setup_->hybridChosenRofPhi() : setup_->chosenRofPhi(); }
  private:
    // bool indicating if hybrid or tmtt being used
    bool hybrid() const { return iConfig_.getParameter<bool>("UseHybrid"); }
    // number of unique data formats
    int numDataFormats_;
    // method to count number of unique data formats
    template<Variable v = Variable::begin, Process p = Process::begin>
    void countFormats();
    // constructs data formats of all unique used variables and flavours
    template<Variable v = Variable::begin, Process p = Process::begin>
    void fillDataFormats();
    // helper (loop) data formats of all unique used variables and flavours
    template<Variable v, Process p, Process it = Process::begin>
    void fillFormats();
    // helper (loop) to convert bits to ntuple of variables
    template<int it = 0, typename ...Ts>
    void extractStub(TTBV& ttBV, std::tuple<Ts...>& data, Process p) const;
    // helper (loop) to convert bits to ntuple of variables
    template<int it = 0, typename ...Ts>
    void extractTrack(TTBV& ttBV, std::tuple<Ts...>& data, Process p) const;
    // helper (loop) to convert ntuple of variables to bits
    template<int it = 0, typename... Ts>
    void attachStub(const std::tuple<Ts...>& data, TTBV& ttBV, Process p) const;
    // helper (loop) to convert ntuple of variables to bits
    template<int it = 0, typename... Ts>
    void attachTrack(const std::tuple<Ts...>& data, TTBV& ttBV, Process p) const;
    // configuration during construction
    edm::ParameterSet iConfig_;
    // stored run-time constants
    const trackerDTC::Setup* setup_;
    // collection of unique formats
    std::vector<DataFormat> dataFormats_;
    // variable flavour mapping
    std::vector<std::vector<DataFormat*>> formats_;
    // number of unused frame bits for a all Stub flavours
    std::vector<int> numUnusedBitsStubs_;
    // number of unused frame bits for a all Track flavours
    std::vector<int> numUnusedBitsTracks_;
    // number of channels of all processes on a TFP
    std::vector<int> numChannel_;
    // number of channels of all processes for whole system
    std::vector<int> numStreams_;
  };

  // base class to represent stubs
  template<typename ...Ts>
  class Stub {
  public:
    // construct Stub from Frame
    Stub(const TTDTC::Frame& frame, const DataFormats* dataFormats, Process p);
    template<typename ...Others>
    // construct Stub from other Stub
    Stub(const Stub<Others...>& stub, Ts... data);
    // construct Stub from TTStubRef
    Stub(const TTStubRef& ttStubRef, const DataFormats* dataFormats, Process p, Ts... data);
    Stub() {}
    ~Stub() {}
    // true if frame valid, false if gap in data stream
    explicit operator bool() const { return frame_.first.isNonnull(); }
    // access to DataFormats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // stub flavour
    Process p() const { return p_; }
    // acess to frame
    const TTDTC::Frame& frame() const { return frame_; }
    // access to TTStubRef
    const TTStubRef& ttStubRef() const { return frame_.first; }
    // access to bitvector
    const TTDTC::BV& bv() const { return frame_.second; }
    // id of collection this stub belongs to
    int trackId() const { return trackId_; }
  protected:
    // number of used bits for given variable
    int width(Variable v) const { return dataFormats_->width(v, p_); }
    // precision of given variable
    double base(Variable v) const { return dataFormats_->base(v, p_); }
    // format of given variable
    const DataFormat& format(Variable v) const { return dataFormats_->format(v, p_); }
    // all dataformats
    const DataFormats* dataFormats_;
    // stub flavour
    Process p_;
    // underlying TTStubRef and bitvector
    TTDTC::Frame frame_;
    // ntuple of variables this stub is assemled of
    std::tuple<Ts...> data_;
    // id of collection this stub belongs to
    int trackId_;
  };

  // class to represent stubs generated by process patch pannel
  class StubPP : public Stub<double, double, double, int, TTBV, int, int, int, int> {
  public:
    // construct StubPP from Frame
    StubPP(const TTDTC::Frame& frame, const DataFormats* dataFormats);
    ~StubPP(){}
    // true if stub belongs to given sector
    bool inSector(int sector) const { return sectors_[sector]; }
    // sectors this stub belongs to
    std::vector<int> sectors() const { return sectors_.ids(); }
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi wrt processing nonant centre
    double phi() const { return std::get<1>(data_); }
    // stub z
    double z() const { return std::get<2>(data_); }
    // reduced layer id
    int layer() const { return std::get<3>(data_); }
    // phi sector map to which this stub belongs to
    TTBV sectorsPhi() const { return std::get<4>(data_); }
    // first eta sector this stub belongs to
    int sectorEtaMin() const { return std::get<5>(data_); }
    // last eta sector this stub belongs to
    int sectorEtaMax() const { return std::get<6>(data_); }
    // first qOverPt bin this stub belongs to
    int qOverPtMin() const { return std::get<7>(data_); }
    // last qOverPt bin this stub belongs to
    int qOverPtMax() const { return std::get<8>(data_); }
  private:
    // sectors this stub belongs to
    TTBV sectors_;
  };

  // class to represent stubs generated by process geometric processor
  class StubGP : public Stub<double, double, double, int, int, int> {
  public:
    // construct StubGP from Frame
    StubGP(const TTDTC::Frame& frame, const DataFormats* dataFormats, int sectorPhi, int sectorEta);
    // construct StubGO from StubPP
    StubGP(const StubPP& stub, int sectorPhi, int sectorEta);
    ~StubGP(){}
    // true if stub belongs to given qOverPt bin
    bool inQoverPtBin(int qOverPtBin) const { return qOverPtBins_[qOverPtBin]; }
    // qOverPt bins  this stub belongs to
    std::vector<int> qOverPtBins() const { return qOverPtBins_.ids(); }
    // stub phi sector
    int sectorPhi() const { return sectorPhi_; }
    // stub eta sector
    int sectorEta() const { return sectorEta_; }
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi wrt phi sector centre
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt eta sector
    double z() const { return std::get<2>(data_); }
    // reduced layer id
    int layer() const { return std::get<3>(data_); }
    // first qOverPt bin this stub belongs to
    int qOverPtMin() const { return std::get<4>(data_); }
    // last qOverPt bin this stub belongs to
    int qOverPtMax() const { return std::get<5>(data_); }
  private:
    // qOverPt bins this stub belongs to
    TTBV qOverPtBins_;
    // stub phi sector
    int sectorPhi_;
    // stub eta sector
    int sectorEta_;
  };

  // class to represent stubs generated by process hough transform
  class StubHT : public Stub<double, double, double, int, int, int, int> {
  public:
    // construct StubHT from Frame
    StubHT(const TTDTC::Frame& frame, const DataFormats* dataFormats, int qOverPt);
    // construct StubHT from StubGP
    StubHT(const StubGP& stub, int qOverPt, int phiT);
    ~StubHT(){}
    // stub qOver pt
    int qOverPt() const { return qOverPt_; }
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); };
    // stub phi residual wrt track parameter
    double phi() const { return std::get<1>(data_); };
    // stub z residual wrt eta sector
    double z() const { return std::get<2>(data_); };
    // reduced layer id
    int layer() const { return std::get<3>(data_); };
    // phi sector
    int sectorPhi() const { return std::get<4>(data_); };
    // eta sector
    int sectorEta() const { return std::get<5>(data_); };
    // stub phi at radius chosenRofPhi wrt phi sector centre
    int phiT() const { return std::get<6>(data_); };
  private:
    // fills track id
    void fillTrackId();
    // stub qOver pt
    int qOverPt_;
  };

  // class to represent stubs generated by process mini hough transform
  class StubMHT : public Stub<double, double, double, int, int, int, int, int> {
  public:
    // construct StubMHT from Frame
    StubMHT(const TTDTC::Frame& frame, const DataFormats* dataFormats);
    // construct StubMHT from StubHT
    StubMHT(const StubHT& stub, int phiT, int qOverPt);
    ~StubMHT(){}
    // stub radius wrt choenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual wrt finer track parameter
    double phi() const { return std::get<1>(data_); }
    // stub z rsidual wrt eta sector
    double z() const { return std::get<2>(data_); }
    // reduced layer id
    int layer() const { return std::get<3>(data_); }
    // phi sector
    int sectorPhi() const { return std::get<4>(data_); }
    // eta sector
    int sectorEta() const { return std::get<5>(data_); }
    // stub phi at radius chosenRofPhi wrt phi sector centre
    int phiT() const { return std::get<6>(data_); }
    // stub qOverpt
    int qOverPt() const { return std::get<7>(data_); }
  private:
    // fills track id
    void fillTrackId();
  };

  // class to represent stubs generated by process seed filter
  class StubSF : public Stub<double, double, double, int, int, int, int, int, int, int> {
  public:
    // construct StubSF from Frame
    StubSF(const TTDTC::Frame& frame, const DataFormats* dataFormats);
    // construct StubSF from StubMHT
    StubSF(const StubMHT& stub, int layer, int zT, int cot, double chi2);
    ~StubSF(){}
    // chi2 contribution of this stub
    double chi2() const {return chi2_;}
    // stub radius wrt chonseRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phiresiudal wrt finer track parameter
    double phi() const { return std::get<1>(data_); }
    // stub z residual to track parameter
    double z() const { return std::get<2>(data_); }
    // reduced layer id
    int layer() const { return std::get<3>(data_); }
    // phi sector
    int sectorPhi() const { return std::get<4>(data_); }
    // eta sector
    int sectorEta() const { return std::get<5>(data_); }
    // stub phi at radius chosenRofPhi wrt phi sector centre
    int phiT() const { return std::get<6>(data_); }
    // stub qOver pt
    int qOverPt() const { return std::get<7>(data_); }
    // stub z at radius chosenRofZ wrt eta sector centre
    int zT() const { return std::get<8>(data_); }
    // stub cotTheta wrt eta sector cotTheta
    int cot() const { return std::get<9>(data_); }
  private:
    // fills track id
    void fillTrackId();
    // chi2 contribution of this stub
    double chi2_;
  };

  // class to represent stubs generated by process kfin
  class StubKFin : public Stub<double, double, double, int> {
  public:
    // construct StubKFin from Frame
    StubKFin(const TTDTC::Frame& frame, const DataFormats* dataFormats, int layer);
    // construct StubKFin from StubSF
    StubKFin(const StubSF& stub, int layer, int trackId);
    // construct StubKFin from TTStubRef
    StubKFin(const TTStubRef& ttStubRef, const DataFormats* dataFormats, double r, double phi, double z, int trackId, int layer);
    ~StubKFin(){}
    // kf layer id
    int layer() const { return layer_; }
    // stub radius wrt chosenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual wrt finer track parameter
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt track parameter
    double z() const { return std::get<2>(data_); }
    // track id this stub belongs to
    int trackId() const { return std::get<3>(data_); }
  private:
    // kf layer id
    int layer_;
  };

  // class to represent stubs generated by process kalman filter
  class StubKF : public Stub<double, double, double> {
  public:
    // construct StubKF from Frame
    StubKF(const TTDTC::Frame& frame, const DataFormats* dataFormats, int layer);
    // construct StubKF from StubKFin
    StubKF(const StubKFin& stub, double qOverPt, double phiT, double cot, double zT);
    ~StubKF(){}
    // kf layer id
    int layer() const { return layer_; }
    // stub radius wrt choenRofPhi
    double r() const { return std::get<0>(data_); }
    // stub phi residual wrt fitted parameter
    double phi() const { return std::get<1>(data_); }
    // stub z residual wrt fitted parameter
    double z() const { return std::get<2>(data_); }
  private:
    // kf layer id
    int layer_;
  };

  // base class to represent tracks
  template<typename ...Ts>
  class Track {
  public:
    // construct Track from Frame
    Track(const FrameTrack& frame, const DataFormats* dataFormats, Process p);
    // construct Track from other Track
    template<typename ...Others>
    Track(const Track<Others...>& track, Ts... data);
    // construct Track from Stub
    template<typename ...Others>
    Track(const Stub<Others...>& stub, const TTTrackRef& ttTrackRef, Ts... data);
    // construct Track from TTTrackRef
    Track(const TTTrackRef& ttTrackRef, const DataFormats* dataFormats, Process p, Ts... data);
    ~Track() {}
    // true if frame valid, false if gap in data stream
    explicit operator bool() const { return frame_.first.isNonnull(); }
    // access to DataFormats
    const DataFormats* dataFormats() const { return dataFormats_; }
    // track flavour
    Process p() const { return p_; }
    // acces to frame
    const FrameTrack& frame() const { return frame_; }
    // access to TTTrackRef
    const TTTrackRef& ttTrackRef() const { return frame_.first; }
    // access to bitvector
    const TTDTC::BV& bv() const { return frame_.second; }
    // access to ntuple of variables this track is assemled of
    const std::tuple<Ts...>& data() const { return data_; }
  protected:
    //number of bits uesd of given variable
    int width(Variable v) const { return dataFormats_->width(v, p_); }
    // precision of given variable
    double base(Variable v) const { return dataFormats_->base(v, p_); }
    // access to run-time constants
    const trackerDTC::Setup* setup() const { return dataFormats_->setup(); }
    // format of given variable
    const DataFormat& format(Variable v) const { return dataFormats_->format(v, p_); }
    // format of given variable and process
    const DataFormat& format(Variable v, Process p) const { return dataFormats_->format(v, p); }
    // all data formats
    const DataFormats* dataFormats_;
    // track flavour
    Process p_;
    // underlying TTTrackRef and bitvector
    FrameTrack frame_;
    // ntuple of variables this track is assemled of
    std::tuple<Ts...> data_;
  };

  // class to represent tracks generated by process kfin
  class TrackKFin : public Track<TTBV, TTBV, TTBV, double, double, double, double, int, int, int> {
  public:
    // construct TrackKFin from Frame
    TrackKFin(const FrameTrack& frame, const DataFormats* dataFormats, const std::vector<StubKFin*>& stubs);
    // construct TrackKFin from StubKFin
    TrackKFin(const StubSF& stub, const TTTrackRef& ttTrackRef, const TTBV& hitPattern, const TTBV& layerMap, const TTBV& maybePattern);
    // construct TrackKFin from TTTrackRef
    TrackKFin(const TTTrackRef& ttTrackRef, const DataFormats* dataFormats, const TTBV& hitPattern, const TTBV& layerMap, const TTBV& maybePattern, double phiT, double qOverPt, double zT, double cot, int sectorPhi, int sectorEta, int trackId);
    ~TrackKFin(){}
    // hitpattern of track
    const TTBV& hitPattern() const { return std::get<0>(data_); }
    // number of stub per layer
    std::vector<int> layerMap() const { return setup()->layerMap(hitPattern(), std::get<1>(data_)); }
    // pattern of layers which are only maybe crossed by found candidate
    const TTBV& maybePattern() const { return std::get<2>(data_); }
    // track phi at radius chosenRofPhi wrt phi sector centre
    double phiT() const { return std::get<3>(data_); }
    // track qOverPt
    double qOverPt() const { return std::get<4>(data_); }
    // track z at radius chosenRofZ wrt eta sector centre
    double zT() const { return std::get<5>(data_); }
    // track cotTheta wrt seta sector cotTheta
    double cot() const { return std::get<6>(data_); }
    // phi sector
    int sectorPhi() const { return std::get<7>(data_); }
    // eta sector
    int sectorEta() const { return std::get<8>(data_); }
    // track id
    int trackId() const { return std::get<9>(data_); }
    // true if given layer has a hit
    bool hitPattern(int index) const { return hitPattern()[index]; }
    // true if given layer has a hit or is a maybe layer
    bool maybePattern(int index) const { return hitPattern()[index] || maybePattern()[index]; }
    // stubs on a given layer
    std::vector<StubKFin*> layerStubs(int layer) const { return stubs_[layer]; }
    // firts stub on a given layer
    StubKFin* layerStub(int layer) const { return stubs_[layer].front(); }
    // selection of ttStubRefs for given hit ids on given layers
    std::vector<TTStubRef> ttStubRefs(const TTBV& hitPattern, const std::vector<int>& layerMap) const;
    // stubs organized in layer
    const std::vector<std::vector<StubKFin*>>& stubs() const { return stubs_; }
    // global cotTheta
    double cotGlobal() const { return cot() + setup()->sectorCot(sectorEta()); }
  private:
    // stubs organized in layer
    std::vector<std::vector<StubKFin*>> stubs_;
  };

  // class to represent tracks generated by process kalman filter
  class TrackKF : public Track<double, double, double, double, int, int, int> {
  public:
    // construct TrackKF from Frame
    TrackKF(const FrameTrack& frame, const DataFormats* dataFormats);
    // construct TrackKF from TrackKFKFin
    TrackKF(const TrackKFin& track, double phiT, double qOverPt, double zT, double cot);
    ~TrackKF(){}
    // track phi at radius chosenRofPhi wrt phi sector centre
    double phiT() const { return std::get<0>(data_); }
    // track qOver pt
    double qOverPt() const { return std::get<1>(data_); }
    // track z at radius chosenRofZ wrt eta sector centre
    double zT() const { return std::get<2>(data_); }
    // track cotTheta wrt eta sector cotTheta
    double cot() const { return std::get<3>(data_); }
    // phi sector
    int sectorPhi() const { return std::get<4>(data_); }
    // eta sector
    int sectorEta() const { return std::get<5>(data_); }
    // true if kf prameter consistent with mht parameter
    bool match() const { return std::get<6>(data_); }
    // conversion to TTTrack with given stubs
    TTTrack<Ref_Phase2TrackerDigi_> ttTrack(const std::vector<StubKF>& stubs) const;
  private:
  };

  // class to represent tracks generated by process duplicate removal
  class TrackDR : public Track<double, double, double, double> {
  public:
    // construct TrackDR from Frame
    TrackDR(const FrameTrack& frame, const DataFormats* dataFormats);
    // construct TrackDR from TrackKF
    TrackDR(const TrackKF& track);
    ~TrackDR(){}
    // track phi at radius 0 wrt processing nonant centre
    double phi0() const { return std::get<0>(data_); }
    // track qOverPt
    double qOverPt() const { return std::get<1>(data_); }
    // track z at radius 0
    double z0() const { return std::get<2>(data_); }
    // track cotThea
    double cot() const { return std::get<3>(data_); }
    // conversion to TTTrack
    TTTrack<Ref_Phase2TrackerDigi_> ttTrack() const;
  private:
  };

} // namespace trackerTFP

EVENTSETUP_DATA_DEFAULT_RECORD(trackerTFP::DataFormats, trackerTFP::DataFormatsRcd);

#endif