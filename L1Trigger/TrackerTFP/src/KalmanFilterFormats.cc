#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iterator>
#include <algorithm>
#include <limits>

using namespace std;
using namespace edm;
using namespace trackerDTC;

namespace trackerTFP {

  void KalmanFilterFormats::endJob() {
    static constexpr int wName = 2;
    static constexpr int wWidth = 3;
    static const vector<int> maxWidths = { 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 18, 18, 10, 10, 18, 18, 18, 18, 27, 27, 18, 18, 18, 18, 18, 18 };
    for (VariableKF v = VariableKF::begin; v != VariableKF::end; v = VariableKF(+v + 1)) {
      const pair<double, double>& range = format(v).rangeActual();
      const int width = ceil(log2(max(abs(range.first), abs(range.second)) * 2. / format(v).base())) + (format(v).twos() ? 0 : 1);
      cout << setw(wName) << +v << " " << setw(wWidth) << width << " " << setw(wWidth) << maxWidths[+v] << " | " << setw(wWidth) << maxWidths[+v] - width << endl;
    }
  }

  KalmanFilterFormats::KalmanFilterFormats() :
    iConfig_(),
    dataFormats_(nullptr),
    setup_(nullptr)
  {
    formats_.reserve(+VariableKF::end);
  }

  KalmanFilterFormats::KalmanFilterFormats(const ParameterSet& iConfig, const DataFormats* dataFormats) :
    iConfig_(iConfig),
    dataFormats_(dataFormats),
    setup_(dataFormats_->setup())
  {
    formats_.reserve(+VariableKF::end);
    fillFormats();
  }

  template<VariableKF it = VariableKF::begin>
  void KalmanFilterFormats::fillFormats() {
    formats_.emplace_back(FormatKF<it>(dataFormats_, iConfig_));
    if constexpr(++it != VariableKF::end)
      fillFormats<++it>();
  }

  DataFormatKF::DataFormatKF(bool twos) :
    twos_(twos),
    width_(0),
    base_(1.),
    range_(0.),
    rangeActual_(numeric_limits<double>::max(), numeric_limits<double>::lowest()) {}

  void DataFormatKF::updateRangeActual(double d) {
    rangeActual_ = make_pair(min(rangeActual_.first, d), max(rangeActual_.second, d));
  }

  template<>
  FormatKF<VariableKF::x0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& input = dataFormats->format(Variable::qOverPt, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx0");
    base_ = pow(2, baseShift) * input.base();
  }

  template<>
  FormatKF<VariableKF::x1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& input = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx1");
    base_ = pow(2, baseShift) * input.base();
  }

  template<>
  FormatKF<VariableKF::x2>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& input = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx2");
    base_ = pow(2, baseShift) * input.base();
  }

  template<>
  FormatKF<VariableKF::x3>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& input = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftx3");
    base_ = pow(2, baseShift) * input.base();
  }

  template<>
  FormatKF<VariableKF::H00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& sf = dataFormats->format(Variable::r, Process::sf);
    base_ = sf.base();
    width_ = sf.width();
    range_ = sf.range();
  }

  template<>
  FormatKF<VariableKF::H12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const Setup* setup = dataFormats->setup();
    const DataFormat& sf = dataFormats->format(Variable::r, Process::sf);
    base_ = sf.base();
    range_ = setup->outerRadius();
    width_ = ceil(log2(range_ / base_));
  }

  template<>
  FormatKF<VariableKF::m0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& sf = dataFormats->format(Variable::phi, Process::sf);
    base_ = sf.base();
    width_ = sf.width();
    range_ = sf.range();
  }

  template<>
  FormatKF<VariableKF::m1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& sf = dataFormats->format(Variable::z, Process::sf);
    base_ = sf.base();
    width_ = sf.width();
    range_ = sf.range();
  }

  template<>
  FormatKF<VariableKF::v0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::sf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftv0");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::v1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::sf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftv1");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
  }

  template<>
  FormatKF<VariableKF::r0>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::sf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr0");
    base_ = pow(2., baseShift) * x1.base();
  }

  template<>
  FormatKF<VariableKF::r1>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::sf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftr1");
    base_ = pow(2., baseShift) * x3.base();
  }

  template<>
  FormatKF<VariableKF::S00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x0 = dataFormats->format(Variable::qOverPt, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS00");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::S01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS01");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::S12>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS12");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
  }

  template<>
  FormatKF<VariableKF::S13>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftS13");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
  }

  template<>
  FormatKF<VariableKF::K00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x0 = dataFormats->format(Variable::qOverPt, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftK00");
    base_ = pow(2., baseShift) * x0.base() / x1.base();
  }

  template<>
  FormatKF<VariableKF::K10>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftK10");
    base_ = pow(2., baseShift);
  }

  template<>
  FormatKF<VariableKF::K21>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftK21");
    base_ = pow(2., baseShift) * x2.base() / x3.base();
  }

  template<>
  FormatKF<VariableKF::K31>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftK31");
    base_ = pow(2., baseShift);
  }

  template<>
  FormatKF<VariableKF::R00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftR00");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::R11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftR11");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
  }

  template<>
  FormatKF<VariableKF::R00Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShiftR00 = iConfig.getParameter<int>("BaseShiftR00");
    const int baseShiftR00Rough = iConfig.getParameter<int>("BaseShiftR00Rough");
    base_ = pow(2., baseShiftR00 + baseShiftR00Rough) * x1.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::R11Rough>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShiftR11 = iConfig.getParameter<int>("BaseShiftR11");
    const int baseShiftR11Rough = iConfig.getParameter<int>("BaseShiftR11Rough");
    base_ = pow(2., baseShiftR11 + baseShiftR11Rough) * x3.base() * x3.base();
  }

  template<>
  FormatKF<VariableKF::invR00Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShiftInvR00 = iConfig.getParameter<int>("BaseShiftInvR00");
    const int baseShiftInvR00Approx = iConfig.getParameter<int>("BaseShiftInvR00Approx");
    base_ = pow(2., baseShiftInvR00 + baseShiftInvR00Approx) / x1.base() / x1.base();
  }

  template<>
  FormatKF<VariableKF::invR11Approx>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShiftInvR11 = iConfig.getParameter<int>("BaseShiftInvR11");
    const int baseShiftInvR11Approx = iConfig.getParameter<int>("BaseShiftInvR11Approx");
    base_ = pow(2., baseShiftInvR11 + baseShiftInvR11Approx) / x3.base() / x3.base();
  }

  template<>
  FormatKF<VariableKF::invR00Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00Cor");
    base_ = pow(2., baseShift);
  }

  template<>
  FormatKF<VariableKF::invR11Cor>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11Cor");
    base_ = pow(2., baseShift);
  }

  template<>
  FormatKF<VariableKF::invR00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR00");
    base_ = pow(2., baseShift) / x1.base() / x1.base();
  }

  template<>
  FormatKF<VariableKF::invR11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftInvR11");
    base_ = pow(2., baseShift) / x3.base() / x3.base();
  }

  template<>
  FormatKF<VariableKF::C00>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x0 = dataFormats->format(Variable::qOverPt, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC00");
    base_ = pow(2., baseShift) * x0.base() * x0.base();
  }

  template<>
  FormatKF<VariableKF::C01>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x0 = dataFormats->format(Variable::qOverPt, Process::kf);
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC01");
    base_ = pow(2., baseShift) * x0.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::C11>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x1 = dataFormats->format(Variable::phiT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC11");
    base_ = pow(2., baseShift) * x1.base() * x1.base();
  }

  template<>
  FormatKF<VariableKF::C22>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC22");
    base_ = pow(2., baseShift) * x2.base() * x2.base();
  }

  template<>
  FormatKF<VariableKF::C23>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(true) {
    const DataFormat& x2 = dataFormats->format(Variable::cot, Process::kf);
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC23");
    base_ = pow(2., baseShift) * x2.base() * x3.base();
  }

  template<>
  FormatKF<VariableKF::C33>::FormatKF(const DataFormats* dataFormats, const edm::ParameterSet& iConfig) : DataFormatKF(false) {
    const DataFormat& x3 = dataFormats->format(Variable::zT, Process::kf);
    const int baseShift = iConfig.getParameter<int>("BaseShiftC33");
    base_ = pow(2., baseShift) * x3.base() * x3.base();
  }

} // namespace trackerTFP