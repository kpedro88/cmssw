#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

template <typename Client>
class TritonGraphProducer : public SonicEDProducer<Client> {
public:
  //needed because base class has dependent scope
  using typename SonicEDProducer<Client>::Input;
  using typename SonicEDProducer<Client>::Output;
  explicit TritonGraphProducer(edm::ParameterSet const& cfg) : SonicEDProducer<Client>(cfg), ctr1_(1), ctr2_(1) {
    //for debugging
    this->setDebugName("TritonGraphProducer");
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    // in lieu of random numbers, just increment counters to generate input dimensions
    ++ctr1_;
    if (ctr1_ >= ctr1max_)
      ctr1_ = 0;
    ++ctr2_;
    if (ctr2_ >= ctr2max_)
      ctr2_ = 0;

    // fill named inputs and set shapes
    auto& input1 = iInput.at("x__0");
    input1.shape() = {ctr1_, input1.dims()[1]};
    input1.vec().resize(input1.shape()[0] * input1.shape()[1], 0.5f);

    auto& input2 = iInput.at("edgeindex__1");
    input2.shape() = {input2.dims()[0], ctr2_};
    input2.vec().resize(input2.shape()[0] * input2.shape()[1], 0.5f);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    //check the results
    const auto& output = iOutput.begin()->second;
    std::stringstream msg;
    for (int i = 0; i < output.shape()[0]; ++i) {
      msg << "output " << i << ": ";
      for (int j = 0; j < output.shape()[1]; ++i) {
        msg << output.vec()[output.shape()[1] * i + j] << " ";
      }
      msg << "\n";
    }
    edm::LogInfo(client_.debugName()) << msg.str();
  }
  ~TritonGraphProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    Client::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  using SonicEDProducer<Client>::client_;

  static constexpr int ctr1max_ = 5, ctr2max_ = 10;
  int ctr1_, ctr2_;
};

using TritonGraphProducerSync = TritonGraphProducer<TritonClientSync>;
using TritonGraphProducerAsync = TritonGraphProducer<TritonClientAsync>;
using TritonGraphProducerPseudoAsync = TritonGraphProducer<TritonClientPseudoAsync>;

DEFINE_FWK_MODULE(TritonGraphProducerSync);
DEFINE_FWK_MODULE(TritonGraphProducerAsync);
DEFINE_FWK_MODULE(TritonGraphProducerPseudoAsync);
