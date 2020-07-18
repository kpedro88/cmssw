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
  explicit TritonGraphProducer(edm::ParameterSet const& cfg) : SonicEDProducer<Client>(cfg), ctr_(1) {
    //for debugging
    this->setDebugName("TritonGraphProducer");
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    // in lieu of random numbers, just increment counters to generate input dimensions
    ++ctr_;
    if (ctr_ >= ctrMax_)
      ctr_ = 1;

    // fill named inputs with proper types and set shapes
    auto& input1 = iInput.at("x__0");
    input1.shape() = {ctr_, input1.dims()[1]};
    data1_.clear();
    data1_.resize(input1.size_shape(), 0.5f);

    auto& input2 = iInput.at("edgeindex__1");
    input2.shape() = {input2.dims()[0], 2*ctr_};
    data2_.clear();
    data2_.resize(input2.size_shape(), 0);
    for (int i = 0; i < input2.shape()[0]; ++i) {
      for (int j = 0; j < input2.shape()[1]; ++j) {
        if (i != j)
          data2_[input2.shape()[1] * i + j] = 1;
      }
    }

    // convert to server format
    input1.to_server(data1_);
    input2.to_server(data2_);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    //check the results
    const auto& output1 = iOutput.begin()->second;
    std::vector<float> tmp;
    // convert from server format
    output1.from_server(tmp);
    std::stringstream msg;
    for (int i = 0; i < output1.shape()[0]; ++i) {
      msg << "output " << i << ": ";
      for (int j = 0; j < output1.shape()[1]; ++j) {
        msg << tmp[output1.shape()[1] * i + j] << " ";
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

  static constexpr int ctrMax_ = 10;
  int ctr_;
  std::vector<float> data1_;
  std::vector<int64_t> data2_;
};

using TritonGraphProducerSync = TritonGraphProducer<TritonClientSync>;
using TritonGraphProducerAsync = TritonGraphProducer<TritonClientAsync>;
using TritonGraphProducerPseudoAsync = TritonGraphProducer<TritonClientPseudoAsync>;

DEFINE_FWK_MODULE(TritonGraphProducerSync);
DEFINE_FWK_MODULE(TritonGraphProducerAsync);
DEFINE_FWK_MODULE(TritonGraphProducerPseudoAsync);
