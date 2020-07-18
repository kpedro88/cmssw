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
#include <random>

template <typename Client>
class TritonGraphProducer : public SonicEDProducer<Client> {
public:
  //needed because base class has dependent scope
  using typename SonicEDProducer<Client>::Input;
  using typename SonicEDProducer<Client>::Output;
  explicit TritonGraphProducer(edm::ParameterSet const& cfg) : SonicEDProducer<Client>(cfg) {
    //for debugging
    this->setDebugName("TritonGraphProducer");
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    //get event-based seed for RNG
    unsigned int runNum_uint = static_cast <unsigned int> (iEvent.id().run());
    unsigned int lumiNum_uint = static_cast <unsigned int> (iEvent.id().luminosityBlock());
    unsigned int evNum_uint = static_cast <unsigned int> (iEvent.id().event());
    std::uint32_t seed = (lumiNum_uint<<10) + (runNum_uint<<20) + evNum_uint;
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> randint1(100, 4000);
    int nnodes = randint1(rng);
    std::uniform_int_distribution<int> randint2(8000, 15000);
    int nedges = randint2(rng);

    //set shapes
    auto& input1 = iInput.at("x__0");
    input1.shape() = {nnodes, input1.dims()[1]};
    data1_.clear();
    data1_.reserve(input1.size_shape());

    auto& input2 = iInput.at("edgeindex__1");
    input2.shape() = {input2.dims()[0], nedges};
    data2_.clear();
    data2_.reserve(input2.size_shape());

    //fill
    std::normal_distribution<float> randx(-10, 4);
    for(unsigned i = 0; i < input1.size_shape(); ++i){
      data1_.push_back(randx(rng));
    }

    std::uniform_int_distribution<int> randedge(0, nnodes);
    for(unsigned i = 0; i < input2.size_shape(); ++i){
      data2_.push_back(randedge(rng));
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

  std::vector<float> data1_;
  std::vector<int64_t> data2_;
};

using TritonGraphProducerSync = TritonGraphProducer<TritonClientSync>;
using TritonGraphProducerAsync = TritonGraphProducer<TritonClientAsync>;
using TritonGraphProducerPseudoAsync = TritonGraphProducer<TritonClientPseudoAsync>;

DEFINE_FWK_MODULE(TritonGraphProducerSync);
DEFINE_FWK_MODULE(TritonGraphProducerAsync);
DEFINE_FWK_MODULE(TritonGraphProducerPseudoAsync);
