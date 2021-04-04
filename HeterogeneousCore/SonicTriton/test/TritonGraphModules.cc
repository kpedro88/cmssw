#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <string>
#include <vector>
#include <random>

class TritonGraphHelper {
public:
  TritonGraphHelper(edm::ParameterSet const& cfg)
      : nodeMin_(cfg.getParameter<unsigned>("nodeMin")),
        nodeMax_(cfg.getParameter<unsigned>("nodeMax")),
        edgeMin_(cfg.getParameter<unsigned>("edgeMin")),
        edgeMax_(cfg.getParameter<unsigned>("edgeMax")) {}
  void makeInput(edm::Event const& iEvent, TritonInputMap& iInput) const {
    //get event-based seed for RNG
    unsigned int runNum_uint = static_cast<unsigned int>(iEvent.id().run());
    unsigned int lumiNum_uint = static_cast<unsigned int>(iEvent.id().luminosityBlock());
    unsigned int evNum_uint = static_cast<unsigned int>(iEvent.id().event());
    std::uint32_t seed = (lumiNum_uint << 10) + (runNum_uint << 20) + evNum_uint;
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> randint1(nodeMin_, nodeMax_);
    int nnodes = randint1(rng);
    std::uniform_int_distribution<int> randint2(edgeMin_, edgeMax_);
    int nedges = randint2(rng);

    //set shapes
    auto& input1 = iInput.at("x__0");
    input1.setShape(0, nnodes);
    auto data1 = std::make_shared<TritonInput<float>>(1);
    auto& vdata1 = (*data1)[0];
    vdata1.reserve(input1.sizeShape());

    auto& input2 = iInput.at("edgeindex__1");
    input2.setShape(1, nedges);
    auto data2 = std::make_shared<TritonInput<int64_t>>(1);
    auto& vdata2 = (*data2)[0];
    vdata2.reserve(input2.sizeShape());

    //fill
    std::normal_distribution<float> randx(-10, 4);
    for (unsigned i = 0; i < input1.sizeShape(); ++i) {
      vdata1.push_back(randx(rng));
    }

    std::uniform_int_distribution<int> randedge(0, nnodes - 1);
    for (unsigned i = 0; i < input2.sizeShape(); ++i) {
      vdata2.push_back(randedge(rng));
    }

    // convert to server format
    input1.toServer(data1);
    input2.toServer(data2);
  }
  void makeOutput(const TritonOutputMap& iOutput, const std::string& debugName) const {
    //check the results
    const auto& output1 = iOutput.begin()->second;
    // convert from server format
    const auto& tmp = output1.fromServer<float>();
    std::stringstream msg;
    for (int i = 0; i < output1.shape()[0]; ++i) {
      msg << "output " << i << ": ";
      for (int j = 0; j < output1.shape()[1]; ++j) {
        msg << tmp[0][output1.shape()[1] * i + j] << " ";
      }
      msg << "\n";
    }
    edm::LogInfo(debugName) << msg.str();
  }
  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<unsigned>("nodeMin", 100);
    desc.add<unsigned>("nodeMax", 4000);
    desc.add<unsigned>("edgeMin", 8000);
    desc.add<unsigned>("edgeMax", 15000);
  }

private:
  //members
  unsigned nodeMin_, nodeMax_;
  unsigned edgeMin_, edgeMax_;
};

class TritonGraphProducer : public TritonEDProducer<> {
public:
  explicit TritonGraphProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg, "TritonGraphProducer"), helper_(cfg) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    helper_.makeInput(iEvent, iInput);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    helper_.makeOutput(iOutput, debugName_);
  }
  ~TritonGraphProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  //member
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphProducer);

#include "HeterogeneousCore/SonicTriton/interface/TritonEDFilter.h"

class TritonGraphFilter : public TritonEDFilter<> {
public:
  explicit TritonGraphFilter(edm::ParameterSet const& cfg) : TritonEDFilter<>(cfg, "TritonGraphFilter"), helper_(cfg) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    helper_.makeInput(iEvent, iInput);
  }
  bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    helper_.makeOutput(iOutput, debugName_);
    return true;
  }
  ~TritonGraphFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  //member
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphFilter);

#include "HeterogeneousCore/SonicTriton/interface/TritonOneEDAnalyzer.h"

class TritonGraphAnalyzer : public TritonOneEDAnalyzer<> {
public:
  explicit TritonGraphAnalyzer(edm::ParameterSet const& cfg)
      : TritonOneEDAnalyzer<>(cfg, "TritonGraphAnalyzer"), helper_(cfg) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    helper_.makeInput(iEvent, iInput);
  }
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    helper_.makeOutput(iOutput, debugName_);
  }
  ~TritonGraphAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  //member
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphAnalyzer);

//this example is intended only for private testing, not for production use
//in production, the dedicated Triton modules should be used, in order to take advantage of ExternalWork (minimize impact of latency)
//and also to minimize duplication of common interfaces and operations
class TritonGraphStandaloneProducer : public edm::stream::EDProducer<> {
public:
  TritonGraphStandaloneProducer(edm::ParameterSet const& cfg)
      : clientPset_(cfg.getParameterSet("Client")), debugName_("TritonGraphStandaloneProducer"), helper_(cfg) {
    //not using ExternalWork, so Sync mode is enforced
    if (clientPset_.getParameter<std::string>("mode") != "Sync") {
      clientPset_.addParameter<std::string>("mode", "Sync");
    }
    edm::Service<TritonService> ts;
    ts->addModel(clientPset_.getParameter<std::string>("modelName"),
                 clientPset_.getParameter<edm::FileInPath>("modelConfigPath").fullPath());
  }

  void beginStream(edm::StreamID) override { makeClient(); }

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) final {
    //set up input
    helper_.makeInput(iEvent, client_->input());
    //inference call
    client_->dispatch();
    //process output
    helper_.makeOutput(client_->output(), debugName_);
    //reset client data
    client_->reset();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

protected:
  //helper
  void makeClient() { client_ = std::make_unique<TritonClient>(clientPset_, debugName_); }

  //members
  edm::ParameterSet clientPset_;
  std::unique_ptr<TritonClient> client_;
  std::string debugName_;
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphStandaloneProducer);
