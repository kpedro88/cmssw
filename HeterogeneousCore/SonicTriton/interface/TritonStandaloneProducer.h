#ifndef HeterogeneousCore_SonicTriton_TritonStandaloneProducer
#define HeterogeneousCore_SonicTriton_TritonStandaloneProducer

#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"

//this example is intended only for private testing, not for production use
//in production, the dedicated Triton modules should be used:
//to take advantage of ExternalWork (minimize impact of latency),
//to ensure necessary steps occur in the correct order, and
//to minimize duplication of common interfaces and operations
template <typename... Capabilities>
class TritonStandaloneProducer : public edm::stream::EDProducer<Capabilities...> {
public:
  TritonStandaloneProducer(edm::ParameterSet const& cfg, const std::string& debugName)
      : clientPset_(cfg.getParameterSet("Client")), debugName_(debugName) {
    //not using ExternalWork, so Sync mode is enforced
    if (clientPset_.getParameter<std::string>("mode") != "Sync") {
      clientPset_.addParameter<std::string>("mode", "Sync");
    }
    edm::Service<TritonService> ts;
    ts->addModel(clientPset_.getParameter<std::string>("modelName"),
                 clientPset_.getParameter<edm::FileInPath>("modelConfigPath").fullPath());
  }

  void beginStream(edm::StreamID) override { makeClient(); }

protected:
  //helper
  void makeClient() { client_ = std::make_unique<TritonClient>(clientPset_, debugName_); }

  //members
  edm::ParameterSet clientPset_;
  std::unique_ptr<TritonClient> client_;
  std::string debugName_;
};

#endif

