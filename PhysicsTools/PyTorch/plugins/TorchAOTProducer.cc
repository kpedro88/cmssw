/*
 * TorchAOTProducer
 * ----------------
 * A minimal CMSSW stream producer that demonstrates running an AOTInductor
 * (torch.export) PyTorch model as a CONST, read-only, thread-shared resource.
 *
 * The model is loaded exactly once per process into an edm::GlobalCache (the
 * cms::torch::ModelAOT wrapper around torch::inductor::AOTIModelPackageLoader).
 * Every stream/thread accesses the SAME const model pointer via globalCache()
 * and calls inference concurrently. Because AOTInductor keeps the weights in a
 * single shared, read-only constant buffer (one std::shared_ptr<ConstantMap>
 * handed to every model instance in the container), adding threads does NOT
 * duplicate the weights -- which is the whole point of the memory-scaling test
 * driven by test/scan_threads.sh.
 *
 * This is a test/demonstration plugin (PhysicsTools/PyTorch is itself a test of
 * the torch interface); it produces a single float per event so the framework
 * has real per-event work to schedule across threads.
 */
#include <atomic>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/PyTorch/interface/ModelAOT.h"
#include "PhysicsTools/PyTorch/interface/TorchInterface.h"

// The GlobalCache payload: the single, shared, const model plus the static
// problem dimensions read from the configuration. (Named namespace so the type
// has external linkage, as required for a GlobalCache template argument.)
namespace torchaot {
  struct AOTCache {
    AOTCache(const std::string& path, int64_t batch, int64_t features)
        : batchSize(batch), nFeatures(features), model(std::make_unique<cms::torch::ModelAOT>(path)) {}
    // The model is loaded ONCE here; every stream uses this same instance.
    // ModelAOT::forward() is non-const (mirrors the torch API), so we hold it by
    // unique_ptr and call through it; the cache object itself is shared as const.
    // The underlying AOTInductor container is internally thread-safe: each run()
    // checks out an idle model instance under a shared lock, and the weights are
    // never modified.
    int64_t batchSize;
    int64_t nFeatures;
    std::unique_ptr<cms::torch::ModelAOT> model;
  };
}  // namespace torchaot
using torchaot::AOTCache;

class TorchAOTProducer : public edm::stream::EDProducer<edm::GlobalCache<AOTCache>> {
public:
  explicit TorchAOTProducer(const edm::ParameterSet&, const AOTCache*);
  ~TorchAOTProducer() override = default;

  static std::unique_ptr<AOTCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const AOTCache*);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const unsigned int nInferencePerEvent_;
  const edm::EDPutTokenT<float> putToken_;
};

std::unique_ptr<AOTCache> TorchAOTProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  const std::string path = iConfig.getParameter<edm::FileInPath>("model_path").fullPath();
  const int64_t batch = iConfig.getParameter<int>("batchSize");
  const int64_t feat = iConfig.getParameter<int>("nFeatures");
  edm::LogInfo("TorchAOTProducer") << "Shared GlobalCache mode: loading the model once: " << path;
  return std::make_unique<AOTCache>(path, batch, feat);
}

void TorchAOTProducer::globalEndJob(const AOTCache*) {}

TorchAOTProducer::TorchAOTProducer(const edm::ParameterSet& iConfig, const AOTCache*)
    : nInferencePerEvent_(iConfig.getParameter<unsigned int>("nInferencePerEvent")), putToken_(produces<float>()) {}

void TorchAOTProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  // Access the single shared, const model. Many streams run this concurrently.
  const AOTCache* cache = globalCache();

  // Disable autograd for inference.
  ::torch::NoGradGuard no_grad;

  // Build an input tensor (no weight allocation here -- just the activations).
  auto input = ::torch::ones({cache->batchSize, cache->nFeatures}, cache->model->device());
  std::vector<at::Tensor> inputs{input};

  float checksum = 0.f;
  for (unsigned int i = 0; i < nInferencePerEvent_; ++i) {
    // The concurrent, logically-const inference call on the shared model.
    auto outputs = cache->model->forward(inputs);
    checksum += outputs[0].flatten()[0].item<float>();
  }

  iEvent.emplace(putToken_, checksum);
}

void TorchAOTProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("model_path");
  desc.add<int>("batchSize", 8);
  desc.add<int>("nFeatures", 10);
  desc.add<unsigned int>("nInferencePerEvent", 1);
  descriptions.add("torchAOTProducer", desc);
  descriptions.setComment(
      "Test producer running an AOTInductor model as a shared const GlobalCache resource across threads.");
}

DEFINE_FWK_MODULE(TorchAOTProducer);
