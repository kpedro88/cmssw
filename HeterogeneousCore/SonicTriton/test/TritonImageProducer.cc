#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <map>

class TritonImageProducer : public TritonEDProducer<> {
public:
  explicit TritonImageProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg, "TritonImageProducer"),
        batchSize_(cfg.getParameter<unsigned>("batchSize")),
        topN_(cfg.getParameter<unsigned>("topN")) {
    //load score list
    std::string imageListFile(cfg.getParameter<edm::FileInPath>("imageList").fullPath());
    std::ifstream ifile(imageListFile);
    if (ifile.is_open()) {
      std::string line;
      while (std::getline(ifile, line)) {
        imageList_.push_back(line);
      }
    } else {
      throw cms::Exception("MissingFile") << "Could not open image list file: " << imageListFile;
    }
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    //get event-based seed for RNG
    unsigned int runNum_uint = static_cast<unsigned int>(iEvent.id().run());
    unsigned int lumiNum_uint = static_cast<unsigned int>(iEvent.id().luminosityBlock());
    unsigned int evNum_uint = static_cast<unsigned int>(iEvent.id().event());
    std::uint32_t seed = (lumiNum_uint << 10) + (runNum_uint << 20) + evNum_uint;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> randfloat(0.f,1.f);

    client_->setBatchSize(batchSize_);
    // create an npix x npix x ncol image w/ arbitrary color value
    // model only has one input, so just pick begin()
    auto& input1 = iInput.begin()->second;
    auto data1 = input1.allocate<float>();
    for (auto& vdata1: *data1){
      edm::LogInfo(debugName_) << "input before = " << triton_utils::printColl(vdata1, ",");
      for (unsigned i = 0; i < input1.sizeDims(); ++i){
        vdata1.push_back(randfloat(rng));
      }
      edm::LogInfo(debugName_) << "input after = " << triton_utils::printColl(vdata1, ",");
    }
    // convert to server format
    input1.toServer(data1);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    // check the results
    findTopN(iOutput.begin()->second);
  }
  ~TritonImageProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    desc.add<unsigned>("batchSize", 1);
    desc.add<unsigned>("topN", 5);
    desc.add<edm::FileInPath>("imageList");
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void findTopN(const TritonOutputData& scores, unsigned n = 5) const {
    const auto& tmp = scores.fromServer<float>();
    auto dim = scores.sizeDims();
    for (unsigned i0 = 0; i0 < scores.batchSize(); i0++) {
      //match score to type by index, then put in largest-first map
      std::map<float, std::string, std::greater<float>> score_map;
      for (unsigned i = 0; i < std::min((unsigned)dim, (unsigned)imageList_.size()); ++i) {
        score_map.emplace(tmp[i0][i], imageList_[i]);
      }
      //get top n
      std::stringstream msg;
      msg << "Scores for image " << i0 << ":\n";
      unsigned counter = 0;
      for (const auto& item : score_map) {
        msg << item.second << " : " << item.first << "\n";
        ++counter;
        if (counter >= topN_)
          break;
      }
      edm::LogInfo(debugName_) << msg.str();
    }
  }

  unsigned batchSize_;
  unsigned topN_;
  std::vector<std::string> imageList_;
};

DEFINE_FWK_MODULE(TritonImageProducer);
