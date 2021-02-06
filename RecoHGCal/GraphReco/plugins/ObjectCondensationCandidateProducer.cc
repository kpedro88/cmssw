/*
 * CMSSW plugin that performs inference of a graph network using RecHits and produces PF candidates.
 *
 * Authors: Gerrit Van Onsem <Gerrit.Van.Onsem@cern.ch>
 *          Marcel Rieger <marcel.rieger@cern.ch>
 *          Jan Kieseler <jan.kieseler@cern.ch>
 *          Kevin Pedro <pedrok@cern.ch>
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

class ObjectCondensationCandidateProducer : public TritonEDProducer<> {
public:
  explicit ObjectCondensationCandidateProducer(edm::ParameterSet const& cfg);

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct HGCRecHitWithPos {
    HGCRecHitWithPos(const HGCRecHit& hit_, const GlobalPoint& pos_) : hit(&hit_), pos(pos_) {}
    const HGCRecHit* hit;
    GlobalPoint pos;
  };

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  hgcal::RecHitTools recHitTools_;
  double minCandEnergy_;

  static constexpr unsigned batchSize_ = 1;
};

ObjectCondensationCandidateProducer::ObjectCondensationCandidateProducer(edm::ParameterSet const& cfg)
    : TritonEDProducer<>(cfg, "ObjectCondensationCandidateProducer"),
      caloGeometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      recHitTokens_(
          edm::vector_transform(cfg.getParameter<std::vector<edm::InputTag>>("recHitCollections"),
                                [this](const edm::InputTag& tag) { return this->consumes<HGCRecHitCollection>(tag); })),
      minCandEnergy_(cfg.getParameter<double>("minCandEnergy")) {
  produces<reco::PFCandidateCollection>();
}

void ObjectCondensationCandidateProducer::acquire(edm::Event const& iEvent,
                                                  edm::EventSetup const& iSetup,
                                                  Input& iInput) {
  recHitTools_.setGeometry(*iSetup.getHandle(caloGeometryToken_));

  //merged collections of rechits w/ positions: + and - eta
  const unsigned nWindows(2);
  std::array<std::vector<HGCRecHitWithPos>, nWindows> allRechits;
  constexpr double minEnergy(0.01);
  for (const auto& token : recHitTokens_) {
    for (const auto& rh : iEvent.get(token)) {
      if (rh.energy() <= minEnergy)
        continue;

      const auto& pos(recHitTools_.getPosition(rh.detid()));
      if (pos.eta() > 0)
        allRechits[0].emplace_back(rh, pos);
      else
        allRechits[1].emplace_back(rh, pos);
    }
  }

  //set input shapes: input_1 has dims [-1, 9]
  const unsigned nRechits = std::accumulate(
      allRechits.begin(), allRechits.end(), 0, [](unsigned sum, const auto& vec) { return sum + vec.size(); });

  auto& input1 = iInput.at("input_1");
  input1.setShape(0, nRechits);
  auto data1 = std::make_shared<TritonInput<float>>(batchSize_);
  auto& vdata1 = (*data1)[0];
  vdata1.reserve(input1.sizeShape());

  auto& input2 = iInput.at("input_2");
  input2.setShape(0, nRechits);
  auto data2 = std::make_shared<TritonInput<int64_t>>(batchSize_);
  auto& vdata2 = (*data2)[0];
  vdata2.resize(input2.sizeShape(), 0);

  //process each endcap
  for (unsigned i = 0; i < nWindows; ++i) {
    auto& arh = allRechits[i];

    //sort according to the energy
    std::sort(arh.begin(), arh.end(), [](const HGCRecHitWithPos& rh1, const HGCRecHitWithPos& rh2) {
      return rh1.hit->energy() > rh2.hit->energy();
    });

    //fill rechit features
    for (const auto& rh : arh) {
      vdata1.insert(vdata1.end(),
                    {rh.hit->energy(),
                     rh.pos.eta(),
                     0,  //ID track or not
                     rh.pos.theta(),
                     rh.pos.mag(),
                     rh.pos.x(),
                     rh.pos.y(),
                     rh.pos.z(),
                     rh.hit->time()});
    }

    //fill second input; this is the row splits: for two windows it is an array (of length the number of rechits in the window) of zeroes,
    //except the 2nd element is nrechits of first window, the 3rd element is nrechits of first+second window,
    //and the last element is 3 (the length of the non-zero-padded array)
    vdata2[i + 1] = vdata2[i] + arh.size();
  }
  vdata2.back() = nWindows + 1;

  //convert to server format
  input1.toServer(data1);
  input2.toServer(data2);
}

void ObjectCondensationCandidateProducer::produce(edm::Event& iEvent,
                                                  edm::EventSetup const& iSetup,
                                                  Output const& iOutput) {
  auto pfCandidates = std::make_unique<reco::PFCandidateCollection>();

  const auto& output1 = iOutput.at("output");
  const unsigned nFeatures = output1.shape()[1];
  //convert from server format
  const auto& vdata = output1.fromServer<float>()[0];

  //regressed x,y measured on HGCal surface (in cm)
  constexpr float zSurface = 320.;
  for (unsigned i = 0; i < output1.shape()[0]; ++i) {
    unsigned index = i * nFeatures;
    float energy = vdata[index + 10];
    //temporary lower threshold on energy of candidates
    if (energy < minCandEnergy_)
      continue;

    float x, y, z;  //t;
    //values relative to input rechit position
    x = vdata[index + 5] + vdata[index + 11];
    y = vdata[index + 6] + vdata[index + 12];
    z = vdata[index + 7] > 0 ? zSurface : -zSurface;
    //FIXME: is t also relative to t of rechit? not used for now
    //t = vdata[index+13];

    //block inspired by calcP4 method in TICL TracksterP4FromEnergySum plugin
    //below assuming (0,0,0) as vertex
    //starts from 'position (x,y,z)'
    math::XYZVector direction(x, y, z);
    direction = direction.Unit() * energy;
    reco::Candidate::LorentzVector p4(direction.X(), direction.Y(), direction.Z(), energy);

    //FIXME: put in real values for these quantities
    const auto charge = 0;
    reco::PFCandidate::ParticleType part_type = reco::PFCandidate::X;

    pfCandidates->emplace_back(charge, p4, part_type);
  }

  iEvent.put(std::move(pfCandidates));
}

void ObjectCondensationCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("recHitCollections",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<double>("minCandEnergy", 1.0);
  TritonClient::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ObjectCondensationCandidateProducer);
