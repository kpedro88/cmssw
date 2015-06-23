#include "RecoParticleFlow/PFClusterProducer/interface/MaskedLayerManager.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"

MaskedLayerManager::MaskedLayerManager(const edm::ParameterSet& conf) {
  const std::vector<unsigned>& ee_mask  = 
    conf.getParameter<std::vector<unsigned> >("EE_layerMask");
  const std::vector<unsigned>& hef_mask = 
    conf.getParameter<std::vector<unsigned> >("HEF_layerMask");
  const std::vector<unsigned>& heb_mask = 
    conf.getParameter<std::vector<unsigned> >("HEB_layerMask");

  allowed_layers.insert(std::make_pair(HGCEE,std::map<unsigned,bool>()));
  for(unsigned i = 0; i < ee_mask.size(); ++i ) {
    allowed_layers[(int)HGCEE].insert(std::make_pair(i+1,(bool)ee_mask[i]));
  }

  allowed_layers.emplace(std::make_pair(HGCHEF,std::map<unsigned,bool>()));
  for(unsigned i = 0; i < hef_mask.size(); ++i ) {
    allowed_layers[(int)HGCHEF].insert(std::make_pair(i+1,(bool)hef_mask[i]));
  }

  allowed_layers.emplace(std::make_pair(HGCHEB,std::map<unsigned,bool>()));
  for(unsigned i = 0; i < heb_mask.size(); ++i ) {
    allowed_layers[(int)HGCHEB].insert(std::make_pair(i+1,(bool)heb_mask[i]));
  }
}

bool MaskedLayerManager::isRecHitDropped( const reco::PFRecHit& hit ) const{
  DetId detid = hit.detId();
  ForwardSubdetector subdet = (ForwardSubdetector)detid.subdetId();
  unsigned layer = std::numeric_limits<unsigned>::max();
  switch(subdet) {
  case HGCEE:
    layer = HGCEEDetId(detid).layer();
    break;
  case HGCHEF:
  case HGCHEB:
    layer = HGCHEDetId(detid).layer();
    break;
  default:
    throw cms::Exception("NotHGC")
      << "You should not being using this case for anything other than HGC.";
  }  
  auto idet = allowed_layers.find((int)subdet);
  if( idet == allowed_layers.end() ) {
    throw cms::Exception("BadDet") 
      << "Couldn't find detector in list of masks";
  }
  auto layer_info = idet->second.find(layer);
  if( layer_info == idet->second.end() ) {
    throw cms::Exception("BadLayer")
      << "Couldn't find the layer for the det id";
  }
  return !(layer_info->second);
}

std::multimap<unsigned,unsigned> 
MaskedLayerManager::buildAbsorberGanging(const ForwardSubdetector& det ) const {
  std::multimap<unsigned,unsigned> result;
  // since layers are consecutive you just progress until 
  // you find a unmasked layer
  auto idet = allowed_layers.find(det);
  if( idet == allowed_layers.end() ) {
    throw cms::Exception("BadDet") 
      << "Couldn't find detector in list of masks";
  }
  std::vector<unsigned> skipped_layers;
  for( const auto& layer : idet->second ) {
    if( layer.second ) {
      result.insert(std::make_pair(layer.first,layer.first));
      for( const unsigned skipped_layer : skipped_layers ) {
        result.insert(std::make_pair(layer.first,skipped_layer));
      }
      skipped_layers.clear();
    } else {
      skipped_layers.push_back(layer.first);
    }
  }
  return result;
}

std::unordered_map<unsigned,unsigned> 
MaskedLayerManager::buildLayerGanging(const ForwardSubdetector& det ) const {
  std::unordered_map<unsigned,unsigned> result;  
  
  // this gives us 2 -> 1,2 , 3 -> 3, 6 -> 4,5,6, etc...
  std::multimap<unsigned,unsigned> ganged_layers = buildAbsorberGanging(det);

  //get the layer mask
  const std::map<unsigned,bool>& mask = layerMask(det);

  // now turn this in to 1,2,3 -> 1, 4,5 -> 2, etc.
  for( unsigned i = 1, j = 1; i <= mask.size(); ++i ) {
    if( ganged_layers.count(i) ) {
      auto range = ganged_layers.equal_range(i);
      for( auto itr = range.first; itr != range.second; ++itr ) {
        result.insert(std::make_pair(itr->second,j));
      }
      ++j;
    }
  }
  
  return result;
}
