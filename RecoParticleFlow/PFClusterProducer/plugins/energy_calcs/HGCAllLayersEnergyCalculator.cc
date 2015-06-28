#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "RecoParticleFlow/PFClusterProducer/interface/MaskedLayerManager.h"

#include "TFile.h"
#include "TGraphErrors.h"

class HGCAllLayersEnergyCalibrator : public PFClusterEnergyCorrectorBase {
 public:
  HGCAllLayersEnergyCalibrator(const edm::ParameterSet& conf) :
    PFClusterEnergyCorrectorBase(conf),
    _mipValueInGeV_ee(conf.getParameter<double>("MipValueInGeV_ee")),
    _mipValueInGeV_hef(conf.getParameter<double>("MipValueInGeV_hef")),    
    _mipValueInGeV_heb(conf.getParameter<double>("MipValueInGeV_heb")),
    // EM calibration
    _coef_a(conf.getParameter<double>("effMip_to_InverseGeV_a")),
    _coef_b(conf.getParameter<double>("effMip_to_InverseGeV_b")),
    _coef_c(conf.getParameter<double>("effMip_to_InverseGeV_c")),
    _coef_d(conf.exists("effMip_to_InverseGeV_d") ? conf.getParameter<double>("effMip_to_InverseGeV_d") : 0),
    _coef_e(conf.exists("effMip_to_InverseGeV_e") ? conf.getParameter<double>("effMip_to_InverseGeV_e") : 1.0),
    // had calibration
    _ee_had_emscale_slope(conf.getParameter<double>("ee_had_emscale_slope")), 
    _ee_had_emscale_offset(conf.getParameter<double>("ee_had_emscale_offset")),
    _hef_had_emscale_slope(conf.getParameter<double>("hef_had_emscale_slope")), 
    _hef_had_emscale_offset(conf.getParameter<double>("hef_had_emscale_offset")),
    _heb_had_emscale_slope(conf.getParameter<double>("heb_had_emscale_slope")), 
    _heb_had_emscale_offset(conf.getParameter<double>("heb_had_emscale_offset")),
    _he_had_correction(conf.getParameter<double>("he_had_correction")), 
    _heb_had_correction(conf.getParameter<double>("heb_had_correction")),
    _pion_energy_slope(conf.getParameter<double>("pion_energy_slope")), 
    _pion_energy_offset(conf.getParameter<double>("pion_energy_offset")),
    _had_residual(conf.getParameter<std::vector<double> >("had_residual")),
    _isEMCalibration(conf.getParameter<bool>("isEMCalibration")),
    _weights_ee(conf.getParameter<std::vector<double> >("weights_ee")),
    _weights_hef(conf.getParameter<std::vector<double> >("weights_hef")),
    _weights_heb(conf.getParameter<std::vector<double> >("weights_heb")),
    _hgcOverburdenParam(nullptr),
    _hgcLambdaOverburdenParam(nullptr)
      { 
	if(conf.exists("hgcOverburdenParamFile"))
	  {
	    edm::FileInPath fp = conf.getParameter<edm::FileInPath>("hgcOverburdenParamFile");	    
	    TFile *fIn=TFile::Open(fp.fullPath().c_str());
	    if(fIn)
	      {
		_hgcOverburdenParam=(const TGraphErrors *) fIn->Get("x0Overburden");
		_hgcLambdaOverburdenParam = (const TGraphErrors *) fIn->Get("lambdaOverburden");
		fIn->Close();
	      }
	  }
        // setup the layer masking information
        const edm::ParameterSet& masking_info = conf.getParameter<edm::ParameterSet>("masking_info");
        mask_.reset(new MaskedLayerManager(masking_info));
        ee_ganging = mask_->buildAbsorberGanging(HGCEE);
        fh_ganging = mask_->buildAbsorberGanging(HGCHEF);
        bh_ganging = mask_->buildAbsorberGanging(HGCHEB);
      }
  HGCAllLayersEnergyCalibrator(const HGCAllLayersEnergyCalibrator&) = delete;
  HGCAllLayersEnergyCalibrator& operator=(const HGCAllLayersEnergyCalibrator&) = delete;
  
  void correctEnergy(reco::PFCluster& c) { correctEnergyActual(c); }
  void correctEnergies(reco::PFClusterCollection& cs) {
    for( unsigned i = 0; i < cs.size(); ++i ) correctEnergyActual(cs[i]);
  }

 private:
  // for EM parameterization
  const double _mipValueInGeV_ee,_mipValueInGeV_hef,_mipValueInGeV_heb,
    _coef_a,_coef_b,_coef_c,_coef_d,_coef_e;
  // for hadronic 
  const double _ee_had_emscale_slope, _ee_had_emscale_offset;
  const double _hef_had_emscale_slope, _hef_had_emscale_offset;
  const double _heb_had_emscale_slope, _heb_had_emscale_offset;
  const double _he_had_correction, _heb_had_correction;
  const double _pion_energy_slope, _pion_energy_offset;
  const std::vector<double> _had_residual; // |eta|^0, |eta|^1 , |eta|^2
  //weights and overburdens
  const bool _isEMCalibration;
  const std::vector<double> _weights_ee,_weights_hef,_weights_heb; 
  const TGraphErrors * _hgcOverburdenParam;
  const TGraphErrors * _hgcLambdaOverburdenParam;
  void correctEnergyActual(reco::PFCluster&) const;

  std::unique_ptr<MaskedLayerManager> mask_;
  std::multimap<unsigned,unsigned> ee_ganging, fh_ganging, bh_ganging;
  
};

DEFINE_EDM_PLUGIN(PFClusterEnergyCorrectorFactory,
		  HGCAllLayersEnergyCalibrator,
		  "HGCAllLayersEnergyCalibrator");

namespace {
  template<typename DETID>
  std::pair<int,int> getlayer(const unsigned rawid) {
    DETID id(rawid);
    return std::make_pair(id.zside(),id.layer());
  }
}

void HGCAllLayersEnergyCalibrator::
correctEnergyActual(reco::PFCluster& cluster) const {  
  //std::cout << "inside HGC EE correct energy" << std::endl;
  double eCorr = 0.0;
  double e_ee(0.0), e_hef(0.0), e_heb(0.0);
  const double clus_eta = cluster.positionREP().eta();
  const double abs_eta = std::abs(clus_eta);
  //const double effMIP_to_InvGeV = _coef_a/(1.0 + std::exp(-_coef_c - _coef_b*std::cosh(clus_eta)));  
  const double effMIP_to_InvGeV = _coef_a*fabs(std::tanh(clus_eta))/(1.-(_coef_c*pow(clus_eta,2)+_coef_d*abs_eta+_coef_e));
  for( const auto& rhf : cluster.recHitFractions() ) {
    const std::vector<double>* weights = nullptr;
    const reco::PFRecHit& hit = *(rhf.recHitRef());
    DetId theid( hit.detId() );
    std::pair<int,int> zside_layer ;    
    double mip_value = 0.0;
    double mip2gev = 0.0;
    double hgcOverburdenWeight = 0.0;
    double energy_in_layer = 0.0;
    switch( theid.subdetId() ) {
    case HGCEE:
      weights = &_weights_ee;
      zside_layer = getlayer<HGCEEDetId>(hit.detId());
      mip_value = _mipValueInGeV_ee;
      mip2gev = effMIP_to_InvGeV;
      if(_isEMCalibration) {
	if(zside_layer.second==1 && _hgcOverburdenParam) {
	  hgcOverburdenWeight = _hgcOverburdenParam->Eval(abs_eta);
	}
      } else {
	if(zside_layer.second==1 && _hgcOverburdenParam) {
	  hgcOverburdenWeight = _hgcLambdaOverburdenParam->Eval(abs_eta);
	}	
      }
      {
        double temp_weight = hgcOverburdenWeight;
        auto range = ee_ganging.equal_range(zside_layer.second);
        for( auto itr = range.first; itr != range.second; ++itr ) {
          const unsigned layer = itr->second;
          temp_weight += (*weights)[layer-1];
        }
        energy_in_layer = temp_weight*hit.energy()/(mip_value*std::tanh(abs_eta));
        e_ee += energy_in_layer;       
      }
      break;
    case HGCHEF:
      weights = &_weights_hef;
      zside_layer = getlayer<HGCHEDetId>(hit.detId());
      mip_value = _mipValueInGeV_hef;
      mip2gev = effMIP_to_InvGeV;
      {
        double temp_weight = 0.0;
        auto range = fh_ganging.equal_range(zside_layer.second);
        for( auto itr = range.first; itr != range.second; ++itr ) {
          const unsigned layer = itr->second;
          temp_weight += (*weights)[layer-1];
        }
        energy_in_layer = temp_weight*hit.energy()/(mip_value*std::tanh(abs_eta));
        e_hef += energy_in_layer;
      }
      break;
    case HGCHEB:
      weights = &_weights_heb;
      zside_layer = getlayer<HGCHEDetId>(hit.detId());
      mip_value = _mipValueInGeV_heb;
      mip2gev = effMIP_to_InvGeV;
      {
        double temp_weight = 0.0;
        auto range = bh_ganging.equal_range(zside_layer.second);
        for( auto itr = range.first; itr != range.second; ++itr ) {
          const unsigned layer = itr->second;
          temp_weight += (*weights)[layer-1];
        }
        energy_in_layer = temp_weight*hit.energy()/(mip_value*std::tanh(abs_eta));;
        e_heb += energy_in_layer;
      }
      break;
    default:
      throw cms::Exception("BadRecHit")
	<< "This module only accepts HGC EE, HEF, or HEB hits" << std::endl;
    }
    //const int layer = zside_layer.second;
    //const double energy_MIP = hit.energy()/mip_value; 
    const double added_energy = energy_in_layer*std::tanh(abs_eta)/mip2gev;
    eCorr += added_energy;
  }

  //final offset correction
  if(_isEMCalibration) {
    eCorr = std::max(0., eCorr-_coef_b/_coef_a); 
    //std::cout << "eCorr EM: " << eCorr << std::endl;
  } else {
    //std::cout << "energy in mips: " 
    //      << e_ee << ' ' << e_hef << ' ' << e_heb << std::endl;
    e_ee  = e_ee*_ee_had_emscale_slope   + _ee_had_emscale_offset;
    e_hef = e_hef*_hef_had_emscale_slope + _hef_had_emscale_offset;
    e_heb = e_heb*_heb_had_emscale_slope + _heb_had_emscale_offset;
    eCorr = e_ee + _he_had_correction*(e_hef + _heb_had_correction*e_heb);
    //std::cout << "eCorr first step: " << eCorr << std::endl;
    eCorr = eCorr*_pion_energy_slope + _pion_energy_offset;
    //std::cout << "eCorr second step: " << eCorr << std::endl;
    const double residual = ( _had_residual[2]*abs_eta*abs_eta +
			      _had_residual[1]*abs_eta +
			      _had_residual[0] );
    eCorr = eCorr*(1-residual);
    //std::cout << "eCorr HAD: " << eCorr << std::endl;
  }
  
  
  cluster.setEnergy(eCorr);
  cluster.setCorrectedEnergy(eCorr);
  
  //std::cout << "cluster pt: " << cluster.energy()/std::cosh(cluster.position().eta()) << std::endl;
}
