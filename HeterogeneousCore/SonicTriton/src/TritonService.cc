#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "grpc_client.h"
#include "grpc_service.pb.h"

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

TritonService::TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg) {
	//loop over input servers: check which models they have
	auto& mapServers = mapServersModels_.by<Server>();
	for(const auto& serverPset : pset.getParameterSetVector("servers")){
		Server tmp(serverPset);
		//ensure uniqueness
		auto it = mapServers.find(tmp.name);
		if (it!=mapServers.end())
			throw cms::Exception("DuplicateServer") << "Not allowed to specify more than one server with same name (" << tmp.name << ")";

		std::unique_ptr<nic::InferenceServerGrpcClient> client;
		triton_utils::throwIfError(nic::InferenceServerGrpcClient::Create(&client, tmp.url, false), "TritonService(): unable to create inference context for "+tmp.name+" ("+tmp.url+")");

		inference::RepositoryIndexResponse repoIndexResponse;
		triton_utils::throwIfError(client->ModelRepositoryIndex(&repoIndexResponse), "TritonService(): unable to get repository index for "+tmp.name+" ("+tmp.url+")");

		for(const auto& modelIndex : repoIndexResponse.models()){
			mapServersModels_.insert(Server,modelIndex.name());
		}
	}
}

std::string TritonService::serverAddress(const std::string& model, const std::string& preferred) const {
	if(!preferred.empty()){
		
	}
	return "";
}
