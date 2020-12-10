#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "HeterogeneousCore/SonicTriton/interface/triton_utils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "grpc_client.h"
#include "grpc_service.pb.h"

namespace ni = nvidia::inferenceserver;
namespace nic = ni::client;

TritonService::TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg) : fallbackOpts_(pset.getParameterSet("Fallback")) {
	//fallback server will be launched (if needed) before beginJob
	areg.watchPreBeginJob(this, &TritonService::preBeginJob);

	//loop over input servers: check which models they have
	for(const auto& serverPset : pset.getUntrackedParameterSetVector("servers")){
		Server tmp(serverPset);
		//ensure uniqueness
		auto sit = servers_.find(tmp);
		if (sit!=servers_.end())
			throw cms::Exception("DuplicateServer") << "Not allowed to specify more than one server with same name (" << tmp.name << ")";

		std::unique_ptr<nic::InferenceServerGrpcClient> client;
		triton_utils::throwIfError(nic::InferenceServerGrpcClient::Create(&client, tmp.url, false), "TritonService(): unable to create inference context for "+tmp.name+" ("+tmp.url+")");

		inference::RepositoryIndexResponse repoIndexResponse;
		triton_utils::throwIfError(client->ModelRepositoryIndex(&repoIndexResponse), "TritonService(): unable to get repository index for "+tmp.name+" ("+tmp.url+")");

		//servers keep track of models and vice versa
		for(const auto& modelIndex : repoIndexResponse.models()){
			const auto& modelName = modelIndex.name();
			auto mit = findModel(modelName);
			if(mit==models_.end())
				mit = models_.emplace(modelName).first;
			mit->servers.insert(tmp.name);
			tmp.models.insert(modelName);
		}
		servers_.insert(tmp);
	}
}

void TritonService::addModel(const std::string& model, const std::string& path) {
	//if model is not in the list, then no specified server provides it
	auto mit = findModel(model);
	if(mit==models_.end())
		unservedModels_.emplace(model,path);
}

//second return value is only true if fallback CPU server is being used
std::pair<std::string,bool> TritonService::serverAddress(const std::string& model, const std::string& preferred) const {
	auto mit = findModel(model);
	if(mit==models_.end())
		throw cms::Exception("MissingModel") << "There are no servers that provide model " << model;

	const auto& modelServers = mit->servers;

	if(!preferred.empty()){
		auto sit = modelServers.find(preferred);
		//todo: add a "strict" parameter to stop execution if preferred server isn't found?
		if(sit==modelServers.end())
			edm::LogWarning("PreferredServer") << "Preferred server " << preferred << " for model " << model << " not available, will choose another server";
		else
			return std::make_pair(findServer(preferred)->url,false);
	}

	//todo: use some algorithm to select server rather than just picking arbitrarily
	return std::make_pair(findServer(*modelServers.begin())->url,false);
}

void TritonService::preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&) {
	if (!fallbackOpts_.enable) return;
}

TritonService::~TritonService() {
	if (!fallbackOpts_.enable) return;
}

void TritonService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	edm::ParameterSetDescription desc;

	edm::ParameterSetDescription validator;
	validator.addUntracked<std::string>("name");
	validator.addUntracked<std::string>("address");
	validator.addUntracked<unsigned>("port");

	desc.addVPSetUntracked("servers", validator);

	edm::ParameterSetDescription fallbackDesc;
	fallbackDesc.addUntracked<bool>("enable",false);
	fallbackDesc.addUntracked<bool>("useDocker",false);
	fallbackDesc.addUntracked<bool>("useGPU",false);
	fallbackDesc.addUntracked<unsigned>("retries",3);
	fallbackDesc.addUntracked<unsigned>("wait",60);
	desc.add<edm::ParameterSetDescription>("Fallback",fallbackDesc);

	descriptions.addWithDefaultLabel(desc);
}
