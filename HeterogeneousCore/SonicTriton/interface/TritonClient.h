#ifndef HeterogeneousCore_SonicTriton_TritonClient
#define HeterogeneousCore_SonicTriton_TritonClient

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientSync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientPseudoAsync.h"
#include "HeterogeneousCore/SonicCore/interface/SonicClientAsync.h"

#include <vector>
#include <string>
#include <exception>

#include "request_grpc.h"

namespace nic = nvidia::inferenceserver::client;
namespace ni = nvidia::inferenceserver;

template <typename Client>
class TritonClient : public Client {
	public:
		typedef ModelInfo = std::pair<std::string, int64_t>;
		struct ServerSideStats {
			uint64_t requestCount_;
			uint64_t cummTimeNs_;
			uint64_t queueTimeNs_;
			uint64_t computeTimeNs_;

			std::map<ModelInfo, ServerSideStats> composingModelsStat_;
		};

		//constructor
		TritonClient(const edm::ParameterSet& params);

		//helper
		std::exception_ptr getResults(const std::unique_ptr<nic::InferContext::Result>& result);

		//accessors
		unsigned nInput() const { return nInput_; }
		unsigned nOutput() const { return nOutput_; }
		unsigned batchSize() const { return batchSize_; }

		//for fillDescriptions
		static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
			edm::ParameterSetDescription descClient;
			descClient.add<unsigned>("nInput");
			descClient.add<unsigned>("nOutput");
			descClient.add<unsigned>("batchSize");
			descClient.add<std::string>("address");
			descClient.add<unsigned>("port");
			descClient.add<unsigned>("timeout");
			descClient.add<std::string>("modelName");
			iDesc.add<edm::ParameterSetDescription>("Client",descClient);
		}

	protected:
		void evaluate() override;

		//helper for common ops
		std::exception_ptr setup();

		//helper to turn triton error into exception
		template <typename F, typename... Args>
		static std::exception_ptr wrap(F&& fn, const std::string& msg, Args&&... args);

		void reportServerSideState(const ServerSideStats& stats) const;
		void summarizeServerStats(
			const ModelInfo model_info,
			const std::map<std::string, ni::ModelStatus>& start_status,
			const std::map<std::string, ni::ModelStatus>& end_status,
			ServerSideStats* server_stats) const;
		void summarizeServerModelStats(
			const std::string& model_name, const int64_t model_version,
			const ni::ModelStatus& start_status, const ni::ModelStatus& end_status,
			ServerSideStats* server_stats) const;

		void getServerSideStatus(std::map<std::string, ni::ModelStatus>* model_status);
		void getServerSideStatus(
			ni::ServerStatus& server_status, const ModelInfo model_info,
			std::map<std::string, ni::ModelStatus>* model_status);

		//members
		std::string url_;
		unsigned timeout_;
		std::string modelName_;
		unsigned batchSize_;
		unsigned nInput_;
		unsigned nOutput_;
		std::unique_ptr<nic::InferContext> context_;
		std::unique_ptr<nic::ServerStatusContext> serverCtx_;
		std::shared_ptr<nic::InferContext::Input> nicInput_; 

		std::map<std::string, ni::ModelStatus> startStatus_, endStatus_;
};

typedef TritonClient<SonicClientSync<std::vector<float>>> TritonClientSync;
typedef TritonClient<SonicClientPseudoAsync<std::vector<float>>> TritonClientPseudoAsync;
typedef TritonClient<SonicClientAsync<std::vector<float>>> TritonClientAsync;

#endif
