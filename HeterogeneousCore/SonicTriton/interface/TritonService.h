#ifndef HeterogeneousCore_SonicTriton_TritonService
#define HeterogeneousCore_SonicTriton_TritonService

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/bimap/bimap.hpp"
#include "boost/bimap/unordered_multiset_of.hpp"

#include <string>
#include <functional>

//forward declarations
namespace edm {
	class ActivityRegistry;
}

class TritonService {
public:
	//classes and defs
	template <typename A, typename B, typename AH = std::hash<A>, typename AE = std::equal_to<A>, typename BH = std::hash<B>, typename BE = std::equal_to<B>>
	using unordered_bimultimap = boost::bimaps::bimap<boost:bimaps::unordered_multiset_of<A,AH,AE>,boost::bimaps::unordered_multiset_of<B,BH,BE>>;

	struct Server {
		Server(const edm::ParameterSet& pset) : name(pset.getUntrackedParameter<std::string>("name")), url(pset.getUntrackedParameter<std::string>("address") + ":" + std::to_string(pset.getUntrackedParameter<unsigned>("port"))) {}

		struct Hash {
			size_t operator()(const Server& obj) const {
				return hashObj(obj.name);
			}
			std::hash<std::string> hashObj;
		};

		struct Equal {
			bool operator()(const Server& lhs, const Server& rhs) const {
				return lhs.name == rhs.name;
			}
		};

		//members
		std::string name;
		std::string url;
	};
	struct Model {};
	using ServerModelMap = unordered_bimultimap<boost::bimaps::tagged<Server,Server>,boost::bimaps::tagged<std::string,Model>,Server::Hash,Server::Equal>;

	TritonService(const edm::ParameterSet& pset, edm::ActivityRegistry& areg);

	//accessors
	std::string serverAddress(const std::string& model, const std::string& preferred="") const;

private:
	//members
	ServerModelMap mapServersModels_;
};

#endif
