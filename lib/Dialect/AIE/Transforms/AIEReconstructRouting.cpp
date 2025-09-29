#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include <fstream>
#include "json.hpp"

#define DEBUG_TYPE "aie-reconstruct-routing"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::ordered_json;

typedef struct MaskValue {
  int mask;
  int value;
} MaskValue;

typedef struct PortConnection {
  Operation *op;
  Port port;
} PortConnection;

typedef struct PortMaskValue {
  Port port;
  MaskValue mv;
} PortMaskValue;

typedef struct WorkItem {
  Operation *op;
  Port port;
  std::vector<TileOp> currentPath;
  MaskValue mv;
} WorkItem;

typedef struct Path {
  TileOp srcTile;
  std::vector<TileOp> dstTiles;
  std::vector<std::pair<std::vector<TileOp>, MaskValue>> paths;

  Path(TileOp src) : srcTile(src) {}
} Path;

class ConnectivityAnalysis {
  DeviceOp &device;
public:
  ConnectivityAnalysis(DeviceOp &d) : device(d) {}

private:
  std::optional<PortConnection> getConnectionThroughWire(Operation *op, 
                                                         Port masterPort) const {
    for (auto wireOp : device.getOps<WireOp>()) {
      if (wireOp.getSource().getDefiningOp() == op &&
          wireOp.getSourceBundle() == masterPort.bundle) {
        Operation *other = wireOp.getDest().getDefiningOp();
        Port otherPort = {wireOp.getDestBundle(), masterPort.channel};
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.bundle) << " "
                                << otherPort.channel << "\n");
        return PortConnection{other, otherPort};
      }
      if (wireOp.getDest().getDefiningOp() == op &&
          wireOp.getDestBundle() == masterPort.bundle) {
        Operation *other = wireOp.getSource().getDefiningOp();
        Port otherPort = {wireOp.getSourceBundle(), masterPort.channel};
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.bundle) << " "
                                << otherPort.channel << "\n");
        return PortConnection{other, otherPort};
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "*** Missing Wire!\n");
    return std::nullopt;
  }
  
  std::vector<PortMaskValue> getConnectionsThroughSwitchbox(Region &r, Port sourcePort) {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");
    Block &b = r.front();
    std::vector<PortMaskValue> portSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        MaskValue maskValue = {0, 0};
        portSet.push_back({connectOp.destPort(), maskValue});
        LLVM_DEBUG(llvm::dbgs() << "To:" << stringifyWireBundle(connectOp.destPort().bundle)
                << " " << connectOp.destPort().channel << "\n");
      }
    }
    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Packet From: "
                   << stringifyWireBundle(connectOp.sourcePort().bundle) << " "
                   << sourcePort.channel << "\n");
        for (auto masterSetOp : b.getOps<MasterSetOp>()) 
          for (Value amsel : masterSetOp.getAmsels())
            for (auto ruleOp : connectOp.getRules().front().getOps<PacketRuleOp>()) {
              if (ruleOp.getAmsel() == amsel) {
                LLVM_DEBUG(llvm::dbgs()
                           << "To:"
                           << stringifyWireBundle(masterSetOp.destPort().bundle)
                           << " " << masterSetOp.destPort().channel << "\n");
                MaskValue maskValue = {ruleOp.maskInt(), ruleOp.valueInt()};
                portSet.push_back({masterSetOp.destPort(), maskValue});
              }
            }
      }
    }
    return portSet;
  }

  std::optional<MaskValue> matchMask(const MaskValue &curr, const MaskValue &next) const {
    int maskConflicts = next.mask & curr.mask;
    if ((maskConflicts & next.value) != (maskConflicts & curr.value))
      return std::nullopt;
    MaskValue newMaskValue = {curr.mask | next.mask,
                              curr.value | (next.mask & next.value)};
    return newMaskValue;
  }

public: 
  Path getPathInfo(TileOp tileOp, Port port) {
    Path pathInfo(tileOp);
    // Start the worklist by traversing from the tile to its connected
    // switchbox.
    auto firstConn = getConnectionThroughWire(tileOp.getOperation(), port);
    if (!firstConn)
      return pathInfo; // no connections, return

    std::vector<WorkItem> worklist;
    worklist.push_back({firstConn->op, firstConn->port, {tileOp}, {0, 0}});

    while(!worklist.empty()) {
      WorkItem wItem = worklist.back();
      worklist.pop_back();

      Operation *other = wItem.op;
      Port otherPort = wItem.port;
      auto currentPath = wItem.currentPath;

      // Find TileOp associated with the current operation
      TileOp currTile = nullptr;
      if (auto t = dyn_cast<TileOp>(other)) {
        currTile = t;
      } else if (auto sw = dyn_cast<SwitchboxOp>(other)) {
        currTile = sw.getTileOp();
      }

      if (currTile && (currentPath.empty() || currentPath.back() != currTile)) {
        currentPath.push_back(currTile);
      }

      // If this is a destination TileOp (flow endpoint)
      if (isa<TileOp>(other) && other != tileOp.getOperation()) {
        auto dstTileOp = dyn_cast<TileOp>(other);
        if (dstTileOp) {
          pathInfo.dstTiles.push_back(dstTileOp);
          pathInfo.paths.push_back({currentPath, wItem.mv});
        }
      } else if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        auto nextPortMVs = getConnectionsThroughSwitchbox(
                            switchOp.getConnections(), otherPort);
        // need to add next ports to check
        for (auto &portMV : nextPortMVs) {
          auto mergedMV = matchMask(wItem.mv, portMV.mv);
          if (!mergedMV)
            continue;
          auto nextConnection = getConnectionThroughWire(switchOp, portMV.port);
          if (nextConnection) {
            // clone path for each branch
            auto newPath = currentPath;
            worklist.push_back({nextConnection->op, nextConnection->port, newPath, *mergedMV});
          }
        }
      } else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        auto nextPortMVs = getConnectionsThroughSwitchbox(
                            switchOp.getConnections(), otherPort);
        // need to add next ports to check
        for (auto &portMV : nextPortMVs) {
          auto mergedMV = matchMask(wItem.mv, portMV.mv);
          if (!mergedMV)
            continue;
          auto nextConnection = getConnectionThroughWire(switchOp, portMV.port);
          if (nextConnection) {
            // clone path for each branch
            auto newPath = currentPath;
            worklist.push_back({nextConnection->op, nextConnection->port, newPath, *mergedMV});
          }
        }
      }
    }
    return pathInfo;
  }
};

struct AIEReconstructRoutingPass : 
    public AIEReconstructRoutingBase<AIEReconstructRoutingPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    ConnectivityAnalysis analysis(device);
    std::vector bundles = {WireBundle::Core, WireBundle::DMA};

    json output;
    output["buffers"] = json::array();
    output["cct_routes"] = json::array();
    output["nbr_routes"] = json::array();
    output["pkt_routes"] = json::array();
    
    // Map of (col, row) to list of allocated buffers
    std::map<std::pair<int, int>, std::vector<int64_t>> bufferMap;

    // Obtain buffer allocation info
    for (auto buffer : device.getOps<BufferOp>()) {
      // no. of bytes allocated for this buffer
      int64_t bufferSize = buffer.getAllocationSize();
      auto tile = buffer.getTileOp();
      int row = tile.getRow(), col = tile.getCol();

      bufferMap[{col, row}].push_back(bufferSize);
    }
    for (auto &[loc, sizes] : bufferMap) {
      int col = loc.first, row = loc.second;
      int64_t totalSize = 0;
      for (auto s : sizes)
        totalSize += s;

      json buf = {
        {"col_x", col},
        {"row_y", row},
        {"sizes_bytes", sizes},
        {"total_size_bytes", totalSize}
      };
      output["buffers"].push_back(buf);
    }

    // Perform connectivity analysis
    for (auto tile : device.getOps<TileOp>()) {
      // Perform analysis on each tile
      for (WireBundle bundle : bundles) {
        for (size_t i = 0; i < tile.getNumSourceConnections(bundle); i++) {
          Path pInfo = analysis.getPathInfo(tile, {bundle, (int)i});
          std::vector<std::vector<std::pair<int, int>>> routes;
          bool isPkt = true;
          if (!pInfo.paths.empty()) {
            for (auto &[path, mv] : pInfo.paths) {
              if (mv.mask == 0 && mv.value == 0) {
                isPkt = false;
              }
              std::vector<std::pair<int, int>> route;
              for (auto t : path) {
                route.push_back({t.getCol(), t.getRow()});
              }
              routes.push_back(route);
            }
            
            // Convert routes into JSON
            json routeGroup = json::object();
            routeGroup["src"] = {{"col_x", pInfo.srcTile.getCol()},
                                 {"row_y", pInfo.srcTile.getRow()}};
            json dstArray = json::array();
            for (auto dst : pInfo.dstTiles) {
              dstArray.push_back({{"col_x", dst.getCol()}, {"row_y", dst.getRow()}});
            }
            routeGroup["dsts"] = dstArray;
            json intermediatesArray = json::array();
            for (auto &route : routes) {
              json routeJson = json::array();
              for (auto &[col, row] : route) {
                routeJson.push_back({{"col_x", col}, {"row_y", row}});
              }
              intermediatesArray.push_back(routeJson);
            }
            routeGroup["intermediates"] = intermediatesArray;
            if (isPkt) {
              output["pkt_routes"].push_back(routeGroup);
            } else {
              output["cct_routes"].push_back(routeGroup);
            }
          }
        }
      }
    }

    std::map<std::pair<TileOp, TileOp>, TileOp> sharedMemConnect;
    for (auto coreOp : device.getOps<CoreOp>()) {
      auto coreTileOp = coreOp.getTileOp();
      
      coreOp.walk([&](UseLockOp useLockOp) {
        auto lock = useLockOp.getLockOp();
        auto lockTileOp = lock.getTileOp();
        bool isProd = lock.hasName() && lock.name().getValue().contains("prod");

        if (coreTileOp != lockTileOp && useLockOp.release()) {
          // Determine connection direction depending on if it's producer lock
          std::pair<TileOp, TileOp> edge = isProd ? std::make_pair(lockTileOp, coreTileOp)
                                                  : std::make_pair(coreTileOp, lockTileOp);

          // Check if overwriting existing entry
          if (sharedMemConnect.count(edge) == 0) {
            sharedMemConnect[edge] = lockTileOp;
          }
        }
      });
    }

    for (auto &[tiles, allocTile] : sharedMemConnect) {
      auto [srcTile, dstTile] = tiles;
      json connJson = {
        {"src", {{"col_x", srcTile.getCol()}, {"row_y", srcTile.getRow()}}},
        {"dst", {{"col_x", dstTile.getCol()}, {"row_y", dstTile.getRow()}}},
        {"allocation_tiles", {{"col_x", allocTile.getCol()}, {"row_y", allocTile.getRow()}}}
      };
      output["nbr_routes"].push_back(connJson);
    }

    // write json to file
    std::ofstream outFile("post_compile_routing_summary.json");
    if (!outFile.is_open()) {
        llvm::errs() << "Could not open post_compile_routing_summary.json for writing\n";
        return;
    }
    outFile << output.dump(2);
    outFile.close();

    llvm::dbgs() << "Route summary written to post_compile_routing_summary.json\n";
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEReconstructRoutingPass() {
  return std::make_unique<AIEReconstructRoutingPass>();
}
