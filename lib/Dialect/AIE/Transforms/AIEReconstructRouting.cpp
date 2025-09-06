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

typedef struct PortConnection {
  Operation *op;
  Port port;
} PortConnection;

typedef struct WorkItem {
  Operation *op;
  Port port;
  std::vector<TileOp> currentPath;
} WorkItem;

typedef struct Path {
  TileOp srcTile;
  std::vector<TileOp> dstTiles;
  std::vector<std::vector<TileOp>> paths;

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
  
  std::vector<Port> getConnectionsThroughSwitchbox(Region &r, Port sourcePort) {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");
    Block &b = r.front();
    std::vector<Port> portSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        portSet.push_back(connectOp.destPort());
        LLVM_DEBUG(llvm::dbgs() << "To:" << stringifyWireBundle(connectOp.destPort().bundle)
                << " " << connectOp.destPort().channel << "\n");
      }
    }
    return portSet;
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
    worklist.push_back({firstConn->op, firstConn->port, {tileOp}});

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
          pathInfo.paths.push_back(currentPath);
        }
      } else if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        auto nextPorts = getConnectionsThroughSwitchbox(
                            switchOp.getConnections(), otherPort);
        // need to add next ports to check
        for (auto &port : nextPorts) {
          auto nextConnection = getConnectionThroughWire(switchOp, port);
          if (nextConnection) {
            // clone path for each branch
            auto newPath = currentPath;
            worklist.push_back({nextConnection->op, nextConnection->port, newPath});
          }
        }
      } else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        auto nextPorts = getConnectionsThroughSwitchbox(
                            switchOp.getConnections(), otherPort);
        // need to add next ports to check
        for (auto &port : nextPorts) {
          auto nextConnection = getConnectionThroughWire(switchOp, port);
          if (nextConnection) {
            // clone path for each branch
            auto newPath = currentPath;
            worklist.push_back({nextConnection->op, nextConnection->port, newPath});
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
          Path pInfo = analysis.getPathInfo(tile, {bundle, i});
          std::vector<std::vector<std::pair<int, int>>> routes;
          if (!pInfo.paths.empty()) {
            for (auto path : pInfo.paths) {
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
            output["cct_routes"].push_back(routeGroup);
          }
        }
      }
    }

    //std::set<Connection, ConnectionComparator> sharedMemConnect;
    std::map<std::pair<TileOp, TileOp>, int> sharedMemConnect;
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
          int shareDir = isProd ? -1 : 1;

          // Check if overwriting existing entry
          if (sharedMemConnect.count(edge)) {
            llvm::errs() << "Overwriting shared mem connection between ("
                        << edge.first.getCol() << "," << edge.first.getRow() << ") and ("
                        << edge.second.getCol() << "," << edge.second.getRow() << ")\n";
          }

          sharedMemConnect[edge] = shareDir;
        }
      });
    }

    for (auto &[tiles, shareDir] : sharedMemConnect) {
      auto [srcTile, dstTile] = tiles;
      json connJson = {
        {"src", {{"col_x", srcTile.getCol()}, {"row_y", srcTile.getRow()}}},
        {"dst", {{"col_x", dstTile.getCol()}, {"row_y", dstTile.getRow()}}},
        {"share_direction", shareDir}
      };
      output["nbr_routes"].push_back(connJson);
    }

    // write json to file
    std::ofstream outFile("route_summary.json");
    if (!outFile.is_open()) {
        llvm::errs() << "Could not open route_summary.json for writing\n";
        return;
    }
    outFile << output.dump(2);
    outFile.close();

    llvm::outs() << "Route summary written to route_summary.json\n";
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEReconstructRoutingPass() {
  return std::make_unique<AIEReconstructRoutingPass>();
}
