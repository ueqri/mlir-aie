#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "json.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#define DEBUG_TYPE "aie-place-tiles"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::json;

static llvm::cl::opt<std::string> jsonFilePath(
    "input-netlist-file",
    llvm::cl::desc("Path to JSON netlist"),
    llvm::cl::init("netlist.json"));

struct AIEPlaceTilesPass : public AIEPlaceTilesBase<AIEPlaceTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody()); 
    std::ifstream jsonFile(jsonFilePath);
    if (!jsonFile.is_open()) {
        llvm::errs() << "Failed to open " << jsonFilePath << "\n";
        return;
    }

    json input = json::parse(jsonFile);

    auto tileOps = llvm::to_vector(device.getOps<TileOp>());
    LLVM_DEBUG(llvm::dbgs() << "Number of tiles: " << tileOps.size() << "\n");
    auto fifoOps = llvm::to_vector(device.getOps<ObjectFifoCreateOp>());
    LLVM_DEBUG(llvm::dbgs() << "Number of FIFOs: " << fifoOps.size() << "\n");

    for (const auto &node : input["nodes"]) {
      int id = node["id"].get<int>();

      // check bounds before indexing
      if (id >= 0 && id < (int)tileOps.size()) {
        TileOp tile = tileOps[id];
        
        int col = node["col_x"];
        int row = node["row_y"];

        tile.setCol(col);
        tile.setRow(row);
        LLVM_DEBUG(llvm::dbgs() << "Matched node id " << id << " -> TileOp at index " << id << "\n");
      } 
      else 
        llvm::errs() << "Warning: node id " << id << " out of range\n";
    }
    LLVM_DEBUG(llvm::dbgs() << "Tiles placed successfully.\n");

    for (size_t i = 0; i < fifoOps.size(); i++) {
      auto fifoOp = fifoOps[i];
      auto tileOps = llvm::to_vector(device.getOps<TileOp>());
      if (!input["nets"][i].contains("routing_info")) {
        fifoOp.emitError("No routing_info key found in JSON for net " + 
            std::to_string(input["nets"][i]["id"].get<int>()));
        return;
      }
      auto route_info = input["nets"][i]["routing_info"];

      if (route_info["connection_type"] == "circuit_switch") {
        SmallVector<Attribute> outerArray;

        for (const auto &hopPath : route_info["intermediates"]) {
          SmallVector<Attribute> innerArray;

          for (const auto &coords : hopPath) {
            SmallVector<Attribute> coordAttrs;
            coordAttrs.push_back(builder.getI32IntegerAttr(coords[0].get<int>()));
            coordAttrs.push_back(builder.getI32IntegerAttr(coords[1].get<int>()));

            auto coordAttr = builder.getArrayAttr(coordAttrs); // 1D array
            innerArray.push_back(coordAttr);
          }
          auto hopPathAttr = builder.getArrayAttr(innerArray); // 2D array
          outerArray.push_back(hopPathAttr);
        }
        auto fullAttr = builder.getArrayAttr(outerArray); // 3D array as plain ArrayAttr
        fifoOp.setHopTileIdsAttr(fullAttr);
        fifoOp.setVia_DMAAttr(builder.getBoolAttr(true));
      }
      else if (route_info["connection_type"] == "neighbor_sharing") {
        SmallVector<Value> delegateTileVals;
        for (const auto &allocTile : route_info["allocation_tiles"]) {
          int col = allocTile["col_x"];
          int row = allocTile["row_y"];
          bool found = false;

          for (auto tile : tileOps) {
            if (tile.getCol() == col && tile.getRow() == row) {
              delegateTileVals.push_back(tile.getResult());
              found = true;
              break;
            }
          }
          if (!found) {
            OpBuilder::InsertionGuard g(builder);
            builder.setInsertionPointAfter(tileOps.back()); // insert after last existing TileOp
            auto newTile =
                builder.create<TileOp>(builder.getUnknownLoc(), col, row);
            delegateTileVals.push_back(newTile.getResult());
          }
        }
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointAfter(fifoOp);
        auto symRef = FlatSymbolRefAttr::get(builder.getContext(), fifoOp.getSymName());
        builder.create<ObjectFifoAllocateOp>(builder.getUnknownLoc(), symRef, delegateTileVals);
      }
      else if (route_info["connection_type"] == "intra_tile") {
        // SmallVector<Attribute> shareDirectionAttrs;
        // shareDirectionAttrs.push_back(builder.getIntegerAttr(
        //     builder.getI32Type(), -1));
        // fifoOp.setViaSharedMemAttr(ArrayAttr::get(fifoOp.getContext(), 
        //     shareDirectionAttrs));
        continue;
      }
      else {
        fifoOp.emitError("Unsupported connection type in JSON");
        return;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "FIFOs configured successfully.\n");
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}
