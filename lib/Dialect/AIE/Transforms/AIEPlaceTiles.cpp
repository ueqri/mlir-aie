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

struct AIEPlaceTilesPass : public AIEPlaceTilesBase<AIEPlaceTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody()); 
    std::ifstream jsonFile("netlist.json");
    if (!jsonFile.is_open()) {
        llvm::errs() << "Failed to open netlist.json\n";
        return;
    }

    json input = json::parse(jsonFile);

    auto tileOps = llvm::to_vector(device.getOps<TileOp>());
    LLVM_DEBUG(llvm::dbgs() << "Number of tiles: " << tileOps.size() << "\n");
    auto fifoOps = llvm::to_vector(device.getOps<ObjectFifoCreateOp>());
    LLVM_DEBUG(llvm::dbgs() << "Number of FIFOs: " << fifoOps.size() << "\n");
    for (size_t i = 0; i < tileOps.size(); i++) {
      auto tileOp = tileOps[i];
      auto node = input["nodes"][i];

      int col = node["col_x"];
      int row = node["row_y"];

      tileOp.setCol(col);
      tileOp.setRow(row);
    }
    LLVM_DEBUG(llvm::dbgs() << "Tiles placed successfully.\n");
    for (size_t i = 0; i < fifoOps.size(); i++) {
      auto fifoOp = fifoOps[i];
      auto route_info = input["nets"][i]["routing_info"];

      if (route_info["connection_type"] == "circuit_switch") {
        SmallVector<IntArray2DAttr> outerArray;

        for (const auto &hopPath : route_info["intermediates"]) {
          SmallVector<IntArray1DAttr> innerArray; 

          for (const auto &coords : hopPath) {
            SmallVector<IntegerAttr> coordAttrs;
            coordAttrs.push_back(builder.getI32IntegerAttr(coords[0].get<int>()));
            coordAttrs.push_back(builder.getI32IntegerAttr(coords[1].get<int>()));

            auto coordAttr = IntArray1DAttr::get(fifoOp.getContext(), coordAttrs);
            innerArray.push_back(coordAttr);
          }

          auto hopPathAttr = IntArray2DAttr::get(fifoOp.getContext(), innerArray); 
          outerArray.push_back(hopPathAttr);
        }

        auto fullAttr = IntArray3DAttr::get(fifoOp.getContext(), outerArray); 
        fifoOp.setHopTileIdsAttr(fullAttr);
        fifoOp.setVia_DMAAttr(builder.getBoolAttr(true));
      }
      else if (route_info["connection_type"] == "neighbor_sharing") {
        SmallVector<Attribute> shareDirectionAttrs;
        for (const auto &dir : route_info["share_directions"]) {
          int shareDirection = dir.get<int>();
          shareDirectionAttrs.push_back(builder.getIntegerAttr(
              builder.getI32Type(), shareDirection));
        }
        fifoOp.setViaSharedMemAttr(ArrayAttr::get(fifoOp.getContext(), 
            shareDirectionAttrs));
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "FIFOs configured successfully.\n");
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}
