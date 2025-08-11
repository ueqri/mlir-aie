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
    llvm::outs() << "Number of tiles: " << tileOps.size() << "\n";
    auto fifoOps = llvm::to_vector(device.getOps<ObjectFifoCreateOp>());
    llvm::outs() << "Number of FIFOs: " << fifoOps.size() << "\n";
    for (size_t i = 0; i < tileOps.size(); i++) {
      auto tileOp = tileOps[i];
      auto node = input["nodes"][i];

      int col = node["col_x"];
      int row = node["row_y"];

      // Replace attributes
      //tileOp->setAttr("col", builder.getI32IntegerAttr(col));
      tileOp.setCol(col);
      tileOp->setAttr("row", builder.getI32IntegerAttr(row));
    }
    llvm::outs() << "Tiles placed successfully.\n";
    for (size_t i = 0; i < fifoOps.size(); i++) {
      auto fifoOp = fifoOps[i];
      auto net  = input["nets"][i];

      if (net["connection_type"] == "circuit_switch") {
        SmallVector<IntArray2DAttr> outerArray; 

        for (const auto &hopPath : net["intermediates"]) {
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
      else if (net["connection_type"] == "neighbour_sharing") {
        int shareDirection = net["share_direction"];
        fifoOp.setViaSharedMemAttr(builder.getI32IntegerAttr(shareDirection));
      }
    }
    llvm::outs() << "FIFOs configured successfully.\n";
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPlaceTilesPass() {
  return std::make_unique<AIEPlaceTilesPass>();
}
