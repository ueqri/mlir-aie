#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include <fstream>
#include "json.hpp"

#define DEBUG_TYPE "pnr-extract-netlist"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::ordered_json;

struct PnRExtractNetlistPass
    : PnRExtractNetlistBase<PnRExtractNetlistPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    int id = 0;

    json output;
    output["nodes"] = json::array();
    output["nets"] = json::array();
    output["links"] = json::array();

    DenseMap<Value, int> tileIdMap;
    for (auto tileOp : device.getOps<TileOp>()) {
      int row = tileOp.getRow(), col = tileOp.getCol();
      std::string type = "COMP";
      if (row == 0)
        type = "SHIM";
      else if (row == 1)
        type = "MEM";

      json node = {
          {"id", id},
          {"type", type},
          {"col_x", col},
          {"row_y", row}};
      output["nodes"].push_back(node);
      tileIdMap[tileOp.getResult()] = id++;
    }

    id = 0;
    DenseMap<ObjectFifoCreateOp, int> fifoIdMap;
    for (auto objectFifo : device.getOps<ObjectFifoCreateOp>()) {
      int sId = tileIdMap[objectFifo.getProducerTile()];
      std::vector<int> dIds;
      for (auto cTile : objectFifo.getConsumerTiles()) {
        dIds.push_back(tileIdMap[cTile]);
      }

      // Extract the memref element type
      auto memrefTy = objectFifo.getElemType().getElementType();

      // Compute number of elements in the memref shape
      int64_t shapeProduct = 1;
      for (auto dim : memrefTy.getShape())
        shapeProduct *= dim;

      // Compute size in bytes
      int64_t bits = memrefTy.getElementType().getIntOrFloatBitWidth();
      int64_t byteSize = shapeProduct * (bits / 8);

      std::vector<int64_t> depths;
      auto elemAttr = objectFifo.getElemNumberAttr();
      if (auto intAttr = dyn_cast<IntegerAttr>(elemAttr)) {
        depths.push_back(intAttr.getInt());
      } else if (auto arrayAttr = dyn_cast<ArrayAttr>(elemAttr)) {
        for (auto val : arrayAttr.getValue()) {
          depths.push_back(cast<IntegerAttr>(val).getInt());
        }
      } else {
        objectFifo.emitError("Unsupported elemNumber format");
        return;
      }

      json net = {
          {"net_id", id},
          {"src_id", sId},
          {"dst_ids", dIds},
          {"depths", depths},
          {"byte_size_per_depth", byteSize}};
      output["nets"].push_back(net);
      fifoIdMap[objectFifo] = id++;
    }

    for (auto linkFifo : device.getOps<ObjectFifoLinkOp>()) {
      auto sFifos = linkFifo.getInputObjectFifos();
      auto dFifos = linkFifo.getOutputObjectFifos();
      std::vector<int> sTIds;
      std::vector<int> dTIds;
      for (auto sFifo : sFifos) {
        sTIds.push_back(fifoIdMap[sFifo]);
      }
      for (auto dFifo : dFifos) {
        dTIds.push_back(fifoIdMap[dFifo]);
      }
      json link = {
        {"src_net_ids", sTIds},
        {"dst_net_ids", dTIds}
      };
      output["links"].push_back(link);
    }

    // write json to file
    std::ofstream outFile(clNetlistFile);
    if (!outFile.is_open()) {
        llvm::errs() << "Error: Could not open " << clNetlistFile
                     << " for writing\n";
        return;
    }
    outFile << output.dump(2);
    outFile.close();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createPnRExtractNetlistPass() {
  return std::make_unique<PnRExtractNetlistPass>();
}
