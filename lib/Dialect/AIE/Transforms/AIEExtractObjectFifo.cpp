#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include <fstream>
#include "json.hpp"

#define DEBUG_TYPE "aie-extract-fifo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::ordered_json;

static llvm::cl::opt<std::string> jsonFilePath(
    "output-netlist-file",
    llvm::cl::desc("Path to JSON netlist"),
    llvm::cl::init("netlist.json"));

struct AIEExtractObjectFifoPass
    : AIEExtractObjectFifoBase<AIEExtractObjectFifoPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    int id = 0;

    json output;
    output["pre_alloc_buffers"] = json::array();
    output["nodes"] = json::array();
    output["nets"] = json::array();
    output["links"] = json::array();
    output["cascades"] = json::array();
    output["pre_alloc_intra_nets"] = json::array();

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

    DenseMap<Value, std::vector<int64_t>> bufferMap;

    for (auto buffer : device.getOps<BufferOp>()) {
      int64_t bufferSize = buffer.getAllocationSize();
      auto tile = buffer.getTileOp();
      bufferMap[tile.getResult()].push_back(bufferSize);
    }

    for (auto &[tileVal, sizes] : bufferMap) {
      int nodeId = tileIdMap[tileVal];
      int64_t totalSize = 0;
      for (auto s : sizes)
        totalSize += s;

      json buf = {
        {"node_id", nodeId},
        {"sizes_bytes", sizes},
        {"total_size_bytes", totalSize}
      };
      output["pre_alloc_buffers"].push_back(buf);
    }

    id = 0;
    DenseMap<ObjectFifoCreateOp, int> fifoIdMap;
    DenseMap<int, std::pair<int64_t, int64_t>> byteSizeMap;
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

      byteSizeMap[id] = {byteSize, depths[0]};

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

    for (auto cascadeOp : device.getOps<CascadeFlowOp>()) {
      auto sTileOp = cascadeOp.getSourceTileOp();
      auto dTileOp = cascadeOp.getDestTileOp();
      int sId = tileIdMap[sTileOp.getResult()];
      int dId = tileIdMap[dTileOp.getResult()];
      json cascade = {
        {"src_node_id", sId},
        {"dst_node_id", dId}
      };
      output["cascades"].push_back(cascade);
    }

    for (auto allocOp : device.getOps<ObjectFifoAllocateOp>()) {
      int fId = fifoIdMap[allocOp.getObjectFifo()];
      auto [byteSize, depth] = byteSizeMap[fId];
      TileOp tileOp = allocOp.getDelegateTileOp();
      int tId = tileIdMap[tileOp.getResult()];
      json alloc = {
        {"net_id", fId},
        {"node_id", tId},
        {"depth", depth},
        {"byte_size_per_depth", byteSize}
      };
      output["pre_alloc_intra_nets"].push_back(alloc);
    }

    // write json to file
    std::ofstream outFile(jsonFilePath);
    if (!outFile.is_open()) {
        llvm::errs() << "Error: Could not open " << jsonFilePath
                     << " for writing\n";
        return;
    }
    outFile << output.dump(2);
    outFile.close();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEExtractObjectFifoPass() {
  return std::make_unique<AIEExtractObjectFifoPass>();
}
