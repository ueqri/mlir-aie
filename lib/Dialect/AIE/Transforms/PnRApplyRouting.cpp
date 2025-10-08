#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include <fstream>
#include "json.hpp"

#define DEBUG_TYPE "pnr-apply-routing"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::ordered_json;

struct PnRApplyRoutingPass : public PnRApplyRoutingBase<PnRApplyRoutingPass> {
  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;

  void initConfigs(int maxCol, int maxRow, DeviceOp &device,
                  const AIETargetModel &targetModel) {
    // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
    for (auto tileOp : device.getOps<TileOp>()) {
      int col, row;
      col = tileOp.colIndex();
      row = tileOp.rowIndex();
      assert(coordToTile.count({col, row}) == 0);
      coordToTile[{col, row}] = tileOp;
    }
    for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
      int col = switchboxOp.colIndex();
      int row = switchboxOp.rowIndex();
      assert(coordToSwitchbox.count({col, row}) == 0);
      coordToSwitchbox[{col, row}] = switchboxOp;
    }
    for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
      int col = shimmuxOp.colIndex();
      int row = shimmuxOp.rowIndex();
      assert(coordToShimMux.count({col, row}) == 0);
      coordToShimMux[{col, row}] = shimmuxOp;
    }
  }
  
  WireBundle getWireBundle(std::string dir) {
    if (dir == "North") return WireBundle::North;
    if (dir == "South") return WireBundle::South;
    if (dir == "East") return WireBundle::East;
    if (dir == "West") return WireBundle::West;
    if (dir == "DMA") return WireBundle::DMA;
    llvm_unreachable("Invalid bundle string");
  }

  TileOp getTile(OpBuilder &builder, int col, int row) {
    if (coordToTile.count({col, row})) {
      return coordToTile[{col, row}];
    }
    auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
    coordToTile[{col, row}] = tileOp;
    
    return tileOp;
  }

  SwitchboxOp getSwitchbox(OpBuilder &builder, int col, int row) {
    if (coordToSwitchbox.count({col, row})) {
      return coordToSwitchbox[{col, row}];
    }
    auto switchboxOp = builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                                  getTile(builder, col, row));
    SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                  builder.getUnknownLoc());
    coordToSwitchbox[{col, row}] = switchboxOp;
    
    return switchboxOp;
  }

  ShimMuxOp getShimMux(OpBuilder &builder, int col) {
    assert(col >= 0);
    int row = 0;
    if (coordToShimMux.count({col, row})) {
      return coordToShimMux[{col, row}];
    }
    assert(getTile(builder, col, row).isShimNOCorPLTile());
    auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                                getTile(builder, col, row));
    SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                  builder.getUnknownLoc());
    coordToShimMux[{col, row}] = switchboxOp;

    return switchboxOp;
  }

  void addConnection(OpBuilder &builder, Interconnect op, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {
    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = builder.saveInsertionPoint();
    builder.setInsertionPoint(b.getTerminator());

    builder.create<ConnectOp>(builder.getUnknownLoc(), inBundle, inIndex,
                              outBundle, outIndex);

    builder.restoreInsertionPoint(point);

    llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << inBundle
               << inIndex << " -> " << outBundle
               << outIndex << "\n";
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const AIETargetModel &targetModel = device.getTargetModel();
    int maxCol = targetModel.columns();
    int maxRow = targetModel.rows();

    initConfigs(maxCol, maxRow, device, targetModel);
    LLVM_DEBUG(llvm::dbgs() << "Begin Path to Routing Pass\n");

    std::ifstream routeFile(clRoutingFilePath);
    if (!routeFile.is_open()) {
        device.emitError() << "Failed to open " << clRoutingFilePath << "\n";
        return signalPassFailure();
    }
    
    json routeJson = json::parse(routeFile);

    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    if (routeJson.contains("switchbox_settings")) {
      for (const auto &sb : routeJson["switchbox_settings"]) {
        int col = sb["col_x"];
        int row = sb["row_y"];
        bool isShim = getTile(builder, col, row).isShimNOCorPLTile();
        // workaround so shimmux comes before shimsb in the IR
        if (isShim) getShimMux(builder, col);
        SwitchboxOp swOp = getSwitchbox(builder, col, row);

        Block &b = swOp.getConnections().front();
        builder.setInsertionPoint(b.getTerminator());

        // Create amsel ops for this sb
        std::map<int, AMSelOp> amselops;
        for (const auto &amsel : sb["amsels"]) {
          int amsID = amsel["ams_id"];
          int arbiterID = amsel["arbiter_id"];
          int select = amsel["select"];

          auto amselOp = builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, select);
          amselops[amsID] = amselOp;
        }

        for (const auto &masterset : sb["mastersets"]){
          std::vector<int> amselIDs = masterset["amsel_ids"];
          WireBundle bundle = getWireBundle(masterset["dst_bundle"]);
          int channel = masterset["dst_channel"];
          SmallVector<Value, 4> amsels;
          for (int amselID : amselIDs) {
            amsels.push_back(amselops[amselID]);
          }

          builder.create<MasterSetOp>(
              builder.getUnknownLoc(), builder.getIndexType(), bundle, channel, amsels, nullptr);
        }

        // Generate packet rules
        for (const auto &group : sb["packet_rules"]) {
          builder.setInsertionPoint(b.getTerminator());
          WireBundle bundle = getWireBundle(group["src_bundle"]);
          int channel = group["src_channel"];

          auto packetRules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                      bundle, channel);
          PacketRulesOp::ensureTerminator(packetRules.getRules(), builder, 
                                          builder.getUnknownLoc());

          Block &rules = packetRules.getRules().front();
          builder.setInsertionPoint(rules.getTerminator());

          for (const auto &rule : group["rules"]) {
            int mask = rule["mask"];
            int value = rule["value"];
            Value amsel = amselops[rule["amsel_id"]];

            builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, value, amsel);
          }
        }

        builder.setInsertionPoint(device.getBody()->getTerminator());
        // Generate cct connections
        DenseSet<std::pair<ShimMuxOp, int>> shimInfo;
        for (const auto &connect : sb["connections"]) {
          WireBundle srcBundle = getWireBundle(connect["src_bundle"]);
          int srcChannel = connect["src_channel"];
          WireBundle dstBundle = getWireBundle(connect["dst_bundle"]);
          int dstChannel = connect["dst_channel"];

          int shimCh = srcChannel;
          // This is to handle shim DMAs. They are technically two
          // switchboxes shimmux -> shimsb 
          // Create shimmux first
          if (isShim && srcBundle == WireBundle::DMA) {
            // must be either DMA0 -> N3 or DMA1 -> N7
            shimCh = srcChannel == 0 ? 3 : 7;
            ShimMuxOp shimMuxOp = getShimMux(builder, col);
            // ensure we only add this shimmux connection once
            if (shimInfo.count({shimMuxOp, shimCh}))
              continue;
            shimInfo.insert({shimMuxOp, shimCh});
            addConnection(builder, cast<Interconnect>(shimMuxOp.getOperation()),
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          }

          if (isShim && srcBundle == WireBundle::DMA) {
            addConnection(builder, cast<Interconnect>(swOp.getOperation()),
                          WireBundle::South, shimCh, dstBundle, dstChannel);
          }
          else if (isShim && dstBundle == WireBundle::DMA) {
            // shim DMAs at end of flows
            // must be either N2 -> DMA0 or N3 -> DMA1
            shimCh = dstChannel == 0 ? 2 : 3;

            ShimMuxOp shimMuxOp = getShimMux(builder, col);
            addConnection(builder, cast<Interconnect>(shimMuxOp.getOperation()),
                          WireBundle::North, shimCh, dstBundle, dstChannel);
            addConnection(builder, cast<Interconnect>(swOp.getOperation()),
                          srcBundle, srcChannel, WireBundle::South, shimCh);
          } else {
            // otherwise, regular switchbox connection
            addConnection(builder, cast<Interconnect>(swOp.getOperation()),
                          srcBundle, srcChannel, dstBundle, dstChannel);
          }
        }
      }
    }

    for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
      auto retVal = switchbox->getOperand(0);
      auto tileOp = retVal.getDefiningOp<TileOp>();

      // Check if it is a shim Tile
      if (!tileOp.isShimNOCTile())
        continue;

      // Check if the switchbox is empty
      if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
        continue;

      Region &r = switchbox.getConnections();
      Block &b = r.front();

      // Find if the corresponding shimmux exsists or not
      int shimExist = 0;
      ShimMuxOp shimOp;
      for (auto shimmux : device.getOps<ShimMuxOp>()) {
        if (shimmux.getTile() == tileOp) {
          shimExist = 1;
          shimOp = shimmux;
          break;
        }
      }

      for (Operation &Op : b.getOperations()) {
        if (auto pktrules = dyn_cast<PacketRulesOp>(Op)) {

          // check if there is MM2S DMA in the switchbox of the 0th row
          if (pktrules.getSourceBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shimExist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp = getShimMux(builder, tileOp.colIndex());
              shimExist = 1;
            }

            Region &r0 = shimOp.getConnections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            pktrules.setSourceBundle(WireBundle::South);
            if (pktrules.getSourceChannel() == 0) {
              pktrules.setSourceChannel(3);
              builder.create<ConnectOp>(builder.getUnknownLoc(), WireBundle::DMA,
                                        0, WireBundle::North, 3);
            }
            if (pktrules.getSourceChannel() == 1) {
              pktrules.setSourceChannel(7);
              builder.create<ConnectOp>(builder.getUnknownLoc(), WireBundle::DMA,
                                        1, WireBundle::North, 7);
            }
          }
        }

        if (auto mtset = dyn_cast<MasterSetOp>(Op)) {

          // check if there is S2MM DMA in the switchbox of the 0th row
          if (mtset.getDestBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shimExist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp = getShimMux(builder, tileOp.colIndex());
              shimExist = 1;
            }

            Region &r0 = shimOp.getConnections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            mtset.setDestBundle(WireBundle::South);
            if (mtset.getDestChannel() == 0) {
              mtset.setDestChannel(2);
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 2, WireBundle::DMA, 0);
            }
            if (mtset.getDestChannel() == 1) {
              mtset.setDestChannel(3);
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 3, WireBundle::DMA, 1);
            }
          }
        }
      }
    }

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<CircuitPathOp, NeighbourPathOp>(op))
        opsToErase.insert(op);
    });
    SmallVector<Operation *> sorted{opsToErase.begin(), opsToErase.end()};
    computeTopologicalSorting(sorted);
    for (auto *op : llvm::reverse(sorted))
      op->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createPnRApplyRoutingPass() {
  return std::make_unique<PnRApplyRoutingPass>();
}
