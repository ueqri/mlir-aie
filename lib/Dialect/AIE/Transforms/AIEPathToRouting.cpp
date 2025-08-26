#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

#define DEBUG_TYPE "aie-path-to-routing"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;


// ONE switchbox setting for ONE path
using SwitchConfig = struct SwitchConfig {
  std::vector<Port> srcs;
  std::vector<Port> dsts;

  void addSrc(Port src) {
    srcs.push_back(src);
  }

  void addDst(Port dst) {
    dsts.push_back(dst);
  }
};

using SwitchConfigs = std::map<TileID, SwitchConfig>;

// Track channel usage for each switchbox
using SwitchFreeChans = struct SwitchFreeChans {
  std::map<WireBundle, std::set<int>> srcChans;
  std::map<WireBundle, std::set<int>> dstChans;
};

using Interconnects = std::map<std::pair<TileID, TileID>, Port>;

struct AIEPathToRoutingPass : public AIEPathToRoutingBase<AIEPathToRoutingPass> {
  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;

  std::map<CircuitPathOp, SwitchConfigs> sbConfigList;
  std::map<TileID, SwitchFreeChans> sbFreeChans;

  void initConfigs(int maxCol, int maxRow, DeviceOp &device,
                  const AIETargetModel &targetModel) {
    // Initialize the channel usage for each switchbox
    for (int col = 0; col < maxCol; col++) {
      for (int row = 0; row < maxRow; row++) {
        TileID tileId{col, row};
        sbFreeChans[tileId] = SwitchFreeChans{};
        std::set<int> channelSet;
        for (int i = 0, e = getMaxEnumValForWireBundle() + 1; i < e; ++i) {
          WireBundle bundle = symbolizeWireBundle(i).value();
          channelSet.clear();
          // get all ports into current switchbox
          int channels =
              targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
          if (channels == 0 && targetModel.isShimNOCorPLTile(col, row)) {
            // workaround for shimMux
            channels = targetModel.getNumSourceShimMuxConnections(col, row, bundle);
          }
          for (int channel = 0; channel < channels; channel++) {
            channelSet.insert(channel);
          }
          
          sbFreeChans[tileId].srcChans[bundle] = channelSet;
          channelSet.clear();

          // get all ports out of current switchbox
          channels = targetModel.getNumDestSwitchboxConnections(col, row, bundle);
          if (channels == 0 && targetModel.isShimNOCorPLTile(col, row)) {
            // workaround for shimMux
            channels = targetModel.getNumDestShimMuxConnections(col, row, bundle);
          }
          for (int channel = 0; channel < channels; channel++) {
            channelSet.insert(channel);
          }
          sbFreeChans[tileId].dstChans[bundle] = channelSet;
        }
      }
    }

    //debug print sbFreeChans at (6, 4)
    if (auto it = sbFreeChans.find({6, 4}); it != sbFreeChans.end()) {
      llvm::outs() << "sbFreeChans at (6, 4):\n";
      for (const auto &[bundle, channels] : it->second.srcChans) {
        llvm::outs() << "  " << stringifyEnum(bundle) << ": ";
        for (int channel : channels) {
          llvm::outs() << channel << " ";
        }
        llvm::outs() << "\n";
      }

      for (const auto &[bundle, channels] : it->second.dstChans) {
        llvm::outs() << "  " << stringifyEnum(bundle) << ": ";
        for (int channel : channels) {
          llvm::outs() << channel << " ";
        }
        llvm::outs() << "\n";
      }
    }

    // Initialize switchbox configs
    for (auto pathOp : device.getOps<CircuitPathOp>()) {
      sbConfigList[pathOp] = {};
    }

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

  // Function to get wire bundles, returns wire bundle pair {a, b}
  std::pair<WireBundle, WireBundle> getWireBundles(TileID a, TileID b) {
    if (a.col == b.col - 1) {
      return {WireBundle::East, WireBundle::West};
    } else if (a.col == b.col + 1) {
      return {WireBundle::West, WireBundle::East};
    } else if (a.row == b.row - 1) {
      return {WireBundle::North, WireBundle::South};
    } else if (a.row == b.row + 1) {
      return {WireBundle::South, WireBundle::North};
    } else {
      llvm::errs() << "Tiles: " << a << " and " << b << 
          " not adjacent, could not acquire wire bundles\n";
      return {};
    }
  }

  int getChannel(TileID tileId, WireBundle bundle, bool isSrc, CircuitPathOp pathOp) {
    auto &chanMap = isSrc ? sbFreeChans.at(tileId).srcChans :
                            sbFreeChans.at(tileId).dstChans;

    if (chanMap.find(bundle) == chanMap.end()) {
      pathOp.emitOpError("Bundle ") << stringifyEnum(bundle) << " not available"
            << " at sb (" << tileId.col << ", " << tileId.row << ")";
    }

    std::set<int> &channels = chanMap[bundle];
    if (channels.empty()) {
      pathOp.emitOpError("Exceeded channels for bundle ") << stringifyEnum(bundle)
          << " at sb: (" << tileId.col << ", " << tileId.row << ")";
    }

    int channel = *channels.begin();
    channels.erase(channels.begin());

    return channel;
  }

  std::vector<std::vector<TileID>> getTilesAlongPath(ArrayRef<IntArray2DAttr> arr3dAttr) {
    std::vector<std::vector<TileID>> hops;

    if (arr3dAttr.empty())
      return hops; // return empty if no hops

    for (auto arr2dAttr : arr3dAttr) {
      std::vector<TileID> hopsPerDst;

      for (auto arr1dAttr : arr2dAttr) {
        assert(arr1dAttr.size() == 2);
        int col = arr1dAttr[0].getInt();
        int row = arr1dAttr[1].getInt();
        hopsPerDst.push_back({col, row});
      }
      hops.push_back(std::move(hopsPerDst));
    }

    return hops;
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

    llvm::outs() << "\t\taddConnection() (" << op.colIndex() << ","
                 << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
                 << inIndex << " -> " << stringifyWireBundle(outBundle)
                 << outIndex << "\n";
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const AIETargetModel &targetModel = device.getTargetModel();
    int maxCol = targetModel.columns();
    int maxRow = targetModel.rows();

    initConfigs(maxCol, maxRow, device, targetModel);
    llvm::outs() << "Begin Path to Routing Pass\n";
    //===------------------------------------------------------------------===//
    // Set up switchbox configs and interconnects for each CircuitPathOp
    //===------------------------------------------------------------------===//
    for (auto pathOp : device.getOps<CircuitPathOp>()) {
      Interconnects interconnects;
      SwitchConfigs &sbConfigs = sbConfigList[pathOp];

      auto srcTile = cast<TileOp>(pathOp.getSource().getDefiningOp());
      TileID srcTileId = {srcTile.colIndex(), srcTile.rowIndex()};
      WireBundle srcBundle = pathOp.getSourceBundle();
      int srcChannel = pathOp.getSourceChannel();
      llvm::ArrayRef<IntArray2DAttr> hops = pathOp.getHopTileIds();
      std::vector<std::vector<TileID>> pathTiles =
          getTilesAlongPath(hops);
      llvm::outs() << "Processing pathOp with Src: " << srcTileId << "\n";
      for (size_t dst_i = 0; dst_i < pathOp.getDests().size(); dst_i++) {
        auto dstTile = cast<TileOp>(pathOp.getDests()[dst_i].getDefiningOp());
        TileID dstTileId = {dstTile.colIndex(), dstTile.rowIndex()};
        WireBundle dstBundle = pathOp.getDestBundle();
        int dstChannel = pathOp.getDestChannels()[dst_i];

        llvm::outs() << "\tSrc: " << srcTileId << " -> "
                     << "Dst[" << dst_i << "]: " << dstTileId << "\n";

        llvm::outs() << "\t\tPath: ";
        for (const auto &hop : pathTiles[dst_i]) {
          llvm::outs() << hop << " ";
        }
        llvm::outs() << "\n";
        // pathTiles is at least [src, dst], loop runs >= 1
        for (size_t i = 0; i < pathTiles[dst_i].size() - 1; i++) {
          const auto &tileA = pathTiles[dst_i][i];
          const auto &tileB = pathTiles[dst_i][i + 1];

          auto [bundleA, bundleB] = getWireBundles(tileA, tileB);
          auto [it, newInsert] = interconnects.try_emplace({tileA, tileB}, Port{});
          Port& pInB = it->second;

          // If new interconnect, initialize ports on both tiles,
          // else reuse first tile's port (multicast)
          if (newInsert) {
            if (i == 0) {
              // if A is src sb, also set input port
              sbConfigs[tileA].addSrc(Port{srcBundle, srcChannel});
            }
            // set output port for tile A
            if (sbConfigs.find(tileA) == sbConfigs.end())
              pathOp.emitOpError("Setting output port for sb with missing input port")
                  << " at sb(" << tileA.col << ", " << tileA.row << ")";
            else
              sbConfigs[tileA].addDst(Port{bundleA, getChannel(tileA, bundleA, false, pathOp)});

            // Initialize input port for tile B
            pInB = Port{bundleB, getChannel(tileB, bundleB, true, pathOp)};
          }
          
          // set input port for tile B
          sbConfigs[tileB].addSrc(pInB);
        }

        // set output port for dst tile
        sbConfigs[dstTileId].addDst(Port{dstBundle, dstChannel});
      }

      //debug
      llvm::outs() << "Finished processing pathOp with Src: " << srcTileId << ", Sb configs:\n";
      for (const auto &[tileId, config] : sbConfigList[pathOp]) {
        llvm::outs() << "\t sb: " << tileId << "\n";
        for (size_t i = 0; i < config.srcs.size(); i++) {
          llvm::outs() << "\t\t" << config.srcs[i] << " -> " 
                       << config.dsts[i] << "\n";
        }
      }
    }
    llvm::outs() << "Building aie connections\n";
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    for (auto pathOp : device.getOps<CircuitPathOp>()) {
      auto srcTile = cast<TileOp>(pathOp.getSource().getDefiningOp());
      TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      auto srcBundle = pathOp.getSourceBundle();
      auto srcChannel = pathOp.getSourceChannel();
      TileID srcSbId = {srcCoords.col, srcCoords.row};

      SwitchConfigs sbConfigs = sbConfigList[pathOp];
      for (const auto &[tileId, setting] : sbConfigs) {
        int col = tileId.col;
        int row = tileId.row;
        SwitchboxOp swOp = getSwitchbox(builder, col, row);

        int shimCh = srcChannel;
        bool isShim = getTile(builder, col, row).isShimNOCorPLTile();

        // This is to handle shim DMAs. They are technically two
        // switchboxes shimmux -> shimsb -> next sb
        if (isShim && tileId == srcSbId) {
          // must be either DMA0 -> N3 or DMA1 -> N7
          shimCh = srcChannel == 0 ? 3 : 7;
          ShimMuxOp shimMuxOp = getShimMux(builder, col);
          addConnection(builder, cast<Interconnect>(shimMuxOp.getOperation()),
                        srcBundle, srcChannel, WireBundle::North, shimCh);
        }
        assert(setting.srcs.size() == setting.dsts.size());
        for (size_t i = 0; i < setting.srcs.size(); i++) {
          Port src = setting.srcs[i];
          Port dest = setting.dsts[i];

          // handle special shim connectivity
          if (isShim && tileId == srcSbId) {
            addConnection(builder, cast<Interconnect>(swOp.getOperation()),
                          WireBundle::South, shimCh, dest.bundle, dest.channel);
          } else if (isShim && (dest.bundle == WireBundle::DMA ||
                                dest.bundle == WireBundle::PLIO ||
                                dest.bundle == WireBundle::NOC)) {
            // shim DMAs at end of flows
            if (dest.bundle == WireBundle::DMA)
              // must be either N2 -> DMA0 or N3 -> DMA1
              shimCh = dest.channel == 0 ? 2 : 3;
            else if (dest.bundle == WireBundle::NOC)
              // must be either N2/3/4/5 -> NOC0/1/2/3
              shimCh = dest.channel + 2;
            else if (dest.bundle == WireBundle::PLIO)
              shimCh = dest.channel;

            ShimMuxOp shimMuxOp = getShimMux(builder, col);
            addConnection(builder, cast<Interconnect>(shimMuxOp.getOperation()),
                          WireBundle::North, shimCh, dest.bundle, dest.channel);
            addConnection(builder, cast<Interconnect>(swOp.getOperation()),
                          src.bundle, src.channel, WireBundle::South, shimCh);
          } else {
            // otherwise, regular switchbox connection
            addConnection(builder, cast<Interconnect>(swOp.getOperation()),
                          src.bundle, src.channel, dest.bundle, dest.channel);
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

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPathToRoutingPass() {
  return std::make_unique<AIEPathToRoutingPass>();
}
