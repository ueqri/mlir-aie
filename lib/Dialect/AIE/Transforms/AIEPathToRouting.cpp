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

  WireBundle getOppositeBundle(WireBundle bundle) {
    switch (bundle) {
      case WireBundle::North: return WireBundle::South;
      case WireBundle::South: return WireBundle::North;
      case WireBundle::East: return WireBundle::West;
      case WireBundle::West: return WireBundle::East;
      default:
        llvm_unreachable("Invalid bundle for opposite");
    }
  }

  int getChannel(TileID tileId, WireBundle bundle, bool isSrc, CircuitPathOp pathOp,
                 const AIETargetModel &targetModel) {
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
    channels.erase(channel);
    if (targetModel.isMemTile(tileId.col, tileId.row) && !isSrc) {
      auto &otherChanMap = sbFreeChans.at(tileId).srcChans;
      otherChanMap[getOppositeBundle(bundle)].erase(channel);
    }

    return channel;
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
    //===------------------------------------------------------------------===//
    // Set up switchbox configs and interconnects for each CircuitPathOp
    //===------------------------------------------------------------------===//
    for (auto pathOp : device.getOps<CircuitPathOp>()) {
      Interconnects interconnects;
      SwitchConfigs &sbConfigs = sbConfigList[pathOp];

      WireBundle srcBundle = pathOp.getSourceBundle();
      int srcChannel = pathOp.getSourceChannel();
      std::vector<std::vector<TileID>> pathTiles = pathOp.getTilesAlongPath();
      
      auto srcTile = cast<TileOp>(pathOp.getSource().getDefiningOp());
      TileID srcTileId = {srcTile.colIndex(), srcTile.rowIndex()};
      llvm::dbgs() << "Processing pathOp with Src: " << srcTileId << "\n";
      
      auto firstInnerMemTileIndex = [&](const std::vector<TileID> &path) {
        for (size_t i = 1; i + 1 < path.size(); i++) {  // skip src and dst
          TileID t = path[i];
          if (targetModel.isMemTile(t.col, t.row)) {
            return static_cast<int>(i);
          }
        }
        return std::numeric_limits<int>::max();
      };

      std::vector<std::pair<std::vector<TileID>, int>> pathChanPairs;
      for (size_t i = 0; i < pathTiles.size(); i++) {
        pathChanPairs.emplace_back(pathTiles[i], pathOp.getDestChannels()[i]);
      }

      std::sort(pathChanPairs.begin(), pathChanPairs.end(),
          [&](auto &a, auto &b) {
            return firstInnerMemTileIndex(a.first) <
                   firstInnerMemTileIndex(b.first);
      });

      pathTiles.clear();
      std::vector<int> destChannels;
      for (auto &pc : pathChanPairs) {
        pathTiles.push_back(std::move(pc.first));
        destChannels.push_back(pc.second);
      }

      for (size_t dst_i = 0; dst_i < pathOp.getDests().size(); dst_i++) {
        TileID dstTileId = pathTiles[dst_i].back();
        WireBundle dstBundle = pathOp.getDestBundle();
        int dstChannel = destChannels[dst_i];

        llvm::dbgs() << "\tSrc: " << srcTileId << " -> "
                      << "Dst[" << dst_i << "]: " << dstTileId << "\n";


        llvm::dbgs() << "\t\tPath: ";
          for (const auto &hop : pathTiles[dst_i])
            llvm::dbgs() << hop << " ";
          llvm::dbgs() << "\n";

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

            // Initialize input port for tile B
            int bChannel = getChannel(tileB, bundleB, true, pathOp, targetModel);
            pInB = Port{bundleB, bChannel};

            if (sbConfigs.find(tileA) == sbConfigs.end())
              pathOp.emitOpError("Setting output port for sb with missing input port")
                  << " at sb(" << tileA.col << ", " << tileA.row << ")";

            // set output port for tile A
            if (targetModel.isMemTile(tileB.col, tileB.row) && bundleB != WireBundle::DMA) {
              llvm::dbgs() << "\t\tInter-mem case: ";
              auto &chanMap = sbFreeChans.at(tileA).dstChans;
              if (chanMap.find(bundleA) == chanMap.end()) {
                pathOp.emitOpError("Bundle ") << stringifyEnum(bundleA) << " not available"
                    << " at sb (" << tileA.col << ", " << tileA.row << ")";
              }
              std::set<int> &channels = chanMap[bundleA];
              if (channels.find(bChannel) != channels.end()) 
                channels.erase(bChannel);
              else {
                pathOp.emitOpError("Channel ") << bChannel << " for bundle "
                    << stringifyEnum(bundleA) << " at sb (" << tileA.col << ", "
                    << tileA.row << ") already used";
              }

              sbConfigs[tileA].addDst(Port{bundleA, bChannel});
            }
            else {
              sbConfigs[tileA].addDst(
                  Port{bundleA, getChannel(tileA, bundleA, false, pathOp, targetModel)}
              );
            }
          }
          
          // set input port for tile B
          sbConfigs[tileB].addSrc(pInB);
        }

        // set output port for dst tile
        sbConfigs[dstTileId].addDst(Port{dstBundle, dstChannel});
      }
    }
    llvm::dbgs() << "Building AIE routing connections\n";
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
