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

  void addSrc(Port src) {srcs.push_back(src); }
  void addDst(Port dst) {dsts.push_back(dst); }
};

using SwitchConfigs = std::map<TileID, SwitchConfig>;

// Track channel usage for each switchbox
using SwitchFreeChans = struct SwitchFreeChans {
  std::map<WireBundle, std::set<int>> availSrcChans;
  std::map<WireBundle, std::set<int>> availDstChans;
};

using Interconnects = std::map<std::pair<TileID, TileID>, Port>;

struct HopInfo {
  TileID tile;
  HopInfo *prevInfo = nullptr;
  std::vector<HopInfo*> nextInfos;
  int dstForPath = -1; // -1 = not a dst, otherwise index of path
  std::optional<std::pair<WireBundle, int>> srcPort;
  DenseMap<WireBundle, int> dstChans;
};

struct Key {
  size_t idx;
  TileID tile;

  bool operator==(const Key &other) const {
      return idx == other.idx && tile == other.tile;
  }
};

struct KeyHash {
  size_t operator()(const Key &k) const noexcept {
    size_t h1 = std::hash<size_t>{}(k.idx);
    size_t h2 = std::hash<int>{}(k.tile.col);
    size_t h3 = std::hash<int>{}(k.tile.row);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

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

          sbFreeChans[tileId].availSrcChans[bundle] = channelSet;
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
          sbFreeChans[tileId].availDstChans[bundle] = channelSet;
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
  
  // returns direction to go from "from" tile to "to" tile
  WireBundle getWireBundleToTo(TileID from, TileID to) {
    int dCol = to.col - from.col;
    int dRow = to.row - from.row;

    if (dCol == 0 && dRow > 0) return WireBundle::North;  
    if (dCol == 0 && dRow < 0) return WireBundle::South;  
    if (dRow == 0 && dCol > 0) return WireBundle::East;  
    if (dRow == 0 && dCol < 0) return WireBundle::West;
    llvm_unreachable("Tiles not adjacent");  
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

  void setFlatPaths(CircuitPathOp op, std::vector<std::vector<HopInfo*>> &flatPaths,
                    std::vector<std::unique_ptr<HopInfo>> &storage,
                    std::unordered_map<Key, HopInfo*, KeyHash> &hopCache) {
    size_t numPaths = op.getTilesAlongPath().size();
    std::vector<std::vector<TileID>> pathTiles = op.getTilesAlongPath();
    for (size_t pIdx = 0; pIdx < numPaths; pIdx++) {
      std::vector<TileID> &path = pathTiles[pIdx];
      HopInfo *prev = nullptr;

      for (size_t j = 0; j < path.size(); ++j) {
        TileID tile = path[j];
        bool dst = (j + 1 == path.size());

        Key key{j, tile};
        HopInfo *hi = nullptr;
        auto it = hopCache.find(key);
        if (it != hopCache.end()) {
          hi = it->second;
        } else {
          storage.push_back(std::make_unique<HopInfo>());
          hi = storage.back().get();
          hi->tile = tile;
          hi->nextInfos.resize(numPaths, nullptr);
          hopCache[key] = hi;
          flatPaths[j].push_back(hi); // traversal by hop
        }
        hi->prevInfo = prev;

        // First hop: all paths share same input bundle/channel
        if (j == 0) 
          hi->srcPort = std::make_pair(op.getSourceBundle(), op.getSourceChannel());

        // Destination hop: per-path output bundle/channel
        if (dst) {
          if (hi->dstForPath != -1)
            llvm_unreachable("One circuit path cannot have two destinations at same tile");
          hi->dstForPath = (int)pIdx;
          hi->dstChans[op.getDestBundle()] = op.getDestChannels()[pIdx];
        }
        if (prev) prev->nextInfos[pIdx] = hi;
        prev = hi;
      }
    }
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
      // Debug //
      auto srcTile = cast<TileOp>(pathOp.getSource().getDefiningOp());
      llvm::dbgs() << "Routing Src: " << "(" << srcTile.colIndex() << "," << srcTile.rowIndex() << ") ";
      for (size_t dst_i = 0; dst_i < pathOp.getDests().size(); dst_i++) {
        auto dstTile = cast<TileOp>(pathOp.getDests()[dst_i].getDefiningOp());
        if (dst_i != 0) llvm::dbgs() << "                   ";
        llvm::dbgs() << "-> (" << dstTile.colIndex() << "," << dstTile.rowIndex() << ") | Path: ";
        auto path = pathOp.getTilesAlongPath()[dst_i];
        for (auto &tile : path) {
          llvm::dbgs() << " (" << tile.col << "," << tile.row << ") ";
        }
        llvm::dbgs() << "\n";
      }
      /////////////////////////////////
      size_t numPaths = pathOp.getTilesAlongPath().size();
      size_t maxDepth = 0;
      for (auto &path : pathOp.getTilesAlongPath())
          maxDepth = std::max(maxDepth, path.size());

      std::vector<std::unique_ptr<HopInfo>> storage;
      std::unordered_map<Key, HopInfo*, KeyHash> hopCache;
      std::vector<std::vector<HopInfo*>> flatPaths(maxDepth);  // outer vector = hop index
      setFlatPaths(pathOp, flatPaths, storage, hopCache);
      
      for (size_t h = 0; h < maxDepth; h++) {
        for (auto &hi : flatPaths[h]) {
          //debug
          // print hopinfo
          llvm::dbgs() << "\tHop " << h << ":\n";
          for (auto &hi : flatPaths[h]) {
            llvm::dbgs() << "\t\tTile (" << hi->tile.col << "," << hi->tile.row << ") ";
            if (hi->prevInfo)
              llvm::dbgs() << " prev: (" << hi->prevInfo->tile.col << "," << hi->prevInfo->tile.row << ") ";
            else
              llvm::dbgs() << " prev: (null) ";
            llvm::dbgs() << " nexts: ";
            for (auto *n : hi->nextInfos) {
              if (n)
                llvm::dbgs() << "(" << n->tile.col << "," << n->tile.row << ") ";
              else
                llvm::dbgs() << "(null) ";
            }
            if (hi->srcPort.has_value()) {
              auto [wb, ch] = hi->srcPort.value();
              llvm::dbgs() << " srcPort: " << stringifyEnum(wb) << ch << " ";
            }
            llvm::dbgs() << " dstForPath: " << hi->dstForPath << " ";
            llvm::dbgs() << " dstChans: ";
            for (auto &[wb, ch] : hi->dstChans) {
              llvm::dbgs() << stringifyEnum(wb) << ch << " ";
            }
            llvm::dbgs() << "\n";
          }
          // end debug

          // start with input bundle and channel
          HopInfo *prev = hi->prevInfo;
          WireBundle srcWB;
          int srcChan;
          if (h == 0) {
            assert (hi->srcPort.has_value());
            std::tie(srcWB, srcChan) = hi->srcPort.value();
          }
          else {
            srcWB = getWireBundleToTo(hi->tile, prev->tile);
            auto it = prev->dstChans.find(getOppositeBundle(srcWB));
            if (it == prev->dstChans.end()) {
              pathOp.emitOpError("Previous tile (")
                  << prev->tile.col << "," << prev->tile.row
                  << ") was not assigned an outCh for bundle "
                  << stringifyEnum(getOppositeBundle(srcWB)) << " to current tile ("
                  << hi->tile.col << "," << hi->tile.row << ")";
              return signalPassFailure();
            }
            srcChan = it->second;
          }
          // // sanity check last outCh is available as inCh
          // auto &srcFree = sbFreeChans[hi->tile].availSrcChans[srcWB];
          // if (srcFree.count(srcChan) == 0) {
          //   pathOp.emitOpError("Sanity: Required inCh ") << srcChan << " for bundle "
          //       << stringifyEnum(srcWB) << " at tile ("
          //       << hi->tile.col << "," << hi->tile.row << ") not available";
          //   return signalPassFailure();
          // }
          
          bool isMem = targetModel.isMemTile(hi->tile.col, hi->tile.row);
          // Work on output channels
          if (isMem && srcWB != WireBundle::DMA) {
            // mem tile: out channels must be same as in channel
            int dstChan = srcChan;
            std::set<TileID> examinedTiles;
            for (size_t pIdx = 0; pIdx < numPaths; pIdx++) {
              if ((int)pIdx == hi->dstForPath) continue;
              if (hi->nextInfos[pIdx]) {
                TileID nextTile = hi->nextInfos[pIdx]->tile;
                if (examinedTiles.count(nextTile)) continue;
                examinedTiles.insert(nextTile);
                WireBundle dstWB = getWireBundleToTo(hi->tile, nextTile);
                auto &dstFree = sbFreeChans[hi->tile].availDstChans[dstWB];
                if (dstFree.count(dstChan) == 0) {
                  pathOp.emitOpError("isNonSrcMem: Required outCh ") << dstChan << " for bundle "
                      << stringifyEnum(dstWB) << " at tile ("
                      << hi->tile.col << "," << hi->tile.row << ") not available";
                  return signalPassFailure();
                }
                auto &nextSrcFree = sbFreeChans[hi->nextInfos[pIdx]->tile]
                                        .availSrcChans[getOppositeBundle(dstWB)];
                if (nextSrcFree.count(dstChan) == 0) {
                  pathOp.emitOpError("isNonSrcMem: Required inCh ") << dstChan << " for bundle "
                      << stringifyEnum(getOppositeBundle(dstWB)) << " at tile ("
                      << nextTile.col << "," << nextTile.row << ") not available";
                  return signalPassFailure();
                }
                hi->dstChans[dstWB] = dstChan;
              } 
            }
          } else {
            // non-mem tile or mem tile as source tile: out channels can be different
            // if next hop is a mem tile and it's not a dst, we need to consider
            // intersect outCh, nextInCh, nextOutCh
            // otherwise intersect outCh, nextInCh
            std::set<TileID> examinedTiles;
            for (size_t pIdx = 0; pIdx < numPaths; pIdx++) {
              if ((int)pIdx == hi->dstForPath) continue;
              if (hi->nextInfos[pIdx]) {
                TileID nextTile = hi->nextInfos[pIdx]->tile;
                bool last = std::all_of(hi->nextInfos[pIdx]->nextInfos.begin(), hi->nextInfos[pIdx]->nextInfos.end(),
                                        [&](HopInfo *n) { return n == nullptr; });
                if (!last && hi->nextInfos[pIdx]->dstForPath == (int)pIdx) continue;
                WireBundle dstWB = getWireBundleToTo(hi->tile, nextTile);
                auto &dstFree = sbFreeChans[hi->tile].availDstChans[dstWB];
                auto &nextSrcFree = sbFreeChans[nextTile].availSrcChans[getOppositeBundle(dstWB)];
                std::set<int> intersection;
                std::set_intersection(dstFree.begin(), dstFree.end(),
                                      nextSrcFree.begin(), nextSrcFree.end(),
                                      std::inserter(intersection, intersection.begin()));
                if (intersection.empty()) {
                  pathOp.emitOpError("No available outCh for bundle ")
                      << stringifyEnum(dstWB) << " at tile ("
                      << hi->tile.col << "," << hi->tile.row << ") to next tile ("
                      << nextTile.col << "," << nextTile.row << ")";
                  return signalPassFailure();
                }
                // If next hop is a mem tile and it's not the dst of that path,
                // further intersect with next hop's outCh`
                if (targetModel.isMemTile(nextTile.col, nextTile.row) && 
                    hi->nextInfos[pIdx]->dstForPath != (int)pIdx) {
                  std::set<int> tmp;
                  auto &nextDstFree = sbFreeChans[nextTile].availDstChans[dstWB];
                  std::set_intersection(intersection.begin(), intersection.end(),
                                        nextDstFree.begin(), nextDstFree.end(),
                                        std::inserter(tmp, tmp.begin()));
                  intersection = std::move(tmp);
                  if (intersection.empty()) {
                    pathOp.emitOpError("No available outCh for bundle ")
                        << stringifyEnum(dstWB) << " at tile ("
                        << hi->tile.col << "," << hi->tile.row << ") to next tile ("
                        << nextTile.col << "," << nextTile.row << ")"
                        << " because outCh unavailable at next tile";
                    return signalPassFailure();
                  }
                }
                int dstChan = *intersection.begin();
                hi->dstChans[dstWB] = dstChan;
              }
            }
          }
          // mark tile srcChan as used
          auto &srcFree = sbFreeChans[hi->tile].availSrcChans[srcWB];
          srcFree.erase(srcChan);
          // Add switchbox config
          for (auto &[dstWB, dstChan] : hi->dstChans) {
            sbConfigList[pathOp][hi->tile].addSrc(Port{srcWB, srcChan});
            sbConfigList[pathOp][hi->tile].addDst(Port{dstWB, dstChan});
            llvm::dbgs() << "\t\tAdded config at (" << hi->tile.col << "," << hi->tile.row
                         << ") " << stringifyEnum(srcWB) << srcChan << " -> " 
                         << stringifyEnum(dstWB) << dstChan << "\n";
            // mark tile dstChan as used
            auto &dstFree = sbFreeChans[hi->tile].availDstChans[dstWB];
            dstFree.erase(dstChan);
          }
        }
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
