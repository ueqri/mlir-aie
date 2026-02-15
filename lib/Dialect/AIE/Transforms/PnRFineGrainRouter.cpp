#include "aie/Dialect/AIE/Transforms/PnRFineGrainRouter.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "pnr-helper"

int findPrevChannel(SwitchConfig &config, WireBundle dstWB) {
  for (auto &dstPort : config.dsts) {
    auto &[wb, ch] = dstPort;
    if (wb == dstWB) return ch;
  }
  return -1;
}

// returns direction to go from "from" tile to "to" tile
WireBundle getDirToAdjTile(TileID from, TileID to) {
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

void fineGrainRouter::initConfigs(int maxCol, int maxRow, DeviceOp &device,
                  const AIETargetModel &targetModel) {
  // Initialize the channel usage for each switchbox
  for (int col = 0; col < maxCol; col++) {
    for (int row = 0; row < maxRow; row++) {
      TileID tileId{col, row};
      sbFreeChans[tileId] = SwitchFreeChans{};
      auto &switchChans = sbFreeChans[tileId];

      for (int i = 0, e = getMaxEnumValForWireBundle() + 1; i < e; ++i) {
        WireBundle bundle = symbolizeWireBundle(i).value();

        // get all ports into current switchbox
        int numSrcChans =
            targetModel.getNumSourceSwitchboxConnections(col, row, bundle);
        if (numSrcChans == 0 && targetModel.isShimNOCorPLTile(col, row)) {
          // workaround for shimMux
          numSrcChans = targetModel.getNumSourceShimMuxConnections(col, row, bundle);
        }
        for (int channel = 0; channel < numSrcChans; channel++) {
          switchChans.freeSrcChans[bundle][channel] = ChanUsage{};
        }

        // get all ports out of current switchbox
        int numDstChans = targetModel.getNumDestSwitchboxConnections(col, row, bundle);
        if (numDstChans == 0 && targetModel.isShimNOCorPLTile(col, row)) {
          // workaround for shimMux
          numDstChans = targetModel.getNumDestShimMuxConnections(col, row, bundle);
        }
        for (int channel = 0; channel < numDstChans; channel++) {
          switchChans.freeDstChans[bundle][channel] = ChanUsage{};
        }
      }
    }
  }

  // Initialize switchbox configs
  for (auto flow : device.getOps<PnRFlowOp>()) {
    sbConfigs[flow] = {};
  }
  for (auto flow : device.getOps<PnRPktFlowOp>()) {
    pktSbConfigs[flow] = {};
  }
}

void fineGrainRouter::debugPrintFlatPaths(const Path &pathTiles,
                                          const std::vector<HopInfo> &hopInfos, 
                                          size_t numPaths, int opIndex, int packetId) {
  llvm::dbgs() << "\n----------------------------------------\n";
  llvm::dbgs() << "=== Router Report for " << (packetId > -1? "Packet Flow " : "Circuit Flow ")
               << opIndex << " ===\n";
  llvm::dbgs() << "[Flow Trace]\n";
  for (size_t pIdx = 0; pIdx < pathTiles.size(); pIdx++) {
    const auto &path = pathTiles[pIdx];
    llvm::dbgs() << "  Path " << pIdx << ": ";
    for (size_t j = 0; j < path.size(); ++j) {
      llvm::dbgs() << path[j];
      if (j + 1 < path.size())
        llvm::dbgs() << " -> ";
    }
    llvm::dbgs() << "\n";
  }
  llvm::dbgs() << "  Total parallel paths: " << numPaths << "\n";
  if (packetId > -1)
    llvm::dbgs() << "  Packet ID: " << packetId << "\n";
  llvm::dbgs() << "\n[Flat Path Trace]\n";
  for (size_t hopIdx = 0; hopIdx < hopInfos.size(); hopIdx++) {
    const auto &hi = hopInfos[hopIdx];
    llvm::dbgs() << "  Hop " << hopIdx << ":\n";
    for (size_t pIdx = 0; pIdx < numPaths; pIdx++) {
      llvm::dbgs() << "    Path " << pIdx << ": ";
      const auto &tile = hi.tilesAtHop[pIdx];
      const auto &dstPort = hi.dstPorts[pIdx];
      if (hopIdx == 0 && hi.srcPort)
        llvm::dbgs() << "SrcPort(" << hi.srcPort->bundle << ": " << hi.srcPort->channel << ") -> ";
      if (tile) {
        llvm::dbgs() << *tile;
        if (dstPort)
          llvm::dbgs() << " -> DstPort(" << dstPort->bundle << ": " << dstPort->channel << ")";
        else
          llvm::dbgs() << " -> ";
      }
      else
        llvm::dbgs() << "[Path previously ended]";
      llvm::dbgs() << "\n";
    }
  }
}

template <typename FlowOpType>
void fineGrainRouter::debugPrintSbConfigs(FlowOpType op) {
  auto printConfigs = [](const auto &configs, FlowOpType op) {
    auto it = configs.find(op);
    if (it == configs.end())
      llvm_unreachable("No switchbox configs found for this flow");
    const auto &switchConfigs = it->second;
    for (const auto &[tile, cfg] : switchConfigs) {
      llvm::dbgs() << "  " << tile << "\n";
      assert(cfg.srcs.size() == cfg.dsts.size());
      for (size_t i = 0; i < cfg.srcs.size(); i++) {
        const auto &src = cfg.srcs[i];
        const auto &dst = cfg.dsts[i];
        llvm::dbgs() << "    " << src.bundle << ":" << src.channel
                      << " -> " << dst.bundle << ":" << dst.channel << "\n";
      }
    }
  };
  llvm::dbgs() << "\n[Switchbox Configs]\n";
  if constexpr (std::is_same_v<FlowOpType, PnRPktFlowOp>)
    printConfigs(pktSbConfigs, op);
  else if constexpr (std::is_same_v<FlowOpType, PnRFlowOp>)
    printConfigs(sbConfigs, op);
  else
    llvm_unreachable("Invalid op type for debugPrintSbConfigs");
  llvm::dbgs() << "----------------------------------------\n";
}

template <typename FlowOpType>
std::vector<HopInfo> fineGrainRouter::createFlatPaths(FlowOpType flowOp,
                                                      const Path &pathTiles,
                                                      size_t numPaths) {
  // Find the length of the longest path to determine the number of hops.
  size_t maxDepth = 0;
  for (const auto &path : pathTiles) {
    maxDepth = std::max(maxDepth, path.size());
  }

  std::vector<HopInfo> hopInfos(maxDepth);

  // Initialize the inner vectors for each hop.
  for (HopInfo &hi : hopInfos) {
    hi.tilesAtHop.resize(numPaths);
    hi.dstPorts.resize(numPaths);
  }

  // Set the source port ONCE for the very first hop.
  hopInfos[0].srcPort = 
      Port{flowOp.getSourceBundle(), flowOp.getSourceChannel()};

  // Transpose the path data into the hop-based structure.
  for (size_t pIdx = 0; pIdx < numPaths; ++pIdx) {
    const std::vector<TileID> &path = pathTiles[pIdx];
    for (size_t j = 0; j < path.size(); ++j) {
      hopInfos[j].tilesAtHop[pIdx] = path[j];

      // If this is the last tile in the current path, set its destination port.
      if (j + 1 == path.size()) {
        hopInfos[j].dstPorts[pIdx] = 
            Port{flowOp.getDestBundle(), flowOp.getDestChannels()[pIdx]};
      }
    }
  }
  return hopInfos;
}

template <typename FlowOpType>
LogicalResult fineGrainRouter::routeFlow(const AIETargetModel &targetModel, 
                                         FlowOpType flowOp, bool isPacket, 
                                         int opIndex, bool debugPrint,
                                         int packetId) {
  auto pathTiles = flowOp.getTilesAlongPath();
  size_t numPaths = pathTiles.size();

  std::vector<HopInfo> hopInfos = createFlatPaths(flowOp, pathTiles, numPaths);
  if (debugPrint)
    debugPrintFlatPaths(pathTiles, hopInfos, numPaths, opIndex, packetId);
  for (size_t hopIdx = 0; hopIdx < hopInfos.size(); hopIdx++) {
    HopInfo &hi = hopInfos[hopIdx];
    std::set<std::pair<TileID, TileID>> seenEdges;
    std::set<std::tuple<TileID, Port, bool>> portsToUse;
    for (size_t pIdx = 0; pIdx < numPaths; pIdx++) {
      LLVM_DEBUG(llvm::dbgs() << "processing hop " << hopIdx << " path " << pIdx << "\n");
      // Each path may be different length and may have
      // ended early; skip those paths.
      if (!hi.tilesAtHop[pIdx].has_value()) 
        continue;
      TileID tile = *hi.tilesAtHop[pIdx];

      // Determine source bundle and channel (from previous hop's decision
      // if not the first hop)
      WireBundle srcWB;
      int srcChan;
      if (hopIdx == 0) {
        assert(hi.srcPort.has_value());
        srcWB = hi.srcPort->bundle;
        srcChan = hi.srcPort->channel;
      }
      else {
        assert(hopInfos[hopIdx - 1].tilesAtHop[pIdx].has_value());
        TileID prevTile = *hopInfos[hopIdx - 1].tilesAtHop[pIdx];
        srcWB = getDirToAdjTile(tile, prevTile);
        int tmpChan = -1;
        if constexpr (std::is_same_v<FlowOpType, PnRPktFlowOp>)
          tmpChan = findPrevChannel(pktSbConfigs[flowOp][prevTile], getOppositeBundle(srcWB));
        else
          tmpChan = findPrevChannel(sbConfigs[flowOp][prevTile], getOppositeBundle(srcWB));
        if (tmpChan == -1) {
          flowOp.emitOpError() << "Flow " << opIndex << " - Hop " << hopIdx
                << " - Path " << pIdx << ": Previous tile ("
              << prevTile.col << "," << prevTile.row
              << ") was not assigned a dstCh for bundle "
              << stringifyEnum(getOppositeBundle(srcWB)) << " to current tile ("
              << tile.col << "," << tile.row << ")";
          return failure();
        }
        srcChan = tmpChan;
      }

      // Determine destination bundle and channel (from path's next tile
      // location)
      if (!hi.dstPorts[pIdx].has_value()) {
        assert(hopIdx < hopInfos.size() - 1 && "Last hop must have dst port");
        HopInfo &nextHi = hopInfos[hopIdx + 1];
        if (!nextHi.tilesAtHop[pIdx].has_value())
          llvm_unreachable("Next tile must exist if current tile is not a dst");
        TileID nextTile = *nextHi.tilesAtHop[pIdx];

        if (seenEdges.count({tile, nextTile})) {
          LLVM_DEBUG(llvm::dbgs() << "  already routed this edge, skipping\n");
          continue;
        }
          
        seenEdges.insert({tile, nextTile});

        WireBundle dstWB = getDirToAdjTile(tile, nextTile);
        
        // Find candidate channels
        // inter tile channel is mapped to same channel
        // ex. tile(col:0, row:0) N3 -> tile(col:0, row:1) S3
        std::set<int> candidateChans;
        std::set<int> freeDstChans = sbFreeChans[tile].getFreeChans(
            dstWB, true, isPacket);
        std::set<int> freeNextSrcChans = sbFreeChans[nextTile].getFreeChans(
            getOppositeBundle(dstWB), /*isDst*/false, isPacket);
        std::set_intersection(freeDstChans.begin(), freeDstChans.end(),
                              freeNextSrcChans.begin(), freeNextSrcChans.end(),
                              std::inserter(candidateChans, candidateChans.begin()));

        bool isCurrentMem = targetModel.isMemTile(tile.col, tile.row);
        bool isNextMem = targetModel.isMemTile(nextTile.col, nextTile.row);
        if (isCurrentMem && srcWB != WireBundle::DMA) {
          // mem tile passthrough: enforce dst chan == src chan
          LLVM_DEBUG(llvm::dbgs() << 
              "  current tile is mem, enforcing dstChan == srcChan\n");
          if (candidateChans.count(srcChan))
            candidateChans = {srcChan};
          else {
            candidateChans.clear();
            flowOp.emitOpError() << "Flow " << opIndex << " - Hop " << hopIdx
                << " - Path " << pIdx << ": Requires " << stringifyEnum(dstWB) 
                << ":" << srcChan << " at tile (" << tile.col
                << "," << tile.row << ") to next tile ("
                << nextTile.col << "," << nextTile.row << ")";
            return failure();
          }
        }
        else if (isNextMem && !hopInfos[hopIdx + 1].dstPorts[pIdx].has_value()) {
          // next tile is mem passthrough: also intersect with next tile's dst chans
          LLVM_DEBUG(llvm::dbgs() << "  next tile is mem, intersecting with its dstChans\n");
          std::set<int> freeNextDstChans = sbFreeChans[nextTile].getFreeChans(
              dstWB, /*isDst*/true, isPacket);
          std::set<int> tmp;
          std::set_intersection(candidateChans.begin(), candidateChans.end(),
                                freeNextDstChans.begin(), freeNextDstChans.end(),
                                std::inserter(tmp, tmp.begin()));
          candidateChans = std::move(tmp);
        }

        if (candidateChans.empty()) {
          flowOp.emitOpError() << "Flow " << opIndex << " - Hop " << hopIdx
              << " - Path " << pIdx << ": No available dstCh for bundle "
              << stringifyEnum(dstWB) << " at tile (" << tile.col 
              << "," << tile.row << ") to next tile ("
              << nextTile.col << "," << nextTile.row << ")";
          return failure();
        }
        int dstChan = *candidateChans.begin();
        portsToUse.insert({tile, Port{srcWB, srcChan}, false});
        portsToUse.insert({tile, Port{dstWB, dstChan}, true});
        LLVM_DEBUG(llvm::dbgs() << "   tile (" << tile.col << "," << tile.row << "): "
                     << "reserving " << stringifyEnum(srcWB) << ":" << srcChan
                     << " -> " << stringifyEnum(dstWB) << ":" << dstChan << "\n");
        //portsToUse.insert({nextTile, Port{getOppositeBundle(dstWB), dstChan}, false});
        // Record in switchbox config
        if constexpr (std::is_same_v<FlowOpType, PnRPktFlowOp>) {
          pktSbConfigs[flowOp][tile].addSrc(Port{srcWB, srcChan});
          pktSbConfigs[flowOp][tile].addDst(Port{dstWB, dstChan});
        } 
        else {
          sbConfigs[flowOp][tile].addSrc(Port{srcWB, srcChan});
          sbConfigs[flowOp][tile].addDst(Port{dstWB, dstChan});
        }
      }
      else {
        // This is a destination tile. Dst port is already known.
        LLVM_DEBUG(llvm::dbgs() << "  This is a destination tile\n");
        portsToUse.insert({tile, Port{srcWB, srcChan}, false});
        portsToUse.insert({tile, *hi.dstPorts[pIdx], true});
        LLVM_DEBUG(llvm::dbgs() << "   tile (" << tile.col << "," << tile.row << "): "
                     << "reserving " << stringifyEnum(srcWB) << ":" << srcChan
                     << " -> " << stringifyEnum(hi.dstPorts[pIdx]->bundle) 
                     << ":" << hi.dstPorts[pIdx]->channel << "\n");
        // Record in switchbox config
        if constexpr (std::is_same_v<FlowOpType, PnRPktFlowOp>) {
          pktSbConfigs[flowOp][tile].addSrc(Port{srcWB, srcChan});
          pktSbConfigs[flowOp][tile].addDst(*hi.dstPorts[pIdx]);
        } 
        else {
          sbConfigs[flowOp][tile].addSrc(Port{srcWB, srcChan});
          sbConfigs[flowOp][tile].addDst(*hi.dstPorts[pIdx]);
        }
      }
    } // end path loop
    // Mark channels after hop is processed to avoid double-marking shared tiles (multi-cast)
    for (auto const &[tile, port, isDst] : portsToUse) {
      if (!sbFreeChans[tile].markUsed(port.bundle, port.channel, 
                                      isDst, isPacket)) {
        flowOp->emitOpError() << stringifyEnum(port.bundle) << " : " 
                              << port.channel << " at tile (" << tile.col 
                              << "," << tile.row << ") not available";
        return failure();
      }
    }
  }
  if (debugPrint)
    debugPrintSbConfigs(flowOp);
  return success();
}

void TileAnalyzer::initConfigs(DeviceOp &device, const AIETargetModel &targetModel) {
  router->initConfigs(getMaxCol(), getMaxRow(), device, targetModel);
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

LogicalResult TileAnalyzer::routeFlow(DeviceOp &device, const AIETargetModel &targetModel,
                                      bool debugPrint) {
  int opIndex = 0;
  for (auto flowOp : device.getOps<PnRFlowOp>()) {
    if (failed(router->routeFlow(targetModel, flowOp, /*isPacket*/false, opIndex++, debugPrint))) 
      return failure();
  }

  for (auto flowOp : device.getOps<PnRPktFlowOp>()) {
    if (failed(router->routeFlow(targetModel, flowOp, /*isPacket*/true,
                                 opIndex++, debugPrint, flowOp.getPacketId())))
      return failure();
  }
  return success();
}

TileOp TileAnalyzer::getTile(OpBuilder &builder, int col, int row) {
  if (coordToTile.count({col, row})) {
    return coordToTile[{col, row}];
  }
  auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
  coordToTile[{col, row}] = tileOp;
  
  return tileOp;
}

SwitchboxOp TileAnalyzer::getSwitchbox(OpBuilder &builder, int col, int row) {
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

ShimMuxOp TileAnalyzer::getShimMux(OpBuilder &builder, int col) {
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
