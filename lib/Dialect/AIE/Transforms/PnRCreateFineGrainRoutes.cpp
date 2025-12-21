#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "aie/Dialect/AIE/Transforms/PnRFineGrainRouter.h"

#define DEBUG_TYPE "pnr-fine-grain-router"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct PnRFineGrainRouterPass : public PnRFineGrainRouterBase<PnRFineGrainRouterPass> {
  void addConnection(OpBuilder &builder, Interconnect op, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {
    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = builder.saveInsertionPoint();
    builder.setInsertionPoint(b.getTerminator());

    builder.create<ConnectOp>(builder.getUnknownLoc(), inBundle, inIndex,
                              outBundle, outIndex);

    builder.restoreInsertionPoint(point);
    if (clPrintReport) {
      llvm::dbgs() << "  TileID(" << op.colIndex() << ","
                 << op.rowIndex() << ") " << inBundle << ":"
                 << inIndex << " -> " << outBundle << ":"
                 << outIndex << "\n";
    }
  }

  void createFlowPhysicals(DeviceOp device, OpBuilder &builder, TileAnalyzer &analyzer) {
    if (clPrintReport)
      llvm::dbgs() << "[AIE Connect Ops]\n";
    for (auto flowOp : device.getOps<PnRFlowOp>()) {
      auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
      TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      auto srcBundle = flowOp.getSourceBundle();
      auto srcChannel = flowOp.getSourceChannel();
      TileID srcSbId = {srcCoords.col, srcCoords.row};

      const SwitchConfigs &sbConfigs = analyzer.router->sbConfigs[flowOp];
      for (const auto &[tileId, settings] : sbConfigs) {
        int col = tileId.col;
        int row = tileId.row;
        SwitchboxOp swOp = analyzer.getSwitchbox(builder, col, row);

        int shimCh = srcChannel;
        bool isShim = analyzer.getTile(builder, col, row).
            isShimNOCorPLTile();

        // This is to handle shim DMAs. They are technically two
        // switchboxes shimmux -> shimsb -> next sb
        if (isShim && tileId == srcSbId) {
          // must be either DMA0 -> N3 or DMA1 -> N7
          shimCh = srcChannel == 0 ? 3 : 7;
          ShimMuxOp shimMuxOp = analyzer.getShimMux(builder, col);
          addConnection(builder, cast<Interconnect>(shimMuxOp.getOperation()),
                        srcBundle, srcChannel, WireBundle::North, shimCh);
        }
        assert(settings.srcs.size() == settings.dsts.size());
        for (size_t i = 0; i < settings.srcs.size(); i++) {
          Port src = settings.srcs[i];
          Port dest = settings.dsts[i];

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

            ShimMuxOp shimMuxOp = analyzer.getShimMux(builder, col);
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
  }
  
  void createPktFlowPhysicals(DeviceOp device, OpBuilder &builder, TileAnalyzer &analyzer) {
    // logical model for switchboxes
    std::map<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
    // Map from a port and flowID to
    std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
    SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
    DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;

    std::map<TileID, mlir::Operation *> tiles;
    for (auto tileOp : device.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      tiles[{col, row}] = tileOp;
    }

    for (auto flowOp : device.getOps<PnRPktFlowOp>()) {
      int flowID = flowOp.getPacketId();
      const SwitchConfigs &sbConfigs = analyzer.router->pktSbConfigs[flowOp];
      for (const auto &[tile, settings] : sbConfigs) {
        for (size_t i = 0; i < settings.srcs.size(); i++) {
          Port src = settings.srcs[i];
          Port dest = settings.dsts[i];
          Connect conn = {{src.bundle, src.channel},
                          {dest.bundle, dest.channel}};
          if (std::find(switchboxes[tile].begin(), switchboxes[tile].end(),
                        std::make_pair(conn, flowID)) == switchboxes[tile].end()) {
            switchboxes[tile].push_back({std::make_pair(conn, flowID)});
          }
        }
      }
    }

    if (clPrintReport)
      llvm::dbgs() << "\n[Packet Logical Connects]\n";
    for (const auto &[tileId, connects] : switchboxes) {
      if (clPrintReport) {
        llvm::dbgs() << "TileID(" << tileId.col << ", "
                     << tileId.row << ")\n";
      }
      for (const auto &[conn, flowID] : connects) {
        Port sourcePort = conn.src;
        Port destPort = conn.dst;
        auto sourceFlow =
            std::make_pair(std::make_pair(tileId, sourcePort), flowID);

        packetFlows[sourceFlow].push_back({tileId, destPort});
        slavePorts.push_back(sourceFlow);
        if (clPrintReport) {
          llvm::dbgs() << "    PktID " << flowID << ": "
                       << stringifyWireBundle(sourcePort.bundle) << ":"
                       << sourcePort.channel << " -> "
                       << stringifyWireBundle(destPort.bundle) << ":"
                       << destPort.channel << "\n";
        }
      }
    }

    // amsel()
    // masterset()
    // packetrules()
    // rule()

    // Compute arbiter assignments. Each arbiter has four msels.
    // Therefore, the number of "logical" arbiters is 6 x 4 = 24
    // A master port can only be associated with one arbiter

    // Constants for arbiter configuration
    constexpr int INVALID_AMSEL_VALUE = -1;
    constexpr int INVALID_ARBITER_VALUE = -1;

    // A map from Tile and master selectValue to the ports targetted by that
    // master select.
    std::map<std::pair<TileID, int>, SmallVector<Port, 4>> masterAMSels;

    // Track which arbiter each port is assigned to (to prevent conflicts)
    std::map<PhysPort, int> portToArbiter;

    // Count of currently used logical arbiters for each tile.
    DenseMap<Operation *, int> amselValues;
    int numMselsPerArbiter = 4;
    int numArbiters = 6;

    // Get arbiter id from amsel
    auto getArbiterIDFromAmsel = [numArbiters](int amsel) {
      return amsel % numArbiters;
    };
    // Get amsel from arbiter id and msel
    auto getAmselFromArbiterIDAndMsel = [numArbiters](int arbiter, int msel) {
      return arbiter + msel * numArbiters;
    };
    // Get a new unique amsel from masterAMSels on tile op. Prioritize on
    // incrementing arbiter id, before incrementing msel
    auto getNewUniqueAmsel = [&](const std::map<std::pair<TileID, int>, 
                                 SmallVector<Port, 4>> &masterAMSels,
                                 TileOp tileOp) {
      for (int i = 0; i < numMselsPerArbiter; i++)
        for (int a = 0; a < numArbiters; a++)
          if (!masterAMSels.count({tileOp.getTileID(), 
                                   getAmselFromArbiterIDAndMsel(a, i)}))
            return getAmselFromArbiterIDAndMsel(a, i);
      tileOp->emitOpError(
          "tile op has used up all arbiter-msel combinations");
      return INVALID_AMSEL_VALUE;
    };
    // Get a new unique amsel from masterAMSels on tile op with given arbiter id
    auto getNewUniqueAmselPerArbiterID = [&](const std::map<std::pair<TileID, int>, 
                                             SmallVector<Port, 4>> &masterAMSels,
                                             TileOp tileOp, int arbiter) {
      for (int i = 0; i < numMselsPerArbiter; i++)
        if (!masterAMSels.count({tileOp.getTileID(),
                                getAmselFromArbiterIDAndMsel(arbiter, i)}))
          return getAmselFromArbiterIDAndMsel(arbiter, i);
      tileOp->emitOpError("tile op arbiter ")
          << std::to_string(arbiter) << " has used up all its msels";
      return INVALID_AMSEL_VALUE;
    };
    
    // Helper function to assign ports to a given amsel value
    // Updates masterAMSels and portToArbiter, skipping ports already on different
    // arbiters
    auto assignPortsToAmsel = [&](TileID tileId, int amselValue,
                                  const SmallVector<PhysPort, 4> &destinations) {
      int targetArbiter = getArbiterIDFromAmsel(amselValue);
      for (auto dest : destinations) {
        Port port = dest.second;
        PhysPort physPort = {tileId, port};

        // Skip this port if it's already assigned to a different arbiter
        if (portToArbiter.count(physPort) &&
            portToArbiter[physPort] != targetArbiter) {
          LLVM_DEBUG(llvm::dbgs()
                    << "  Skipping port " << stringifyWireBundle(port.bundle)
                    << ":" << port.channel << " - already on arbiter "
                    << portToArbiter[physPort] << ", target is " << targetArbiter
                    << "\n");
          continue;
        }

        masterAMSels[{tileId, amselValue}].push_back(port);
        portToArbiter[physPort] = targetArbiter;
      }
    };

    // Helper function to find existing arbiter assignment for ports in a flow
    // Returns -1 if no existing assignment found
    auto findExistingArbiter =
        [&](TileID tileId, const SmallVector<PhysPort, 4> &destinations) -> int {
      for (auto dest : destinations) {
        Port port = dest.second;
        PhysPort physPort = {tileId, port};
        if (portToArbiter.count(physPort)) {
          return portToArbiter[physPort];
        }
      }
      return INVALID_ARBITER_VALUE;
    };

    // Check all multi-cast flows (same source, same ID). They should be
    // assigned the same arbiter and msel so that the flow can reach all the
    // destination ports at the same time For destination ports that appear in
    // different (multicast) flows, it should have a different <arbiterID, msel>
    // value pair for each flow
    for (const auto &packetFlow : packetFlows) {
      // The Source Tile of the flow
      TileID tileId = packetFlow.first.first.first;
      TileOp tileOp = analyzer.getTile(builder, tileId.col, tileId.row);
      if (amselValues.count(tileOp) == 0)
        amselValues[tileOp] = 0;

      // arb0: 6*0,   6*1,   6*2,   6*3
      // arb1: 6*0+1, 6*1+1, 6*2+1, 6*3+1
      // arb2: 6*0+2, 6*1+2, 6*2+2, 6*3+2
      // arb3: 6*0+3, 6*1+3, 6*2+3, 6*3+3
      // arb4: 6*0+4, 6*1+4, 6*2+4, 6*3+4
      // arb5: 6*0+5, 6*1+5, 6*2+5, 6*3+5

      int amselValue = amselValues[tileOp];
      assert(amselValue < numArbiters && "Could not allocate new arbiter!");

      // Find existing arbiter and amsel assignments for this flow
      // Strategy: Look for existing amsel entries that match the flow's
      // destinations
      // - Complete match: reuse existing amsel
      // - Partial match: create new amsel on same arbiter
      // - No match: create new amsel (on existing arbiter if ports already
      // assigned)
      bool hasMatchingAmselEntry = false;
      int partialMatchArbiterID = INVALID_ARBITER_VALUE;

      // Check if any ports in this flow already have arbiter assignments
      int existingArbiter = findExistingArbiter(tileId, packetFlow.second);

      // Search for matching amsel entries, prioritizing those that match the
      // existing arbiter
      for (const auto &amselEntry : masterAMSels) {
        if (amselEntry.first.first != tileId)
          continue;
        amselValue = amselEntry.first.second;
        int thisArbiter = getArbiterIDFromAmsel(amselValue);

        // If we have an existing arbiter assignment, only consider amsels from
        // that arbiter
        if (existingArbiter != INVALID_ARBITER_VALUE &&
            existingArbiter != thisArbiter) {
          continue;
        }

        // Check if destinations match (completely or partially)
        const SmallVector<Port, 4> &existingPorts =
            masterAMSels[{tileId, amselValue}];
        bool hasOverlap = false;
        bool hasNonOverlap = false;

        for (auto dest : packetFlow.second) {
          Port port = dest.second;
          if (std::find(existingPorts.begin(), existingPorts.end(), port) ==
              existingPorts.end())
            hasNonOverlap = true;
          else
            hasOverlap = true;
        }

        if (hasOverlap) {
          hasMatchingAmselEntry = true;
          // Partial match if some ports don't overlap or sizes differ
          if (hasNonOverlap || existingPorts.size() != packetFlow.second.size())
            partialMatchArbiterID = thisArbiter;
          break;
        }
      }

      if (!hasMatchingAmselEntry) {
        // This packet flow switchbox's output ports completely mismatches with
        // any existing amsel. Creating a new amsel.

        // Determine target arbiter: use existing if available, otherwise allocate
        // based on priority
        int targetArbiter = existingArbiter;

        if (targetArbiter == INVALID_ARBITER_VALUE) {
          amselValue = getNewUniqueAmsel(masterAMSels, tileOp);
        } else {
          // Use existing arbiter to maintain consistency
          amselValue =
              getNewUniqueAmselPerArbiterID(masterAMSels, tileOp, targetArbiter);
          if (amselValue == INVALID_AMSEL_VALUE) {
            // No more msels available on this arbiter - routing conflict
            tileOp->emitOpError("cannot assign flow: arbiter ")
                << targetArbiter
                << " has no free msels, but flow requires this arbiter due to "
                  "existing port assignments";
            return;
          }
        }

        // Update masterAMSels with new amsel, skipping ports already assigned to
        // different arbiters
        assignPortsToAmsel(tileId, amselValue, packetFlow.second);
      } else if (partialMatchArbiterID != INVALID_ARBITER_VALUE) {
        // This packet flow switchbox's output ports partially overlaps with
        // some existing amsel. Create a NEW amsel with the SAME arbiter for this
        // flow. The comment states: "destination ports that appear in different
        // (multicast) flows should have a different <arbiterID, msel> value pair
        // for each flow" but use the same arbiter to maintain the constraint.

        // Use the arbiter we found (which should match existingArbiter if set)
        int targetArbiter = partialMatchArbiterID;
        if (existingArbiter != INVALID_ARBITER_VALUE &&
            existingArbiter != targetArbiter) {
          // Conflict detected - should've been caught earlier, but add safety check
          tileOp->emitOpError(
              "internal error: arbiter conflict in partial match");
          return;
        }

        amselValue =
            getNewUniqueAmselPerArbiterID(masterAMSels, tileOp, targetArbiter);

        // Update masterAMSels with new amsel, skipping ports already assigned to
        // different arbiters
        assignPortsToAmsel(tileId, amselValue, packetFlow.second);
      } else {
        // Complete match - reuse the existing amsel
        // Track arbiter assignments for all ports in this flow
        int arbiter = getArbiterIDFromAmsel(amselValue);
        for (auto dest : packetFlow.second) {
          Port port = dest.second;
          PhysPort physPort = {tileId, port};
          // Update tracking even for reused amsels
          if (!portToArbiter.count(physPort)) {
            portToArbiter[physPort] = arbiter;
          }
        }
      }

      slaveAMSels[packetFlow.first] = amselValue;
      amselValues[tileOp] = getArbiterIDFromAmsel(amselValue);
    }

    // Compute the master set IDs
    // A map from a switchbox output port to its associated amsel values
    std::map<PhysPort, SmallVector<int, 4>> mastersets;
    for (const auto &[physPort, ports] : masterAMSels) {
      TileID tileId = physPort.first;
      int amselValue = physPort.second;
      for (auto port : ports) {
        PhysPort physPort = {tileId, port};
        mastersets[physPort].push_back(amselValue);
      }
    }

    // Validate that each port only has amsels from a single arbiter
    for (const auto &[physPort, amselList] : mastersets) {
      int assignedArbiter = INVALID_ARBITER_VALUE;
      for (auto amsel : amselList) {
        int thisArbiter = getArbiterIDFromAmsel(amsel);
        if (assignedArbiter != INVALID_ARBITER_VALUE &&
            assignedArbiter != thisArbiter) {
          TileID tileId = physPort.first;
          Port port = physPort.second;
          TileOp tileOp = analyzer.getTile(builder, tileId.col, tileId.row);
          tileOp->emitOpError("port ")
              << stringifyWireBundle(port.bundle) << ":" << port.channel
              << " assigned to multiple arbiters: " << assignedArbiter << " and "
              << thisArbiter << " (amsels: " << amselList[0];
          for (size_t i = 1; i < amselList.size(); i++) {
            llvm::errs() << ", " << amselList[i];
          }
          llvm::errs() << ")\n";
          return signalPassFailure();
        }
        assignedArbiter = thisArbiter;
      }
    }
    if (clPrintReport)
      llvm::dbgs() << "\n[Mastersets]\n";
    for (const auto &[physPort, values] : mastersets) {
      TileID tileId = physPort.first;
      WireBundle bundle = physPort.second.bundle;
      int channel = physPort.second.channel;
      if (clPrintReport) {
        llvm::dbgs()
          << "  " << tileId << " " << stringifyWireBundle(bundle)
          << ":" << channel << '\n';
        for (auto value : values)
          llvm::dbgs() << "    Amsel: " << value << '\n';
      }
    }

    // Compute mask values
    // Merging as many stream flows as possible
    // The flows must originate from the same source port and have different IDs
    // Two flows can be merged if they share the same destinations
    SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slaveGroups;
    SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
    while (!workList.empty()) {
      auto slave1 = workList.pop_back_val();
      Port slavePort1 = slave1.first.second;

      bool foundgroup = false;
      for (auto &group : slaveGroups) {
        auto slave2 = group.front();
        if (Port slavePort2 = slave2.first.second; slavePort1 != slavePort2)
          continue;

        bool matched = true;
        auto dests1 = packetFlows[slave1];
        auto dests2 = packetFlows[slave2];
        if (dests1.size() != dests2.size())
          continue;

        for (auto dest1 : dests1) {
          if (std::find(dests2.begin(), dests2.end(), dest1) == dests2.end()) {
            matched = false;
            break;
          }
        }

        if (matched) {
          group.push_back(slave1);
          foundgroup = true;
          break;
        }
      }

      if (!foundgroup) {
        SmallVector<std::pair<PhysPort, int>, 4> group({slave1});
        slaveGroups.push_back(group);
      }
    }

    std::map<std::pair<PhysPort, int>, int> slaveMasks;
    for (const auto &group : slaveGroups) {
      // Iterate over all the ID values in a group
      // If bit n-th (n <= 5) of an ID value differs from bit n-th of another ID
      // value, the bit position should be "don't care", and we will set the
      // mask bit of that position to 0
      int mask[5] = {-1, -1, -1, -1, -1};
      for (auto port : group) {
        int ID = port.second;
        for (int i = 0; i < 5; i++) {
          if (mask[i] == -1)
            mask[i] = ID >> i & 0x1;
          else if (mask[i] != (ID >> i & 0x1))
            mask[i] = 2; // found bit difference --> mark as "don't care"
        }
      }

      int maskValue = 0;
      for (int i = 4; i >= 0; i--) {
        if (mask[i] == 2) // don't care
          mask[i] = 0;
        else
          mask[i] = 1;
        maskValue = (maskValue << 1) + mask[i];
      }
      for (auto port : group)
        slaveMasks[port] = maskValue;
    }

    // Realize the routes in MLIR

    // Update tiles map if any new tile op declaration is needed for constructing
    // the flow.
    for (const auto &swMap : mastersets) {
      TileID tileId = swMap.first.first;
      TileOp tileOp = analyzer.getTile(builder, tileId.col, tileId.row);
      if (std::none_of(tiles.begin(), tiles.end(),
                      [&tileOp](const std::pair<const xilinx::AIE::TileID,
                                                Operation *> &tileMapEntry) {
                        return tileMapEntry.second == tileOp.getOperation();
                      })) {
        tiles[{tileOp.colIndex(), tileOp.rowIndex()}] = tileOp;
      }
    }

    for (auto map : tiles) {
      Operation *tileOp = map.second;
      TileOp tile = cast<TileOp>(map.second);
      TileID tileId = tile.getTileID();

      // Create a switchbox for the routes and insert inside it.
      builder.setInsertionPointAfter(tileOp);
      SwitchboxOp swbox =
          analyzer.getSwitchbox(builder, tile.colIndex(), tile.rowIndex());
      SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                    builder.getUnknownLoc());
      Block &b = swbox.getConnections().front();
      builder.setInsertionPoint(b.getTerminator());

      std::vector<bool> amselOpNeededVector(numMselsPerArbiter * numArbiters);
      for (const auto &map : mastersets) {
        if (tileId != map.first.first)
          continue;

        for (auto value : map.second) {
          amselOpNeededVector[value] = true;
        }
      }
      // Create all the amsel Ops
      std::map<int, AMSelOp> amselOps;
      for (int i = 0; i < numMselsPerArbiter; i++) {
        for (int a = 0; a < numArbiters; a++) {
          auto amselValue = getAmselFromArbiterIDAndMsel(a, i);
          if (amselOpNeededVector[amselValue]) {
            int arbiterID = a;
            int msel = i;
            auto amsel =
                builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
            amselOps[amselValue] = amsel;
          }
        }
      }
      // Create all the master set Ops
      // First collect the master sets for this tile.
      SmallVector<Port, 4> tileMasters;
      for (const auto &map : mastersets) {
        if (tileId != map.first.first)
          continue;
        tileMasters.push_back(map.first.second);
      }
      // Sort them so we get a reasonable order
      std::sort(tileMasters.begin(), tileMasters.end());
      for (auto tileMaster : tileMasters) {
        WireBundle bundle = tileMaster.bundle;
        int channel = tileMaster.channel;
        SmallVector<int, 4> msels = mastersets[{tileId, tileMaster}];
        SmallVector<Value, 4> amsels;
        for (auto msel : msels) {
          assert(amselOps.count(msel) == 1);
          amsels.push_back(amselOps[msel]);
        }

        builder.create<MasterSetOp>(
            builder.getUnknownLoc(), builder.getIndexType(), bundle, channel,
            amsels, nullptr);
      }

      // Generate the packet rules
      DenseMap<Port, PacketRulesOp> slaveRules;
      for (auto group : slaveGroups) {
        builder.setInsertionPoint(b.getTerminator());

        auto port = group.front().first;
        if (tileId != port.first)
          continue;

        WireBundle bundle = port.second.bundle;
        int channel = port.second.channel;
        auto slave = port.second;

        int mask = slaveMasks[group.front()];
        int ID = group.front().second & mask;

        Value amsel = amselOps[slaveAMSels[group.front()]];

        PacketRulesOp packetrules;
        if (slaveRules.count(slave) == 0) {
          packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                      bundle, channel);
          PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                          builder.getUnknownLoc());
          slaveRules[slave] = packetrules;
        } else
          packetrules = slaveRules[slave];

        Block &rules = packetrules.getRules().front();

        // Verify ID mapping against all other rules of the same slave.
        for (auto rule : rules.getOps<PacketRuleOp>()) {
          auto verifyMask = rule.maskInt();
          auto verifyValue = rule.valueInt();
          if ((group.front().second & verifyMask) == verifyValue) {
            rule->emitOpError("can lead to false packet id match for id ")
                << ID << ", which is not supposed to pass through this port.";
            rule->emitRemark("Please consider changing all uses of packet id ")
                << ID << " to avoid deadlock.";
          }
        }

        builder.setInsertionPoint(rules.getTerminator());
        builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
      }
    }

    // Add support for shimDMA
    // From shimDMA to BLI: 1) shimDMA 0 --> North 3
    //                      2) shimDMA 1 --> North 7
    // From BLI to shimDMA: 1) North   2 --> shimDMA 0
    //                      2) North   3 --> shimDMA 1

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
              shimOp = analyzer.getShimMux(builder, tileOp.colIndex());
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
              shimOp = analyzer.getShimMux(builder, tileOp.colIndex());
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
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const AIETargetModel &targetModel = device.getTargetModel();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    int maxCol = targetModel.columns();
    int maxRow = targetModel.rows();
    TileAnalyzer analyzer(maxCol, maxRow);
    analyzer.initConfigs(device, targetModel);

    if (clPrintReport)
      llvm::dbgs() << "=== " << "PnR Fine Grain Router Pass" << " ===\n";
    //===------------------------------------------------------------------===//
    // Set up switchbox configs and interconnects for each flowOp
    //===------------------------------------------------------------------===//
    if (failed(analyzer.routeFlow(device, targetModel, clPrintReport)))
      return signalPassFailure();
    if (clPrintReport) {
      llvm::dbgs() << "\n----------------------------------------\n";
      llvm::dbgs() << "=== Physical Components Report ===\n";
    }
    createFlowPhysicals(device, builder, analyzer);
    createPktFlowPhysicals(device, builder, analyzer);
    if (clPrintReport)
      llvm::dbgs() << "----------------------------------------\n";

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<PnRFlowOp, PnRPktFlowOp>(op))
        opsToErase.insert(op);
    });
    SmallVector<Operation *> sorted{opsToErase.begin(), opsToErase.end()};
    computeTopologicalSorting(sorted);
    for (auto *op : llvm::reverse(sorted))
      op->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>AIE::createPnRFineGrainRouterPass() {
  return std::make_unique<PnRFineGrainRouterPass>();
}