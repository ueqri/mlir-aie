#ifndef PNR_ROUTING_H
#define PNR_ROUTING_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

namespace xilinx::AIE {

using Path = std::vector<std::vector<TileID>>;
typedef std::pair<TileID, Port> PhysPort;

// ONE switchbox setting for ONE flow
using SwitchConfig = struct SwitchConfig {
  std::vector<Port> srcs;
  std::vector<Port> dsts;

  void addSrc(Port src) {srcs.push_back(src); }
  void addDst(Port dst) {dsts.push_back(dst); }
};
using SwitchConfigs = std::map<TileID, SwitchConfig>;

// Track channel usage for each switchbox channel
struct ChanUsage {
  bool usedByCircuit = false;             // set when a circuit path reserves this channel
  bool freeForCircuit() const { return !usedByCircuit; }
  bool freeForPacket()  const { return !usedByCircuit; } // packets may share with other packets
};

using SwitchFreeChans = struct SwitchFreeChans {
  // map bundle -> map channel -> usage info
  std::map<WireBundle, std::map<int, ChanUsage>> freeSrcChans;
  std::map<WireBundle, std::map<int, ChanUsage>> freeDstChans;

  std::set<int> getFreeChans(WireBundle wb, bool isDst, bool isPacket) {
    std::set<int> freeChans;
    auto &mapRef = isDst ? freeDstChans : freeSrcChans;
    for (auto &[ch, usage] : mapRef[wb]) {
      if (isPacket ? usage.freeForPacket() : usage.freeForCircuit())
        freeChans.insert(ch);
    }
    return freeChans;
  }

  bool markUsed(WireBundle wb, int ch, bool isDst, bool isPacket) {
    auto &mapRef = isDst ? freeDstChans : freeSrcChans;
    ChanUsage &u = mapRef[wb][ch];
    if (isPacket) {
      // We assume pkt-switched channels can be used by any packet flow
    } else {
      // circuit flow
      if (u.usedByCircuit) return false;
      u.usedByCircuit = true;
    }
    return true;
  }
};

struct HopInfo {
  llvm::SmallVector<std::optional<TileID>> tilesAtHop; // tiles at this hop (for branching paths)
  std::optional<Port> srcPort;
  llvm::SmallVector<std::optional<Port>> dstPorts;
};

class fineGrainRouter {
public:
  std::map<PnRFlowOp, SwitchConfigs> sbConfigs;
  std::map<PnRPktFlowOp, SwitchConfigs> pktSbConfigs;
  std::map<TileID, SwitchFreeChans> sbFreeChans;

  void initConfigs(int maxCol, int maxRow, DeviceOp &device, 
                   const AIETargetModel &targetModel);
  template <typename FlowOpType> 
  std::vector<HopInfo> createFlatPaths(FlowOpType op,
                                       const Path &pathTiles,
                                       size_t numPaths);
  template <typename FlowOpType> 
  llvm::LogicalResult routeFlow(const AIETargetModel &targetModel, 
                                FlowOpType flowOp, 
                                bool isPacket,
                                int opIndex,
                                int packetId = -1);
  void debugPrintFlatPaths(const Path &pathTiles,
                           const std::vector<HopInfo> &hopInfos, 
                           size_t numPaths, int opIndex, int packetId);
  template <typename FlowOpType>
  void debugPrintSbConfigs(FlowOpType op);
};

class TileAnalyzer {
public:
  int maxCol, maxRow;
  std::shared_ptr<fineGrainRouter> router;
  llvm::DenseMap<TileID, TileOp> coordToTile;
  llvm::DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  llvm::DenseMap<TileID, ShimMuxOp> coordToShimMux;

  TileAnalyzer(int maxCol, int maxRow) :  maxCol(maxCol), maxRow(maxRow) {
    router = std::make_shared<fineGrainRouter>();
  }

  int getMaxCol() const { return maxCol; }
  int getMaxRow() const { return maxRow; }

  void initConfigs(DeviceOp &device, const AIETargetModel &targetModel);
  llvm::LogicalResult routeFlow(DeviceOp &device, const AIETargetModel &targetModel);

  TileOp getTile(mlir::OpBuilder &builder, int col, int row);

  SwitchboxOp getSwitchbox(mlir::OpBuilder &builder, int col, int row);

  ShimMuxOp getShimMux(mlir::OpBuilder &builder, int col);
};
} // namespace xilinx::AIE
#endif