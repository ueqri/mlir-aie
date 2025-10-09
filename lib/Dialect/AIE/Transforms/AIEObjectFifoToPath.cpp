#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"

#include "json.hpp"
#include <variant>
#include <fstream>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using json = nlohmann::ordered_json;

#define DEBUG_TYPE "aie-objectFifo-to-path"

#define LOOP_VAR_DEPENDENCY (-2)

//===----------------------------------------------------------------------===//
// Lock Analysis
//===----------------------------------------------------------------------===//
class LockAnalysis {
  DenseMap<std::pair<Value, int>, int> locksPerTile;

public:
  LockAnalysis(DeviceOp &device) {
    // go over the locks created for each tile and update the index in
    // locksPerTile
    device.walk([&](LockOp lockOp) {
      auto tile = lockOp.getTile();
      auto lockID = lockOp.getLockIDValue();
      locksPerTile[{tile, lockID}] = 1;
    });
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(TileOp &tileOp) {
    const auto &targetModel = getTargetModel(tileOp);
    for (unsigned i = 0;
         i < targetModel.getNumLocks(tileOp.getCol(), tileOp.getRow()); i++)
      if (int usageCnt = locksPerTile[{tileOp, i}]; usageCnt == 0) {
        locksPerTile[{tileOp, i}] = 1;
        return i;
      }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// DMA Channel Analysis
//===----------------------------------------------------------------------===//
class DMAChannelAnalysis {
  DenseMap<std::tuple<Value, DMAChannelDir, int>, int> channelsPerTile;

public:
  DMAChannelAnalysis(DeviceOp &device) {
    // go over the channels used for each tile and update channel map
    for (auto memOp : device.getOps<MemOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                           op.getChannelIndex()}] = 1;
        }
      }
    }
    for (auto memOp : device.getOps<MemTileDMAOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                           op.getChannelIndex()}] = 1;
        }
      }
    }
    for (auto memOp : device.getOps<ShimDMAOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          channelsPerTile[{memOp.getTile(), op.getChannelDir(),
                           op.getChannelIndex()}] = 1;
        }
      }
    }
  }

  /// Given a tile and DMAChannelDir, returns next usable channel index for
  /// that tile.
  int getDMAChannelIndex(TileOp tileOp, DMAChannelDir dir) {
    int maxChannelNum = 0;
    if (dir == DMAChannelDir::MM2S)
      maxChannelNum = tileOp.getNumSourceConnections(WireBundle::DMA);
    else
      maxChannelNum = tileOp.getNumDestConnections(WireBundle::DMA);
    for (int i = 0; i < maxChannelNum; i++)
      if (int usageCnt = channelsPerTile[{tileOp.getResult(), dir, i}];
          usageCnt == 0) {
        channelsPerTile[{tileOp.getResult(), dir, i}] = 1;
        return i;
      }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// Create objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoToPathPass : public AIEObjectFifoToPathBase<AIEObjectFifoToPathPass> {
  DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>>
      buffersPerFifo; // maps each objFifo to its corresponding buffer
  DenseMap<ObjectFifoCreateOp, std::vector<ExternalBufferOp>>
      externalBuffersPerFifo; // maps each objFifo to its corresponding
  // external buffers
  DenseMap<ObjectFifoCreateOp, std::vector<std::vector<LockOp>>>
      locksPerFifo; // maps each objFifo to its corresponding locks
  std::vector<std::pair<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>>
      splitDmaFifos; // maps each objFifo between non-adjacent tiles to its
  // corresponding consumer objectFifos
  std::vector<std::pair<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>>
      splitNbrFifos; // maps orig objFifo to sink-based consumer objFifos
  DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp>
      objFifoLinks; // maps each ObjectFifoLinkOp to objFifo whose elements
  // have been created and should be used
  std::vector<ObjectFifoCreateOp> originalFifoOps; // list of original
  // ObjectFifoCreateOps in the device
  DenseMap<TileOp, int> pktFlowChannelPerTile; // maps each tile to the index
  // of the channel used for packet flow

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * -1 if the shared memory module is that of the first input tile,
  ///   * 1 if it is that of the second input tile,
  ///   * 0 is no memory module is shared.
  bool isSharedMemory(TileOp a, TileOp b, int *share_direction) {
    const auto &targetModel = getTargetModel(a.getOperation());

    if ((a.isShimTile() && !b.isShimTile()) ||
        (!a.isShimTile() && b.isShimTile())) {
      *share_direction = 0;
      return false;
    }
    if ((targetModel.isMemTile(a.getCol(), a.getRow()) &&
         !targetModel.isMemTile(b.getCol(), b.getRow())) ||
        (!targetModel.isMemTile(a.getCol(), a.getRow()) &&
         targetModel.isMemTile(b.getCol(), b.getRow()))) {
      *share_direction = 0;
      return false;
    }
    bool rightShared = targetModel.isLegalMemAffinity(
        a.colIndex(), a.rowIndex(), b.colIndex(), b.rowIndex());

    bool leftShared = targetModel.isLegalMemAffinity(
        b.colIndex(), b.rowIndex(), a.colIndex(), a.rowIndex());

    if (leftShared)
      *share_direction = -1;
    else if (rightShared)
      *share_direction = 1;
    else
      *share_direction = 0;

    return leftShared || rightShared;
  }

  /// Function to retrieve ObjectFifoLinkOp of ObjectFifoCreateOp,
  /// if it belongs to one.
  std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
    auto device = op->getParentOfType<DeviceOp>();
    for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
      for (ObjectFifoCreateOp in : linkOp.getInputObjectFifos())
        if (in == op)
          return {linkOp};
      for (ObjectFifoCreateOp out : linkOp.getOutputObjectFifos())
        if (out == op)
          return {linkOp};
    }
    return {};
  }

  ObjectFifoCreateOp
  createObjectFifo(OpBuilder &builder, AIEObjectFifoType datatype,
                   std::string name, Value prodTile, Value consTile,
                   Attribute depth, BDDimLayoutArrayAttr dimensionsToStream,
                   BDDimLayoutArrayArrayAttr dimensionsFromStreamPerConsumer,
                   bool allocate = false) {
    auto ofName = builder.getStringAttr(name);
    auto fifo = builder.create<ObjectFifoCreateOp>(
        builder.getUnknownLoc(), ofName, prodTile, consTile, depth, datatype,
        dimensionsToStream, dimensionsFromStreamPerConsumer);
    if (allocate) {
      auto symRef = FlatSymbolRefAttr::get(builder.getContext(), fifo.getSymName());
      builder.create<ObjectFifoAllocateOp>(builder.getUnknownLoc(), symRef, 
                                           ArrayRef<Value>{consTile});
    }
    return fifo;
  }

  /// Function used to create objectFifo locks 
  /// Called by createObjectFifoElements().
  std::vector<std::vector<LockOp>> createObjectFifoLocks(OpBuilder &builder,
                                            LockAnalysis &lockAnalysis,
                                            ObjectFifoCreateOp op, int numElem,
                                            int joinDistribFactor,
                                            TileOp creation_tile,
                                            int repeatCount) {
    std::vector<std::vector<LockOp>> allLocks;
    if (op.getDisableSynchronization())
      return allLocks;
    // if shimTile external buffers are collected from input code
    // create as many locks as there are external buffers
    if (creation_tile.isShimTile()) {
      numElem = 1;
      if (!externalBuffersPerFifo[op].empty())
        numElem = externalBuffersPerFifo[op].size();
    }
    int numConsumers = 1;
    std::optional<ObjectFifoAllocateOp> opAlloc = getOptionalAllocateOp(op);
    if (opAlloc.has_value() && opAlloc->getNumSrcDelegates() > 1)
      numConsumers = op.getConsumerTiles().size();
    // create corresponding aie2 locks
    for (int c = 0; c < numConsumers; c++) {
      std::vector<LockOp> locks;
      std::string consSuffix = "";
      if (numConsumers > 1)
        consSuffix = "_nbr_src_" + std::to_string(c);
      for (int i = 0; i < joinDistribFactor; i++) {
        auto initValues = op.getInitValues().has_value()
                              ? op.getInitValues().value().size()
                              : 0;
        int prodLockID = lockAnalysis.getLockID(creation_tile);
        assert(prodLockID >= 0 && "No more locks to allocate!");
        int prodLockValue = (numElem - initValues) * repeatCount;
        auto prodLock = builder.create<LockOp>(
            builder.getUnknownLoc(), creation_tile, prodLockID, prodLockValue);
        prodLock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + consSuffix + 
                                  "_prod_lock_" + std::to_string(i)));
        locks.push_back(prodLock);

        int consLockID = lockAnalysis.getLockID(creation_tile);
        assert(consLockID >= 0 && "No more locks to allocate!");
        int consLockValue = initValues * repeatCount;
        auto consLock = builder.create<LockOp>(
            builder.getUnknownLoc(), creation_tile, consLockID, consLockValue);
        consLock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + consSuffix +
                                  "_cons_lock_" + std::to_string(i)));
        locks.push_back(consLock);
      }
      allLocks.push_back(locks);
    }

    return allLocks;
  }

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, LockAnalysis &lockAnalysis,
                                ObjectFifoCreateOp op, int share_direction) {
    if (!op.size())
      return;

    std::vector<BufferOp> buffers;
    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int numElem = op.size();
    int of_elem_index = 0; // used to give objectFifo elements a symbolic name

    // if this objectFifo is linked to another, check if the other's elements
    // have already been created: if none of the output objectfifos of the link
    // have initValues, then the elements that are created are those of the
    // objFifo with elements of bigger size
    bool linked = false;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp) {
      auto fifoIn = linkOp->getInputObjectFifos()[0];
      auto fifoOut = linkOp->getOutputObjectFifos()[0];
      linked = true;
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        return; // elements have already been created
      if (linkOp->isJoin()) {
        // if join, fifoOut has bigger size
        if (op.name() != fifoOut.name())
          return;
      } else if (linkOp->isDistribute()) {
        // if distribute, fifoIn has bigger size
        if (op.name() != fifoIn.name())
          return;
      } else {
        // check if output objectfifo has initValues
        if (fifoOut.getInitValues().has_value()) {
          if (fifoOut.name() != op.name())
            return;
        } else {
          // check which objectfifo of the link has bigger size
          auto fifoInType = llvm::cast<AIEObjectFifoType>(fifoIn.getElemType());
          auto elemInType = llvm::cast<MemRefType>(fifoInType.getElementType());
          int inSize = elemInType.getNumElements();

          auto fifoOutType =
              llvm::cast<AIEObjectFifoType>(fifoOut.getElemType());
          auto elemOutType =
              llvm::cast<MemRefType>(fifoOutType.getElementType());

          if (int outSize = elemOutType.getNumElements(); inSize >= outSize) {
            if (op.name() != fifoIn.name())
              return;
          } else {
            if (fifoOut.name() != op.name())
              return;
          }
        }
      }
    }

    // When using circuit-switch (share_dir = 0), the FIFO has already 
    // been split, so each FIFO creates its elements on the producer tile.
    // (Note: cons-Fifo's producer tile is cons itself, see line 1102)
    // If PnR decides that neighbour sharing should use a buffer on producer 
    // side (share_dir = -1), we also create the element on the producer tile.
    // Otherwise (share_dir = 1), the element is created on the consumer side.
    TileOp creation_tile;
    if (share_direction == 0 || share_direction == -1)
      creation_tile = op.getProducerTileOp();
    else {
      auto consumerTileOp =
          dyn_cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
      creation_tile = consumerTileOp;
    }
    std::optional<ObjectFifoAllocateOp> opAlloc = getOptionalAllocateOp(op);
    if (opAlloc.has_value() && opAlloc->getDelegateTileOps().size() == 1) {
      TileOp delegate = opAlloc->getDelegateTileOps()[0];
      creation_tile = delegate;
    }
    // Reset opbuilder location to after the last tile declaration
    Operation *t = nullptr;
    auto dev = op->getParentOfType<DeviceOp>();
    for (auto tile_op : dev.getBody()->getOps<TileOp>()) {
      t = tile_op.getOperation();
    }
    builder.setInsertionPointAfter(t);
    for (int i = 0; i < numElem; i++) {
      mlir::ElementsAttr initValues = nullptr;
      if (!creation_tile.isShimTile()) {
        if (op.getInitValues().has_value()) {
          initValues =
              llvm::cast<mlir::ElementsAttr>(op.getInitValues().value()[i]);
        }
        auto buff = builder.create<BufferOp>(
            builder.getUnknownLoc(), elemType, creation_tile,
            builder.getStringAttr(op.name().str() + "_buff_" +
                                  std::to_string(of_elem_index)),
            /*address*/ nullptr, initValues,
            /*mem_bank*/ nullptr);
        buffers.push_back(buff);
      }
      of_elem_index++;
    }

    int repeatCount = 1;
    int joinDistribFactor = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();
    if (linked) {
      if (linkOp->getRepeatCount().has_value())
        repeatCount = linkOp->getRepeatCount().value();
      if (linkOp->isDistribute())
        joinDistribFactor *= linkOp->getFifoOuts().size();
      else if (linkOp->isJoin())
        joinDistribFactor *= linkOp->getFifoIns().size();
      objFifoLinks[*linkOp] = op;
    }
    std::vector<std::vector<LockOp>> locks = createObjectFifoLocks(
        builder, lockAnalysis, op, numElem, joinDistribFactor, creation_tile, repeatCount);
    buffersPerFifo[op] = buffers;
    locksPerFifo[op] = locks;
  }

  /// Function that returns a pointer to the block of a Region
  /// that contains the AIEEndOp.
  Block *findEndOpBlock(Region &r) {
    Block *endBlock = nullptr;
    for (auto &bl : r.getBlocks())
      if (!bl.getOps<EndOp>().empty())
        endBlock = &bl;
    return endBlock;
  }

  /// Function used to create a Bd block.
  template <typename MyOp>
  void createBd(OpBuilder &builder, LockOp acqLock, int acqMode,
                LockAction acqLockAction, LockOp relLock, int relMode,
                MyOp buff, int offset, int len, Block *succ,
                BDDimLayoutArrayAttr dims, BDPadLayoutArrayAttr padDimensions,
                std::optional<PacketInfoAttr> bdPacket) {
    if (acqLock)
      builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock, acqLockAction,
                                acqMode);
    if (bdPacket)
      builder.create<DMABDPACKETOp>(builder.getUnknownLoc(), bdPacket->getPktType(),
                                    bdPacket->getPktId());
    if (!dims.getValue().empty() && padDimensions) {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims,
                              padDimensions);
    } else if (!dims.getValue().empty()) {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims);
    } else {
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len);
    }
    if (acqLock)
      builder.create<UseLockOp>(builder.getUnknownLoc(), relLock,
                                LockAction::Release, relMode);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  template <typename MyOp>
  void createBdBlock(OpBuilder &builder, ObjectFifoCreateOp op, int lockMode,
                     int acqNum, int relNum, MyOp buff, int offset, int len,
                     DMAChannelDir channelDir, size_t lockIndex, Block *succ,
                     BDDimLayoutArrayAttr dims,
                     BDPadLayoutArrayAttr padDimensions,
                     std::optional<PacketInfoAttr> bdPacket,
                     bool distribOrJoin = false) {
    LockOp acqLock;
    LockOp relLock;
    int acqMode = 1;
    int relMode = 1;
    auto acqLockAction = LockAction::Acquire;
    if (locksPerFifo[op][0].size() > 0) {
      auto dev = op->getParentOfType<DeviceOp>();
      if (auto &target = dev.getTargetModel();
          target.getTargetArch() == AIEArch::AIE1) {
        acqMode = lockMode == 0 ? 1 : 0;
        relMode = lockMode == 0 ? 0 : 1;
        acqLock = locksPerFifo[op][0][lockIndex];
        relLock = locksPerFifo[op][0][lockIndex];
      } else {
        acqMode = acqNum;
        relMode = relNum;
        acqLockAction = LockAction::AcquireGreaterEqual;
        int prodLockIndex = 0;
        int consLockIndex = 1;
        if (distribOrJoin) {
          prodLockIndex = lockIndex * 2;
          consLockIndex = lockIndex * 2 + 1;
        }
        acqLock = channelDir == DMAChannelDir::S2MM
                      ? locksPerFifo[op][0][prodLockIndex]
                      : locksPerFifo[op][0][consLockIndex];
        relLock = channelDir == DMAChannelDir::S2MM
                      ? locksPerFifo[op][0][consLockIndex]
                      : locksPerFifo[op][0][prodLockIndex];
      }
    }
    createBd(builder, acqLock, acqMode, acqLockAction, relLock, relMode, buff,
             offset, len, succ, dims, padDimensions, bdPacket);
  }

  void mergeDMA(DeviceOp &device, OpBuilder &builder,
                 ObjectFifoCreateOp op, DMAChannelDir channelDir,
                 int channelIndex, int lockMode,
                 BDDimLayoutArrayAttr dims,
                 std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int len = elemType.getNumElements();

    // check for repeat count
    int repeatCount = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();

    // search for the buffers/locks (based on if this objFifo has a link)
    ObjectFifoCreateOp target = op;
    if (std::optional<ObjectFifoLinkOp> linkOp = getOptionalLinkOp(op);
        linkOp.has_value()) {
      if (objFifoLinks.find(linkOp.value()) != objFifoLinks.end()) {
        target = objFifoLinks[linkOp.value()];
        if (target == op) {
          if (linkOp->getRepeatCount().has_value()) {
            acqNum *= linkOp->getRepeatCount().value();
            relNum *= linkOp->getRepeatCount().value();
          }
        }
      }
    }
    // search for MemOp
    Operation *producerMem = nullptr;
    for (auto memOp : device.getOps<MemOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = memOp.getOperation();
        break;
      }
    }
    if (producerMem == nullptr) {
      op.emitOpError("No MemOp found on producer tile, cannot merge packet-flow DMA into it.");
      return signalPassFailure();
    }
    DMAStartOp startOp = nullptr;
    if (auto mem = dyn_cast<MemOp>(producerMem)) {
      // Traverse *all* blocks in the mem region
      for (Block &block : mem.getRegion()) {
        for (Operation &op : block) {
          if (auto dmaStart = dyn_cast<DMAStartOp>(op)) {
            if (dmaStart.getChannelIndex() == channelIndex &&
                dmaStart.getChannelDir() == channelDir) {
              startOp = dmaStart;
              break;
            }
          }
        }
        if (startOp)
          break;
      }
    }

    if (!startOp) {
      op.emitOpError("DMAStartOp was not initialized for channel ")
          << channelIndex << " dir " << (channelDir == DMAChannelDir::MM2S ? "MM2S" : "S2MM")
          << ", cannot merge packet-flow DMA into it.";
      return signalPassFailure();
    }
    Block *entryBd = startOp.getSuccessor(0);
    Block *chainBd = startOp.getSuccessor(1);
    Region &region = producerMem->getRegion(0);

    Block *tailBd = nullptr;
    for (Block &blk : region) {
      if (auto nextBd = dyn_cast<NextBDOp>(blk.getTerminator())) {
        if (nextBd.getSuccessor() == entryBd) {
          tailBd = &blk;
          break;
        }
      }
    }
    if (!tailBd)
      llvm_unreachable("Could not find tail BD block");

    Block *bdBlock = builder.createBlock(chainBd);
    tailBd->getTerminator()->setSuccessor(bdBlock, 0);
    // create new Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    size_t totalBlocks = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= buffersPerFifo[target].size())
        break;
      for (int r = 0; r < repeatCount; r++) {
        if (totalBlocks == numBlocks * repeatCount - 1)
          succ = entryBd;
        else
          succ = builder.createBlock(chainBd);

        builder.setInsertionPointToStart(curr);
        createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                                buffersPerFifo[target][elemIndex], /*offset*/ 0,
                                len, channelDir, elemIndex, succ, dims,
                                nullptr, bdPacket);
        curr = succ;
        totalBlocks++;
      }
      elemIndex++;
    }
  }

  /// Function that either calls createAIETileDMA(), createShimDMA() or
  /// createMemTileDMA() based on op tile row value.
  void createDMA(DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp op,
                 DMAChannelDir channelDir, int channelIndex, int lockMode,
                 BDDimLayoutArrayAttr dims, BDPadLayoutArrayAttr pad_dims,
                 std::optional<PacketInfoAttr> bdPacket) {
    if (op.getProducerTileOp().isShimTile()) {
      createShimDMA(device, builder, op, channelDir, channelIndex, lockMode,
                    dims, bdPacket);
    } else if (op.getProducerTileOp().isMemTile()) {
      BDPadLayoutArrayAttr padDims = nullptr;
      if (channelDir == DMAChannelDir::MM2S && pad_dims)
        padDims = pad_dims;
      createMemTileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims, padDims, bdPacket);
    } else {
      createAIETileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims, bdPacket);
      if (channelDir == DMAChannelDir::MM2S && bdPacket.has_value()) {
        llvm::dbgs() << "Marking channel " << channelIndex
                     << " on tile (" << op.getProducerTileOp().colIndex()
                     << "," << op.getProducerTileOp().rowIndex()
                     << ") as used for packet flow\n";
        // record channel index as configured to packet flow for this objFifo
        // can be used later to pack other packet flows on the same channel
        pktFlowChannelPerTile[op.getProducerTileOp()] = channelIndex;
      }
    }
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createAIETileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims,
                        std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int len = elemType.getNumElements();

    // check for repeat count
    int repeatCount = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();

    // search for the buffers/locks (based on if this objFifo has a link)
    ObjectFifoCreateOp target = op;
    if (std::optional<ObjectFifoLinkOp> linkOp = getOptionalLinkOp(op);
        linkOp.has_value()) {
      if (objFifoLinks.find(linkOp.value()) != objFifoLinks.end()) {
        target = objFifoLinks[linkOp.value()];
        if (target == op) {
          if (linkOp->getRepeatCount().has_value()) {
            acqNum *= linkOp->getRepeatCount().value();
            relNum *= linkOp->getRepeatCount().value();
          }
        }
      }
    }

    // search for MemOp
    Operation *producerMem = nullptr;
    for (auto memOp : device.getOps<MemOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = memOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerMem == nullptr) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(device.getBody()->getTerminator());
      auto newMemOp =
          builder.create<MemOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerMem = newMemOp.getOperation();
    }
    Block *endBlock = findEndOpBlock(producerMem->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    size_t totalBlocks = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= buffersPerFifo[target].size())
        break;
      for (int r = 0; r < repeatCount; r++) {
        if (totalBlocks == numBlocks * repeatCount - 1)
          succ = bdBlock;
        else
          succ = builder.createBlock(endBlock);

        builder.setInsertionPointToStart(curr);
        createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                                buffersPerFifo[target][elemIndex], /*offset*/ 0,
                                len, channelDir, elemIndex, succ, dims,
                                nullptr, bdPacket);
        curr = succ;
        totalBlocks++;
      }
      elemIndex++;
    }
  }

  /// Function used to create a ShimDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createShimDMA(DeviceOp &device, OpBuilder &builder,
                     ObjectFifoCreateOp op, DMAChannelDir channelDir,
                     int channelIndex, int lockMode,
                     BDDimLayoutArrayAttr dims,
                     std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = externalBuffersPerFifo[op].size();
    if (numBlocks == 0)
      return;

    int acqNum = 1;
    int relNum = 1;

    // search for ShimDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<ShimDMAOp>()) {
      if (dmaOp.getTile() == op.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = op.getProducerTileOp();
    if (producerDMA == nullptr) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(device.getBody()->getTerminator());
      auto newDMAOp = builder.create<ShimDMAOp>(
          builder.getUnknownLoc(), builder.getIndexType(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= externalBuffersPerFifo[op].size())
        break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      MemRefType buffer = externalBuffersPerFifo[op][elemIndex].getType();
      int len = buffer.getNumElements();
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(builder, op, lockMode, acqNum, relNum,
                                      externalBuffersPerFifo[op][elemIndex],
                                      /*offset*/ 0, len, channelDir, elemIndex,
                                      succ, dims, nullptr, bdPacket);
      curr = succ;
      elemIndex++;
    }
  }

  /// Function used to create a MemTileDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createMemTileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims,
                        BDPadLayoutArrayAttr padDimensions,
                        std::optional<PacketInfoAttr> bdPacket) {
    size_t numBlocks = op.size();
    if (numBlocks == 0)
      return;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int lenOut = elemType.getNumElements();
    int acqNum = 1;
    int relNum = 1;

    // check for repeat count
    int repeatCount = 1;
    if (op.getRepeatCount().has_value())
      repeatCount = op.getRepeatCount().value();

    // search for the buffers/locks (based on if this objFifo has a link)
    // identify size difference between input and output memrefs
    ObjectFifoCreateOp target = op;
    bool isDistribute = false;
    bool isJoin = false;
    int extraOffset = 0;
    int joinDistribFactor = 1;
    int joinDistribLockIndex = 0;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp) {
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end()) {
        target = objFifoLinks[*linkOp];
        auto srcOffsets = linkOp->getSrcOffsets();
        auto dstOffsets = linkOp->getDstOffsets();

        if (linkOp->getRepeatCount().has_value())
          if (linkOp->getInputObjectFifos()[0] == op) {
            acqNum *= linkOp->getRepeatCount().value();
            relNum *= linkOp->getRepeatCount().value();
          }

        if (linkOp->isJoin()) {
          // compute offset and length
          isJoin = true;
          if (target == op) {
            joinDistribFactor *= linkOp->getFifoIns().size();
          } else {
            int i = 0;
            for (auto fifoIn : linkOp->getInputObjectFifos()) {
              if (fifoIn.name() == op.name())
                break;
              i++;
            }
            extraOffset = *getConstantIntValue(srcOffsets[i]);
            lenOut = linkOp->getJoinTransferLengths()[i];
            joinDistribLockIndex = i;
          }
        } else if (linkOp->isDistribute()) {
          // compute offset and length
          isDistribute = true;
          if (target == op) {
            joinDistribFactor *= linkOp->getFifoOuts().size();
          } else {
            int i = 0;
            for (auto fifoOut : linkOp->getOutputObjectFifos()) {
              if (fifoOut.name() == op.name())
                break;
              i++;
            }
            extraOffset = *getConstantIntValue(dstOffsets[i]);
            lenOut = linkOp->getDistributeTransferLengths()[i];
            joinDistribLockIndex = i;
          }
        } else {
          if (target != op) {
            auto targetFifo =
                llvm::cast<AIEObjectFifoType>(target.getElemType());
            auto targetElemType =
                llvm::cast<MemRefType>(targetFifo.getElementType());
            lenOut = targetElemType.getNumElements();
          }
        }

        // check if current op is of smaller size in link
        if (target != op)
          numBlocks = target.size();
      }
    }

    // search for MemTileDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<MemTileDMAOp>()) {
      if (dmaOp.getTile() == target.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerDMA == nullptr) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(device.getBody()->getTerminator());
      auto newDMAOp =
          builder.create<MemTileDMAOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t elemIndex = 0;
    size_t lockIndex = 0;
    size_t totalBlocks = 0;
    bool distribOrJoin = false;
    for (size_t i = 0; i < numBlocks; i++) {
      if (elemIndex >= buffersPerFifo[target].size())
        break;
      for (int r = 0; r < repeatCount * joinDistribFactor; r++) {
        if (totalBlocks == numBlocks * repeatCount * joinDistribFactor - 1)
          succ = bdBlock;
        else
          succ = builder.createBlock(endBlock);

        builder.setInsertionPointToStart(curr);
        int offset = 0;
        if (isDistribute || isJoin) {
          distribOrJoin = true;
          if (target == op) {
            if (isDistribute) {
              offset = *getConstantIntValue(linkOp->getDstOffsets()[r]);
              lenOut = linkOp->getDistributeTransferLengths()[r];
            } else {
              offset = *getConstantIntValue(linkOp->getSrcOffsets()[r]);
              lenOut = linkOp->getJoinTransferLengths()[r];
            }
            lockIndex = r % joinDistribFactor;
          } else {
            offset = extraOffset;
            lockIndex = joinDistribLockIndex;
          }
        } else {
          lockIndex = elemIndex;
        }
        createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                                buffersPerFifo[target][elemIndex], offset,
                                lenOut, channelDir, lockIndex, succ, dims,
                                padDimensions, bdPacket, distribOrJoin);
        curr = succ;
        totalBlocks++;
      }
      elemIndex++;
    }
  }

  // Function that computes the Least Common Multiplier of the values
  // of a vector.
  int computeLCM(std::set<int> values) {
    int lcm = 1;
    for (int i : values)
      lcm = i * lcm / std::gcd(i, lcm);
    return lcm;
  }

  // Function that unrolls for-loops that contain objectFifo operations.
  LogicalResult unrollForLoops(DeviceOp &device, OpBuilder &builder,
                               std::set<TileOp> objectFifoTiles) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (objectFifoTiles.count(coreOp.getTileOp()) > 0) {
        std::vector<scf::ForOp> unrolledLoops;
        std::map<Operation *, bool> foundMap;
        std::map<Operation *, int64_t> remainderMap;
        std::map<Operation *, int64_t> tripCountMap;
        WalkResult res = coreOp.walk([&](scf::ForOp forLoop) {
          // look for operations on objectFifos
          // when multiple fifos in same loop, must use the smallest
          // common multiplier as the unroll factor
          foundMap[forLoop.getOperation()] = false;
          std::set<int> objFifoSizes;
          Block *body = forLoop.getBody();
          remainderMap[forLoop.getOperation()] = 0;
          for (auto acqOp : body->getOps<ObjectFifoAcquireOp>()) {
            if (acqOp.getOperation()->getParentOp() == forLoop) {
              foundMap[forLoop.getOperation()] = true;
              ObjectFifoCreateOp op = acqOp.getObjectFifo();
              objFifoSizes.insert(op.size());
            }
          }
          // If the loop doesn't have acquire and release locks
          // Push it to the unrolledLoops to avoid unrolling
          if (!foundMap[forLoop.getOperation()]) {
            unrolledLoops.push_back(forLoop);
            return WalkResult::advance();
          }
          // Walk in the loop region to unroll the loop and its remainder
          Region *region = forLoop->getParentRegion();
          scf::ForOp prevLoop;
          prevLoop = forLoop;
          tripCountMap[prevLoop.getOperation()] = 0;
          while (remainderMap[prevLoop.getOperation()] > 1 ||
                 foundMap[prevLoop.getOperation()]) {
            region->walk([&](scf::ForOp remLoop) {
              bool skipLoop = false;
              int64_t tripCount = 0;
              if (remLoop.getSingleLowerBound() &&
                  remLoop.getSingleUpperBound() && remLoop.getSingleStep()) {
                tripCount = constantTripCount(*(remLoop.getSingleLowerBound()),
                                              *(remLoop.getSingleUpperBound()),
                                              *(remLoop.getSingleStep()))
                                .value_or(0);
              }
              int unrollFactor =
                  computeLCM(objFifoSizes); // also counts original loop body
              // Loop ids are not unique.
              // Sometimes, immediately after unrolling, the unrolled loop
              // and the one next to it (can be the remainder loop or an
              // independent loop) will have the same ID. This makes it
              // difficult to identify which loop needs to be unrolled.
              // Once it restarts walking from start, it ends up allocating
              // new ID to each loop.
              if (remainderMap[prevLoop.getOperation()] > 1 &&
                  foundMap[remLoop.getOperation()] == false &&
                  prevLoop != remLoop) {
                skipLoop = true;
              }
              if (std::count(unrolledLoops.begin(), unrolledLoops.end(),
                             remLoop) == 0 &&
                  !skipLoop) {
                tripCountMap[remLoop.getOperation()] = tripCount;
                // if loop iterations < unrollFactor, unroll the loop fully
                if (tripCountMap[remLoop.getOperation()] < unrollFactor)
                  unrollFactor = tripCountMap[remLoop.getOperation()];
                // If unrollFactor = 0,divide by zero
                if (unrollFactor == 0) {
                  remLoop.emitOpError()
                      << "could not be unrolled with unrollFactor = 0, check "
                         "loop boundaries."
                      << "\n";
                  return WalkResult::interrupt();
                }
                remainderMap[remLoop.getOperation()] =
                    tripCountMap[remLoop.getOperation()] % unrollFactor;
                auto step = remLoop.getStep()
                                .getDefiningOp<arith::ConstantOp>()
                                .getValue();
                int64_t step_value = llvm::dyn_cast<IntegerAttr>(step).getInt();

                if (step_value < unrollFactor ||
                    foundMap[remLoop.getOperation()]) {
                  // Process the for loop
                  if (failed(mlir::loopUnrollByFactor(remLoop, unrollFactor))) {
                    remLoop.emitOpError()
                        << "could not be unrolled with unrollFactor: "
                        << unrollFactor << "\n";
                    return WalkResult::interrupt();
                  }
                  unrolledLoops.push_back(remLoop);
                  foundMap[remLoop.getOperation()] = false;
                } else {
                  remainderMap[remLoop.getOperation()] = 0;
                  foundMap[remLoop.getOperation()] = false;
                }
              } else {
                remainderMap[remLoop.getOperation()] = 0;
                foundMap[remLoop.getOperation()] = false;
              }
              prevLoop = remLoop;
              return WalkResult::advance();
            });
          }
          return WalkResult::advance();
        });
        if (res.wasInterrupted())
          return failure();
      }
    }
    return success();
  }

  /// Function used to create a UseLockOp based on input parameters.
  /// acc is an accumulator map that tracks the indices of the next locks to
  /// acquire (or release). Uses op to find index of acc for next lockID.
  /// Updates acc.
  void createUseLocks(OpBuilder &builder, ObjectFifoCreateOp op,
                      int idx, ObjectFifoPort port,
                      DenseMap<std::tuple<ObjectFifoCreateOp, int, int>, int> &acc,
                      int numLocks, LockAction lockAction) {
    ObjectFifoCreateOp target = op;
    auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
    if (auto linkOp = getOptionalLinkOp(op))
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        target = objFifoLinks[*linkOp];

    if (numLocks == 0)
      return;

    if (locksPerFifo[target][idx].size() == 0) {
      acc[{op, idx, portNum}] = (acc[{op, idx, portNum}] + numLocks) %
                            op.size(); // update to next objFifo elem
      return;
    }

    // search for the correct lock based on the port of the acq/rel
    // operation e.g. acq as consumer is the read lock (second)
    LockOp lock;
    if (lockAction == LockAction::AcquireGreaterEqual) {
      if (port == ObjectFifoPort::Produce)
        lock = locksPerFifo[target][idx][0];
      else
        lock = locksPerFifo[target][idx][1];
    } else {
      if (port == ObjectFifoPort::Produce)
        lock = locksPerFifo[target][idx][1];
      else
        lock = locksPerFifo[target][idx][0];
    }
    builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockAction,
                              numLocks);
    acc[{op, idx, portNum}] = (acc[{op, idx, portNum}] + numLocks) %
                               op.size(); // update to next objFifo elem
  }

  /// Function used to check whether op is already contained in map.
  /// If it is then return the associated int, if not create new entry and
  /// return 0.
  int updateAndReturnIndex(
      DenseMap<std::tuple<ObjectFifoCreateOp, int, int>, int> &map,
      std::tuple<ObjectFifoCreateOp, int, int> key) {
    if (map.find(key) == map.end()) {
      map[key] = 0;
      return 0;
    }
    return map[key];
  }

  /// Function used to add an external buffer to the externalBuffersPerFifo map.
  void addExternalBuffer(ObjectFifoCreateOp fifo, ExternalBufferOp buff) {
    if (externalBuffersPerFifo.find(fifo) == externalBuffersPerFifo.end()) {
      std::vector<ExternalBufferOp> buffs;
      externalBuffersPerFifo[fifo] = buffs;
    }
    externalBuffersPerFifo[fifo].push_back(buff);
  }

  /// Function used to detect all external buffers associated with parent
  /// objectFifo and tile then map them to child objectFifo.
  void detectExternalBuffers(DeviceOp &device, ObjectFifoCreateOp parent,
                             ObjectFifoCreateOp child, Value tile) {
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
      if (auto objFifo = regOp.getObjectFifo();
          regOp.getTile() == tile && objFifo == parent)
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>());
  }

  /// Function used to replace uses of split objectFifos.
  /// objectFifos that are split are renamed, need to replace name in core uses.
  void replaceSplitFifo(ObjectFifoCreateOp originalOp, ObjectFifoCreateOp newOp,
                        TileOp tile) {
    auto original =
        originalOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newSymbol =
        newOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    for (auto user : tile->getUsers())
      if (isa<CoreOp>(user))
        if (auto res =
                SymbolTable::replaceAllSymbolUses(original, newSymbol, user);
            res.failed())
          llvm_unreachable("unreachable");
  }

  /// Function used to find the size of an objectFifo after split based on
  /// the maximum number of elements (of the original objectFifo) acquired
  /// by a process running on given tile. If no CoreOp exists for this tile
  /// return 0.
  int findObjectFifoSize(DeviceOp &device, Value tile,
                         ObjectFifoCreateOp objFifo) {
    if (objFifo.size() == 0)
      return 0;

    // if memTile, size is equal to objFifo size
    if (tile.getDefiningOp<TileOp>().isMemTile())
      return objFifo.size();

    // if shimTile, size is equal to number of external buffers
    if (tile.getDefiningOp<TileOp>().isShimTile())
      for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
        if (regOp.getTile() == tile)
          return regOp.getExternalBuffers().size();
      }

    int maxAcquire = 0;
    for (auto coreOp : device.getOps<CoreOp>())
      if (coreOp.getTile() == tile)
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          if (auto createOp = acqOp.getObjectFifo(); createOp == objFifo)
            if (acqOp.acqNumber() > maxAcquire)
              maxAcquire = acqOp.acqNumber();
        });

    if (maxAcquire > 0) {
      if (maxAcquire == 1 && objFifo.size() == 1)
        return 1;
      return maxAcquire + 1;
      // +1 because objectFifo size is always 1 bigger than maxAcquire to allow
      // for prefetching: simplest case scenario is at least a ping-pong buffer
    }

    return objFifo.size();
  }

  /// Function used to generate, from an objectFifo with a shimTile endpoint, a
  /// shimDMAAllocationOp containing the channelDir, channelIndex and
  /// shimTile col assigned by the objectFifo lowering.
  void createObjectFifoAllocationInfo(OpBuilder &builder, MLIRContext *ctx,
                                      FlatSymbolRefAttr obj_fifo, int colIndex,
                                      DMAChannelDir channelDir,
                                      int channelIndex, std::optional<PacketInfoAttr> packet) {
    PacketInfoAttr packetInfo = nullptr;
    if (packet)
      packetInfo = *packet;
    builder.create<ShimDMAAllocationOp>(builder.getUnknownLoc(), obj_fifo,
                                        DMAChannelDirAttr::get(ctx, channelDir),
                                        builder.getI64IntegerAttr(channelIndex),
                                        builder.getI64IntegerAttr(colIndex),
                                        builder.getBoolAttr(false), packetInfo);
  }

  /// Function used to verify that an objectfifo is present in at most one
  /// ObjectFifoLinkOp.
  void verifyObjectFifoLinks(DeviceOp &device) {
    DenseSet<ObjectFifoCreateOp> objectfifoset;
    for (ObjectFifoLinkOp link : device.getOps<ObjectFifoLinkOp>()) {
      for (ObjectFifoCreateOp inOf : link.getInputObjectFifos()) {
        if (objectfifoset.count(inOf))
          inOf.emitOpError("objectfifo cannot be in more than one "
                           "ObjectFifoLinkOp");
        objectfifoset.insert(inOf);
      }
      for (ObjectFifoCreateOp outOf : link.getOutputObjectFifos()) {
        if (objectfifoset.count(outOf))
          outOf.emitOpError("objectfifo cannot be in more than one "
                            "ObjectFifoLinkOp");
        objectfifoset.insert(outOf);
      }
    }
  }

  // Function to clone FIFO usage in the Producer core for multicast cases where 
  // consumer-side buffers are required. The function rewrites the FIFO 
  // as multiple 1-to-1 FIFOs and duplicates Producer operations, cloning 
  // all users of acquire/subview results and recursively cloning their 
  // dependent ops until no users remain. If all consumers of FIFO uses
  // consumer-side buffer, function returns set of ops to erase as we will
  // erase prodFifo.
  SetVector<Operation*> cloneProdCoreUse(OpBuilder &builder, MLIRContext *ctx,
                                         CoreOp coreOp, ObjectFifoCreateOp prodFifo,
                                         std::vector<ObjectFifoCreateOp> consFifos) {
    SetVector<Operation*> opsToErase;
    for (auto consFifo : consFifos) {
      // track all values produced by acquire/subview 
      // so we can find their users in core
      DenseMap<Value, Value> trackedValues;

      coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
        if (acqOp.getObjectFifo() == prodFifo) {
          builder.setInsertionPointAfter(acqOp);
          auto newAcq = builder.create<ObjectFifoAcquireOp>(
              acqOp.getLoc(), acqOp.getResult().getType(),
              acqOp.getPortAttr(), 
              FlatSymbolRefAttr::get(ctx, consFifo.getSymName()),
              acqOp.getSizeAttr());
          trackedValues[acqOp.getResult()] = newAcq.getResult();
          opsToErase.insert(acqOp.getOperation());
        }
      });

      coreOp.walk([&](ObjectFifoReleaseOp relOp) {
        if (relOp.getObjectFifo() == prodFifo) {
          builder.setInsertionPointAfter(relOp);
          builder.create<ObjectFifoReleaseOp>(
              relOp.getLoc(), relOp.getPortAttr(),
              FlatSymbolRefAttr::get(ctx, consFifo.getSymName()),
              relOp.getSizeAttr());
          opsToErase.insert(relOp.getOperation());
        }
      });

      // find subviewAccessOps. They don't use objectFifo
      // directly, instead use result of acquire of FIFO
      coreOp.walk([&](ObjectFifoSubviewAccessOp subOp) {
        if (trackedValues.contains(subOp.getSubview())) {
          builder.setInsertionPointAfter(subOp);
          auto newSub = builder.create<ObjectFifoSubviewAccessOp>(
              subOp.getLoc(), subOp.getResult().getType(),
              trackedValues.lookup(subOp.getSubview()), subOp.getIndexAttr());
          trackedValues[subOp.getResult()] = newSub.getResult();
          opsToErase.insert(subOp.getOperation());
        }
      });

      // clone other ops in core that use acquire/subview 
      // results of FIFO
      coreOp.walk([&](Operation *op) {
        if (isa<func::CallOp>(op) ||
            op->getDialect()->getNamespace() == "memref" ||
            op->getDialect()->getNamespace() == "arith") {
          bool isUser = llvm::any_of(op->getOperands(), [&](Value v){
            return trackedValues.count(v);
          });

          if (!isUser)
            return;

          IRMapping mapping;
          for (auto operand : op->getOperands()) {
            if (trackedValues.count(operand))
              mapping.map(operand, trackedValues.lookup(operand));
          }

          builder.setInsertionPointAfter(op);
          Operation* newOp = op->clone(mapping);
          builder.insert(newOp);

          for (auto it : llvm::zip(op->getResults(), newOp->getResults())) {
            trackedValues[std::get<0>(it)] = std::get<1>(it);
          }

          opsToErase.insert(op);
        }
      });
    }

    return opsToErase;
  }

  // Function to get lock indices that must be handled for FIFOs with multiple 
  // consumers. In this case, the FIFO uses shared memory on the producer, so 
  // the producer core must process one lock pair per consumer, while a consumer 
  // core must process its own lock index within the FIFO. If FIFO is single cast
  // returns index 0.
  SmallVector<int> getLockIndices(ObjectFifoCreateOp op, TileOp coreTile, TileOp fifoProdTile) {
    std::optional<ObjectFifoAllocateOp> opAlloc = getOptionalAllocateOp(op);
    // one-to-one or one-to-many DMAs should not have allocate (for now)
    if (!opAlloc.has_value())
      return {0};

    auto delegates = opAlloc->getDelegateTileOps();
    SmallVector<int> lockIndices;
    if (delegates.size() > 1) {
      if (coreTile == fifoProdTile) {
        // Producer core with mem-on-src for multiple consumers
        // process all locks
        for (size_t i = 0; i < op.getConsumerTiles().size(); i++)
          lockIndices.push_back(static_cast<int>(i));
      } 
      else {
        // Process on only matching consumer core
        int i = 0;
        for (auto cons : op.getConsumerTiles()) {
          if (coreTile == cons.getDefiningOp<TileOp>()) {
            lockIndices.push_back(i);
            break;
          }
          i++;
        }
      }
    }
    else {
      // one-to-one/one-to-many DMA or mem-on-sink FIFO
      lockIndices.push_back(0);
    }
    return lockIndices;  
  }

  /// Function to retrieve ObjectFifoAllocateOp of ObjectFifoCreateOp,
  /// if it exists.
  std::optional<ObjectFifoAllocateOp>
  getOptionalAllocateOp(ObjectFifoCreateOp op) {
    ObjectFifoAllocateOp allocOp;
    auto device = op->getParentOfType<DeviceOp>();
    bool foundAlloc = false;
    for (ObjectFifoAllocateOp alloc : device.getOps<ObjectFifoAllocateOp>()) {
      if (alloc.getObjectFifo() == op) {
        if (foundAlloc)
          op.emitOpError("has more than one allocate operation");
        allocOp = alloc;
        foundAlloc = true;
      }
    }
    if (foundAlloc)
      return {allocOp};
    return {};
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    auto ctx = device->getContext();
    auto producerWireType = WireBundle::DMA;
    auto consumerWireType = WireBundle::DMA;
    std::set<TileOp> objectFifoTiles; // track cores to check for loops during unrolling

    verifyObjectFifoLinks(device);

    auto range = device.getOps<ObjectFifoCreateOp>();
    originalFifoOps.insert(originalFifoOps.end(), range.begin(), range.end());
    
    //===------------------------------------------------------------------===//
    // Split objectFifos into a consumer end and producer end if needed
    //===------------------------------------------------------------------===//
    // We are going to create additional createObjectFifoOps, so get a copy of
    // all "original" ones before the loop to avoid looping over newly created
    // ones.
    // Split: ObjectfifoCreateOp(prod, cons) -> ObjectFifoCreateOp(prod, cons) 
    //                                          ObjectFifoCreateOp(cons, cons)
    for (auto createOp : originalFifoOps) {
      std::vector<ObjectFifoCreateOp> splitConsumerFifos;
      int consumerIndex = 0;
      auto producerTile = createOp.getProducerTile();
      auto producerTileOp = createOp.getProducerTileOp();
      ArrayRef<BDDimLayoutArrayAttr> consumerDims =
          createOp.getDimensionsFromStreamPerConsumer();
      std::optional<ObjectFifoAllocateOp> opAlloc =
          getOptionalAllocateOp(createOp);
      // Only FIFOs using DMA or multi-cast via shared memory 
      // are split into two ends; skip in shared memory one-to-one case
      if (!createOp.getVia_DMA() && 
          createOp.getConsumerTiles().size() <= 1)
        continue;

      SmallVector<Value> memOnSrcConsumers;
      for (auto consumerTile : createOp.getConsumerTiles()) {
        if (!createOp.getVia_DMA() && opAlloc.has_value() && 
            opAlloc->getDelegateTileOps()[consumerIndex] == producerTileOp) {
          memOnSrcConsumers.push_back(consumerTile);
          consumerIndex++;
          continue;
        }
        
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());

        builder.setInsertionPointAfter(createOp);

        // Fifo creation parameters
        auto datatype = llvm::cast<AIEObjectFifoType>(createOp.getElemType());
        std::string conSuffix;
        int consumerDepth = createOp.size();
        std::string consumerFifoName;

        if (createOp.getVia_DMA()) {
          if (isa<ArrayAttr>(createOp.getElemNumber())) {
            // +1 to account for 1st depth (producer)
            consumerDepth = createOp.size(consumerIndex + 1);
          } else {
            consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);
          }
          conSuffix = "_cons";
        } else {
          conSuffix = "_nbr_sink";
          consumerDepth = createOp.size();
        } 

        auto consumerObjFifoSize =
            builder.getIntegerAttr(builder.getI32Type(), consumerDepth);
        // rename and replace split objectFifo
        if (createOp.getConsumerTiles().size() > 1) {
          consumerFifoName = createOp.name().str() + "_" +
                             std::to_string(consumerIndex) + conSuffix;
        } else {
          consumerFifoName = createOp.name().str() + conSuffix;
        }
        BDDimLayoutArrayAttr emptyDims =
            BDDimLayoutArrayAttr::get(builder.getContext(), {});
        BDDimLayoutArrayAttr singletonFromStreamDims =
            BDDimLayoutArrayAttr::get(
                builder.getContext(),
                ArrayRef<BDDimLayoutAttr>{consumerDims[consumerIndex]});
        BDDimLayoutArrayArrayAttr fromStreamDims =
            BDDimLayoutArrayArrayAttr::get(builder.getContext(),
                                           singletonFromStreamDims);
        ObjectFifoCreateOp consumerFifo;
        if (createOp.getVia_DMA()) {
          consumerFifo = createObjectFifo(
              builder, datatype, consumerFifoName, consumerTile, consumerTile,
              consumerObjFifoSize, emptyDims, fromStreamDims);
        }
        else {
          consumerFifo = createObjectFifo(
              builder, datatype, consumerFifoName, producerTile, consumerTile,
              consumerObjFifoSize, emptyDims, fromStreamDims, true);
        }

        if (createOp.getDisableSynchronization())
          consumerFifo.setDisableSynchronization(true);
        replaceSplitFifo(createOp, consumerFifo, consumerTileOp);

        // identify external buffers that were registered to the consumer fifo
        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile);
        
        // record that this objectFifo was split
        splitConsumerFifos.push_back(consumerFifo);

        // update the linkOp if the split objFifo was originally its start point
        if (auto linkOp = getOptionalLinkOp(createOp))
          for (ObjectFifoCreateOp fifoIn : linkOp->getInputObjectFifos())
            if (fifoIn.name() == createOp.name() &&
                consumerTile == *linkOp->getOptionalSharedTile())
              if (failed(SymbolTable::replaceAllSymbolUses(
                      createOp, consumerFifo.name(), linkOp->getOperation())))
                llvm::report_fatal_error("unable to update all symbol uses");

        consumerIndex++;
      }
      
      if (!splitConsumerFifos.empty()) {
        if (!createOp.getVia_DMA()) {
          splitNbrFifos.emplace_back(createOp, splitConsumerFifos);
          // update original fifo to be only mem-on-src consumers
          createOp.getConsumerTilesMutable().assign(memOnSrcConsumers);
        } 
        else
          splitDmaFifos.emplace_back(createOp, splitConsumerFifos);
      }
    }
    //===------------------------------------------------------------------===//
    // - Handle multicast using shared memory connection.
    // - Duplicate FIFO usage in Producer core if shared memory is intended on
    //   consumer.
    // - Remove original FIFO if all consumers are consumer-side memory
    //===------------------------------------------------------------------===//
    SetVector<Operation*> nbrOpsToErase;
    for (auto &[prodFifo, splitConsFifos] : splitNbrFifos) {
      auto tileOp = prodFifo.getProducerTileOp();
      for (auto tileUser : tileOp->getUsers()) {
        if (auto coreOp = dyn_cast<CoreOp>(tileUser)) {
          SetVector<Operation*> opsToErase = cloneProdCoreUse(builder, ctx, 
                                                              coreOp, prodFifo, 
                                                              splitConsFifos);
          if (opsToErase.empty()) 
            prodFifo.emitOpError("Duplication failed in producer core.");

          // if all consumer tiles use buffer-on-cons, mark removal for prod side
          if (prodFifo.getConsumerTiles().size() == 0) {
            LLVM_DEBUG(llvm::dbgs() << "Removing source side for " << prodFifo << "\n");
            opsToErase.insert(prodFifo);
            nbrOpsToErase.insert(opsToErase.begin(), opsToErase.end());
          }

          LLVM_DEBUG(llvm::dbgs() << "Showing core after clone for " << prodFifo << "\n");
        }
      }
    }

    // remove marked operations
    SmallVector<Operation*> nbrSorted{nbrOpsToErase.begin(), nbrOpsToErase.end()};
    computeTopologicalSorting(nbrSorted);
    for (auto *op : llvm::reverse(nbrSorted))
      op->erase();
    //===------------------------------------------------------------------===//
    // - Create objectFifo buffers and locks.
    // - Populate a list of tiles containing objectFifos for later processing of
    //   the acquires/releases (uses of the FIFO).
    // - Global release counter tracker to keep track of the objectFifo state
    //===------------------------------------------------------------------===//
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      // add all tiles that contain an objectFifo to objectFifoTiles for later
      // loop unrolling pass
      objectFifoTiles.insert(createOp.getProducerTileOp());
      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());
        objectFifoTiles.insert(consumerTileOp);
      }

      // identify external buffers that were registered to
      // the producer objectFifo
      if (createOp.getProducerTileOp().isShimTile())
        detectExternalBuffers(device, createOp, createOp,
                              createOp.getProducerTile());
      
      // if using 1-to-1 shared_memory, PnR decides if buffer will be on producer or
      // consumer side. Shared memory FIFOs with multiple consumers have to be
      // mem-on-src consumers, therefore direction = -1
      if (!createOp.getVia_DMA()) {
          createObjectFifoElements(builder, lockAnalysis, createOp, -1);
      }
      else {
        // check if split fifo is the copy or original; 
        // if original (this will become producer side buffers/locks), need
        // to update depth to be only its own size (original has depths for
        // producer + all consumers)
        if (isa<ArrayAttr>(createOp.getElemNumber()))
          createOp.setElemNumberAttr(
              builder.getI32IntegerAttr(createOp.size()));
        // otherwise buffer size for producer fifo might change
        else {
          if (!createOp.getInitValues().has_value()) {
            int prodMaxAcquire = findObjectFifoSize(
                  device, createOp.getProducerTileOp(), createOp);
            createOp.setElemNumberAttr(
                  builder.getI32IntegerAttr(prodMaxAcquire));
          }
        }
        createObjectFifoElements(builder, lockAnalysis, createOp, 0);
      }
    }
    //===------------------------------------------------------------------===//
    // Create tile DMAs and build non-neighbour paths
    //===------------------------------------------------------------------===//
    // Only the objectFifos we split above require DMA communication; the others
    // rely on shared memory and share the same buffers.
    for (auto &[producer, consumers] : splitDmaFifos) {
      int producerChanIndex = -1;
      bool canMerge= false;
      auto prodTileOp = producer.getProducerTileOp();

      std::optional<PacketInfoAttr> bdPacket = {};
      if (producer.getPacketId().has_value()) {
        bdPacket = {AIE::PacketInfoAttr::get(ctx,
            /*pkt_type*/ 0, /*pkt_id*/ *producer.getPacketId())};
        llvm::dbgs() << "Using packet id " << *producer.getPacketId() << " for fifo " 
                     << producer.name() << "\n";
      }
      
      // check if packet switched and if there is an existing DMA channel
      // configured for pkt-switched comms on this tile, we can merge dma
      // ONLY for core tile and MM2S direction for now
      if (!prodTileOp.isShimTile() && !prodTileOp.isMemTile() &&
          bdPacket.has_value() && pktFlowChannelPerTile.count(prodTileOp) > 0) {
        producerChanIndex = pktFlowChannelPerTile[prodTileOp];
        canMerge = true;
        llvm::dbgs() << "Merging producer DMA for fifo " << producer.name() 
                     << " on tile (" << prodTileOp.colIndex() << ","
                     << prodTileOp.rowIndex() << ") to channel "
                     << producerChanIndex << "\n";
      }
      else {
        // create producer tile DMA
        llvm::dbgs() << "Creating new MM2S DMA for fifo " 
                     << producer.name() << " on tile (" 
                     << prodTileOp.colIndex() << "," 
                     << prodTileOp.rowIndex() << ")\n";
        producerChanIndex = dmaAnalysis.getDMAChannelIndex(
            producer.getProducerTileOp(), DMAChannelDir::MM2S);
      }
      if (producerChanIndex == -1)
        producer.getProducerTileOp().emitOpError(
            "number of output DMA channel exceeded!");
      DMAChannel producerChan = {DMAChannelDir::MM2S, producerChanIndex};

      if (canMerge)
        mergeDMA(device, builder, producer, producerChan.direction,
                 producerChan.channel, 0, producer.getDimensionsToStreamAttr(),
                 bdPacket);
      else
        createDMA(device, builder, producer, producerChan.direction,
                  producerChan.channel, 0, producer.getDimensionsToStreamAttr(),
                  producer.getPadDimensionsAttr(), bdPacket);
      // generate objectFifo allocation info
      builder.setInsertionPoint(device.getBody()->getTerminator());

      if (producer.getProducerTileOp().isShimTile())
        createObjectFifoAllocationInfo(
            builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
            producer.getProducerTileOp().colIndex(), producerChan.direction,
            producerChan.channel, bdPacket);
      
      SmallVector<int32_t> consChannels;
      for (auto consumer : consumers) {
        // create consumer tile DMA
        int consumerChanIndex = dmaAnalysis.getDMAChannelIndex(
            consumer.getProducerTileOp(), DMAChannelDir::S2MM);
        if (consumerChanIndex == -1)
          consumer.getProducerTileOp().emitOpError(
              "number of input DMA channel exceeded!");
        DMAChannel consumerChan = {DMAChannelDir::S2MM, consumerChanIndex};
        consChannels.push_back(consumerChan.channel);
        BDDimLayoutArrayAttr consumerDims =
            consumer.getDimensionsFromStreamPerConsumer()[0];
        createDMA(device, builder, consumer, consumerChan.direction,
                  consumerChan.channel, 1, consumerDims, nullptr, {});
        // generate objectFifo allocation info
        builder.setInsertionPoint(device.getBody()->getTerminator());

        if (consumer.getProducerTileOp().isShimTile())
          createObjectFifoAllocationInfo(
              builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
              consumer.getProducerTileOp().colIndex(), consumerChan.direction,
              consumerChan.channel, {});
      }

      IntegerAttr pktIdAttr;
      if (bdPacket)
        pktIdAttr = builder.getIntegerAttr(builder.getI8Type(), bdPacket->getPktId());
      builder.setInsertionPointAfter(producer);
      builder.create<PnRFlowOp>(builder.getUnknownLoc(),
                              producer.getProducerTile(),
                              producerWireType, producerChan.channel,
                              producer.getConsumerTiles(),
                              consumerWireType, consChannels, pktIdAttr);
    }
    //===------------------------------------------------------------------===//
    // Statically unroll for loops 
    //===------------------------------------------------------------------===//
    std::set<TileOp> unrollTiles;
    for (auto c : device.getOps<CoreOp>()) {
      TileOp t = c.getTileOp();
      if (objectFifoTiles.count(t) > 0)
        unrollTiles.insert(t);
    }
    if (failed(unrollForLoops(device, builder, unrollTiles)))
      signalPassFailure();
    //===------------------------------------------------------------------===//
    // Replace ops
    //===------------------------------------------------------------------===//
    for (auto coreOp : device.getOps<CoreOp>()) {
      DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp *>>
          subviews; // maps each "subview" to its buffer references (subviews
      // are created by AcquireOps)
      DenseMap<std::tuple<ObjectFifoCreateOp, int, int>, std::vector<int>>
          acquiresPerFifo; // maps each objFifo to indices of buffers acquired
      // in latest subview of that objFifo (useful to
      // cascade acquired elements to next AcquireOp)
      DenseMap<std::tuple<ObjectFifoCreateOp, int, int>,
               std::vector<ObjectFifoReleaseOp>>
          releaseOps; // useful to check which ReleaseOp has taken place before
      // an AcquireOp per objFifo
      DenseMap<std::tuple<ObjectFifoCreateOp, int, int>, int>
          acqPerFifo; // maps each objFifo to its next index to acquire within
      // this CoreOp
      DenseMap<std::tuple<ObjectFifoCreateOp, int, int>, int>
          relPerFifo; // maps each objFifo to its next index to release within
      // this CoreOp

      //===----------------------------------------------------------------===//
      // Replace objectFifo.release ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        builder.setInsertionPointAfter(releaseOp);
        ObjectFifoCreateOp op = releaseOp.getObjectFifo();
        auto fifoProdTile = op.getProducerTileOp();
        auto port = releaseOp.getPort();
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = releaseOp->getParentOfType<CoreOp>();

        if (auto linkOp = getOptionalLinkOp(op)) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
            releaseOp->emitOpError("currently cannot access objectFifo used in "
                                   "ObjectFifoLinkOp");
            return;
          }
        }

        SmallVector<int> lockIndices = getLockIndices(op, core.getTileOp(), fifoProdTile);
        for (auto idx : lockIndices) {
          // update index of next element to release for this objectFifo
          updateAndReturnIndex(relPerFifo, {op, idx, portNum});

          // release locks
          int numLocks = releaseOp.relNumber();
          // account for repetition
          if (op.getRepeatCount().has_value())
            numLocks *= op.getRepeatCount().value();
          createUseLocks(builder, op, idx, port, relPerFifo, numLocks,
                        LockAction::Release);

          // register release op
          if (releaseOps.find({op, idx, portNum}) != releaseOps.end()) {
            releaseOps[{op, idx, portNum}].push_back(releaseOp);
          } else {
            std::vector release = {releaseOp};
            releaseOps[{op, idx, portNum}] = release;
          }
        }
      });

      //===----------------------------------------------------------------===//
      // Replace objectFifo.acquire ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        ObjectFifoCreateOp op = acquireOp.getObjectFifo();
        auto fifoProdTile = op.getProducerTileOp();
        builder.setInsertionPointAfter(acquireOp);
        auto port = acquireOp.getPort();
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = acquireOp->getParentOfType<CoreOp>();

        auto linkOp = getOptionalLinkOp(op);
        if (linkOp) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
            acquireOp->emitOpError("currently cannot access objectFifo used in "
                                   "ObjectFifoLinkOp");
            return;
          }
        }

        SmallVector<int> lockIndices = getLockIndices(op, core.getTileOp(), fifoProdTile);
        bool makeBuffers = true;
        for (auto idx : lockIndices) {
          // index of next element to acquire for this objectFifo
          int start = updateAndReturnIndex(
              acqPerFifo, {op, idx, portNum}); // useful for keeping track of which
          // indices are acquired

          // check how many elements have been released in between this AcquireOp
          // and the previous one
          // !!! operations may not be in the same block !!!
          int numRel = 0;
          for (std::vector<ObjectFifoReleaseOp>::iterator relOp =
                  releaseOps[{op, idx, portNum}].begin();
              relOp != releaseOps[{op, idx,portNum}].end();) {
            bool erased = false;
            Operation *acqBlockDefOp = acquireOp.getOperation();
            do {
              Operation *relBlockDefOp = (*relOp).getOperation();
              do {
                if (acqBlockDefOp->getBlock() == relBlockDefOp->getBlock()) {
                  if (relBlockDefOp->isBeforeInBlock(acqBlockDefOp)) {
                    numRel += (*relOp).relNumber();
                    relOp = releaseOps[{op, idx, portNum}].erase(relOp);
                    // to ensure that we do not account
                    // the ReleaseOps again later,
                    // after the subview is created
                    erased = true;
                  }
                }
              } while ((relBlockDefOp = relBlockDefOp->getParentOp()) &&
                      !isa<DeviceOp>(relBlockDefOp) && !erased);
            } while ((acqBlockDefOp = acqBlockDefOp->getParentOp()) &&
                    !isa<DeviceOp>(acqBlockDefOp) && !erased);
            if (!erased)
              ++relOp;
          }

          // track indices of elements to acquire
          std::vector<int> acquiredIndices;
          if (!acquiresPerFifo[{op, idx, portNum}].empty()) {
            // take into account what has already been acquired by previous
            // AcquireOp in program order
            acquiredIndices = acquiresPerFifo[{op, idx, portNum}];
            // take into account what has been released in-between
            if (static_cast<size_t>(numRel) > acquiredIndices.size()) {
              acquireOp->emitOpError("cannot release more elements than are "
                                    "already acquired");
              return;
            }
            for (int i = 0; i < numRel; i++)
              acquiredIndices.erase(acquiredIndices.begin());
          }

          // acquire locks
          int numLocks = acquireOp.acqNumber();
          int alreadyAcq = acquiredIndices.size();
          int numCreate;
          if (numLocks > alreadyAcq)
            numCreate = numLocks - alreadyAcq;
          else
            numCreate = 0;

          // account for repetition
          if (op.getRepeatCount().has_value())
            numCreate *= op.getRepeatCount().value();

          createUseLocks(builder, op, idx, port, acqPerFifo, numCreate,
                          LockAction::AcquireGreaterEqual);

          // if objFifo was linked with others, find which objFifos
          // elements to use
          ObjectFifoCreateOp target = op;
          if (linkOp)
            if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
              target = objFifoLinks[*linkOp];
          
          if (makeBuffers) {
            // create subview: buffers that were already acquired + new acquires
            for (int i = 0; i < numCreate; i++) {
              acquiredIndices.push_back(start);
              start = (start + 1) % op.size();
            }
            std::vector<BufferOp *> subviewRefs;
            subviewRefs.reserve(acquiredIndices.size());
            for (auto index : acquiredIndices)
              subviewRefs.push_back(&buffersPerFifo[target][index]);

            subviews[acquireOp] = subviewRefs;

            makeBuffers = false; // only need to make buffers once
          }
          
          acquiresPerFifo[{op, idx, portNum}] = acquiredIndices;
        }
      });

      //===----------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        auto acqOp = accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        if (ObjectFifoCreateOp op = acqOp.getObjectFifo()) {
          if (auto linkOp = getOptionalLinkOp(op); linkOp.has_value()) {
            if (!linkOp->isDistribute() && !linkOp->isJoin()) {
              for (auto consumerTile : op.getConsumerTiles()) {
                if (auto consumerTileOp =
                        dyn_cast<TileOp>(consumerTile.getDefiningOp())) {
                  int share_dir_value = 0;
                  bool sharing = isSharedMemory(
                      op.getProducerTileOp(), consumerTileOp, &share_dir_value);
                  if (!sharing)
                    accessOp->emitOpError(
                        "currently cannot access objectFifo used in "
                        "ObjectFifoLinkOp if the tiles don't share memory");
                }
              }
            }
          }
        }
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()]->getBuffer());
      });
    }
    // make global symbols to replace the to be erased ObjectFifoCreateOps
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      builder.setInsertionPointToStart(device.getBody());
      auto sym_name = createOp.getName();
      createOp->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr("__erase_" + sym_name));
      auto memrefType = llvm::cast<AIEObjectFifoType>(createOp.getElemType())
                            .getElementType();
      builder.create<memref::GlobalOp>(builder.getUnknownLoc(), sym_name,
                                       builder.getStringAttr("public"),
                                       memrefType, nullptr, false, nullptr);
    }
    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoCreateOp, ObjectFifoLinkOp,
              ObjectFifoRegisterExternalBuffersOp, ObjectFifoAcquireOp,
              ObjectFifoSubviewAccessOp, ObjectFifoReleaseOp, ObjectFifoAllocateOp>(op))
        opsToErase.insert(op);
    });
    SmallVector<Operation *> sorted{opsToErase.begin(), opsToErase.end()};
    computeTopologicalSorting(sorted);
    for (auto *op : llvm::reverse(sorted))
      op->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoToPathPass() {
  return std::make_unique<AIEObjectFifoToPathPass>();
}
