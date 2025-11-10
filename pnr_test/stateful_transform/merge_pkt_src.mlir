// RUN: aie-opt --pnr-stateful-transform %s 2>/dev/null | filecheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(3, 4)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 5)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of2_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of2_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of2_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of2_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]], 6) {init = 2 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 7) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_0]], 4) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_0]], 5) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_1]], 2) {init = 2 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_25:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_0]], 2) {init = 2 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_27:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           %[[VAL_28:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "ofc_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_29:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "ofc_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_30:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "ofc_cons_prod_lock_0"}
// CHECK:           %[[VAL_31:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "ofc_cons_cons_lock_0"}
// CHECK:           %[[VAL_32:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ofc_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_33:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "ofc_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_34:.*]] = aie.lock(%[[VAL_0]], 0) {init = 2 : i32, sym_name = "ofc_prod_lock_0"}
// CHECK:           %[[VAL_35:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "ofc_cons_lock_0"}
// CHECK:           aie.pnr_flow(%[[VAL_0]], DMA : 0, {%[[VAL_1]]}, DMA : [0]) {hop_tile_ids = []}
// CHECK:           aie.pnr_pktflow(%[[VAL_0]], DMA : 1, {%[[VAL_1]]}, DMA : [1], packet_id : 0) {hop_tile_ids = []}
// CHECK:           aie.pnr_pktflow(%[[VAL_0]], DMA : 1, {%[[VAL_2]]}, DMA : [0], packet_id : 1) {hop_tile_ids = []}
// CHECK:           aie.pnr_pktflow(%[[VAL_0]], DMA : 1, {%[[VAL_3]]}, DMA : [0], packet_id : 2) {hop_tile_ids = []}
// CHECK:           %[[VAL_36:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_35]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_32]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_34]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_35]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_33]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_34]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_38:.*]] = aie.dma_start(MM2S, 1, ^bb4, ^bb10)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_27]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd_packet(0, 0)
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_26]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd_packet(0, 1)
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd_packet(0, 2)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[VAL_27]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd_packet(0, 0)
// CHECK:             aie.dma_bd(%[[VAL_25]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_26]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd_packet(0, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_18]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd_packet(0, 2)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb10:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

// This test checks that multiple pkt-switched fifos leaving from the same source leave on the same
// dma port. Since a cct-switched fifo is processed first, the pkt-switched fifos should leave on
// DMA port 1.
module @merge_pkt_src {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)
        %tile34 = aie.tile(3, 4)
        %tile35 = aie.tile(3, 5)
        // 1 circuit-switched fifo
        aie.objectfifo @ofc (%tile12, {%tile33}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<256xi32>>
        // 3 packet-switched fifos
        aie.objectfifo @of0 (%tile12, {%tile33}, 2 : i32) {via_DMA = true, pkt = true} : !aie.objectfifo<memref<256xi32>>
        aie.objectfifo @of1 (%tile12, {%tile34}, 2 : i32) {via_DMA = true, pkt = true} : !aie.objectfifo<memref<256xi32>>
        aie.objectfifo @of2 (%tile12, {%tile35}, 2 : i32) {via_DMA = true, pkt = true} : !aie.objectfifo<memref<256xi32>>
    }
}
