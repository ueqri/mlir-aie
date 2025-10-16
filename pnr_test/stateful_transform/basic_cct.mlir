// RUN: aie-opt --pnr-stateful-transform %s 2>/dev/null | filecheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 2) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 3) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_0]], 2) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_0_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of0_0_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "of0_0_cons_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of0_0_cons_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of0_1_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of0_1_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_18:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of0_1_cons_prod_lock_0"}
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of0_1_cons_cons_lock_0"}
// CHECK:           %[[VAL_20:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of0_2_cons_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of0_2_cons_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "of0_2_cons_prod_lock_0"}
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "of0_2_cons_cons_lock_0"}
// CHECK:           %[[VAL_24:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_25:.*]] = aie.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_0]], 0) {init = 2 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:           %[[VAL_27:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:           aie.pnr_flow(%[[VAL_0]], DMA : 0, {%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]}, DMA : [0, 0, 0]) {hop_tile_ids = [[[1, 2], [1, 3]], [[1, 2], [0, 2]], [[1, 2], [1, 1]]]}
// CHECK:           aie.pnr_flow(%[[VAL_0]], DMA : 1, {%[[VAL_1]]}, DMA : [1]) {hop_tile_ids = [[[1, 2], [1, 3]]]}
// CHECK:           %[[VAL_28:.*]] = aie.mem(%[[VAL_0]]) {
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_27]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_24]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_26]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_27]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_25]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_26]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_18]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = aie.memtile_dma(%[[VAL_3]]) {
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_20]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_23]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<256xi32>, 0, 256)
// CHECK:             aie.use_lock(%[[VAL_23]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

// Basic test for circuit-switched single and multi-cast fifos
module @basic_cct {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile02 = aie.tile(0, 2)
        %tile11 = aie.tile(1, 1)

        aie.objectfifo @of0 (%tile12, {%tile13, %tile02, %tile11}, [2 : i32, 2 : i32, 2 : i32, 2 : i32]) {via_DMA = true, hop_tile_ids=[[[1,2],[1,3]],[[1,2],[0,2]],[[1,2],[1,1]]]}: !aie.objectfifo<memref<256xi32>>
        aie.objectfifo @of1 (%tile12, {%tile13}, [2 : i32, 2 : i32]) {via_DMA = true, hop_tile_ids=[[[1,2],[1,3]]] }: !aie.objectfifo<memref<256xi32>>
    }
}
