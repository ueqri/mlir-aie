// RUN: aie-opt --pnr-fine-grain-router %s 2>/dev/null | filecheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_4:.*]] = aie.tile(1, 0)
// CHECK:           %[[VAL_5:.*]] = aie.tile(3, 2)
// CHECK:           %[[VAL_6:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<DMA : 0, North : 0>
// CHECK:             aie.connect<South : 1, DMA : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_9:.*]] = aie.switchbox(%[[VAL_8]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.shim_mux(%[[VAL_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_3]]) {
// CHECK:             aie.connect<North : 0, DMA : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = aie.shim_mux(%[[VAL_4]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:         }
module {
    aie.device(npu2) {
        %shim_noc_tile_0_0 = aie.tile(0, 0)
        %mem_tile_0_1 = aie.tile(0, 1)
        %tile_0_2 = aie.tile(0, 2)
        %mem_tile_1_1 = aie.tile(1, 1)
        %shim_noc_tile_1_0 = aie.tile(1, 0)
        %tile_3_2 = aie.tile(3, 2)

        aie.pnr_flow(%mem_tile_0_1, DMA : 0, {%tile_3_2}, DMA : [0]) {hop_tile_ids = [[[0 , 1], [0, 2], [1, 2], [2, 2], [3, 2]]]}
        aie.pnr_flow(%shim_noc_tile_0_0, DMA : 0, {%mem_tile_0_1, %tile_0_2}, DMA : [0, 0]) {hop_tile_ids = [[[0 , 0], [0 , 1]], [[0 , 0], [0 , 1], [0 , 2]]]}
        aie.pnr_flow(%tile_3_2, DMA : 0, {%mem_tile_1_1}, DMA : [0]) {hop_tile_ids = [[[3 , 2], [2 , 2], [1 , 2], [1 , 1]]]}
        aie.pnr_flow(%mem_tile_1_1, DMA : 0, {%shim_noc_tile_1_0}, DMA : [0]) {hop_tile_ids = [[[1 , 1], [1 , 0]]]}
    }
}
