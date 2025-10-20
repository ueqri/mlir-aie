// RUN: aie-opt --pnr-fine-grain-router %s 2>/dev/null | filecheck %s 
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 0>
// CHECK:             aie.connect<DMA : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_6:.*]] = aie.switchbox(%[[VAL_5]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_8:.*]] = aie.switchbox(%[[VAL_7]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_10:.*]] = aie.switchbox(%[[VAL_9]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:           }
// CHECK:         }
module @basic_cct {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile43 = aie.tile(4, 3)
        %tile03 = aie.tile(0, 3)

        aie.pnr_flow(%tile12, DMA : 0, {%tile43, %tile03}, DMA : [0, 0]) {hop_tile_ids = [[[1,2],[1,3],[2,3],[3,3],[4,3]], [[1,2],[1,3],[0,3]]]}
        aie.pnr_flow(%tile12, DMA : 1, {%tile43}, DMA : [1]) {hop_tile_ids = [[[1,2],[1,3],[2,3],[3,3],[4,3]]]}
    }
}
