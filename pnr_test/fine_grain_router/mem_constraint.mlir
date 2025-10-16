// RUN: aie-opt --pnr-fine-grain-router %s 2>/dev/null | filecheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.shim_mux(%[[VAL_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<DMA : 0, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_6:.*]] = aie.tile(0, 2)
// CHECK:           %[[VAL_7:.*]] = aie.switchbox(%[[VAL_6]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.switchbox(%[[VAL_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:         }
module @mem_constraint {
    aie.device(npu2) {
        %shim = aie.tile(0, 0)
        %mem = aie.tile(0, 1)
        %core = aie.tile(0, 3)

        aie.pnr_flow(%shim, DMA : 0, {%core, %mem}, DMA : [0, 0]) {hop_tile_ids = [[[0,0],[0,1],[0,2],[0,3]], [[0,0],[0,1]]]}
        aie.pnr_flow(%mem, DMA : 0, {%core}, DMA : [1]) {hop_tile_ids = [[[0,1],[0,2],[0,3]]]}
    }
}
