// RUN: aie-opt --pnr-apply-routing="routing-file-path=%S/route_check.json" %s | FileCheck %s 

// CHECK-LABEL: module @route_check {
// CHECK:         aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[VAL_2:.*]] = aie.shim_mux(%[[VAL_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             %[[VAL_4:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_5:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_6:.*]] = aie.masterset(North : 0, %[[VAL_4]])
// CHECK:             %[[VAL_7:.*]] = aie.masterset(South : 2, %[[VAL_5]])
// CHECK:             aie.packet_rules(South : 3) {
// CHECK:               aie.rule(31, 0, %[[VAL_4]])
// CHECK:               aie.rule(31, 16, %[[VAL_4]])
// CHECK:             }
// CHECK:             aie.packet_rules(North : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_5]])
// CHECK:             }
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = aie.switchbox(%[[VAL_1]]) {
// CHECK:             %[[VAL_9:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_10:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_11:.*]] = aie.masterset(DMA : 0, %[[VAL_9]])
// CHECK:             %[[VAL_12:.*]] = aie.masterset(South : 0, %[[VAL_10]])
// CHECK:             aie.packet_rules(South : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_9]])
// CHECK:               aie.rule(31, 16, %[[VAL_9]])
// CHECK:             }
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_10]])
// CHECK:             }
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:         }
// CHECK:       }

module @route_check{
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
  }
}