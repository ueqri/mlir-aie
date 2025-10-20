// RUN: aie-opt --pnr-fine-grain-router %s 2>/dev/null | filecheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.masterset(East : 0, %[[VAL_2]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_2]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_8:.*]] = aie.masterset(DMA : 0, %[[VAL_6]])
// CHECK:             %[[VAL_9:.*]] = aie.masterset(South : 0, %[[VAL_7]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_7]])
// CHECK:               aie.rule(31, 0, %[[VAL_6]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.tile(2, 4)
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             %[[VAL_12:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_13:.*]] = aie.masterset(South : 0, %[[VAL_12]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_12]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.tile(4, 2)
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             %[[VAL_16:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_17:.*]] = aie.masterset(DMA : 0, %[[VAL_16]])
// CHECK:             aie.packet_rules(North : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_16]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_19:.*]] = aie.switchbox(%[[VAL_18]]) {
// CHECK:             %[[VAL_20:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_21:.*]] = aie.masterset(North : 0, %[[VAL_20]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_20]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_22:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_23:.*]] = aie.switchbox(%[[VAL_22]]) {
// CHECK:             %[[VAL_24:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_25:.*]] = aie.masterset(East : 0, %[[VAL_24]])
// CHECK:             aie.packet_rules(North : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_24]])
// CHECK:             }
// CHECK:             aie.packet_rules(South : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_24]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_27:.*]] = aie.switchbox(%[[VAL_26]]) {
// CHECK:             %[[VAL_28:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_29:.*]] = aie.masterset(East : 0, %[[VAL_28]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(30, 0, %[[VAL_28]])
// CHECK:             }
// CHECK:           }
// CHECK:         }
module @merge {
  aie.device(npu2) {
    %tile12 = aie.tile(1, 2)
    %tile43 = aie.tile(4, 3)
    %tile24 = aie.tile(2, 4)
    %tile42 = aie.tile(4, 2)

    aie.pnr_pktflow(%tile12, DMA : 0, {%tile43}, DMA : [0], packet_id : 0) {hop_tile_ids = [[[1,2],[2,2],[2,3],[3,3],[4,3]]]}
    aie.pnr_pktflow(%tile24, DMA : 0, {%tile42}, DMA : [0], packet_id : 1) {hop_tile_ids = [[[2,4],[2,3],[3,3],[4,3],[4,2]]]}
  }
}
