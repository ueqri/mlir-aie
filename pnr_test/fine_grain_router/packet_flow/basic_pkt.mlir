// RUN: aie-opt --pnr-fine-grain-router %s 2>/dev/null | filecheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.masterset(North : 0, %[[VAL_2]])
// CHECK:             aie.packet_rules(DMA : 1) {
// CHECK:               aie.rule(31, 1, %[[VAL_2]])
// CHECK:             }
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_2]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = aie.tile(4, 3)
// CHECK:           %[[VAL_5:.*]] = aie.switchbox(%[[VAL_4]]) {
// CHECK:             %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_8:.*]] = aie.masterset(DMA : 0, %[[VAL_6]])
// CHECK:             %[[VAL_9:.*]] = aie.masterset(DMA : 1, %[[VAL_7]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_7]])
// CHECK:               aie.rule(31, 0, %[[VAL_6]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_11:.*]] = aie.switchbox(%[[VAL_10]]) {
// CHECK:             %[[VAL_12:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_13:.*]] = aie.masterset(DMA : 0, %[[VAL_12]])
// CHECK:             aie.packet_rules(East : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_12]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_15:.*]] = aie.switchbox(%[[VAL_14]]) {
// CHECK:             %[[VAL_16:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_17:.*]] = aie.amsel<0> (1)
// CHECK:             %[[VAL_18:.*]] = aie.masterset(West : 0, %[[VAL_16]])
// CHECK:             %[[VAL_19:.*]] = aie.masterset(East : 0, %[[VAL_16]], %[[VAL_17]])
// CHECK:             aie.packet_rules(South : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_17]])
// CHECK:               aie.rule(31, 0, %[[VAL_16]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_21:.*]] = aie.switchbox(%[[VAL_20]]) {
// CHECK:             %[[VAL_22:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_23:.*]] = aie.masterset(East : 0, %[[VAL_22]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(30, 0, %[[VAL_22]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_25:.*]] = aie.switchbox(%[[VAL_24]]) {
// CHECK:             %[[VAL_26:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_27:.*]] = aie.masterset(East : 0, %[[VAL_26]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(30, 0, %[[VAL_26]])
// CHECK:             }
// CHECK:           }
// CHECK:         }
module @basic_pkt {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile43 = aie.tile(4, 3)
        %tile03 = aie.tile(0, 3)

        aie.pnr_pktflow(%tile12, DMA : 0, {%tile43, %tile03}, DMA : [0, 0], packet_id : 0) {hop_tile_ids = [[[1,2],[1,3],[2,3],[3,3],[4,3]], [[1,2],[1,3],[0,3]]]}
        aie.pnr_pktflow(%tile12, DMA : 1, {%tile43}, DMA : [1], packet_id : 1) {hop_tile_ids = [[[1,2],[1,3],[2,3],[3,3],[4,3]]]}
    }
}
