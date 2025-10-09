// RUN: aie-opt --aie-objectFifo-to-path %s | FileCheck %s
// CHECK-LABEL:   aie.device(npu2) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 3)
// CHECK:           %[[VAL_1:.*]] = aie.tile(1, 4)
// CHECK:           %[[VAL_2:.*]] = aie.tile(0, 3)
// CHECK:           %[[VAL_3:.*]] = aie.tile(1, 2)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_0_nbr_sink_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of_0_nbr_sink_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "of_0_nbr_sink_prod_lock_0"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of_0_nbr_sink_cons_lock_0"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of_1_nbr_sink_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of_1_nbr_sink_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of_1_nbr_sink_prod_lock_0"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of_1_nbr_sink_cons_lock_0"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of_2_nbr_sink_buff_0"} : memref<256xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "of_2_nbr_sink_buff_1"} : memref<256xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "of_2_nbr_sink_prod_lock_0"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "of_2_nbr_sink_cons_lock_0"}
// CHECK:           %[[VAL_16:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_17:.*]] = arith.constant 14 : i32
// CHECK:             %[[VAL_18:.*]] = arith.constant 3 : index
// CHECK:             memref.store %[[VAL_17]], %[[VAL_12]]{{\[}}%[[VAL_18]]] : memref<256xi32>
// CHECK:             memref.store %[[VAL_17]], %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<256xi32>
// CHECK:             memref.store %[[VAL_17]], %[[VAL_4]]{{\[}}%[[VAL_18]]] : memref<256xi32>
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_20:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_20]]] : memref<256xi32>
// CHECK:             %[[VAL_22:.*]] = arith.constant 100 : i32
// CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i32
// CHECK:             %[[VAL_24:.*]] = arith.constant 5 : index
// CHECK:             memref.store %[[VAL_23]], %[[VAL_4]]{{\[}}%[[VAL_24]]] : memref<256xi32>
// CHECK:             aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_26:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_27:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_26]]] : memref<256xi32>
// CHECK:             %[[VAL_28:.*]] = arith.constant 100 : i32
// CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:             %[[VAL_30:.*]] = arith.constant 5 : index
// CHECK:             memref.store %[[VAL_29]], %[[VAL_8]]{{\[}}%[[VAL_30]]] : memref<256xi32>
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             %[[VAL_32:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_33:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_32]]] : memref<256xi32>
// CHECK:             %[[VAL_34:.*]] = arith.constant 100 : i32
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_33]], %[[VAL_34]] : i32
// CHECK:             %[[VAL_36:.*]] = arith.constant 5 : index
// CHECK:             memref.store %[[VAL_35]], %[[VAL_12]]{{\[}}%[[VAL_36]]] : memref<256xi32>
// CHECK:             aie.use_lock(%[[VAL_14]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }


// This test is for verifying that a shared-memory multicast fifo is lowered correctly. In this case all buffers
// are located on sink tiles,
module @nbr_all_sink {
    aie.device(npu2) {
        %tile13 = aie.tile(1, 3)
        %tile14 = aie.tile(1, 4)
        %tile03 = aie.tile(0, 3)
        %tile12 = aie.tile(1, 2)

        aie.objectfifo @of (%tile13, {%tile14, %tile03, %tile12}, [2 : i32, 2 : i32, 2 : i32, 2 : i32]) : !aie.objectfifo<memref<256xi32>>
        aie.objectfifo.allocate @of (%tile14, %tile03, %tile12)

        %core13 = aie.core(%tile13) {
            %inputSubview = aie.objectfifo.acquire @of (Produce, 1) : !aie.objectfifosubview<memref<256xi32>>
            %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

            %val = arith.constant 14 : i32
            %idx = arith.constant 3 : index
            memref.store %val, %input[%idx] : memref<256xi32>

            aie.objectfifo.release @of (Produce, 1)
            aie.end
        }

        %core14 = aie.core(%tile14) {
            %inputSubview = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
            %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

            %idx1 = arith.constant 3 : index
            %d1   = memref.load %input[%idx1] : memref<256xi32>
            %c1   = arith.constant 100 : i32
            %d2   = arith.addi %d1, %c1 : i32
            %idx2 = arith.constant 5 : index
            memref.store %d2, %input[%idx2] : memref<256xi32>

            aie.objectfifo.release @of (Consume, 1)
            aie.end
        }

        %core03 = aie.core(%tile03) {
            %inputSubview = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
            %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

            %idx1 = arith.constant 3 : index
            %d1   = memref.load %input[%idx1] : memref<256xi32>
            %c1   = arith.constant 100 : i32
            %d2   = arith.addi %d1, %c1 : i32
            %idx2 = arith.constant 5 : index
            memref.store %d2, %input[%idx2] : memref<256xi32>

            aie.objectfifo.release @of (Consume, 1)
            aie.end
        }

        %core12 = aie.core(%tile12) {
            %inputSubview = aie.objectfifo.acquire @of (Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
            %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

            %idx1 = arith.constant 3 : index
            %d1   = memref.load %input[%idx1] : memref<256xi32>
            %c1   = arith.constant 100 : i32
            %d2   = arith.addi %d1, %c1 : i32
            %idx2 = arith.constant 5 : index
            memref.store %d2, %input[%idx2] : memref<256xi32>

            aie.objectfifo.release @of (Consume, 1)
            aie.end
        }
    }
}
