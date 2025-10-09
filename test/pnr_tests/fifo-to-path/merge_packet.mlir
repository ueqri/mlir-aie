module @merge_packet {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile33 = aie.tile(3, 3)
        %tile34 = aie.tile(3, 4)
        %tile35 = aie.tile(3, 5)
        // 1 circuit-switched fifo
        aie.objectfifo @ofc (%tile12, {%tile33}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<256xi32>>
        // 3 packet-switched fifos
        aie.objectfifo @of0 (%tile12, {%tile33}, 2 : i32) {via_DMA = true, packet_id = 0 : i8} : !aie.objectfifo<memref<256xi32>>
        aie.objectfifo @of1 (%tile12, {%tile34}, 2 : i32) {via_DMA = true, packet_id = 1 : i8} : !aie.objectfifo<memref<256xi32>>
        aie.objectfifo @of2 (%tile12, {%tile35}, 2 : i32) {via_DMA = true, packet_id = 2 : i8} : !aie.objectfifo<memref<256xi32>>
    }
}