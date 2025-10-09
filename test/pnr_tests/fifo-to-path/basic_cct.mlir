module @basic_cct {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile02 = aie.tile(0, 2)
        %tile11 = aie.tile(1, 1)

        aie.objectfifo @of0 (%tile12, {%tile13, %tile02, %tile11}, [2 : i32, 2 : i32, 2 : i32, 2 : i32]) {via_DMA = true}: !aie.objectfifo<memref<256xi32>>
        aie.objectfifo @of1 (%tile12, {%tile13}, [2 : i32, 2 : i32]) {via_DMA = true}: !aie.objectfifo<memref<256xi32>>
    }
}