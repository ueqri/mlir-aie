module @nbr_mixed {
    aie.device(npu2) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        %tile02 = aie.tile(0, 2)
        %tile11 = aie.tile(1, 1)

        aie.objectfifo @of (%tile12, {%tile13, %tile02, %tile11}, [2 : i32, 2 : i32, 2 : i32, 2 : i32]) : !aie.objectfifo<memref<256xi32>>
        aie.objectfifo.allocate @of (%tile12, %tile02, %tile11)
    }
}