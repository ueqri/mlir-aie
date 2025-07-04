#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

add_mlir_dialect_library(
  AIETransforms
  AIEAssignBuffers.cpp
  AIEAssignBufferDescriptorIDs.cpp
  AIEAssignLockIDs.cpp
  AIEFindFlows.cpp
  AIEPathFinder.cpp
  AIECreatePathFindFlows.cpp
  AIECoreToStandard.cpp
  AIECanonicalizeDevice.cpp
  AIEExtractObjectFifo.cpp
  AIEPlaceTiles.cpp
  AIELocalizeLocks.cpp
  AIENormalizeAddressSpaces.cpp
  AIEVectorOpt.cpp
  AIEObjectFifoStatefulTransform.cpp
  AIEObjectFifoRegisterProcess.cpp
  AIELowerCascadeFlows.cpp
  AIEGenerateColumnControlOverlay.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIENormalizeAddressSpacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIEPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect)

target_include_directories(AIETransforms PRIVATE ${AIE_SOURCE_DIR}/third_party/json)