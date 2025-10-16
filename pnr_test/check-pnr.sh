#!/bin/bash
# Aggregate lit results for all .mlir files in pnr_test
lit "$(dirname "$0")" -v