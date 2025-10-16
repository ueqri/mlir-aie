# -*- Python -*-
import os
import lit.formats
import tempfile

# Basic config
config.name = "pnr-test"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']

# Environment setup
config.environment['PATH'] = os.environ['PATH']

# Use temporary folder to avoid polluting source tree
config.test_exec_root = tempfile.mkdtemp(prefix="lit_exec_")

# Keep test_source_root as the directory containing lit.cfg.py
config.test_source_root = os.path.dirname(__file__)

# Make test output fully visible in terminal
config.maxIndividualTestTime = 0
config.singlePhase = True
config.useProgressBar = True

# Enable showing stdout/stderr on failure
config.showOutput = True

# Prevent timing log creation
config.noTiming = True  # Custom hint for compatibility with --no-timing
