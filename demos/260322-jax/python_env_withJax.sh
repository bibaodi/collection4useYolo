#!/bin/env bash
_pyfile=${1:-nothing}
test ! -f "${_pyfile}" && (echo "file not exist."; exit 0)

conda run -n trainLinkNet18_env python ${_pyfile}
