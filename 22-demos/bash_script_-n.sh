#!/bin/bash

set -e

# Parse command line arguments
TARGET="$1"  # Optional target parameter

echo "[$TARGET]"
if [[ -n "$TARGET" ]] ; then
    echo "Usage: $0 " >&2
fi

# Validate target if provided
if [[ -n "$TARGET" ]] && [[ ! "$TARGET" =~ ^(stable|latest|[0-9]+\.[0-9]+\.[0-9]+(-[^[:space:]]+)?)$ ]]; then
    echo "Usage: $0 [stable|latest|VERSION]" >&2
    exit 1
fi

echo "..."
