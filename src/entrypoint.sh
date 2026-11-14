#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Starting GraphRAG chatbot"
exec streamlit run main.py "$@"
