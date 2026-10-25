#!/usr/bin/env bash
# Canonical Dashboard Launch Script for Financial News Sentiment + RAG Analyst

PORT=8501
URL="http://localhost:${PORT}"

# Check if TCP port 8501 is actively listening
PORT_PID=$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null | head -n 1)

if [ -n "$PORT_PID" ]; then
    IS_STREAMLIT=$(ps -p "$PORT_PID" -o command= 2>/dev/null | grep -E "streamlit|app\.py")
    if [ -n "$IS_STREAMLIT" ]; then
        echo "=================================================="
        echo "Dashboard is already running on canonical port ${PORT}:"
        echo "${URL}"
        echo "=================================================="
        exit 0
    else
        echo "Error: Port ${PORT} is occupied by another process (PID: ${PORT_PID})."
        echo "Please free port ${PORT} to launch the dashboard on the canonical URL."
        exit 1
    fi
fi

mkdir -p .streamlit
cat << 'EOF' > .streamlit/credentials.toml
[general]
email = ""
[credentials]
email = ""
EOF

cat << 'EOF' > .streamlit/config.toml
[server]
headless = true
enableCORS = false
fileWatcherType = "none"

[browser]
gatherUsageStats = false
EOF

echo "Launching Financial News Sentiment + RAG Analyst Dashboard..."
echo "Canonical URL: ${URL}"
echo "=================================================="

# Preserve host Hugging Face model cache path before redirecting HOME
REAL_HOME="${HOME:-/Users/domaajay}"
export HF_HOME="${HF_HOME:-$REAL_HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

# Direct HOME to workspace directory so Streamlit writes credentials locally
export HOME="$PWD"
export STREAMLIT_CONFIG_DIR="$PWD/.streamlit"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"

./venv/bin/streamlit run app.py --server.port ${PORT} --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
