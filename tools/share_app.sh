#!/usr/bin/env bash
# Run the app and put it on a public URL, without Streamlit Cloud and without installing anything.
#
#     bash tools/share_app.sh
#
# It starts Streamlit on localhost:8501, opens an SSH tunnel through localhost.run, and prints the
# public address. The address changes every run and dies when you press Ctrl-C, which is what you
# want for a demo. Your laptop has to stay awake and online while it is up.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$HOME/.venvs/isaf/bin/python}"
PORT="${PORT:-8501}"

"$PYTHON" -m streamlit run app/streamlit_app.py \
    --server.port "$PORT" --server.headless true > /tmp/isaf_app.log 2>&1 &
APP=$!
trap 'kill $APP 2>/dev/null || true' EXIT

printf "starting the app"
for _ in $(seq 40); do
    if curl -fs "http://localhost:$PORT/_stcore/health" > /dev/null 2>&1; then break; fi
    printf "."; sleep 1
done
echo " ready on http://localhost:$PORT"
echo "opening the public tunnel, the address appears below"
echo

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 -R 80:localhost:"$PORT" nokey@localhost.run
