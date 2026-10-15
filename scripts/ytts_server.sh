#!/bin/bash

pushd $CBOX_VENV
source ./bin/activate
popd
nohup python3 ./server/ytts_server.py "Your personal Y-TTS server is now running, enabled and ready for operation!" "./voices/ytts_ref_1.wav" 0.35 0.8 0.7 --server --port $CBOX_PORT --vmap "./voices/vmap.json" > ./ytts_server.log 2>&1 &


