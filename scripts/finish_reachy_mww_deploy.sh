#!/usr/bin/env bash
# Finish microWakeWord runtime deployment to Reachy Mini.
# Run this on the Mac once ~/.taterwakewordtrainer/app/current/trained_wake_words/reachy.tflite exists.
set -euo pipefail

MODEL_SRC="${HOME}/.taterwakewordtrainer/app/current/trained_wake_words/reachy.tflite"
JUMP="${JUMP:?set JUMP to HA host, e.g. JUMP=root@homeassistant.tail023c3.ts.net}"
REACHY_USER="pollen"
REACHY_HOST="${REACHY_HOST:?set REACHY_HOST to Reachy LAN IP, e.g. REACHY_HOST=192.168.x.y}"
REMOTE_MODEL_PATH="/opt/volc-stt-adapter/models/reachy.tflite"
ENV_FILE="/etc/volc-stt-adapter/adapter.env"

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "error: model not found at ${MODEL_SRC}"
  echo "Check trainer status: curl -s http://127.0.0.1:8789/api/train_status | python3 -m json.tool | tail -30"
  exit 1
fi

echo "== 1/4 uploading model to HA host =="
scp -o StrictHostKeyChecking=no "${MODEL_SRC}" "${JUMP}:/tmp/reachy.tflite"

echo "== 2/4 relaying model to Reachy =="
ssh -o StrictHostKeyChecking=no "${JUMP}" \
  "scp -o StrictHostKeyChecking=no /tmp/reachy.tflite ${REACHY_USER}@${REACHY_HOST}:/tmp/reachy.tflite && rm /tmp/reachy.tflite"

echo "== 3/4 installing model + flipping KWS_RUNTIME on Reachy =="
ssh -o StrictHostKeyChecking=no "${JUMP}" "ssh -o StrictHostKeyChecking=no ${REACHY_USER}@${REACHY_HOST} '
set -euo pipefail
echo root | sudo -S install -o pollen -g pollen -m 0644 /tmp/reachy.tflite ${REMOTE_MODEL_PATH}
rm /tmp/reachy.tflite
echo root | sudo -S cp ${ENV_FILE} ${ENV_FILE}.bak.\$(date +%Y%m%d_%H%M%S)
echo root | sudo -S python3 - <<PYEOF
import re, pathlib
p = pathlib.Path(\"${ENV_FILE}\")
txt = p.read_text()
def upsert(k, v):
    global txt
    pat = re.compile(rf\"^{re.escape(k)}=.*$\", re.M)
    if pat.search(txt):
        txt = pat.sub(f\"{k}={v}\", txt)
    else:
        txt += f\"\n{k}={v}\n\"
upsert(\"KWS_RUNTIME\", \"microwakeword\")
upsert(\"KWS_MODEL_PATH\", \"${REMOTE_MODEL_PATH}\")
p.write_text(txt)
PYEOF
echo --- new KWS env ---
grep -E \"^KWS_RUNTIME|^KWS_MODEL_PATH\" ${ENV_FILE}
echo --- restart ---
echo root | sudo -S systemctl restart volc-stt-adapter.service
sleep 3
systemctl status volc-stt-adapter.service --no-pager | head -15
echo --- recent logs ---
echo root | sudo -S journalctl -u volc-stt-adapter.service -n 20 --no-pager
'"
echo "== 4/4 done =="
