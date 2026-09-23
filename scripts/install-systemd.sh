#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  printf 'error: this installer must run as root\n' >&2
  exit 1
fi

start_service=false
case ${1-} in
  '') ;;
  --start) start_service=true ;;
  *) printf 'usage: %s [--start]\n' "$0" >&2; exit 2 ;;
esac

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
INSTALL_DIR=/opt/volc-stt-adapter
VENV_DIR=${INSTALL_DIR}/.venv
ENV_DIR=/etc/volc-stt-adapter
ENV_FILE=${ENV_DIR}/adapter.env
SERVICE_FILE=/etc/systemd/system/volc-stt-adapter.service
PYTHON=/venvs/mini_daemon/bin/python

if [[ ! -x ${PYTHON} ]]; then
  printf 'error: Python executable not found: %s\n' "${PYTHON}" >&2
  exit 1
fi
if ! getent passwd pollen >/dev/null; then
  printf 'error: required user pollen does not exist\n' >&2
  exit 1
fi

install -d -o pollen -g pollen -m 0755 "${INSTALL_DIR}"
install -d -o pollen -g pollen -m 0755 /var/cache/volc-stt-adapter
install -d -o pollen -g pollen -m 0755 /var/cache/volc-stt-adapter/tts

install_file() {
  local source=$1 destination=$2 mode=${3:-0644}
  if [[ -e ${destination} && ${source} -ef ${destination} ]]; then
    return
  fi
  install -o pollen -g pollen -m "${mode}" "${source}" "${destination}"
}

# Copy the explicit runtime set only. This is safe when run from INSTALL_DIR and
# deliberately never removes an existing .venv or other on-device state.
for file in volc_stt_adapter.py audio_gate.py wake_word.py reachy_speaker.py requirements.txt; do
  install_file "${SOURCE_DIR}/${file}" "${INSTALL_DIR}/${file}"
done
install -d -o pollen -g pollen -m 0755 "${INSTALL_DIR}/models/openwakeword"
for file in reechy-spk150-steps100k-acc98.15-rec97.50.onnx; do
  install_file "${SOURCE_DIR}/models/${file}" "${INSTALL_DIR}/models/${file}"
done
for file in embedding_model.onnx melspectrogram.onnx; do
  install_file "${SOURCE_DIR}/models/openwakeword/${file}" "${INSTALL_DIR}/models/openwakeword/${file}"
done

if [[ ! -x ${VENV_DIR}/bin/python ]]; then
  "${PYTHON}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --timeout 120 --retries 10 -r "${INSTALL_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install --timeout 120 --retries 10 --no-deps openwakeword==0.6.0
chown -R pollen:pollen "${VENV_DIR}"

install -d -o root -g pollen -m 0750 "${ENV_DIR}"
if [[ ! -e ${ENV_FILE} ]]; then
  install -o root -g pollen -m 0640 "${SOURCE_DIR}/deploy/systemd/adapter.env.example" "${ENV_FILE}"
else
  chown root:pollen "${ENV_FILE}"
  chmod 0640 "${ENV_FILE}"
fi
install -o root -g root -m 0644 "${SOURCE_DIR}/deploy/systemd/volc-stt-adapter.service" "${SERVICE_FILE}"

systemctl daemon-reload
systemctl enable volc-stt-adapter.service
if [[ ${start_service} == true ]]; then
  systemctl restart volc-stt-adapter.service
else
  printf 'Installed and enabled; not started. Configure %s, then run:\n  systemctl start volc-stt-adapter.service\n' "${ENV_FILE}"
fi
