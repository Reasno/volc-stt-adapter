# Reachy Mini 火山引擎 STT Adapter

该服务代理 Reachy Mini 的 OpenAI Realtime WebSocket：控制、模型回复、TTS 与工具调用继续透传到原 Hugging Face Realtime 上游；输入的 16 kHz、单声道、PCM16 little-endian 音频由本地 openWakeWord 检测，再交给火山流式 ASR v3。火山 `definite=true` 中文分句以 `input_text` 注入原上游。

## KWS v2 模式

`KWS_MODE` 严格支持三个值，默认 `shadow`：

- `off`：不加载 KWS 模型，音频持续发送火山，所有 definite 分句放行。
- `shadow`：音频仍持续发送火山，KWS 与 speaker gate 只计算并记录 `enforce_allow/reason`，实际全部放行。用于灰度观察。
- `enforce`：唤醒前不创建火山 stream；本地命中后创建 stream，并发送按样本索引保存的 1.5 秒预滚。只有符合授权规则的 definite 分句才注入 Reachy/HF 上游。

唤醒记录当前火山 stream `generation`、本地 sample index 和时间轴。只有时间区间（含 `KWS_MATCH_TOLERANCE_MS`）覆盖唤醒点的 definite 分句才能绑定 `speaker_id`。授权身份是 `(stream_generation, speaker_id)`；同 speaker 的合法分句将 30 秒窗口续期，其他或缺失 speaker 的后续分句被丢弃。重连、`input_audio_buffer.clear` 或 generation 变化立即撤销授权。`TRIGGERED` 默认 3 秒超时，之后经过短暂 `CLOSING` 再回到 `SLEEPING`，不会形成永久开门反馈环。

若唤醒后的首个 definite 分句缺少可靠时间轴，enforce 会保守地只放行该句，不创建 30 秒 speaker 授权，并输出 warning。KWS 模型加载失败会输出 ERROR，并把该连接明确降级为 `off`，不会保留半初始化 detector。

## 安装

### 原生 Python 3.11 / aarch64

openWakeWord 的包元数据依赖 `tflite-runtime`，但本项目只使用 ONNX；Python 3.11/aarch64 通常没有对应 TFLite wheel，因此必须跳过其依赖并显式安装 ONNX 依赖：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps openwakeword==0.6.0
cp .env.example .env
# 原生运行默认模型可写相对路径：./models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx
.venv/bin/python volc_stt_adapter.py
```

### Docker / Compose

```bash
cp .env.example .env
# 容器内设置 KWS_MODEL_PATH=/app/models/reechy-spk150-steps100k-acc98.15-rec97.50.onnx
# 所有 Compose KWS 项均以 ${VAR:-default} 读取，不会硬覆盖 .env。
docker compose up -d --build
```

Dockerfile 同样使用 Python 3.11、显式 `onnxruntime` 依赖和 `openwakeword==0.6.0 --no-deps`，无需 TFLite wheel。

## 配置

核心参数见 [`.env.example`](.env.example)：

- `KWS_MODE=shadow`
- `KWS_MODEL_PATH`：原生路径与容器路径不同，见上文。
- `KWS_THRESHOLD=0.5`
- `KWS_PREROLL_SECONDS=1.5`
- `KWS_TRIGGER_TIMEOUT_SECONDS=3`
- `KWS_SPEAKER_WINDOW_SECONDS=30`
- `KWS_MATCH_TOLERANCE_MS=400`
- `KWS_QUEUE_FRAMES=32`：每连接有界推理队列；满时显式丢弃新帧并记录 warning，避免无限积压。

火山凭据可直接设置 `VOLC_APP_KEY/VOLC_ACCESS_KEY/VOLC_RESOURCE_ID`，也可设置 `HA_CONFIG_ENTRIES` 从 Home Assistant storage 读取。服务不会打印凭据。

## 灰度与指标

建议流程：

1. `off` 验证现有火山 transient reconnect 与 Reachy 上游代理行为。
2. 使用真实房间、家庭成员、设备电机噪声录制正负样本，通过 `scripts/kws_eval.py` 调阈值。
3. `shadow` 观察至少一个完整业务周期，比较 gate decision 与实际注入文本。
4. 满足指标后切换 `enforce`；保留快速回退 `shadow/off` 的配置能力。

重点监控：KWS score/触发数、正样本 recall、负样本 FPR、推理队列丢帧、trigger timeout、缺失 timeline/speaker、speaker 拒绝数、generation reset、模型加载失败降级次数。

离线评测会对尾帧补零，并输出每条峰值及 TP/FN/FP/TN、recall、FPR：

```bash
.venv/bin/python scripts/kws_eval.py \
  --positive samples/reachy-1.wav --positive samples/reachy-2.wav \
  --negative samples/noise-1.wav --negative samples/conversation-1.wav \
  --threshold 0.5
```

## 模型许可

Reachy 社区唤醒模型来自 `andyjmorgan/reachy-wake-word`，采用 **CC BY-NC-SA 4.0**，包含**非商业（NonCommercial）约束**；衍生与共享还须遵守署名及相同方式共享。openWakeWord 特征模型与来源说明见 [`models/README.md`](models/README.md)。在任何商业场景启用前必须完成许可评估或替换模型。

## 协议与运维

- 监听：`ws://0.0.0.0:8765/v1/realtime`
- 除输入音频外，客户端事件原样转发；输入 audio append/commit/clear 在 adapter 终止。
- 上游 input-audio transcription 事件被过滤，避免重复转录。
- 火山继续使用 `bigmodel_async`、原生 VAD、SSD speaker 信息和既有 transient reconnect。
- `LOG_LEVEL=DEBUG` 可查看原始火山结果；INFO 可查看 KWS 与 gate decision。
- 服务默认无客户端认证，仅建议可信 LAN；跨网络应增加 TLS 与鉴权。

## 主动播报 HTTP API

服务在 STT WebSocket 同一进程中另启 HTTP 监听（默认 `0.0.0.0:8766`）：

- `GET /health`：只报告 adapter HTTP 服务存活，不访问 Reachy、火山或 daemon。
- `POST /speak`：JSON 为 `{"text":"要播报的文本","bypass_gate":true}`；文本 trim 后不能为空，UTF-8 编码不超过 4000 字节。`bypass_gate` 必须是 JSON boolean，默认 `false`。
- 成功返回包含 `ok`、`route=conversation`、`gate_requested/gate_opened/gate_reason` 和 `request_id`。`bypass_gate=true` 会在播报成功后同时绕过 KWS 与火山 transcript keyword gate：enforce 且恰有一个 live Reachy 连接时立即启动火山流，并将下一条任意说话人的 utterance 直接放行、绑定为活跃 speaker，随后进入正常 30 秒窗口。0 个、多连接或 off/shadow 只返回对应 reason，不使成功播报失败。
- `/speak` 只调用 Reachy `conversation.say`；conversation 不可用或调用失败时返回 502，不再调用 Volc TTS 或 daemon `play_sound`。
- 输入错误返回 400；播报阶段总超时返回 502。播报成功后的 gate opening 超时/异常仍返回 200，避免 HA 重试造成重复播报。

请求由进程内 `asyncio.Lock` 串行，且受 `SPEAK_TOTAL_TIMEOUT_SECONDS` 总超时约束。

### curl

```bash
curl -sS http://127.0.0.1:8766/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"晚饭准备好了","bypass_gate":true}'
```

`/health` 与 `/speak` 均供可信内网直接访问。请求体不能指定 daemon、conversation 或 TTS URL，避免形成 SSRF 入口。

### Home Assistant `rest_command`

Home Assistant 可直接调用可信内网中的 adapter：

```yaml
# configuration.yaml
rest_command:
  reachy_speak:
    url: "http://127.0.0.1:8766/speak"
    method: POST
    headers:
      content-type: "application/json"
    payload: '{"text": {{ text | tojson }}, "bypass_gate": {{ bypass_gate | default(false) | tojson }} }'

# automation/script 调用
# action: rest_command.reachy_speak
# data:
#   text: "晚饭准备好了"
#   bypass_gate: true
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SPEAK_HTTP_HOST` / `SPEAK_HTTP_PORT` | `0.0.0.0` / `8766` | HTTP 监听地址 |
| `SPEAK_TOTAL_TIMEOUT_SECONDS` | `45` | 排队、conversation、fallback 的请求总超时 |
| `REACHY_CONVERSATION_RPC_URL` | `ws://192.168.31.94:7860/rpc` | conversation JSON-RPC WebSocket |
| `REACHY_DAEMON_URL` | `http://192.168.31.94:8000` | daemon API base URL |
| `VOLC_TTS_URL` | `wss://openspeech.bytedance.com/api/v3/tts/bidirection` | Seed-TTS v3 WebSocket |
| `VOLC_TTS_RESOURCE_ID` | `seed-tts-2.0` | TTS resource id |
| `VOLC_TTS_VOICE` | `zh_female_vv_uranus_bigtts` | HA 集成中 `seed-tts-2.0` 的默认中文音色，可通过环境变量覆盖；显式设为空时仅 fallback 明确报错，不阻断启动 |
| `VOLC_TTS_TIMEOUT_SECONDS` | `30` | TTS 整体超时 |
| `VOLC_TTS_MAX_AUDIO_BYTES` | `16777216` | 单次合成音频硬上限；超过上限或空结果不会缓存 |
| `VOLC_TTS_CACHE_ENTRIES` | `100` | fallback 完整音频的进程内 LRU 条目上限；`0` 禁用。缓存键包含 resource、voice、音频参数和文本，命中后仍会上传并播放 |
| `DAEMON_SOUND_TIMEOUT_SECONDS` | `10` | daemon upload、play、delete 各自的请求超时 |
| `DAEMON_SOUND_CLEANUP_DELAY_SECONDS` | `300` | 上传音频延迟清理秒数 |

TTS 复用 STT 的 `VOLC_APP_KEY/VOLC_ACCESS_KEY`，并按现有 HA 集成使用 `X-Api-App-Key/X-Api-Access-Key` 请求头；仓库不保存 secret。当前开发环境无法连接 `192.168.31.94` 实机，本功能只进行了协议解析和全 mock 网络验证；conversation handler、实际音色授权、daemon 上传/播放/延迟删除仍需在受控实机环境验证。


## Reachy Mini 原生 systemd 部署（Debian aarch64）

此方案直接使用 Reachy Mini 上的 Python 3.12（`/venvs/mini_daemon/bin/python`）创建应用独立 venv，**不使用 Docker**。应用安装到 `/opt/volc-stt-adapter`，秘密配置保存在 `/etc/volc-stt-adapter/adapter.env`；STT 只监听 `127.0.0.1:8765`，`/speak` 监听 `0.0.0.0:8766` 供 LAN 调用。

### 安装

在仓库根目录执行：

```bash
sudo ./scripts/install-systemd.sh
sudoedit /etc/volc-stt-adapter/adapter.env
# 至少填写 VOLC_APP_KEY、VOLC_ACCESS_KEY。
sudo systemctl start volc-stt-adapter.service
```

安装器幂等同步明确列出的程序、requirements 和模型，不删除现场 `.venv`。它会执行 `daemon-reload` 和 `enable`，但默认不启动，避免示例凭据为空时反复失败；只有显式执行 `sudo ./scripts/install-systemd.sh --start` 才会立即 restart/start。已有 `adapter.env` 不会被覆盖。

systemd unit 通过 `EnvironmentFile=/etc/volc-stt-adapter/adapter.env` 注入配置；程序本身不会自动查找这个系统级文件。若需要在 shell 中手工运行，必须先显式导出其中的变量：

```bash
set -a
source /etc/volc-stt-adapter/adapter.env
set +a
/opt/volc-stt-adapter/.venv/bin/python /opt/volc-stt-adapter/volc_stt_adapter.py
```

### 将 Conversation 上游切到 localhost adapter

创建或更新 daemon drop-in：

```bash
sudo install -d -m 0755 /etc/systemd/system/reachy-mini-daemon.service.d
sudoedit /etc/systemd/system/reachy-mini-daemon.service.d/stt-adapter.conf
```

内容如下：

```ini
[Service]
Environment="HF_REALTIME_CONNECTION_MODE=local"
Environment="HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime"
```

应用 drop-in（先启动 adapter，再重启 daemon）：

```bash
sudo systemctl daemon-reload
sudo systemctl start volc-stt-adapter.service
sudo systemctl restart reachy-mini-daemon.service
```

adapter 自身通过 `ws://127.0.0.1:7860/rpc` 调用 conversation，并通过 `http://127.0.0.1:8000` 调用 daemon。unit 仅使用 `NoNewPrivileges` 和 `PrivateTmp` 做基础加固，不启用会阻断 localhost、外部火山/HF 服务或 `/opt/volc-stt-adapter/models` 的网络/文件系统 sandbox。

### 验证与日志

```bash
systemctl is-enabled volc-stt-adapter.service
systemctl status volc-stt-adapter.service reachy-mini-daemon.service
journalctl -u volc-stt-adapter.service -n 100 --no-pager
curl -fsS http://127.0.0.1:8766/health
ss -lnt | grep -E '127\.0\.0\.1:8765|0\.0\.0\.0:8766'
curl -sS http://127.0.0.1:8766/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"systemd 部署验证"}'
```

预期 STT 不对 LAN 暴露，而 `/speak` 可从可信 LAN 通过 `http://<reachy-mini-ip>:8766/speak` 访问。防火墙如已启用，只需为可信网段放行 TCP 8766，不要放行 8765。

### Home Assistant `/speak` URL 迁移

如果 Home Assistant 不在 Reachy Mini 本机，把 `rest_command` URL 从旧 adapter/Docker 地址迁移为：

```yaml
url: "http://<reachy-mini-ip>:8766/speak"
```

`/health` 与 `/speak` 均无需 Authorization，仅应暴露给可信内网。若 Home Assistant 与服务确实同机，才使用 `http://127.0.0.1:8766/speak`。

### 回滚

先让 daemon 恢复内置 Hugging Face 上游，再停用 adapter：

```bash
sudo rm -f /etc/systemd/system/reachy-mini-daemon.service.d/stt-adapter.conf
sudo systemctl daemon-reload
sudo systemctl restart reachy-mini-daemon.service
sudo systemctl disable --now volc-stt-adapter.service
```

上述操作保留 `/etc/volc-stt-adapter/adapter.env`、`/opt/volc-stt-adapter` 和独立 `.venv`，便于再次启用。确认无需保留后再人工删除；回滚不需要 Docker，也不应修改 Reachy daemon 的 Python venv。
