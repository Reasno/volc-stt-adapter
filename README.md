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
- `POST /speak`：JSON 为 `{"text":"要播报的文本","open_gate":true,"allow_tts_fallback":true}`；文本 trim 后不能为空，UTF-8 编码不超过 4000 字节。两个 option 都必须是 JSON boolean：`open_gate` 默认 `false`，`allow_tts_fallback` 默认 `true`；字符串 `"true"/"false"` 会返回 400。
- 成功返回包含 `ok`、实际 `route`、`fallback_allowed`、`gate_requested/gate_opened/gate_reason` 和 `request_id`。`open_gate=true` 只在播报完整成功后生效：enforce 且恰有一个 live Reachy 连接时，为下一句回复开启现有 30 秒 speaker 窗口；0 个、多连接或 off/shadow 只返回对应 reason，不使成功播报失败。
- `allow_tts_fallback=false` 时只调用 `conversation.say`；若 conversation 不可用则返回 502、`reason=fallback_disabled`，不会调用 Volc TTS/daemon，也不会开 gate。
- 输入错误返回 400；鉴权失败返回 401；conversation 与 TTS/daemon 两条路径均失败或播报阶段总超时返回 502。播报成功后的 gate opening 超时/异常仍返回 200，避免 HA 重试造成重复播报。

请求由进程内 `asyncio.Lock` 串行，且受 `SPEAK_TOTAL_TIMEOUT_SECONDS` 总超时约束。路由首先短连接 `REACHY_CONVERSATION_RPC_URL` 调用 `conversation.say`；只有收到匹配 JSON-RPC id 的成功 result 才结束。连接失败、`not_running` 或其他 RPC error 时，才使用 Seed-TTS 2.0 合成完整 MP3，上传 daemon 并调用 `play_sound`。conversation 成功时绝不会调用 TTS。上传文件使用唯一名称，播放成功后默认延迟 300 秒删除，避免播放中删除。

### curl

```bash
curl -sS http://127.0.0.1:8766/speak \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${SPEAK_API_TOKEN}" \
  -d '{"text":"晚饭准备好了","open_gate":true,"allow_tts_fallback":true}'
```

`SPEAK_API_TOKEN` 为空时允许可信内网免鉴权调用，并在进程生命周期内警告一次；非空时必须发送 Bearer token。请求体不能指定 daemon、conversation 或 TTS URL，避免形成 SSRF 入口。

### Home Assistant `rest_command`

将 token 放入 HA `secrets.yaml`，不要硬编码：

```yaml
# configuration.yaml
rest_command:
  reachy_speak:
    url: "http://127.0.0.1:8766/speak"
    method: POST
    headers:
      authorization: "Bearer {{ token }}"
      content-type: "application/json"
    payload: '{"text": {{ text | tojson }}, "open_gate": {{ open_gate | default(false) | tojson }}, "allow_tts_fallback": {{ allow_tts_fallback | default(true) | tojson }} }'

# automation/script 调用
# action: rest_command.reachy_speak
# data:
#   text: "晚饭准备好了"
#   open_gate: true
#   allow_tts_fallback: true
#   token: !secret reachy_speak_api_token
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SPEAK_HTTP_HOST` / `SPEAK_HTTP_PORT` | `0.0.0.0` / `8766` | HTTP 监听地址 |
| `SPEAK_API_TOKEN` | 空 | 可选 Bearer token |
| `SPEAK_TOTAL_TIMEOUT_SECONDS` | `45` | 排队、conversation、fallback 的请求总超时 |
| `REACHY_CONVERSATION_RPC_URL` | `ws://192.168.31.94:7860/rpc` | conversation JSON-RPC WebSocket |
| `REACHY_DAEMON_URL` | `http://192.168.31.94:8000` | daemon API base URL |
| `VOLC_TTS_URL` | `wss://openspeech.bytedance.com/api/v3/tts/bidirection` | Seed-TTS v3 WebSocket |
| `VOLC_TTS_RESOURCE_ID` | `seed-tts-2.0` | TTS resource id |
| `VOLC_TTS_VOICE` | `zh_female_vv_uranus_bigtts` | HA 集成中 `seed-tts-2.0` 的默认中文音色，可通过环境变量覆盖；显式设为空时仅 fallback 明确报错，不阻断启动 |
| `VOLC_TTS_TIMEOUT_SECONDS` | `30` | TTS 整体超时 |
| `VOLC_TTS_MAX_AUDIO_BYTES` | `16777216` | 单次合成音频硬上限 |
| `DAEMON_SOUND_TIMEOUT_SECONDS` | `10` | daemon upload、play、delete 各自的请求超时 |
| `DAEMON_SOUND_CLEANUP_DELAY_SECONDS` | `300` | 上传音频延迟清理秒数 |

TTS 复用 STT 的 `VOLC_APP_KEY/VOLC_ACCESS_KEY`，并按现有 HA 集成使用 `X-Api-App-Key/X-Api-Access-Key` 请求头；仓库不保存 secret。当前开发环境无法连接 `192.168.31.94` 实机，本功能只进行了协议解析和全 mock 网络验证；conversation handler、实际音色授权、daemon 上传/播放/延迟删除仍需在受控实机环境验证。
