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
