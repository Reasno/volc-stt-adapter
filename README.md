# Reachy Mini 火山引擎 STT Adapter

把 Reachy Mini Conversation App 使用的 OpenAI Realtime WebSocket 连接代理到原 Hugging Face Realtime 上游，同时在本地截流 16 kHz PCM 音频并交给火山引擎流式 ASR v3。火山中文转录作为用户 `input_text` 注入上游，因此保留原有大模型回复、TTS 和工具调用能力。

## 数据流

```text
Reachy ──OpenAI Realtime──> Adapter ──文本/控制事件──> Hugging Face Realtime
                                └────PCM──> 火山 ASR
Reachy <──音频/TTS/工具事件──── Adapter <───────────────────────┘
       <──火山中文转录事件─────┘
```

## 协议兼容范围

- 监听：`ws://0.0.0.0:8765/v1/realtime`
- 除输入音频外，客户端事件原样转发给 Hugging Face Realtime
- 上游的回复音频、文本、工具调用和生命周期事件原样返回 Reachy
- `session.update` 中输入转录语言强制改为 `zh-CN`
- `input_audio_buffer.append/commit/clear` 在 adapter 终止，不转发给上游
- 火山最终中文转录以 `conversation.item.create(input_text)` 注入上游，并触发 `response.create`
- 上游 input-audio transcription 事件被过滤，避免重复或英文转录
- 音频：16 kHz、单声道、PCM16 little-endian、Base64
- VAD：不再使用 adapter 本地能量阈值；连续音频交给火山 `bigmodel_async`，仅接受二遍识别返回的 `definite=true` 分句

## 凭据

程序按以下顺序读取：

1. 同目录 `.env`（也可用 `ENV_FILE` 指定）及进程环境变量：`VOLC_APP_KEY`、`VOLC_ACCESS_KEY`、`VOLC_RESOURCE_ID`
2. 如果凭据不完整且设置了 `HA_CONFIG_ENTRIES`，从 HA `core.config_entries` 的首个 `volcengine_voice_assistant` / `stt` subentry 读取

程序不会打印凭据。推荐在 HA 小主机直接设置：

```dotenv
HA_CONFIG_ENTRIES=/var/lib/homeassistant/homeassistant/.storage/core.config_entries
```

这样不需要把密钥复制到第二个文件。若选择生成 `.env`，务必执行 `chmod 600 .env`。

## 安装与运行

```bash
cd /var/lib/homeassistant/volc_stt_adapter
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# 编辑 .env；推荐只启用 HA_CONFIG_ENTRIES
.venv/bin/python volc_stt_adapter.py
```

### systemd

将仓库中的 `volc-stt-adapter.service` 安装到 `/etc/systemd/system/` 后：

```bash
systemctl daemon-reload
systemctl enable --now volc-stt-adapter.service
journalctl -u volc-stt-adapter.service -f
```

## Reachy 配置

`/etc/systemd/system/reachy-mini-daemon.service.d/stt-adapter.conf`：

```ini
[Service]
Environment=HF_REALTIME_CONNECTION_MODE=local
Environment=HF_REALTIME_WS_URL=ws://192.168.31.111:8765/v1/realtime
```

然后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart reachy-mini-daemon
```

## 调试

- 查看火山原生 VAD 确认的分句：`journalctl -u volc-stt-adapter.service -f | grep "native VAD definite"`
- 查看详细火山结果：临时设置 `LOG_LEVEL=DEBUG`
- 正常底噪、舵机和电机振动不应产生 `speech_started`；只有 `definite=true` 且文本非空时才发送该事件

## 安全说明

服务默认监听整个局域网且没有客户端认证。只建议在可信 LAN 使用；如需跨网络暴露，应增加反向代理 TLS 和鉴权。

## 实施与验证报告（2026-09-21）

### 1. 源码与协议确认

- 使用密码 SSH 读取 Reachy 实机 snapshot：`ddc309630448a664b0283812ff80048c36966c35`。
- 实机 `huggingface_realtime.py` SHA-256 为 `d033d17d301f83fe89364f05432ab08136c351ae87d3d0d9dbc970273d5f43cd`，与本地上游仓库 commit `f58523b` 完全一致。
- Reachy 发送 `session.update` 和连续 `input_audio_buffer.append`；音频是 16 kHz / mono / PCM16。
- Hugging Face 会话分配器允许匿名硬件会话分配，返回一次性 `connect_url`；adapter 不需要复制 Reachy 的 HF token。
- HA 火山组件使用自定义二进制 v3 协议；鉴权为 `X-Api-App-Key`、`X-Api-Access-Key`、`X-Api-Resource-Id` 和随机 `X-Api-Connect-Id`。
- HA STT subentry 字段确认：`access_key`、`app_key`、`name`、`resource_id`、`url`；未输出字段值。

### 2. 实现

完整实现见 [`volc_stt_adapter.py`](volc_stt_adapter.py)。它包含：

- 上游 Hugging Face Realtime 会话自动分配和全双工代理；
- 除输入音频外的 OpenAI Realtime 控制、回复音频、工具调用事件双向透传；
- 输入 PCM 在 adapter 终止，避免上游英文 STT/VAD 与火山结果竞争；
- 火山最终转录注入为上游 `input_text`，自动触发 `response.create`；
- 输入转录语言强制为 `zh-CN`，并过滤上游 transcription 事件；
- 使用火山 `bigmodel_async` 二遍识别和原生 VAD 分句，不再保留本地 RMS 阈值、前置缓冲或强制 finalize；
- 只有火山返回非空且 `definite=true` 的分句，才向 Reachy 发送 `speech_started`、transcription 和 completed 事件；
- `.env` 与 HA config entries 两种无硬编码凭据读取方式；
- 连接清理、超时、错误事件和结构化日志。

### 3. HA 小主机部署

- 目录：`/var/lib/homeassistant/volc_stt_adapter`
- Python：3.11 venv；因主机原先缺少 `ensurepip`，安装了 Debian `python3-venv`。
- 依赖：`websockets 15.0.1`、`aiohttp 3.14.3`
- 服务：`volc-stt-adapter.service`，已 enable 且 active
- 监听：`0.0.0.0:8765`
- 凭据：`.env` 只配置 `HA_CONFIG_ENTRIES` 路径，运行时读取 HA storage；没有复制或写死实际密钥。
- 上游：`UPSTREAM_MODE=allocator`，每个 Reachy 连接自动获得独立 Hugging Face Realtime 会话。

### 4. Reachy 配置

- `/etc/systemd/system/reachy-mini-daemon.service.d/stt-adapter.conf` 已安装。
- 原有 `transcription-language.conf` 已统一更新为 `zh-CN`，避免 drop-in 文件排序覆盖。
- systemd 生效环境：

```text
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://192.168.31.111:8765/v1/realtime
REALTIME_TRANSCRIPTION_LANGUAGE=zh-CN
```

- `reachy_mini_conversation_app` 当前为 `running`。
- 实机日志确认使用 adapter URL、Realtime session 更新成功、17 个工具已注册。

### 5. 已完成验证

- TCP、systemd、上游会话分配和 WebSocket 连接：通过。
- OpenAI Python SDK 2.28.0 兼容性：通过。
- 火山鉴权、中间结果、最终结果和 VAD 事件顺序：通过。
- 中文完整链路：输入“你好，我是瑞奇机器人，今天天气不错。”，火山 completed 文本完全一致。
- 上游大模型回复：收到中文“收到。”。
- 上游 TTS：收到 10 KB 以上 PCM 回复音频。
- 工具调用透传：测试函数 `turn_on_light` 收到 `response.function_call_arguments.done`。
- Reachy 真人语音：实机日志已记录多句中文并注入上游，包括“现在几点了？”“你看看左边。”“左边有什么？”。

### 6. VAD 误触发修复（2026-09-21 02:17）

根因不是 Reachy 的运动状态，而是 adapter 原先使用 RMS 能量阈值（最初 450）自行切分音频，并在 800 ms“静音”后主动结束火山请求。机械振动很容易超过能量阈值；强制结束请求又会促使 ASR 对噪声做最终解码，生成伪文本，随后 adapter 注入 `response.create`，Reachy 因此进入 talking。原 Conversation App 自身并不做这个本地能量判断，只消费上游服务端事件。

修复：

- 完全删除 adapter 本地 RMS VAD、阈值、静音计数、前置缓冲和强制 finalize。
- 火山端点从 `bigmodel` 切换为双向流式优化版 `bigmodel_async`。
- 开启 `enable_nonstream=true`，使用火山默认 800 ms 原生 VAD 分句和二遍识别。
- 只接受 `utterances[].definite == true` 且文本非空的结果。
- 只有收到上述 definite 分句后才向 Reachy 依次发送 `speech_started`、transcription delta、`speech_stopped` 和 completed，再将文本注入 Hugging Face 上游。
- 按用户要求，没有增加运动状态静默窗口。

验证：

- 合成中文语音无手工 commit 即得到 definite 文本，完整对话回复成功。
- 110 Hz、幅度 6000、持续 5 秒的高能机械振动模拟音频，随后 2 秒静音：`speech_started=0`、completed=0、`response.created=0`。
- 生产服务已重启并处于 active；日志无 ERROR/Traceback。
