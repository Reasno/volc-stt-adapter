# Reachy wake-word models

`reechy-spk150-steps100k-acc98.15-rec97.50.onnx` 来自 [andyjmorgan/reachy-wake-word](https://github.com/andyjmorgan/reachy-wake-word)，目标词为 `reechy`，用于 openWakeWord ONNX 推理。

## 许可（重要）

- Reachy 社区唤醒模型：**CC BY-NC-SA 4.0**。
- 该许可包含 **NonCommercial（非商业）约束**，并要求署名和相同方式共享。
- 上游代码许可：Apache-2.0。

因此，不应在未完成许可确认的商业产品或商业服务中直接使用该社区模型；商业使用前应取得额外授权或替换为许可兼容的自有模型。

`openwakeword/melspectrogram.onnx` 和 `openwakeword/embedding_model.onnx` 是随项目固定的 openWakeWord 特征模型，避免容器构建或运行时下载。当前运行依赖 `openwakeword==0.6.0`，仅启用 ONNX 路径。

上线前请使用实际设备麦克风、房间、说话人及电机噪声的正负样本运行 `scripts/kws_eval.py`，根据 recall/FPR 确定阈值。
