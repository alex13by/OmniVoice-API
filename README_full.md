# OmniVoice Full-Featured Server

完整功能版 OmniVoice 服务器，保留官方所有 Web UI 功能，同时添加 OpenAI 兼容 API。

## 特点

- **完整 Web UI**：声音克隆、声音设计、自动声音三个模式
- **高级参数**：扩散步数、引导比例、语速、固定时长、去噪等
- **OpenAI API**：兼容 AnythingLLM 等工具
- **模型共享**：Web UI 和 API 共用模型，节省内存

## 安装

1. 将 `omnivoice_full_server.py` 放到 `D:\AI\OmniVoice` 目录
2. 确保已安装依赖：
   ```bash
   uv pip install fastapi uvicorn python-multipart
   ```

## 启动

双击 `start_full.bat` 或运行：

```bash
cd D:\AI\OmniVoice
uv run python omnivoice_full_server.py
```

## 功能说明

### Web UI 标签页

1. **Voice Cloning（声音克隆）**
   - 上传参考音频（3-15秒）
   - 可选输入参考文本（留空自动识别）
   - 克隆声音合成新文本

2. **Voice Design（声音设计）**
   - 通过描述创建声音
   - 支持：性别、年龄、音调、风格、口音、方言
   - 例：`female, young adult, high pitch, british accent`

3. **Auto Voice（自动声音）**
   - 随机选择声音
   - 适合快速测试

### 高级参数

- **Diffusion Steps**：扩散步数（4-64，默认32，越大质量越好速度越慢）
- **Guidance Scale**：引导比例（0-10，默认2.0）
- **Speed**：语速（0.5-2.0，默认1.0）
- **Fixed Duration**：固定时长秒数（0=自动）
- **Add denoise token**：添加去噪token（默认开启）

### API 端点

- `POST /v1/audio/speech` - TTS（OpenAI兼容）
- `POST /v1/audio/clone` - 声音克隆（支持文件上传）
- `GET /v1/models` - 模型列表
- `GET /health` - 健康检查

## AnythingLLM 配置

1. 设置 → Voice & Speech → TTS Provider: **Generic OpenAI**
2. 配置：
   - **API Endpoint**: `http://localhost:5050/v1`
   - **API Key**: `omnivoice`
   - **Model**: `tts-1`
   - **Voice**: `alloy` 或描述如 `female, low pitch`

## 命令行参数

```bash
# 自定义端口
uv run python omnivoice_full_server.py --web_port 8080 --api_port 5051

# 创建公开链接（Gradio）
uv run python omnivoice_full_server.py --share
```
