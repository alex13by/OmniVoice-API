import os

# ==================== 1. 强制模型路径锁定 (必须放在最前面) ====================
project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, "models")

if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

# 强制设置 HuggingFace 缓存路径，防止占用C盘
os.environ["HF_HOME"] = model_dir
os.environ["HF_HUB_CACHE"] = model_dir
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# =========================================================================

import io
import gc
import json
import shutil
import torch
import torchaudio
import tempfile
import argparse
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from threading import Thread

# 导入 FastAPI 相关
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 导入硬件录音库
try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("提示: 若需使用服务器硬件录音，请安装: pip install sounddevice soundfile")

# 导入 Gradio 和 OmniVoice
import gradio as gr
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ==================== 2. 基础配置与持久化管理 ====================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_FILE = os.path.join(project_root, "api_config.json")
SAVED_VOICES_DIR = os.path.join(project_root, "saved_voices")
if not os.path.exists(SAVED_VOICES_DIR):
    os.makedirs(SAVED_VOICES_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device == "cuda" else torch.float32

# 加载模型
print(f"正在加载 OmniVoice 模型至 {device}...")
print(f"模型存储路径锁定在: {model_dir}")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=dtype,
    load_asr=True,
    cache_dir=model_dir
)
sampling_rate = model.sampling_rate
print("模型加载完成。")

# 官方列表定义
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "gender": {"label": "性别 / Gender", "choices": ["Male / 男", "Female / 女"]},
    "age": {"label": "年龄 / Age", "choices": ["Child / 儿童", "Teenager / 少年", "Young Adult / 青年", "Middle-aged / 中年", "Elderly / 老年"]},
    "pitch": {"label": "音调 / Pitch", "choices": ["Very Low Pitch / 极低音调", "Low Pitch / 低音调", "Moderate Pitch / 中音调", "High Pitch / 高音调", "Very High Pitch / 极高音调"]},
    "style": {"label": "风格 / Style", "choices": ["Whisper / 耳语"]},
    "accent": {"label": "英文口音 / English Accent", "choices": ["American Accent / 美式口音", "Australian Accent / 澳大利亚口音", "British Accent / 英国口音", "Chinese Accent / 中国口音", "Canadian Accent / 加拿大口音", "Indian Accent / 印度口音", "Korean Accent / 韩国口音", "Russian Accent / 俄罗斯口音", "Japanese Accent / 日本口音"]},
    "dialect": {"label": "中文方言 / Chinese Dialect", "choices": ["Henan Dialect / 河南话", "Shaanxi Dialect / 陕西话", "Sichuan Dialect / 四川话", "Guizhou Dialect / 贵州话", "Yunnan Dialect / 云南话", "Gansu Dialect / 甘肃话", "Qingdao Dialect / 青岛话", "Northeast Dialect / 东北话"]}
}

def load_api_config():
    """从 JSON 加载 API 全局配置"""
    default_config = {
        "mode": "design", 
        "voice_design": {k: "Auto" for k in _CATEGORIES.keys()},
        "voice_clone": {"ref_audio_path": "", "ref_text": ""},
        "generation": {"num_step": 32, "guidance_scale": 2.0, "speed": 1.0, "language": "Auto", "denoise": True, "preprocess": True, "postprocess": True}
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                if "mode" in user_cfg: default_config["mode"] = user_cfg["mode"]
                if "voice_design" in user_cfg: default_config["voice_design"].update(user_cfg["voice_design"])
                if "voice_clone" in user_cfg: default_config["voice_clone"].update(user_cfg["voice_clone"])
                if "generation" in user_cfg: default_config["generation"].update(user_cfg["generation"])
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    return default_config

def save_api_config(config_dict):
    """保存配置到磁盘"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        return "✅ 配置保存成功！API 立即生效。"
    except Exception as e:
        return f"❌ 保存失败: {e}"

# ==================== 3. 核心推理与录音逻辑 ====================

def build_instruct_text(values_dict: dict):
    """将属性字典转为模型可识别的指令"""
    parts = []
    for key, val in values_dict.items():
        if val and val != "Auto":
            if " / " in val:
                en, zh = val.split(" / ", 1)
                # 官方逻辑：方言传中文名，其余传英文名
                parts.append(zh.strip() if key == "dialect" else en.strip())
            else:
                parts.append(val)
    return ", ".join(parts) if parts else None

def get_hardware_mics():
    """获取服务器本地输入设备"""
    try:
        devices = sd.query_devices()
        return [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    except:
        return ["未检测到硬件设备"]

def record_from_hardware(device_str, duration):
    """服务器硬件录音"""
    try:
        device_id = int(device_str.split(":")[0])
        fs = 24000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device_id)
        sd.wait()
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, recording, fs)
        return temp_path, f"录音成功: {device_str}"
    except Exception as e:
        return None, f"录音失败: {str(e)}"

def run_generate_core(text, language, mode, ref_audio, ref_text, instruct, num_step, guidance_scale, denoise, speed, duration, preprocess, postprocess):
    if not text or not text.strip(): return None, "文本内容不能为空。"

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step),
        guidance_scale=float(guidance_scale),
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess),
        postprocess_output=bool(postprocess),
    )

    lang = language if (language and language != "Auto") else None
    kw = {"text": text.strip(), "language": lang, "generation_config": gen_config}

    if duration and float(duration) > 0:
        kw["duration"] = float(duration)
    else:
        kw["speed"] = float(speed)

    if mode == "clone":
        if not ref_audio or not os.path.exists(ref_audio): return None, "参考音频无效。"
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    elif mode == "design" or instruct:
        if instruct: kw["instruct"] = instruct

    try:
        audio = model.generate(**kw)
        waveform = audio[0].cpu().squeeze(0).numpy()
        # 返回 (采样率, int16数组)
        return (sampling_rate, (waveform * 32767).astype(np.int16)), "Success"
    except Exception as e:
        logger.error(f"推理失败: {e}")
        return None, f"发生错误: {str(e)}"

# ==================== 4. FastAPI 接口 (5050 端口) ====================

api_app = FastAPI()
api_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class TTSRequest(BaseModel):
    input: str
    voice: str = "alloy"
    language: str = "Auto"
    speed: Optional[float] = None
    num_step: Optional[int] = None
    guidance_scale: Optional[float] = None

@api_app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    cfg = load_api_config()
    openai_defaults = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    final_speed = request.speed if request.speed is not None else cfg["generation"]["speed"]
    final_lang = request.language if request.language != "Auto" else cfg["generation"]["language"]

    if request.voice not in openai_defaults:
        mode, instruct, ref_aud, ref_txt = "design", request.voice, None, None
    else:
        mode = cfg["mode"]
        if mode == "clone":
            instruct, ref_aud, ref_txt = None, cfg["voice_clone"]["ref_audio_path"], cfg["voice_clone"]["ref_text"]
        else:
            instruct, ref_aud, ref_txt = build_instruct_text(cfg["voice_design"]), None, None

    result, status = run_generate_core(
        text=request.input, language=final_lang, mode=mode,
        ref_audio=ref_aud, ref_text=ref_txt, instruct=instruct,
        num_step=cfg["generation"]["num_step"],
        guidance_scale=cfg["generation"]["guidance_scale"],
        denoise=cfg["generation"]["denoise"],
        speed=final_speed, duration=0,
        preprocess=cfg["generation"]["preprocess"],
        postprocess=cfg["generation"]["postprocess"]
    )

    if result is None: raise HTTPException(status_code=500, detail=status)
    
    # 修复点：result 是元组 (sampling_rate, wav_data)，wav_data 是 result[1]
    sr_out = result[0]
    wav_out = result[1]
    
    buffer = io.BytesIO()
    # 将 int16 转回 float32 tensor 以便 torchaudio 处理
    audio_tensor = torch.from_numpy(wav_out.astype(np.float32) / 32767.0).unsqueeze(0)
    torchaudio.save(buffer, audio_tensor, sr_out, format="mp3")
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/mpeg")

@api_app.get("/v1/models")
async def list_models():
    return {"data": [{"id": "tts-1", "object": "model"}]}

# ==================== 5. Gradio Web UI (8001 端口) ====================

def create_ui():
    css = ".gradio-container {max-width: 100% !important;}"
    theme = gr.themes.Soft(font=["Inter", "sans-serif"])

    def get_mic_devices():
        try:
            return [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
        except: return ["未检测到硬件设备"]

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Advanced Server") as demo:
        gr.Markdown(f"# 🎙️ OmniVoice 全功能控制中心\n**设备:** `{device}` | **API 端口:** `5050` | **Web 端口:** `8001`")

        with gr.Tabs():
            # --- Tab 1: API 全局配置 ---
            with gr.TabItem("⚙️ API 全局配置"):
                gr.Markdown("在此配置 API 接口（如 AnythingLLM）的默认音色行为。")
                current_cfg = load_api_config()
                with gr.Row():
                    with gr.Column():
                        api_mode = gr.Radio([("声音设计 (属性组合)", "design"), ("声音克隆 (固定音频)", "clone")], value=current_cfg["mode"], label="API 响应模式")
                        api_design_comps = {}
                        for k, info in _CATEGORIES.items():
                            api_design_comps[k] = gr.Dropdown(label=info["label"], choices=["Auto"] + info["choices"], value=current_cfg["voice_design"].get(k, "Auto"))
                    with gr.Column():
                        api_lang = gr.Dropdown(_ALL_LANGUAGES, label="默认语种", value=current_cfg["generation"]["language"])
                        api_speed = gr.Slider(0.5, 2.0, value=current_cfg["generation"]["speed"], label="默认语速")
                        api_steps = gr.Slider(4, 64, value=current_cfg["generation"]["num_step"], step=1, label="推理步数")
                        api_gs = gr.Slider(0.0, 5.0, value=current_cfg["generation"]["guidance_scale"], label="CFG系数")
                        api_dn = gr.Checkbox(label="启用去噪", value=current_cfg["generation"]["denoise"])
                        api_pp = gr.Checkbox(label="启用预处理", value=current_cfg["generation"]["preprocess"])
                        api_po = gr.Checkbox(label="启用后处理", value=current_cfg["generation"]["postprocess"])
                        
                        save_btn = gr.Button("💾 保存 API 全局配置", variant="primary")
                        save_status = gr.Textbox(label="状态")

                def handle_api_save(mode, lang, speed, steps, gs, dn, pp, po, *design_vals):
                    keys = list(_CATEGORIES.keys())
                    cfg = load_api_config()
                    cfg.update({
                        "mode": mode,
                        "voice_design": {keys[i]: design_vals[i] for i in range(len(keys))},
                        "generation": {**cfg["generation"], "language": lang, "speed": speed, "num_step": steps, "guidance_scale": gs, "denoise": dn, "preprocess": pp, "postprocess": po}
                    })
                    return save_api_config(cfg)

                save_btn.click(handle_api_save, [api_mode, api_lang, api_speed, api_steps, api_gs, api_dn, api_pp, api_po] + [api_design_comps[k] for k in _CATEGORIES.keys()], save_status)

            # --- Tab 2: 声音克隆 ---
            with gr.TabItem("👤 声音克隆"):
                with gr.Row():
                    with gr.Column():
                        vc_text = gr.Textbox(label="待合成文本", lines=3, value="你好，这是测试声音克隆的效果。")
                        
                        with gr.Group():
                            gr.Markdown("### 🎤 音频输入区")
                            hw_mic = gr.Dropdown(choices=get_mic_devices(), label="指定服务器物理麦克风设备", info="解决浏览器录音权限问题")
                            hw_dur = gr.Slider(3, 15, 5, step=1, label="录音时长 (秒)")
                            hw_rec_btn = gr.Button("🔴 开始硬件录制", variant="stop")
                            vc_audio = gr.Audio(label="参考音频 (上传或录制)", type="filepath", sources=["upload", "microphone"])
                            vc_ref_text = gr.Textbox(label="参考文本 (建议填写)")
                        
                        with gr.Accordion("生成高级参数", open=False):
                            vc_lang = gr.Dropdown(_ALL_LANGUAGES, label="语言", value="Auto")
                            vc_ns = gr.Slider(4, 64, 32, step=1, label="步数")
                            vc_sp = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速")
                        
                        vc_btn = gr.Button("▶️ 预览生成", variant="secondary")
                        vc_save_btn = gr.Button("⭐ 设为 API 默认音色", variant="primary")
                    
                    with gr.Column():
                        vc_out = gr.Audio(label="输出结果")
                        vc_status = gr.Textbox(label="消息")

                hw_rec_btn.click(record_from_hardware, [hw_mic, hw_dur], [vc_audio, vc_status])
                
                vc_btn.click(lambda t, l, a, rt, ns, sp: run_generate_core(t, l, "clone", a, rt, None, ns, 2.0, True, sp, 0, True, True), 
                             [vc_text, vc_lang, vc_audio, vc_ref_text, vc_ns, vc_sp], [vc_out, vc_status])

                def set_clone_api(aud, txt):
                    if not aud: return "错误：缺少音频文件。"
                    target_path = os.path.join(SAVED_VOICES_DIR, f"api_clone_voice{os.path.splitext(aud)[1]}")
                    shutil.copy(aud, target_path)
                    cfg = load_api_config()
                    cfg.update({"mode": "clone", "voice_clone": {"ref_audio_path": target_path, "ref_text": txt}})
                    return save_api_config(cfg)
                
                vc_save_btn.click(set_clone_api, [vc_audio, vc_ref_text], vc_status)

            # --- Tab 3: 声音设计 ---
            with gr.TabItem("🎨 声音设计"):
                with gr.Row():
                    with gr.Column():
                        vd_text = gr.Textbox(label="待合成文本", lines=3)
                        vd_lang = gr.Dropdown(_ALL_LANGUAGES, label="语言", value="Auto")
                        vd_drops = [gr.Dropdown(label=info["label"], choices=["Auto"] + info["choices"], value="Auto") for k, info in _CATEGORIES.items()]
                        with gr.Accordion("高级参数", open=False):
                            vd_ns = gr.Slider(4, 64, 32, step=1, label="步数")
                            vd_sp = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速")
                        vd_btn = gr.Button("▶️ 预览生成", variant="secondary")
                    with gr.Column():
                        vd_out = gr.Audio(label="输出结果")

                vd_btn.click(lambda t, l, ns, sp, *args: run_generate_core(t, l, "design", None, None, build_instruct_text({list(_CATEGORIES.keys())[i]: args[i] for i in range(len(_CATEGORIES))}), ns, 2.0, True, sp, 0, True, True),
                             [vd_text, vd_lang, vd_ns, vd_sp] + vd_drops, [vd_out, gr.State()])

    return demo

if __name__ == "__main__":
    # 启动 API 服务
    Thread(target=lambda: uvicorn.run(api_app, host="0.0.0.0", port=5050, log_level="info"), daemon=True).start()
    # 启动 Web UI
    create_ui().launch(server_name="0.0.0.0", server_port=8001)