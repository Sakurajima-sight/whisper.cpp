import os
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json
import tempfile
import re
from fastapi.responses import JSONResponse

# 创建 FastAPI 实例
app = FastAPI()

class WhisperAPI:
    def __init__(self, whisper_cli_path="/whisper.cpp/build/bin/whisper-cli"):
        self.whisper_cli_path = whisper_cli_path
        self.models_dir = "/whisper.cpp/models/"
        self.default_language = "auto"  # 默认语言为自动检测
        self.default_model = "base"  # 默认选择 base 模型
        self.language = self.default_language
        self.model = None
        self.threads = 4  # 默认线程数为 4
        self.output_format = None
        self.duration = 0  # 默认不限制音频时长
        self.offset = 0  # 默认不设置偏移
        self.vad_enabled = False  # 默认关闭 VAD（语音活动检测）
        self.max_len = 0  # 默认不限制最大字符数

    def set_model(self, model_name):
        """设置使用的模型"""
        supported_models = ["tiny", "base", "small", "medium", "large-v1", "large-v3-turbo"]
        if model_name not in supported_models:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {', '.join(supported_models)}")
        self.model = os.path.join(self.models_dir, f"ggml-{model_name}.bin")
        if not os.path.exists(self.model):
            raise FileNotFoundError(f"Model file not found: {self.model}")

    def set_language(self, language_code):
        """设置语言"""
        self.language = language_code

    def set_threads(self, num_threads):
        """设置线程数"""
        self.threads = num_threads

    def set_duration(self, duration_ms):
        """设置处理音频的时长（单位：毫秒）"""
        self.duration = duration_ms

    def set_offset(self, offset_ms):
        """设置音频的时间偏移（单位：毫秒）"""
        self.offset = offset_ms

    def set_output(self, output_format, output_file):
        """设置输出格式和文件"""
        self.output_format = output_format
        self.output_file = output_file

    def enable_vad(self, enable=True):
        """启用或禁用 VAD（语音活动检测）"""
        self.vad_enabled = enable

    def set_max_len(self, max_len):
        """设置每个转录段的最大字符数"""
        self.max_len = max_len

    def run(self, audio_file):
        """运行转录处理"""
        if not os.path.exists(audio_file):
            raise ValueError(f"Audio file {audio_file} does not exist")

        if self.model is None:
            raise ValueError("Model is not set. Please set a model using set_model()")

        # 构建命令
        command = [self.whisper_cli_path]
        command.extend(["-f", audio_file])
        command.extend(["-l", self.language])
        command.extend(["-m", self.model])
        command.extend(["-t", str(self.threads)])

        if self.duration > 0:
            command.extend(["-d", str(self.duration)])

        if self.offset > 0:
            command.extend(["-ot", str(self.offset)])

        if self.vad_enabled:
            command.append("--vad")  # 启用 VAD

        if self.max_len > 0:
            command.extend(["-ml", str(self.max_len)])  # 设置最大字符数

        if self.output_format:
            if self.output_format == 'txt':
                command.append("-otxt")
            elif self.output_format == 'json':
                command.append("-oj")

            if self.output_file:
                command.extend(["--output-file", self.output_file])

        # 执行命令并捕获输出
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")

# 定义请求体模型，用于接收用户输入的参数
class TranscribeRequest(BaseModel):
    model: str = "base"  # 默认模型为 base
    language: str = "auto"  # 默认语言为自动检测
    threads: int = 4  # 默认线程数为 4
    duration: Optional[int] = 0  # 默认不设置音频处理时长
    offset: Optional[int] = 0  # 默认不设置音频偏移
    vad_enabled: Optional[bool] = False  # 默认禁用 VAD
    max_len: Optional[int] = 0  # 默认不限制最大字符数
    output_format: Optional[str] = "txt"  # 默认输出格式为 txt
    output_file: Optional[str] = None  # 默认不输出文件

from fastapi.responses import JSONResponse, FileResponse

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), request: str = File(...)):
    try:
        request_data = json.loads(request)
        whisper = WhisperAPI()
        whisper.set_model(request_data['model'])
        whisper.set_language(request_data['language'])
        whisper.set_threads(request_data['threads'])
        whisper.set_duration(request_data.get('duration', 0))
        whisper.set_offset(request_data.get('offset', 0))
        whisper.enable_vad(request_data.get('vad_enabled', False))
        whisper.set_max_len(request_data.get('max_len', 0))
        whisper.set_output(request_data.get('output_format', 'txt'), request_data.get('output_file'))

        temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        result = whisper.run(temp_file_path)
        cleaned_result = re.sub(r'\[.*?-->.*?\]\s*', '', result).strip()

        if request_data.get('output_file'):
            output_file_path = os.path.join(tempfile.gettempdir(), request_data['output_file'])
            output_dir = os.path.dirname(output_file_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file_path, "w") as f:
                f.write(cleaned_result)
            return FileResponse(output_file_path, media_type='application/octet-stream', filename=request_data['output_file'])

        # 返回标准 JSON 格式
        return {"transcription": cleaned_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
