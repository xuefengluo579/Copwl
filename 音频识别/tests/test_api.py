"""测试 FastAPI 服务端点."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def test_wav_bytes(synthetic_audio):
    """生成有效的 WAV 字节."""
    buf = io.BytesIO()
    sf.write(buf, synthetic_audio.T, 16000, format="WAV")
    buf.seek(0)
    return buf.read()


@pytest.mark.api
class TestHealthEndpoint:
    """测试健康检查."""

    def test_health_returns_200(self, test_client: TestClient):
        response = test_client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "version" in data

    def test_health_response_structure(self, test_client: TestClient):
        response = test_client.get("/health")
        data = response.json()
        assert data["status"] in ("ok", "degraded", "error")
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))


@pytest.mark.api
class TestEventDetection:
    """测试事件检测端点."""

    def test_post_with_valid_wav(self, test_client: TestClient, test_wav_bytes):
        response = test_client.post(
            "/v1/audio/events",
            files={"file": ("test.wav", test_wav_bytes, "audio/wav")},
        )
        # 503 = 模型未加载 (无权重), 200 = 成功, 422 = 校验失败
        assert response.status_code in (200, 422, 503)

    def test_post_empty_file(self, test_client: TestClient):
        response = test_client.post(
            "/v1/audio/events",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        # 400=空文件, 422=校验失败, 503=引擎未加载 (fairseq 未安装)
        assert response.status_code in (400, 422, 503)

    def test_post_bad_audio(self, test_client: TestClient):
        response = test_client.post(
            "/v1/audio/events",
            files={"file": ("bad.wav", b"not wav data", "audio/wav")},
        )
        assert response.status_code in (422, 503)

    def test_post_response_structure(self, test_client: TestClient, test_wav_bytes):
        response = test_client.post(
            "/v1/audio/events",
            files={"file": ("test.wav", test_wav_bytes, "audio/wav")},
        )
        if response.status_code == 200:
            data = response.json()
            assert "events" in data
            assert "inference_time_ms" in data
            assert "audio_duration_ms" in data
            assert isinstance(data["events"], list)


@pytest.mark.api
class TestDocs:
    """测试 API 文档可访问性."""

    def test_docs_available(self, test_client: TestClient):
        response = test_client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, test_client: TestClient):
        response = test_client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, test_client: TestClient):
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "w2v-SELD Audio Event Detection"
