# AudioStream

服务端：

```bash
uvicorn main:app
```

客户端：

```bash
curl -X POST "http://127.0.0.1:8000/audio_stream" \
-H "Content-Type: application/json" \
-d '{
  "output_format": "mp3"
}' --output output_audio.mp3


curl -X POST "http://127.0.0.1:8000/audio_stream" \
-H "Content-Type: application/json" \
-d '{
  "output_format": "wav"
}' --output output_audio.wav
```



注：音频文件来自[wenet](https://github.com/wenet-e2e/wenet/tree/main/runtime/gpu/client/test_wavs)。

