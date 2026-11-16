"""Regression: a healthy slow synchronous policy must not trip websocket pings."""
import asyncio
import threading
import time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("openpi_client")
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.sync.client
from autoresearch.cosmos3_droid_window import run_server_window

def test_slow_healthy_inference_keeps_connection(tmp_path, monkeypatch):
    ready=threading.Event()
    stop=threading.Event()
    port=[]
    async def handler(ws):
        await ws.send(msgpack_numpy.packb({}))
        await ws.recv()
        time.sleep(.25)  # upstream synchronous inference blocks ping service
        await ws.send(msgpack_numpy.packb({"action":np.zeros((32,8),dtype=np.float32),
            "worldolympiad_request_index":0,"worldolympiad_request_seed":123,
            "worldolympiad_format_prompt_as_json":False}))
        await ws.wait_closed()
    async def serve():
        async with websockets.asyncio.server.serve(handler,"127.0.0.1",0,ping_interval=None) as server:
            port.append(server.sockets[0].getsockname()[1]);ready.set()
            while not stop.is_set(): await asyncio.sleep(.01)
    thread=threading.Thread(target=lambda:asyncio.run(serve()),daemon=True)
    thread.start();assert ready.wait(5)
    original=websockets.sync.client.connect
    def short_defaults(*args,**kwargs):
        kwargs.setdefault("ping_interval",.02)
        kwargs.setdefault("ping_timeout",.02)
        return original(*args,**kwargs)
    monkeypatch.setattr(websockets.sync.client,"connect",short_defaults)
    image=tmp_path/"input.png"
    Image.new("RGB",(640,540)).save(image)
    try:
        actions,raw,metadata=run_server_window(window_root=tmp_path/"window",
          input_image=image,joints=np.zeros(7,dtype=np.float32),gripper=0,prompt="test",
          host="127.0.0.1",port=port[0],request_index=0,seed_schedule_seed=0,expected_request_seed=123)
        assert actions.shape==(33,8) and metadata["request_seed"]==123
    finally:
        stop.set();thread.join(5)
        assert not thread.is_alive()
