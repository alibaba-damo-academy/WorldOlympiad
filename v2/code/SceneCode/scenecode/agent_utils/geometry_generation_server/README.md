# Geometry Generation Server

Flask server for converting 2D images to 3D geometry (GLB files). Supports two
backends:

- **Hunyuan3D**: Diffusion-based image-to-3D (faster, lower memory)
- **SAM3D**: Segmentation + Gaussian splatting reconstruction (higher quality)

Uses queue-based processing to handle concurrent requests sequentially. Drake
SDF conversion is handled client-side for better parallel processing.

## Architecture

```
┌─────────────────────────────────────┐
│         Client Applications         │  ← AssetManager, Tests, Tools
│      (GeometryGenerationClient)     │
└─────────────────┬───────────────────┘
                  │ HTTP Requests
                  ▼
┌─────────────────────────────────────┐
│       Flask Server (Port 7000)      │  ← Request Handling & Routing
│   • /generate_geometry (POST)       │
│   • /health (GET)                   │
└─────────────────┬───────────────────┘
                  │ Queue Processing
                  ▼
┌─────────────────────────────────────┐
│        Request Queue Thread         │  ← Sequential Processing
│   • Thread-safe queue handling      │
│   • Error isolation per request     │
│   • Pipeline lifecycle management   │
└─────────────────┬───────────────────┘
                  │ Backend Selection
                  ▼
┌───────────────────┬─────────────────┐
│  Hunyuan3D        │  SAM3D          │
│  Pipeline Manager │  Pipeline Mgr   │  ← GPU Model Management
│  • Diffusion      │  • SAM3 segm.   │
│  • Image → mesh   │  • 3DGS recon   │
└───────────────────┴─────────────────┘
                  │ Return Geometry
                  ▼
┌─────────────────────────────────────┐
│           Client Side               │  ← Drake SDF Conversion
│   • GLB → Drake geometry format     │      (AssetManager)
│   • SDF file generation             │
│   • Parallel CPU processing         │
└─────────────────────────────────────┘
```

## Usage

The server can be used in two ways:

### 1. Standalone Mode (Testing/Debugging/Microservice)
For independent testing, debugging, or microservice deployment:
```bash
# Hunyuan3D backend (default)
python -m scenecode.agent_utils.geometry_generation_server.standalone_server

# SAM3D backend
python -m scenecode.agent_utils.geometry_generation_server.standalone_server \
  --backend sam3d \
  --sam3-checkpoint external/checkpoints/sam3.pt \
  --sam3d-checkpoint external/checkpoints/pipeline.yaml
```

### 2. Programmatic Mode (Recommended for Applications)
For integration within experiments or applications:
```python
from scenecode.agent_utils.geometry_generation_server import GeometryGenerationServer

# Hunyuan3D backend
server = GeometryGenerationServer(host="127.0.0.1", port=7000)

# SAM3D backend
server = GeometryGenerationServer(
    host="127.0.0.1",
    port=7000,
    backend="sam3d",
    sam3d_config={
        "sam3_checkpoint": "external/checkpoints/sam3.pt",
        "sam3d_checkpoint": "external/checkpoints/pipeline.yaml",
    },
)

server.start()
server.wait_until_ready()
# ... use server via GeometryGenerationClient ...
server.stop()
```

### Client Example
```python
from scenecode.agent_utils.geometry_generation_server import (
    GeometryGenerationClient,
    GeometryGenerationServerRequest
)

client = GeometryGenerationClient()

# For batch requests with backend selection
requests = [
    GeometryGenerationServerRequest(
        image_path="/path/to/chair.png",
        output_dir="/path/to/output",
        prompt="Modern wooden chair",
        backend="sam3d",  # or "hunyuan3d"
        sam3d_config={
            "sam3_checkpoint": "external/checkpoints/sam3.pt",
            "sam3d_checkpoint": "external/checkpoints/pipeline.yaml",
            "mode": "foreground",  # or "text"
        },
    ),
]

# Process results as they stream back (enables parallel GPU/CPU processing)
for index, response in client.generate_geometries(requests):
    print(f"Asset {index} completed: {response.geometry_path}")
    # Can start Drake conversion here while GPU processes next asset
```

## API

- `POST /generate_geometries` - Convert batch of 2D images to 3D geometry (streaming NDJSON response)
- `GET /health` - Server health status and queue size

## Backend Comparison

| Feature | Hunyuan3D | SAM3D |
|---------|-----------|-------|
| Quality | Good | Higher |
| Speed | Faster | Slower |
| GPU Memory | ~24GB | ~32GB |
| Textures | Basic | UV-mapped |
| Best for | Development | Production |
