"""Export actual per-action Isaac RGB evidence; never substitute imagined frames."""
from pathlib import Path
import json
import subprocess


def observation_paths(root, frame_count, short):
    paths = [Path(root) / f'step_{i:06d}_{short}_image.png' for i in range(frame_count)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing native simulator RGB evidence: {missing[0]}')
    return paths


def export(capture, output, frame_count, fps=15):
    import imageio_ffmpeg
    capture, output = Path(capture), Path(output)
    metadata = {'schema_version': 'native_isaac_export.v1', 'frame_count': frame_count,
                'fps': fps, 'source': 'executed_simulator_cameras', 'videos': {}}
    for view, short in [('exterior_left', 'left'), ('exterior_right', 'right'), ('wrist', 'wrist')]:
        observation_paths(capture / 'observations', frame_count, short)
        destination = output / view / 'imagination_640x360.mp4'
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                        '-framerate', str(fps), '-start_number', '0', '-i',
                        str(capture / 'observations' / f'step_%06d_{short}_image.png'),
                        '-frames:v', str(frame_count), '-vf', 'scale=640:360',
                        '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p',
                        str(destination)], check=True)
        metadata['videos'][view] = str(destination.relative_to(output))
    (output / 'export_metadata.json').write_text(json.dumps(metadata, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame-count', type=int, required=True)
    args = parser.parse_args()
    export(args.capture, args.output, args.frame_count)


if __name__ == '__main__':
    main()
