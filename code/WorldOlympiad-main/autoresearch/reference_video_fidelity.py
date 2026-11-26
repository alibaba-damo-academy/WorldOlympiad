"""Reference media integrity against renderer-owned frames, independent of speed."""
from pathlib import Path
import cv2
import numpy as np
from .io import sha256_file


def inspect_source_fidelity(video, sources, *, fps):
    paths=[Path(path) for path in sources]
    if len(paths)<2 or not np.isfinite(fps) or fps<=0:
        raise ValueError("reference fidelity needs a declared positive multi-frame clock")
    capture=cv2.VideoCapture(str(video))
    decoded_fps=float(capture.get(cv2.CAP_PROP_FPS))
    count=0; max_mse=0.; max_local_error=0.; max_local_motion=0.; first=None
    shapes_match=True; digests=[]
    try:
        for path in paths:
            source=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if source is None: raise ValueError(f"missing renderer-owned source frame: {path}")
            digests.append(sha256_file(path))
            original=source.astype(np.float32)
            if first is None: first=original.copy()
            elif original.shape!=first.shape: raise ValueError("source render dimensions change mid-episode")
            motion=np.abs(original-first).mean(axis=2)
            max_local_motion=max(max_local_motion,float(cv2.boxFilter(motion,-1,(16,16)).max()))
            okay,decoded=capture.read()
            if not okay: break
            count+=1
            if decoded.shape!=source.shape:
                shapes_match=False;continue
            error=original-decoded.astype(np.float32)
            max_mse=max(max_mse,float(np.mean(error**2)))
            local=cv2.boxFilter(np.abs(error).mean(axis=2),-1,(16,16))
            max_local_error=max(max_local_error,float(local.max()))
        extra_frame=capture.read()[0]
    finally:
        capture.release()
    checks={"source_clock_matches":abs(decoded_fps-fps)<1e-6,
            "exact_source_frame_count":count==len(paths) and not extra_frame,
            "source_resolution_matches":shapes_match,
            "decoded_frames_match_renderer":max_mse<=65.025 and max_local_error<=8.0}
    return {"passed":all(checks.values()),"checks":checks,
            "source_frame_sha256":digests,"expected_frame_count":len(paths),"decoded_frame_count":count,
            "maximum_frame_mse_8bit":max_mse,"maximum_local_codec_mae_8bit":max_local_error,
            "maximum_local_source_motion_from_first_8bit":max_local_motion,
            "source_has_local_motion":max_local_motion>=4.,
            "policy":{"codec_minimum_frame_psnr_db":30.,"codec_maximum_local_mae_8bit":8.,
                      "local_patch_size_px":16,"minimum_local_source_motion_8bit":4.,
                      "motion_measured_over_whole_episode":True}}
