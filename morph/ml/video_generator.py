import os, uuid, cv2
import numpy as np
from django.conf import settings


def save_frames(frames):
    run_id = uuid.uuid4().hex[:10]
    out_dir = os.path.join(settings.MEDIA_ROOT, "frames", run_id)
    os.makedirs(out_dir, exist_ok=True)

    paths, urls = [], []
    for i, f in enumerate(frames):
        img = f.squeeze().detach().cpu().numpy()
        img = (img * 255 if img.max() <= 1 else img).astype(np.uint8)

        name = f"frame_{i:03d}.png"
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, img)

        paths.append(path)
        urls.append(f"{settings.MEDIA_URL}frames/{run_id}/{name}")

    return paths, urls, run_id


def create_video(frame_paths, run_id, fps=10):
    out_dir = os.path.join(settings.MEDIA_ROOT, "output", run_id)
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "morph.mp4")
    first = cv2.imread(frame_paths[0], 0)
    h, w = first.shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for p in frame_paths:
        img = cv2.imread(p, 0)
        if img is not None:
            video.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    video.release()
    return f"{settings.MEDIA_URL}output/{run_id}/morph.mp4"
