import os
import uuid
import cv2
import numpy as np
from django.conf import settings


# --------------------------------------------------
# SINGLE DIGIT: SAVE FRAMES
# --------------------------------------------------
def save_frames(frames):
    """
    Saves single-digit frames to MEDIA_ROOT/frames/<run_id>/
    Returns:
        - frame_paths (filesystem)
        - frame_urls (browser)
        - run_id
    """
    run_id = uuid.uuid4().hex[:10]
    out_dir = os.path.join(settings.MEDIA_ROOT, "frames", run_id)
    os.makedirs(out_dir, exist_ok=True)

    frame_paths = []
    frame_urls = []

    for i, frame in enumerate(frames):
        img = frame.squeeze().detach().cpu().numpy()

        # Scale if needed
        if img.max() <= 1:
            img = img * 255

        img = img.astype(np.uint8)

        filename = f"frame_{i:03d}.png"
        path = os.path.join(out_dir, filename)

        cv2.imwrite(path, img)

        frame_paths.append(path)
        frame_urls.append(f"{settings.MEDIA_URL}frames/{run_id}/{filename}")

    return frame_paths, frame_urls, run_id


# --------------------------------------------------
# SINGLE DIGIT: CREATE VIDEO
# --------------------------------------------------
def create_video(frame_paths, run_id, fps=10):
    """
    Creates MP4 video from frame paths
    """
    out_dir = os.path.join(settings.MEDIA_ROOT, "output", run_id)
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "morph.mp4")

    first = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    h, w = first.shape

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
        isColor=True
    )

    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    writer.release()

    return f"{settings.MEDIA_URL}output/{run_id}/morph.mp4"


# --------------------------------------------------
# MULTI DIGIT: MERGE TWO DIGIT FRAMES
# --------------------------------------------------
def merge_digit_frames(frame1, frame2):
    """
    Horizontally stacks two digit tensors into one image
    """
    img1 = frame1.squeeze().detach().cpu().numpy()
    img2 = frame2.squeeze().detach().cpu().numpy()

    if img1.max() <= 1:
        img1 = img1 * 255
    if img2.max() <= 1:
        img2 = img2 * 255

    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)

    merged = np.hstack((img1, img2))
    return merged


# --------------------------------------------------
# MULTI DIGIT: SAVE MERGED FRAMES
# --------------------------------------------------
def save_multi_digit_frames(all_frames):
    run_id = uuid.uuid4().hex[:10]
    out_dir = os.path.join(settings.MEDIA_ROOT, "frames", run_id)
    os.makedirs(out_dir, exist_ok=True)

    frame_paths = []
    frame_urls = []
    steps = len(all_frames[0])

    for i in range(steps):
        merged_img = merge_digit_frames(
            all_frames[0][i],
            all_frames[1][i]
        )

        filename = f"frame_{i:03d}.png"
        path = os.path.join(out_dir, filename)

        cv2.imwrite(path, merged_img)

        frame_paths.append(path)
        frame_urls.append(f"{settings.MEDIA_URL}frames/{run_id}/{filename}")

    return frame_paths, frame_urls, run_id

#--------------------------------------------------
# generate GIF
#--------------------------------------------------

import imageio.v2 as imageio


def create_gif(frame_paths, run_id, duration=0.1):
    """
    Creates GIF from frame paths
    duration = time per frame in seconds
    """
    out_dir = os.path.join(settings.MEDIA_ROOT, "output", run_id)
    os.makedirs(out_dir, exist_ok=True)

    gif_path = os.path.join(out_dir, "morph.gif")

    images = []
    for path in frame_paths:
        img = imageio.imread(path)
        images.append(img)

    imageio.mimsave(gif_path, images, duration=duration)

    return f"{settings.MEDIA_URL}output/{run_id}/morph.gif"
