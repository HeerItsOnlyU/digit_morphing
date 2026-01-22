from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from .ml.pipeline import MorphPipeline
from .ml.video_generator import (
    save_frames,
    save_multi_digit_frames,
    create_video,create_gif,
)


def upload_view(request):
    """
    Handles:
    - Single digit morphing
    - Two digit morphing
    """

    if request.method == "POST":
        image_a = request.FILES.get("image_a")
        image_b = request.FILES.get("image_b")
        steps = int(request.POST.get("steps", 10))

        # Optional mode selector (default = single digit)
        mode = request.POST.get("mode", "single")

        if not image_a or not image_b:
            return render(request, "upload.html", {
                "message": "Please upload both images!"
            })

        # Save uploaded images
        fs = FileSystemStorage()
        path_a = fs.save(image_a.name, image_a)
        path_b = fs.save(image_b.name, image_b)

        full_a = fs.path(path_a)
        full_b = fs.path(path_b)

        # --------------------------------------------------
        # SINGLE DIGIT MODE
        # --------------------------------------------------
        if mode == "single":
            pipeline = MorphPipeline(task="digit")

            frames = pipeline.run(full_a, full_b, steps)

            # Save frames + create video
            frame_paths, frame_urls, run_id = save_frames(frames)
            video_url = create_video(frame_paths, run_id)
            gif_url = create_gif(frame_paths, run_id)


            return render(request, "upload.html", {
                "message": "Single-digit morphing completed!",
                "video_url": video_url,
                "gif_url": gif_url,
                "frames": frame_urls,
                "mode": "single"
            })

        # --------------------------------------------------
        # TWO DIGIT MODE
        # --------------------------------------------------
        elif mode == "multi":
            pipeline = MorphPipeline(task="digit_multi")

            all_frames = pipeline.run_multi_digit(full_a, full_b, steps)

            # Save merged frames + create video
            frame_paths, frame_urls, run_id = save_multi_digit_frames(all_frames)
            video_url = create_video(frame_paths, run_id)
            gif_url = create_gif(frame_paths, run_id)



            return render(request, "upload.html", {
            "message": "Two-digit morphing completed!",
            "video_url": video_url,
            "gif_url": gif_url,
            "frames": frame_urls,   # ✅ THIS WAS MISSING
            "mode": "multi"
            })


        else:
            return render(request, "upload.html", {
                "message": "Invalid mode selected!"
            })

    return render(request, "upload.html")
