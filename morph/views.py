from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .ml.pipeline import MorphPipeline
from .ml.video_generator import save_frames, create_video


def upload_view(request):
    if request.method == "POST":
        image_a = request.FILES.get("image_a")
        image_b = request.FILES.get("image_b")
        steps = int(request.POST.get("steps", 10))

        if not image_a or not image_b:
            return render(request, "upload.html", {
                "message": "Please upload both images!"
            })

        fs = FileSystemStorage()

        path_a = fs.save(image_a.name, image_a)
        path_b = fs.save(image_b.name, image_b)

        full_a = fs.path(path_a)
        full_b = fs.path(path_b)

        pipeline = MorphPipeline(task="digit")

        frames = pipeline.run(full_a, full_b, steps)

        # ✅ disk paths for video + urls for html
        frame_disk_paths, frame_urls, run_id = save_frames(frames)

        # ✅ create video
        video_url = create_video(frame_disk_paths, run_id=run_id)

        return render(request, "upload.html", {
            "message": "Morphing video generated successfully!",
            "video_url": video_url,
            "frames": frame_urls,
        })

    return render(request, "upload.html")
