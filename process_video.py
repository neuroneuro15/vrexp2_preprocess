from moviepy.editor import VideoFileClip
from moviepy.video import fx

def reduce_video(fname):
    """Return moviepy.VideoFileClip, resized to 50% and at 30 fps."""

    return (VideoFileClip(fname) 
        .subclip(600, 620) 
        .resize(0.5)
        .fx(fx.all.colorx, 3) 
        .fx(fx.all.supersample, .017, 3)
        .set_fps(30)
    ) 

