from os import path
from glob import glob
from moviepy.editor import VideoFileClip
from moviepy import video
from skimage import filters
import numpy as np
from scipy import ndimage

def process_video(fname_old, fname_new):    
    final_video = (VideoFileClip(fname_old, resize_algorithm='bilinear')
        .resize(0.5)
        .fl_image(lambda img: np.clip((img.astype(np.uint16) * 6), 0, 255).astype(np.uint8))
        .fx(video.fx.all.supersample, .017, 7)
        .set_fps(30)
    ) 
    final_video.write_videofile(fname_new, audio=False)


parent_dir = '/home/nickdg/theta_storage/data/VR_Experiments_Round_2/Converted Motive Files'

def task_rescale_video():
    for currpath, dirs, fnames in os.walk(parent_dir):
        for avi_file in glob(path.join(currpath, '*.avi')):
            fname = path.join(currpath, avi_file)
            fname_new = path.splitext(fname)[0] + '_f.mp4'
            yield {
                'actions': [(process_video, fname, fname_new)],
                'targets': [fname_new],
                'file_dep': [fname],
            }
    

    

    

