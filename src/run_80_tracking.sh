python demo.py tracking --exp_id driving1002_fixeddist_track_color --load_model ../models/coco_tracking.pth \
  --demo /data/rmullapu/video_distillation/driving1/driving1002.mp4 \
  --debug 1 --save_video --resize_video --video_w 1280 --video_h 720 --pre_hm --show_track_color
