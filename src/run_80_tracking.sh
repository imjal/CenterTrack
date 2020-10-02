python demo.py tracking --exp_id stitched_driving$1 --load_model ../models/coco_tracking.pth --demo /data2/jl5/driving1/driving$1.mp4 \
  --debug 1 --record_mAP --pre_hm --debug 1 --teacher_labels /data2/jl5/bdd100k_pred/train/instances_driving$1.pkl
