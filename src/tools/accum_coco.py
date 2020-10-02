import numpy as np

class AccumCOCODetResult:
    def __init__(self):
        self.cocoDt = []
        self.dt_counter = 0
        # self.remap_coco2bdd = get_remap(coco2bdd_class_groups)

    def add_det_to_coco(self, iter_id, dets):
        '''
        convert a CenterNet detection to coco image instance
        '''
        for i in range(len(dets)):
            bbox = dets[i]['bbox']
            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]- bbox[1]]
            res = {
                "image_id": int(iter_id), 
                "category_id": dets[i]['class'] - 1, 
                "bbox": bbox,
                "score": dets[i]['score'],
                "area": bbox[2] * bbox[3] # box area
            }
            res['id'] = self.dt_counter
            self.dt_counter+=1
            self.cocoDt += [res]

    def get_dt(self):
        return self.cocoDt