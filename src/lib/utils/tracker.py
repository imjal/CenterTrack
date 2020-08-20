import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy

class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.reset()

  def init_track(self, results):
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        # active and age are never used in the paper
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.tracks.append(item)

  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step(self, results, public_det=None, mask_rcnn_pred=None):
    N = len(results)
    M = len(self.tracks)
    dets = np.array(
      [det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    reg_dets = np.array(
      [det['ct'] for det in results], np.float32) # N x 2
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
      (track['bbox'][3] - track['bbox'][1])) \
      for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
      (item['bbox'][3] - item['bbox'][1])) \
      for item in results], np.float32) # N
    item_cat = np.array([item['class'] for item in results], np.int32) # N
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2
    dist = (((tracks.reshape(1, -1, 2) - \
              dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M
    invalid = ((dist > track_size.reshape(1, M)) + \
      (dist > item_size.reshape(N, 1))) # + \
    # (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0 # for now get rid of category matching, solely distance & size
    dist = dist + invalid * 1e18

    if mask_rcnn_pred is not None:
      L = len(mask_rcnn_pred['boxes'])
      mask_rcnn_pred['boxes'] = np.array(mask_rcnn_pred['boxes']) / 3
      mask_centers = np.array([ [ (x[0] + x[2])/2, (x[1] + x[3])/2] for x in mask_rcnn_pred['boxes']])
      dist2 = ((reg_dets.reshape(-1, 1, 2) - mask_centers.reshape(1, -1, 2)) ** 2).sum(axis=2) # L2 distance between the dets and mask_centers
      mask_item_size = np.array([((item[2] - item[0]) * (item[3] - item[1])) for item in mask_rcnn_pred['boxes']], np.float32)
      invalid_mask = ((dist2 > mask_item_size.reshape(1, L)) + (dist2 > item_size.reshape(N, 1)))
      dist2 = dist2 + invalid_mask * 1e18 # make invalids a super high distance

    if self.opt.hungarian:
      item_score = np.array([item['score'] for item in results], np.float32) # N
      dist[dist > 1e18] = 1e18
      matched_indices = linear_assignment(dist)
    else:
      matched_indices = greedy_assignment(copy.deepcopy(dist))
    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])] # one of these will be empty

    if self.opt.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
          unmatched_tracks.append(m[1])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    # Public detection: only create tracks from provided detections
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      track['class'] = self.tracks[m[1]]['class'] # continue the same class track
      ret.append(track)

    if self.opt.public_det and len(unmatched_dets) > 0:
      # Public detection: only create tracks from provided detections
      pub_dets = np.array([d['ct'] for d in public_det], np.float32)
      dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(
        axis=2)
      matched_dets = [d for d in range(dets.shape[0]) \
        if not (d in unmatched_dets)]
      dist3[matched_dets] = 1e18
      for j in range(len(pub_dets)):
        i = dist3[:, j].argmin()
        if dist3[i, j] < item_size[i]:
          dist3[i, :] = 1e18
          track = results[i]
          if track['score'] > self.opt.new_thresh:
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)
    else:
      # # Private detection: create tracks for all un-matched detections
      if mask_rcnn_pred is not None:
        new_matched_indices = greedy_assignment(copy.deepcopy(dist2))
        matched_dets = [x[0] for x in matches]
        for i, m in enumerate(new_matched_indices):
          if m[0] not in unmatched_dets:
            continue
          track = results[m[0]]
          if track['score'] > self.opt.new_thresh:
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
            track['class'] = mask_rcnn_pred['classes'][m[1]] + 1
            ret.append(track)
    self.tracks = ret
    return ret

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)
