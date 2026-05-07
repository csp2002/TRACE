# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
        allow_pad_short_videos=False,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob
        # If True, short videos (D < num_frames) are handled by repeating the
        # boundary frame (last in forward order, first in reverse) instead of
        # raising. Useful when we don't want to drop patients with few slices.
        self.allow_pad_short_videos = allow_pad_short_videos

    def sample(self, video, segment_loader, epoch=None):

        for retry in range(MAX_RETRIES):
            D = len(video.frames)
            if D >= self.num_frames:
                start = random.randrange(0, D - self.num_frames + 1)
                frames = [video.frames[start + step] for step in range(self.num_frames)]
            elif self.allow_pad_short_videos and D >= 1:
                # Take all D frames in order, pad by repeating the last frame.
                frames = list(video.frames)
                frames = frames + [frames[-1]] * (self.num_frames - D)
            else:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {D} annotated frames."
                )
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time (for the padded case this places the duplicated
                # frames at the beginning — still valid conditioning).
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class MiddleAnchoredSampler(VOSSampler):
    """
    Samples `num_frames` consecutive frames with the volume's **middle slice
    always at position 0**.

    Direction is randomized (50% forward: middle, middle+1, ..., middle+T-1;
    50% reverse: middle, middle-1, ..., middle-T+1). If one side has fewer
    than `num_frames - 1` slices, we fall back to the other direction.

    This matches the test-time protocol of `test_medsam2_2d_video.py`, which
    splits the volume at middle and runs the same box prompt forward and
    backward. `reverse_time_prob` from RandomUniformSampler is intentionally
    replaced by the direction-random logic here.
    """

    def __init__(self, num_frames, max_num_objects, allow_pad_short_videos=False):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        # When True, short videos (fewer than `num_frames` slices available in
        # the chosen direction from middle) are padded by repeating the far-end
        # real frame. When False, raise an Exception (legacy behavior).
        self.allow_pad_short_videos = allow_pad_short_videos

    def _make_window(self, video, middle, direction, T):
        """Build a T-frame window starting at middle, moving in `direction`
        (\"fwd\" or \"bwd\"). Pad by repeating the far-end frame when there
        are not enough real slices in that direction."""
        D = len(video.frames)
        if direction == "fwd":
            available = D - middle   # slices from middle to D-1 inclusive
            real = [video.frames[middle + i] for i in range(min(available, T))]
        else:
            available = middle + 1   # slices from middle to 0 inclusive
            real = [video.frames[middle - i] for i in range(min(available, T))]
        if len(real) < T:
            real = real + [real[-1]] * (T - len(real))
        return real

    def sample(self, video, segment_loader, epoch=None):
        D = len(video.frames)
        T = self.num_frames
        middle = D // 2

        space_fwd = D - 1 - middle      # slices after middle
        space_bwd = middle              # slices before middle

        # Candidate directions with "enough" real slices (T-1 real frames after middle).
        candidate_dirs = []
        if space_fwd >= T - 1:
            candidate_dirs.append("fwd")
        if space_bwd >= T - 1:
            candidate_dirs.append("bwd")

        if not candidate_dirs:
            if not self.allow_pad_short_videos:
                raise Exception(
                    f"MiddleAnchoredSampler: volume {video.video_name} has D={D} "
                    f"which is insufficient for T={T} in either direction and "
                    f"allow_pad_short_videos=False."
                )
            # Both directions will need padding. Randomize direction for
            # training-sample diversity; _make_window handles repeat-padding.
            direction = random.choice(["fwd", "bwd"])
        else:
            direction = random.choice(candidate_dirs)
        frames = self._make_window(video, middle, direction, T)

        # Object-id filter mirrors RandomUniformSampler.
        visible_object_ids = []
        loaded = segment_loader.load(frames[0].frame_idx)
        if isinstance(loaded, LazySegments):
            visible_object_ids = list(loaded.keys())
        else:
            for oid, seg in loaded.items():
                if seg.sum():
                    visible_object_ids.append(oid)
        if len(visible_object_ids) == 0:
            raise Exception(
                f"MiddleAnchoredSampler: middle slice of video "
                f"{video.video_name} has no visible object."
            )
        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """
    VOS Sampler for evaluation: sampling all the frames and all the objects in a video
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(self, video, segment_loader, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        object_ids = segment_loader.load(frames[0].frame_idx).keys()
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
