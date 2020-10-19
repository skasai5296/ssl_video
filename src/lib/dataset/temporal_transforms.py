import random
from typing import List, Iterable


class Compose:
    def __init__(self, transform: Iterable):
        self.transform = transform

    def __call__(self, frame_indices: List[int]) -> List[int]:
        for i, t in enumerate(self.transform):
            frame_indices = t(frame_indices)
        return frame_indices


class LoopPadding:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, frame_indices: List[int]) -> List[int]:
        out = []
        for i in range(self.size):
            out.append(frame_indices[i % len(frame_indices)])

        return out


class TemporalBeginCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, frame_indices: List[int]) -> List[int]:
        out = frame_indices[: self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, frame_indices: List[int]) -> List[int]:
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop:
    def __init__(self, size: int):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices: List[int]) -> List[int]:
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        out = self.loop(out)

        return out


# class TemporalEvenCrop:
#    def __init__(self, size: int, n_samples: int = 1):
#        self.size = size
#        self.n_samples = n_samples
#        self.loop = LoopPadding(size)
#
#    def __call__(self, frame_indices: List[int]) -> List[int]:
#        n_frames = len(frame_indices)
#        stride = max(1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))
#
#        out: List[int] = []
#        for begin_index in frame_indices[::stride]:
#            if len(out) >= self.n_samples:
#                break
#            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
#            sample = list(range(begin_index, end_index))
#
#            out.append(self.loop(sample))
#
#        return out


# class SlidingWindow:
#    def __init__(self, size: int, stride: int = 0):
#        self.size = size
#        if stride == 0:
#            self.stride = self.size
#        else:
#            self.stride = stride
#        self.loop = LoopPadding(size)
#
#    def __call__(self, frame_indices: List[int]) -> List[int]:
#        out = []
#        for begin_index in frame_indices[:: self.stride]:
#            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
#            sample = list(range(begin_index, end_index))
#
#            out.append(self.loop(sample))
#
#        return out


class TemporalSubsampling:
    def __init__(self, stride: int):
        self.stride = stride

    def __call__(self, frame_indices: List[int]) -> List[int]:
        return frame_indices[:: self.stride]


class Shuffle:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def __call__(self, frame_indices: List[int]) -> List[int]:
        random.shuffle(frame_indices)
        return frame_indices
