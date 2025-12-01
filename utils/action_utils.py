import torch
from typing import List, Tuple

YAW_LIST = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
PITCH_LIST = [-20.0, 0.0, 20.0, 40.0]

YAW_DICT = {
    '0.0': 0,
    '45.0': 1,
    '90.0': 2,
    '135.0': 3,
    '180.0': 4,
    '225.0': 5,
    '270.0': 6,
    '315.0': 7
}

PITCH_DICT = {
    '-20.0': 0,
    '0.0': 1,
    '20.0': 2,
    '40.0': 3
}


def build_view_sequence(yaw_list: List[float] = YAW_LIST,
                        pitch_list: List[float] = PITCH_LIST) -> List[Tuple[float, float]]:
    seq: List[Tuple[float, float]] = []
    for yaw in yaw_list:
        for pitch in pitch_list:
            seq.append((yaw, pitch))
    return seq


def build_action_tensor(yaw_list: List[float] = YAW_LIST,
                        pitch_list: List[float] = PITCH_LIST) -> torch.Tensor:
    view_seq = build_view_sequence(yaw_list, pitch_list)
    T = len(view_seq)

    actions = []
    prev_yaw, prev_pitch = view_seq[0]
    actions.append([0.0, 0.0])

    for t in range(1, T):
        yaw_t, pitch_t = view_seq[t]
        dyaw = yaw_t - prev_yaw
        dpitch = pitch_t - prev_pitch
        actions.append([dyaw, dpitch])
        prev_yaw, prev_pitch = yaw_t, pitch_t

    action_tensor = torch.tensor(actions, dtype=torch.float32)  # [T, 2]
    return action_tensor

class ActionTokenizer:
    def __init__(self, yaw_dict=YAW_DICT, pitch_dict=PITCH_DICT):
        self.yaw_dict = yaw_dict
        self.pitch_dictx = pitch_dict

    def yaw_to_id(self, angle):
        return self.yaw_dict.get(str(angle))

    def pitch_to_id(self, angle):
        return self.pitch_dictx.get(str(angle))
