import torch
from typing import List, Tuple

YAW_LIST = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
PITCH_LIST = [-20.0, 0.0, 20.0, 40.0]

DYAW_DICT = {
    '0.0': 0,
    '45.0': 1
}

DPITCH_DICT = {
    '0.0': 0,
    '-60.0': 1,
    '20.0': 2
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
    def __init__(self, yaw_dict=DYAW_DICT, pitch_dict=DPITCH_DICT):
        self.yaw_dict = yaw_dict
        self.pitch_dictx = pitch_dict

    def yaw_to_id(self, angle):
        return self.yaw_dict.get(str(float(angle)))

    def pitch_to_id(self, angle):
        return self.pitch_dictx.get(str(float(angle)))

    def encode_pair(self, dyaw, dpitch):
        return torch.tensor(
            [self.yaw_to_id(dyaw), self.pitch_to_id(dpitch)],
            dtype=torch.long
        )

    def encode_sequence(self, action_seq):
        if isinstance(action_seq, torch.Tensor):
            action_seq = action_seq.cpu().numpy()

        ids = []
        for dyaw, dpitch in action_seq:  # T loops
            dyaw = float(dyaw)
            dpitch = float(dpitch)
            ids.append([
                self.yaw_to_id(dyaw),
                self.pitch_to_id(dpitch),
            ])

        return torch.tensor(ids, dtype=torch.long)