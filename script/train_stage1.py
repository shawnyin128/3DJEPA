from torch.utils.data import DataLoader

from model.JEPA import JEPAModel
from utils.action_utils import ActionTokenizer, build_action_tensor
from utils.dataset_utils import ShapeNetDataset


# define model
hidden_size = 1024
head_dim = 128
head_num = 32
kv_head_num = 8
num_yaw = 2
num_pitch = 3
num_layers = 8

jepa = JEPAModel(hidden_size=hidden_size,
                 head_dim=head_dim,
                 head_num=head_num,
                 kv_head_num=kv_head_num,
                 num_yaw=num_yaw,
                 num_pitch=num_pitch,
                 num_layers=num_layers)

# load dataset
dataset = ShapeNetDataset(root="../data/3D")
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

# define action
action_tokenizer = ActionTokenizer()
action_sequence = build_action_tensor()
action_tensor = action_tokenizer.encode_sequence(action_sequence)

# train
for batch in loader:
    image, meta = batch
    loss, stats = jepa(image, action_tensor)
    print(loss, stats)
    break