import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import time
import IPython

e = IPython.embed
from pathlib import Path
from torchvision import transforms
from PIL import Image

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, episode_len, history_stack=0):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_pad_len = 200
        action_str = 'qpos_action'

        self.history_stack = history_stack

        self.dataset_paths = []
        self.roots = []
        self.is_sims = []
        self.original_action_shapes = []

        self.states = []
        self.image_dict = dict()
        for cam_name in self.camera_names:
            self.image_dict[cam_name] = []
        self.actions = []

        for i, episode_id in enumerate(self.episode_ids):
            self.dataset_paths.append(os.path.join(self.dataset_dir, f'processed_episode_{episode_id}.hdf5'))
            root = h5py.File(self.dataset_paths[i], 'r')
            self.roots.append(root)
            self.is_sims.append(root.attrs['sim'])
            self.original_action_shapes.append(root[action_str].shape)

            self.states.append(np.array(root['observation.state']))
            for cam_name in self.camera_names:
                self.image_dict[cam_name].append(root[f'observation.image.{cam_name}'])
            self.actions.append(np.array(root[action_str]))

        self.is_sim = self.is_sims[0]

        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)

        # self.__getitem__(0) # initialize self.is_sim

    # def __len__(self):
    #     return len(self.episode_ids)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts

    def __getitem__(self, ts_index):
        sample_full_episode = False  # hardcode

        index, start_ts = self._locate_transition(ts_index)

        original_action_shape = self.original_action_shapes[index]
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.states[index][start_ts]
        # qvel = root['/observations/qvel'][start_ts]

        if self.history_stack > 0:
            last_indices = np.maximum(0, np.arange(start_ts - self.history_stack, start_ts)).astype(int)
            last_action = self.actions[index][last_indices, :]

        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = self.image_dict[cam_name][index][start_ts]
        # get all actions after and including start_ts
        all_time_action = self.actions[index][:]

        all_time_action_padded = np.zeros((self.max_pad_len + original_action_shape[0], original_action_shape[1]),
                                          dtype=np.float32)
        all_time_action_padded[:episode_len] = all_time_action
        all_time_action_padded[episode_len:] = all_time_action[-1]

        padded_action = all_time_action_padded[start_ts:start_ts + self.max_pad_len]
        real_len = episode_len - start_ts

        is_pad = np.zeros(self.max_pad_len)
        is_pad[real_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            # Images are now stored in CHW format (channels, height, width)
            all_cam_images.append(image_dict[cam_name])

        # Stack camera images: (num_cameras, channels, height, width)
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Keep the multi-camera format as (num_cameras, channels, height, width)
        # ACT expects this format and will handle the cameras internally

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.history_stack > 0:
            last_action_data = torch.from_numpy(last_action).float()

        # Images are already normalized to [0, 1] in the conversion script
        # normalize image and change dtype to float
        # image_data = image_data / 255.0  # Remove this since images are already normalized
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.history_stack > 0:
            last_action_data = (last_action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            qpos_data = torch.cat((qpos_data, last_action_data.flatten()))
        # print(f"qpos_data: {qpos_data.shape}, action_data: {action_data.shape}, image_data: {image_data.shape}, is_pad: {is_pad.shape}")
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    action_str = 'qpos_action'
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'processed_episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['observation.state'][()]
            action = root[action_str][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)  # (episode, timstep, action_dim)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats, all_episode_len


def find_all_processed_episodes(path):
    episodes = [f for f in os.listdir(path)]
    return episodes


def BatchSampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


def load_data(dataset_path, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_path}\n')

    if dataset_path.endswith('.hdf5') and os.path.isfile(dataset_path):
        print("Loading from a single HDF5 file.")
        with h5py.File(dataset_path, 'r') as f:
            demo_keys = sorted([k for k in f['/data'].keys() if k.startswith('demo_')])
            num_episodes = len(demo_keys)

        train_ratio = 0.99
        indices = np.arange(num_episodes)
        np.random.shuffle(indices)
        train_ids = indices[:int(train_ratio * num_episodes)]
        val_ids = indices[int(train_ratio * num_episodes):]

        # 计算 norm stats
        norm_stats = compute_single_file_norm_stats(dataset_path, demo_keys)

        train_dataset = SingleHDF5Dataset(dataset_path, demo_keys, train_ids, camera_names, norm_stats)
        val_dataset = SingleHDF5Dataset(dataset_path, demo_keys, val_ids, camera_names, norm_stats)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=4)
        return train_dataloader, val_dataloader, norm_stats, False
    else:
        all_eps = find_all_processed_episodes(dataset_path)
        num_episodes = len(all_eps)

        train_ratio = 0.99
        shuffled_indices = np.random.permutation(num_episodes)
        train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
        val_indices = shuffled_indices[int(train_ratio * num_episodes):]

        norm_stats, all_episode_len = get_norm_stats(dataset_path, num_episodes)
        train_episode_len_l = [all_episode_len[i] for i in train_indices]
        val_episode_len_l = [all_episode_len[i] for i in val_indices]
        batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l)
        batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

        train_dataset = EpisodicDataset(train_indices, dataset_path, camera_names, norm_stats, train_episode_len_l)
        val_dataset = EpisodicDataset(val_indices, dataset_path, camera_names, norm_stats, val_episode_len_l)
        train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=24,
                                      prefetch_factor=2)
        val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=16,
                                    prefetch_factor=2)
        return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_id(base_dir, prefix):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"The provided base directory does not exist or is not a directory: \n{base_path}")

    # Loop through all subdirectories of the base path
    for subfolder in base_path.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith(prefix):
            return str(subfolder), subfolder.name

    # If no matching subfolder is found
    return None, None


def find_all_ckpt(base_dir, prefix="policy_epoch_"):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError("The provided base directory does not exist or is not a directory.")

    ckpt_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.startswith(prefix):
            ckpt_files.append(file.name)
    # find latest ckpt
    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split(prefix)[-1].split('_')[0]), reverse=True)
    epoch = int(ckpt_files[0].split(prefix)[-1].split('_')[0])
    return ckpt_files[0], epoch

class SingleHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, demo_keys, indices, camera_names, norm_stats):
        self.hdf5_path = hdf5_path
        self.demo_keys = [demo_keys[i] for i in indices]
        self.camera_names = camera_names
        self.norm_stats = norm_stats

        self.index_map = []
        self.lengths = []

        # Set up image preprocessing (resize to 224x224)
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # output shape: (3, 224, 224)
        ])

        with h5py.File(hdf5_path, 'r') as f:
            for ep in self.demo_keys:
                l = f[f"data/{ep}/actions"].shape[0]
                self.lengths.append(l)
                self.index_map.extend([(ep, t) for t in range(l)])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ep, t = self.index_map[idx]
        with h5py.File(self.hdf5_path, 'r') as f:
            ep_root = f[f"data/{ep}"]
            qpos = ep_root['obs/joint_positions'][t]
            actions = ep_root['actions'][()]

            image_list = []
            for cam in self.camera_names:
                img = ep_root[f"obs/{cam}_image"][t]  # (H, W, C)
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img_tensor = self.image_transform(img)  # (3, 224, 224), float32 in [0,1]
                image_list.append(img_tensor.numpy())

            all_cam_images = np.stack(image_list, axis=0)  # (num_cams, 3, 224, 224)

        # Pad actions
        padded_actions = np.zeros((200 + actions.shape[0], actions.shape[1]), dtype=np.float32)
        padded_actions[:actions.shape[0]] = actions
        padded_actions[actions.shape[0]:] = actions[-1]
        padded = padded_actions[t:t + 200]

        # is_pad mask
        is_pad = np.zeros(200)
        is_pad[actions.shape[0] - t:] = 1 if actions.shape[0] - t < 200 else 0

        # Final tensors
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(
            ((qpos - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std'])
        ).float()
        action_data = torch.from_numpy(
            ((padded - self.norm_stats['action_mean']) / self.norm_stats['action_std'])
        ).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad


def compute_single_file_norm_stats(hdf5_path, demo_keys):
    all_qpos = []
    all_actions = []
    with h5py.File(hdf5_path, 'r') as f:
        for ep in demo_keys:
            qpos = f[f"data/{ep}/obs/joint_positions"][()]
            actions = f[f"data/{ep}/actions"][()]
            all_qpos.append(torch.from_numpy(qpos))
            all_actions.append(torch.from_numpy(actions))
    all_qpos = torch.cat(all_qpos)
    all_actions = torch.cat(all_actions)
    return {
        "qpos_mean": all_qpos.mean(0).numpy(),
        "qpos_std": all_qpos.std(0).clamp(min=1e-2).numpy(),
        "action_mean": all_actions.mean(0).numpy(),
        "action_std": all_actions.std(0).clamp(min=1e-2).numpy(),
    }
