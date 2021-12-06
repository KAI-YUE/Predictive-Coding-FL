import os
import numpy as np
import pickle
import logging

# PyTorch Libraries
import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, images, labels, type_="mnist"):
        """Construct a user train_dataset and convert ndarray 
        """
        images = self._normalize(images, type_)
        labels = (labels).astype(np.int64)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return dict(image=image, label=label)

    def _normalize(self, images, type_):
        if type_ == "mnist":
            images = images.astype(np.float32)/255
            images = (images - 0.1307)/0.3081
        elif type_ == "fmnist":
            images = images.astype(np.float32)/255
            images = (images - 0.2860)/0.3530
        elif type_ == "cifar":
            image_area = 32**2
            images = images.astype(np.float32)/255
            images[:, :image_area] = (images[:, :image_area] - 0.4914) / 0.247                              # r channel 
            images[:, image_area:2*image_area] = (images[:, image_area:2*image_area] - 0.4822) / 0.243      # g channel
            images[:, -image_area:] = (images[:, -image_area:] - 0.4465) / 0.261                            # b channel
        else: 
            images = images.astype(np.float32)/255
        
        return images

def assign_noniid_data(train_dataset, alpha=1, num_users=1, **kwargs):
    """
    Assign train_dataset to multiple users.

    Args:
        train_dataset (dict):   a train_dataset which contains training samples and labels. 
        alpha (float):          the parameter of DIrichlet distribution 
        num_users (int):        the number of users.

    Returns:
        dict:  keys denote userID ranging from [0,...,num_users-1] and values are sampleID
               ranging from [0,...,num_samples]
    
    """

    num_classes = 10
    N = train_dataset["labels"].shape[0]
    y_train = train_dataset["labels"]

    samples_per_user = int(y_train.shape[0]/num_users)
    samples_per_class = int(y_train.shape[0]/num_classes)
    user_dataidx_map = {}

    idxs_ascending_labels = np.argsort(y_train)
    labels_idx_map = np.zeros((num_classes, samples_per_class))
    for i in range(num_classes):
        labels_idx_map[i] = idxs_ascending_labels[i*samples_per_class:(i+1)*samples_per_class]
        np.random.shuffle(labels_idx_map[i])
        
    for user_id in range(num_users):
        current_user_dataidx = []
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        histogram = samples_per_user*proportions
        histogram = histogram.astype(np.int)
        
        for i in range(num_classes):
            current_user_dataidx.append(labels_idx_map[i][:histogram[i]])
            np.random.shuffle(labels_idx_map[i])
            
        user_dataidx_map[user_id] = np.hstack(current_user_dataidx).astype(np.int).flatten()
    
    return user_dataidx_map


def assign_iid_data(train_dataset, num_users=1, **kwargs):
    """
    Assign train_dataset to multiple users.

    Args:
        train_dataset (dict):     a train_dataset which contains training samples and labels. 
        num_users (int):     the number of users.
        labels_per_user:      number of labels assigned to the user in no-iid setting.

    Returns:
        dict:  keys denote userID ranging from [0,...,num_users-1] and values are sampleID
               ranging from [0,...,num_samples]
    
    """
    
    try:
        num_samples = train_dataset["labels"].shape[0]
        samples_per_user = num_samples // num_users
    except KeyError:
        logging.error("Input train_dataset dictionary doesn't cotain key 'labels'.")

    user_with_data = {}
    userIDs = np.arange(num_users)
    sampleIDs = np.arange(num_samples)
    np.random.shuffle(userIDs)
    np.random.shuffle(sampleIDs)
    
    # Assign the train_dataset in an iid fashion
    for userID in userIDs:
        user_with_data[userID] = sampleIDs[userID*samples_per_user: (userID+1)*samples_per_user].tolist()

    return user_with_data


def assign_user_data(config, logger):
    """
    Load data and generate user_with_data dict given the configuration.

    Args:
        config (class):    a configuration class.
    
    Returns:
        dict: a dict contains train_data, test_data and user_with_data[userID:sampleID].
    """
    
    with open(config.train_data_dir, "rb") as fp:
        train_data = pickle.load(fp)
    
    with open(config.test_data_dir, "rb") as fp:
        test_data = pickle.load(fp)

    if os.path.exists(config.user_with_data):
        logger.info("Pre-defined data distribution")
        with open(config.user_with_data, "rb") as fp:
            user_with_data = pickle.load(fp)
    else:
        logger.info("Non-IID data distribution with alpha {:.2f}".format(config.alpha))
        user_with_data = assign_noniid_data(train_dataset=train_data, alpha=config.alpha, num_users=config.users)

    return dict(train_data=train_data,
                test_data=test_data,
                user_with_data=user_with_data)


def assign_user_resource(config, userID, train_dataset, user_with_data):
    """Simulate one user resource by assigning the dataset and configurations.
    """
    user_resource = {}
    batch_size = config.local_batch_size
    tau = config.tau
    user_resource["lr"] = config.lr
    user_resource["device"] = config.device
    user_resource["batch_size"] = batch_size

    sampleIDs = user_with_data[userID][:tau*batch_size]
    user_resource["images"] = train_dataset["images"][sampleIDs]
    user_resource["labels"] = train_dataset["labels"][sampleIDs]

    classes, class_count = np.unique(user_resource["labels"], return_counts=True)
    sampling_weight = np.zeros(user_resource["labels"].shape[0])
    for i, class_ in enumerate(classes):
        class_idx = (user_resource["labels"] == class_)
        sampling_weight[class_idx] = 1/class_count[i]
    
    user_resource["sampling_weight"] = sampling_weight

    # shuffle the sampleIDs
    np.random.shuffle(user_with_data[userID])

    return user_resource
