### Communication-Efficient Federated Learning via Predictive Coding

![](https://img.shields.io/badge/status-maintained-green) ![](https://img.shields.io/badge/Pytorch-1.9.0-blue)

#### Introduction
In this [paper](https://arxiv.org/abs/2108.00918), we propose a predictive coding based communication scheme for federated learning. 
The scheme has shared prediction functions among all devices and allows each worker to transmit a compressed residual vector derived from the reference.

#### Prerequisites
```bash
pip3 install -r requirements.txt
```

<!-- The dataset is included in the repository. Cloning the repository may take some time. You may fork it and modify `.gitignore` if this behavior undesirable. -->

####  Example

- Run [FedPAQ](http://proceedings.mlr.press/v108/reisizadeh20a.html) for 10 communication rounds:
    ```bash
    python3 fedpaq.py
    ```
    You can modify the hyperparameters by changing the configuration file [fedpaq.yaml](config/fedpaq.yaml).

<br />

- Run the predictive coding method for 10 communication rounds:
    ```bash
    python3 main.py
    ```

<br />

#### Change the Degree of Heterogeneity

The `user_with_data` files predefine the [Dirichlet non-IID partitions [HQB19]](https://arxiv.org/abs/1909.06335)  with different degrees of heterogeneity.  If you want to generate different partitions, you can use the following code snippets:

```python
"""
For each client, sample q~Dir(alpha, p).
"""
alpha = 0.5
num_users = 30
num_classes = 10
num_datapoints = 50000

samples_per_user = int(y_train[:num_datapoints].shape[0]/num_users)
samples_per_class = int(y_train[:num_datapoints].shape[0]/num_classes)
user_dataidx_map = {}

idxs_ascending_labels = np.argsort(y_train[:num_datapoints])
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
``` 