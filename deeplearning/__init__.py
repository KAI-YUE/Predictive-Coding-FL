from .dataset import UserDataset, assign_user_data
from .networks import *

nn_registry = {
"lenet":        LeNet_5,
"vgg":          VGG_7
}
