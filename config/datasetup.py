import torch
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from more_itertools import chunked

def cifar_10(num_nodes, hetero_k,b_size):
    all_classes = list(range(10))
    node_label = [[] for _ in range(num_nodes)]
    # each label is allocated at least one node and each node has max k classes
    for label in all_classes:
        append_flag = True
        while append_flag:
            selected_node = random.randint(0,num_nodes-1)
            if len(node_label[selected_node])< hetero_k:
                node_label[selected_node].append(label)
                append_flag = False

    for node in range(num_nodes):
        remaining_classes = [label for label in all_classes if label not in node_label[node]]
        current_classes = len(node_label[node])

        num_additional_classes = max(0,min(hetero_k-current_classes,len(remaining_classes)))
        additional_classes = random.sample(remaining_classes,k = num_additional_classes)
        node_label[node]+=additional_classes


    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])

    train_dataset = datasets.CIFAR10('./data',
                        train=True,
                        download=True,
                        transform=transform_train)

    test_dataset = datasets.CIFAR10('./data',
                        train=False,
                        transform=transform_test)

    n_train = len(train_dataset)
    n_node = len(node_label)

    # train_dataset = torch.utils.data.random_split(train_dataset,n_train)
    test_loader = DataLoader(test_dataset,batch_size=b_size,shuffle=False)
    label_node = []

    for label in range(10):
        label_node.append([])
        for node in range(n_node):
            if label in node_label[node]:
                label_node[-1].append(node)

    node_indices = [[] for i in range(n_node)]

    for label in range(10):
        indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i]==label]
        random.shuffle(indices)
        chunked_indices = list(chunked(indices, int(len(indices)/len(label_node[label]))))

        for i in range(len(label_node[label])):
            node_indices[label_node[label][i]] += chunked_indices[i]

    for i in range(n_node):
        random.shuffle(node_indices[i])
    n_data = min([len(node_indices[i]) for i in range(n_node)])
        
    train_subset_list = [Subset(train_dataset, node_indices[i][:n_data]) for i in range(n_node)]
    train_iters = [torch.utils.data.DataLoader(train_subset_list[i], batch_size=b_size, shuffle=True) for i in range(num_nodes)]

    return train_iters,test_loader