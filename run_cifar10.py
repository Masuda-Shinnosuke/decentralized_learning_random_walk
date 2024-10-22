from config.settings import parse_argument, fix_seeds
from config.graph import ring_graph,get_neighbors
from models.cnn import CNN_C
from config.datasetup import cifar_10
import torch.optim as optim
import torch.nn.functional as F
import random
import json


def train(model,train_loader,optimizer):
    for image_train, target_train in train_loader:
        optimizer.zero_grad()
        output = model(image_train)
        loss = F.cross_entropy(output,target_train)
        loss.backward()
        optimizer.step()

def test_model(model,test_loader):
    
    with torch.no_grad():

        for image_test, target_test in test_loader:

            prediction_test = model(image_test)
            test_accuracy += (prediction_test.max(1)[1]==target_test).sum().item()

        test_accuracy /= len(test_loader.dataset)
        print(test_accuracy)

    return test_accuracy

def run_simulation(seed, num_nodes, graph_type, data_set, hetero_k, batch_size, learning_rate,iteration):
    fix_seeds(seed)

    if graph_type == "ring":
        graph = ring_graph(num_nodes)
    
    if data_set == "cifar-10":
        model = CNN_C(10,128)
        optimizer = optim.SGD(model.parameters(),lr= learning_rate,momentum=0.9)
        train_iters, test_loader = cifar_10(num_nodes, hetero_k, batch_size)

    test_accuracies = []
    current_state = random.randint(0, num_nodes-1)
    for step in range(iteration):
        print(f"iteration : {step}")
        train(model,train_iters[current_state],optimizer)
        test_accuracy = test_model(model,test_loader)
        data = {
            f"{step}":test_accuracy
        }
        test_accuracies.append(data)

    with open(f"{data_set}_diffchange.json","w") as f:
                    json.dump(transition_accuracy,f)
                    f.write("\n")

