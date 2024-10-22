from config.settings import parse_argument, fix_seeds
from config.graph import ring_graph,get_neighbors
from models.cnn import CNN_C
from walk_method.walk import simpleRandomwalk
from config.datasetup import cifar_10
import torch.optim as optim
import torch.nn.functional as F
import torch
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
        test_accuracy = 0.0

        for image_test, target_test in test_loader:
            prediction_test = model(image_test)
            test_accuracy += (prediction_test.max(1)[1]==target_test).sum().item()

        test_accuracy /= len(test_loader.dataset)
        print(test_accuracy)

    return test_accuracy

def run_simulation(seed, num_nodes, graph_type, data_set, hetero_k, batch_size, learning_rate,iteration,walk_type):
    fix_seeds(seed)

    if graph_type == "ring":
        graph = ring_graph(num_nodes)
    
    if data_set == "cifar-10":
        model = CNN_C(10,128)
        optimizer = optim.SGD(model.parameters(),lr= learning_rate,momentum=0.9)
        train_iters, test_loader = cifar_10(num_nodes, hetero_k, batch_size)

    if walk_type == "srw":
         method = simpleRandomwalk

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
        next_state = method(graph,current_state)
        current_state = next_state


    with open(f"{data_set}_{graph_type}_num_nodes{num_nodes}_hetero_k{hetero_k}_method_{walk_type}.json","w") as f:
                    json.dump(test_accuracies,f)
                    f.write("\n")



if __name__ == "__main__":
    args = parse_argument()
    run_simulation(int(args.seed),int(args.num_nodes),args.graph_type,args.data_set,int(args.hetero_k),int(args.batch_size),float(args.learning_rate), int(args.iteration),args.walk)
