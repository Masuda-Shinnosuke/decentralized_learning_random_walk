import torch.optim as optim
import torch.nn.functional as F
import torch


def get_parameters(model):
    params = []
    for param in model.parameters():
        flattened_param = param.view(-1)
        params.append(flattened_param)
    concatenated_params = torch.cat(params)
    return concatenated_params


def normal_train(model,train_loader,optimizer):
    for image_train, target_train in train_loader:
        optimizer.zero_grad()
        output = model(image_train)
        loss = F.cross_entropy(output,target_train)
        loss.backward()
        optimizer.step()

def adaptive_train(model, train_loader, optimizer, momentum, v):
    theta = 0.9
    eta = 0.01
    delta = 1e-8
    for image_train, target_train in train_loader:
        optimizer.zero_grad()
        output = model(image_train)
        loss = F.cross_entropy(output, target_train)
        loss.backward()

    with torch.no_grad():
        for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_param.data = theta*m_param.data+(1-theta)*param.grad.data
            v_param.data = v_param.data+(param.grad.data)**2
            param.data = param.data - eta*m_param.data/torch.sqrt(v_param.data+delta)

    return momentum,v

def gradient_clipping(model, train_loader, optimizer, momentum, v, lamda):
    theta = 0.9
    eta = 0.01
    delta = 1e-8
    for image_train, target_train in train_loader:
        optimizer.zero_grad()
        output = model(image_train)
        loss = F.cross_entropy(output, target_train)
        loss.backward()

    p_mom = []
    p_v = []

    for param1, param2 in zip(momentum.parameters(), v.parameters()):
        p_mom.append(param1)
        p_v.append(param2)

    with torch.no_grad():
        for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_param.data = theta*m_param.data+(1-theta)*param.grad.data
            v_param.data = v_param.data+(param.grad.data)**2
            # param.data = param.data - eta*m_param.data/torch.sqrt(v_param.data+delta)

    m_distance = 0.0
    v_distance = 0.0

    for current_m, pre_m , current_v, pre_v in zip(momentum.parameters(), p_mom, v.parameters(), p_v):
        m_distance += torch.sum((current_m - pre_m)**2)
        v_distance += torch.sum((current_v - pre_v)**2)

    m_distance = torch.square(m_distance)
    v_distance = torch.square(v_distance)

    if m_distance >= lamda:
        for m_param in momentum.parameters():
            m_param.data = (m_param.data/m_distance)*lamda

    if v_distance >= lamda:
        for v_param in v.parameters():
            v_param.data = (v_param.data/v_distance)*lamda

    for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_param.data = theta*m_param.data+(1-theta)*param.grad.data
            v_param.data = v_param.data+(param.grad.data)**2
            param.data = param.data - eta*m_param.data/torch.sqrt(v_param.data+delta)

    return momentum, v
