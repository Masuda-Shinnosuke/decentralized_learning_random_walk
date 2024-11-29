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
        break
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
        break
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

def adaptive_mom(model, train_loader, optimizer, momentum):
    theta = 0.9
    eta = 0.01
    
    for image_train, taeget_train in train_loader:
        break
    optimizer.zero_grad()
    output = model(image_train)
    loss = F.cross_entropy(output, taeget_train)
    loss.backward()

    with torch.no_grad():
        for param, m_param in zip(model.parameters(),momentum.parameters()):
            m_param.data = theta*m_param.data +(1-theta)*param.grad.data
            param.data = param.data - eta*m_param.data

    return momentum

def gradient_clipping(model, train_loader, optimizer, momentum, v, lamda):
    theta = 0.9
    eta = 0.01
    delta = 1e-8
    for image_train, target_train in train_loader:
        break
    optimizer.zero_grad()
    output = model(image_train)
    loss = F.cross_entropy(output, target_train)
    loss.backward()


    with torch.no_grad():
        for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_param.data = theta*m_param.data+(1-theta)*param.grad.data
            v_param.data = v_param.data+(param.grad.data)**2
            # param.data = param.data - eta*m_param.data/torch.sqrt(v_param.data+delta)

    m_distance = 0.0
    v_distance = 0.0

    for current_m in momentum.parameters():
        m_distance += torch.sum((current_m )**2)
    m_distance = torch.sqrt(m_distance)


    for param, m_param in zip(model.parameters(),momentum.parameters()):
        param.data = param.data -eta*(m_param.data/m_distance)
        

    return momentum, v
