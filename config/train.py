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
            v_param.data = theta*v_param.data+(1-theta)*(param.grad.data)**2
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

def clipped_mom(model, train_loader, optimizer, momentum,v):
    theta = 0.9
    eta = 0.01
    delta = 1e-8
    
    for image_train, taeget_train in train_loader:
        break
    optimizer.zero_grad()
    output = model(image_train)
    loss = F.cross_entropy(output, taeget_train)
    loss.backward()

    m_distance = 0
    with torch.no_grad():
        for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_param.data = theta*m_param.data +(1-theta)*param.grad.data
            v_param.data = v_param.data+(param.grad.data)**2

        
        for m_param in momentum.parameters():
            m_distance += torch.sum((m_param)**2)

        m_distance = torch.sqrt(m_distance)

        for param,m_param,v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_clipped = m_param.data/m_distance
            param.data = param.data - eta*m_clipped/torch.sqrt(v_param+delta)

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

    gradient_distance = 0.0

    with torch.no_grad():
        for gradient_param in model.parameters():
            gradient_distance += torch.sum((gradient_param)**2)
        
        gradient_distance = torch.sqrt(gradient_distance)

        for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
            m_param.data = theta*m_param.data+(1-theta)*param.grad.data/gradient_distance
            v_param.data = v_param.data+(param.grad.data/gradient_distance)**2


    for param, m_param, v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
        param.data = param.data -eta*(m_param.data/torch.sqrt(v_param+delta))
        

    return momentum, v

def clipped_adaptive(model, train_loader, optimizer, momentum, v):
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

    m_distance = 0.0
    v_distance = 0.0

    for current_m, current_v in zip(momentum.parameters(),v.parameters()):
        m_distance += torch.sum((current_m )**2)
        v_distance += torch.sum((current_v)**2)
    m_distance = torch.sqrt(m_distance)
    v_distance = torch.sqrt(v_distance)

    for param, m_param,v_param in zip(model.parameters(),momentum.parameters(),v.parameters()):
        m_clipped = m_param.data/m_distance
        v_clipped = v_param.data/v_distance
        param.data = param.data -eta*m_clipped/torch.sqrt(v_clipped.data+delta)
        

    return momentum, v
