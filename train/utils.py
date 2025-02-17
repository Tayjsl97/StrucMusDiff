import time
import math
import pickle
import torch
import numpy as np
import random
from torch.utils.data import DataLoader,TensorDataset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def timeSince(since):
    now=time.time()
    s=now-since
    h=math.floor(s/3600)
    s-=h*3600
    m=math.floor(s/60)
    s-=m*60

    return '%dh_%dm_%ds' % (h, m, s)


def onset_to_type(data,onset2type):
    new_data=torch.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            new_data[i][j][0]=onset2type[data[i][j][0].item()]
            new_data[i][j][1]=data[i][j][1]
    return new_data


def normalize(x):
    mean = 4.3653
    std = 22.6885
    x -= mean
    x /= (std+1e-8)
    return x


def unnormalize(x):
    mean = 4.3653
    std = 22.6885
    x *= (std+1e-8)
    x += mean
    return x


def get_train_val_dataloaders(data_path,batch_size,num_workers=0,pin_memory=False,is_latent=False):
    file=open(data_path,'rb')
    data=pickle.load(file)
    input_data=torch.Tensor(data['input']).to(device)#.narrow(0,0,20)
    simi_data=torch.Tensor(data['sim_data']).to(device)#.narrow(0,0,20)
    simi_value=torch.Tensor(np.round(data['sim_value'],2)).to(device)#.narrow(0,0,20)
    simi_index = torch.LongTensor(data['sim_index']).to(device)  # .narrow(0,0,20)
    phrase_mark=torch.LongTensor(data['phrase_mark']).to(device)#.narrow(0,0,20)
    emotion=torch.LongTensor(data['emotion']).to(device)#.narrow(0,0,20)
    input_data=normalize(input_data)
    simi_data=normalize(simi_data)
    train_input_data, eval_input_data=train_test_split_by_emo(input_data,emotion,test_size=0.1, random_state=77)
    train_simi_data, eval_simi_data = train_test_split_by_emo(simi_data, emotion,test_size=0.1, random_state=77)
    train_simi_value, eval_simi_value = train_test_split_by_emo(simi_value, emotion,test_size=0.1, random_state=77)
    train_simi_index, eval_simi_index = train_test_split_by_emo(simi_index, emotion, test_size=0.1, random_state=77)
    train_phrase_mark, eval_phrase_mark = train_test_split_by_emo(phrase_mark, emotion,test_size=0.1, random_state=77)
    train_emotion, eval_emotion = train_test_split_by_emo(emotion, emotion,test_size=0.1, random_state=77)
    train_dataset = TensorDataset(train_input_data, train_simi_data, train_simi_index, train_simi_value,
                                  train_phrase_mark,train_emotion)
    eval_dataset = TensorDataset(eval_input_data, eval_simi_data, eval_simi_index, eval_simi_value,
                                 eval_phrase_mark,eval_emotion)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        # drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        eval_dataset,
        batch_size,
        # drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_dl,val_dl


def get_train_val_dataloaders_latent(data_path,batch_size,num_workers=0,pin_memory=False):
    # data = np.load(data_path).to(device)  # (1071,15,2)
    file=open(data_path,'rb')
    data=pickle.load(file)
    input_data=torch.Tensor(data['input']).to(device)#.narrow(0,0,20)
    simi_data_z=torch.Tensor(data['sim_data_z']).to(device)#.narrow(0,0,20)
    simi_data = torch.Tensor(data['sim_data']).to(device)  # .narrow(0,0,20)
    simi_value=torch.Tensor(data['sim_value']).to(device)#.narrow(0,0,20)
    simi_index = torch.LongTensor(data['sim_index']).to(device)  # .narrow(0,0,20)
    phrase_mark=torch.LongTensor(data['phrase_mark']).to(device)#.narrow(0,0,20)
    emotion=torch.LongTensor(data['emotion']).to(device)#.narrow(0,0,20)
    # print("input_data: ",input_data[0])
    input_data=normalize(input_data)
    simi_data=normalize(simi_data)
    train_input_data, eval_input_data = train_test_split_by_emo(input_data,emotion,test_size=0.1, random_state=77)
    train_simi_data_z, eval_simi_data_z = train_test_split_by_emo(simi_data_z, emotion, test_size=0.1, random_state=77)
    train_simi_data, eval_simi_data = train_test_split_by_emo(simi_data, emotion,test_size=0.1, random_state=77)
    train_simi_value, eval_simi_value = train_test_split_by_emo(simi_value, emotion,test_size=0.1, random_state=77)
    train_simi_index, eval_simi_index = train_test_split_by_emo(simi_index, emotion, test_size=0.1, random_state=77)
    train_phrase_mark, eval_phrase_mark = train_test_split_by_emo(phrase_mark, emotion,test_size=0.1, random_state=77)
    train_emotion, eval_emotion = train_test_split_by_emo(emotion, emotion,test_size=0.1, random_state=77)
    # train_input_data, eval_input_data = train_test_split(input_data, test_size=0.1, random_state=77)
    # train_simi_data, eval_simi_data = train_test_split(simi_data, test_size=0.1, random_state=77)
    # train_simi_value, eval_simi_value = train_test_split(simi_value, test_size=0.1, random_state=77)
    # train_phrase_mark, eval_phrase_mark = train_test_split(phrase_mark, test_size=0.1, random_state=77)
    # train_emotion, eval_emotion = train_test_split(emotion, test_size=0.1, random_state=77)
    train_dataset = TensorDataset(train_input_data,train_simi_data_z,train_simi_data, train_simi_index, train_simi_value,train_phrase_mark,train_emotion)
    eval_dataset = TensorDataset(eval_input_data, eval_simi_data_z,eval_simi_data, eval_simi_index, eval_simi_value,eval_phrase_mark,eval_emotion)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        # drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        eval_dataset,
        batch_size,
        # drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_dl,val_dl
