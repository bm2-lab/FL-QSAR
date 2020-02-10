import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,random_split
import numpy as np
import pandas as pd
import crypten

def normalize(target_array):
    return (target_array-data_max)/(data_max-data_min)

def logarithmic(target_array):
    return np.log(target_array+1)

def calculate_r_square(output,target):
    return 1-torch.div(torch.sum((output-target).pow(2)),torch.sum((target-target.mean()).pow(2)))
    
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1)
        self.dropout1=nn.Dropout(p=0.25)
        self.dropout2=nn.Dropout(p=0.25)
        self.dropout3=nn.Dropout(p=0.25)
        self.dropout4=nn.Dropout(p=0.1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout1(x)))
        x = F.relu(self.fc3(self.dropout2(x)))
        x = F.relu(self.fc4(self.dropout3(x)))
        x = self.fc5(self.dropout4(x))
        return x

    


def preprocess(x_origin,y_origin):
    y_origin=y_origin.reshape((-1,1))
    y=normalize(y_origin)
    x=logarithmic(x_origin)
    x=torch.from_numpy(x)
    y=torch.from_numpy(y)
    x=torch.tensor(x, dtype=torch.float32)
    y=torch.tensor(y, dtype=torch.float32)
    
    return x,y

crypten.init()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_list=['LOGD','METAB','NK1','OX1','OX2','PGP','PPB','RAT_F','TDI','THROMBIN','CB1','DPP4','HIVINT','HIVPROT','3A4']
for filename in file_list:
    for count in range(10):
        torch.manual_seed(count+1)
        compute_nodes = ['bob', 'alice','ray','rita']
        train_file=f"./data/{filename}_training_disguised.csv"
        test_file=f"./data/{filename}_test_disguised2.csv"
        data = pd.read_csv(train_file)
        test_data=pd.read_csv(test_file)
        input_size=data.shape[1]-2
        y_origin=np.array(data.iloc[:,1].values)
        x_origin=np.array(data.iloc[:,2:].values)
        y_test_origin=np.array(test_data.iloc[:,1].values)
        x_test_origin=np.array(test_data.iloc[:,2:].values)
        y_all_origin=np.concatenate((y_origin,y_test_origin),axis=0)
        data_max=y_origin.max() if y_origin.max() > y_test_origin.max() else y_test_origin.max()
        data_min=y_origin.min() if y_origin.min() < y_test_origin.min() else y_test_origin.min()


        BATCH_SIZE=128
        criterion = nn.MSELoss()

        x,y=preprocess(x_origin,y_origin)
        x_test,y_test=preprocess(x_test_origin,y_test_origin)
        x_test,y_test=x_test.to(device),y_test.to(device)
        train = TensorDataset(x, y)
        train_1,train_2,train_3,train_4 = random_split(train,[len(train)//4,len(train)//4,len(train)//4,len(train)-3*(len(train)//4)])
        train_loader_1 = DataLoader(train_1, batch_size=BATCH_SIZE)
        train_loader_2 = DataLoader(train_2, batch_size=BATCH_SIZE)
        train_loader_3 = DataLoader(train_3, batch_size=BATCH_SIZE)
        train_loader_4 = DataLoader(train_4, batch_size=BATCH_SIZE)
        train_loader_list = [train_loader_1,train_loader_2,train_loader_3,train_loader_4]

        ray_R_squared_list=[]
        alice_R_squared_list=[]
        bob_R_squared_list=[]
        rita_R_squared_list=[]
        R_squared_list=[]

        f=open(f'./4_result1/{filename}_result_{count+1}.txt','w')
        torch.manual_seed(1)
        solo_bobs_model = Net().to(device)
        solo_alices_model = Net().to(device)
        solo_rays_model = Net().to(device)
        solo_ritas_model = Net().to(device)
        solo_bobs_optimizer = optim.SGD(solo_bobs_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_alices_optimizer = optim.SGD(solo_alices_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_rays_optimizer = optim.SGD(solo_rays_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_ritas_optimizer = optim.SGD(solo_ritas_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_optimizers=[solo_bobs_optimizer, solo_alices_optimizer,solo_rays_optimizer,solo_ritas_optimizer]
        solo_models = [solo_bobs_model, solo_alices_model,solo_rays_model,solo_ritas_model]
        params = [list(solo_bobs_model.parameters()), list(solo_alices_model.parameters()),list(solo_rays_model.parameters()),list(solo_ritas_model.parameters())]
        for i in range(180):
            new_params = list()
            for k in range(len(train_loader_list)):
                train_loss = 0
                for batch_idx, (data,target) in enumerate(train_loader_list[k]):
                    solo_optimizers[k].zero_grad()
                    data,target = data.to(device),target.to(device)
                    output = solo_models[k](data)      
                    loss = criterion(output, target)
                    loss.backward()
                    train_loss += loss.item()*len(output)
                    solo_optimizers[k].step()
                f.write(compute_nodes[k]+"_train_loss:"+str(train_loss/len(train_loader_list[k].dataset))+"\n")

            with torch.no_grad():
                bob_test_output=solo_models[0](x_test)
                bob_test_loss=criterion(bob_test_output,y_test)
                f.write('bob_test loss:'+str(bob_test_loss.item())+"\n")
                f.write("bob_R_squared:"+str(calculate_r_square(bob_test_output.detach(),y_test).item())+"\n")
                bob_R_squared_list.append(calculate_r_square(bob_test_output.detach(),y_test))

                alice_test_output=solo_models[1](x_test)
                alice_test_loss=criterion(alice_test_output,y_test)
                f.write('alice_test loss:'+str(alice_test_loss.item())+"\n")
                f.write("alice_R_squared:"+str(calculate_r_square(alice_test_output.detach(),y_test).item())+"\n")
                alice_R_squared_list.append(calculate_r_square(alice_test_output.detach(),y_test))

                ray_test_output=solo_models[2](x_test)
                ray_test_loss=criterion(ray_test_output,y_test)
                f.write('ray_test loss:'+str(ray_test_loss.item())+"\n")
                f.write("ray_R_squared:"+str(calculate_r_square(ray_test_output.detach(),y_test).item())+"\n")
                ray_R_squared_list.append(calculate_r_square(ray_test_output.detach(),y_test))

                rita_test_output=solo_models[3](x_test)
                rita_test_loss=criterion(rita_test_output,y_test)
                f.write('rita_test loss:'+str(rita_test_loss.item())+"\n")
                f.write("rita_R_squared:"+str(calculate_r_square(rita_test_output.detach(),y_test).item())+"\n")
                rita_R_squared_list.append(calculate_r_square(rita_test_output.detach(),y_test))
        f.write("\n"+"bob_max_R_squared:"+str(max(bob_R_squared_list).item())+"\n")
        f.write("alice_max_R_squared:"+str(max(alice_R_squared_list).item())+"\n")
        f.write("ray_max_R_squared:"+str(max(ray_R_squared_list).item())+"\n")
        f.write("rita_max_R_squared:"+str(max(rita_R_squared_list).item())+"\n")
        f.close()

        f=open(f'./4_result/{filename}_result_{count+1}.txt','w')
        torch.manual_seed(1)
        solo_bobs_model = Net().to(device)
        solo_alices_model = Net().to(device)
        solo_rays_model = Net().to(device)
        solo_ritas_model = Net().to(device)

        solo_bobs_optimizer = optim.SGD(solo_bobs_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_alices_optimizer = optim.SGD(solo_alices_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_rays_optimizer = optim.SGD(solo_rays_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_ritas_optimizer = optim.SGD(solo_ritas_model.parameters(), lr=0.05,momentum=0.9,weight_decay=0.0001)
        solo_optimizers=[solo_bobs_optimizer, solo_alices_optimizer,solo_rays_optimizer,solo_ritas_optimizer]
        solo_models = [solo_bobs_model, solo_alices_model,solo_rays_model,solo_ritas_model]
        params = [list(solo_bobs_model.parameters()), list(solo_alices_model.parameters()),list(solo_rays_model.parameters()),list(solo_ritas_model.parameters())]
        for i in range(180):
            new_params = list()
            for k in range(len(train_loader_list)):
                train_loss = 0
                for batch_idx, (data,target) in enumerate(train_loader_list[k]):
                    solo_optimizers[k].zero_grad()
                    data,target = data.to(device),target.to(device)
                    output = solo_models[k](data)      
                    loss = criterion(output, target)
                    loss.backward()
                    train_loss += loss.item()*len(output)
                    solo_optimizers[k].step()
                f.write(compute_nodes[k]+"_train_loss:"+str(train_loss/len(train_loader_list[k].dataset))+"\n")
            for param_i in range(len(params[0])):
                spdz_params = list()
                for remote_index in range(len(compute_nodes)):
                    clone_param=params[remote_index][param_i].clone().cpu()
                    spdz_params.append(crypten.cryptensor(torch.tensor(clone_param)))
                new_param = ((spdz_params[0] + spdz_params[1]+spdz_params[2]+spdz_params[3])/4).get_plain_text()
                new_params.append(new_param)

            with torch.no_grad():
                for model in params:
                    for param in model:
                        param *= 0
                
                for remote_index in range(len(compute_nodes)):
                    for param_index in range(len(params[remote_index])):
                        new_params[param_index] = new_params[param_index].to(device)
                        params[remote_index][param_index].set_(new_params[param_index])

                bob_test_output=solo_models[0](x_test)
                bob_test_loss=criterion(bob_test_output,y_test)
                f.write('bob_test loss:'+str(bob_test_loss.item())+"\n")
                f.write("bob_R_squared:"+str(calculate_r_square(bob_test_output.detach(),y_test).item())+"\n")
                R_squared_list.append(calculate_r_square(bob_test_output.detach(),y_test))

                alice_test_output=solo_models[1](x_test)
                alice_test_loss=criterion(alice_test_output,y_test)
                f.write('alice_test loss:'+str(alice_test_loss.item())+"\n")
                f.write("alice_R_squared:"+str(calculate_r_square(alice_test_output.detach(),y_test).item())+"\n")
                R_squared_list.append(calculate_r_square(alice_test_output.detach(),y_test))

                ray_test_output=solo_models[2](x_test)
                ray_test_loss=criterion(ray_test_output,y_test)
                f.write('ray_test loss:'+str(ray_test_loss.item())+"\n")
                f.write("ray_R_squared:"+str(calculate_r_square(ray_test_output.detach(),y_test).item())+"\n")
                R_squared_list.append(calculate_r_square(ray_test_output.detach(),y_test))

                rita_test_output=solo_models[3](x_test)
                rita_test_loss=criterion(rita_test_output,y_test)
                f.write('rita_test loss:'+str(rita_test_loss.item())+"\n")
                f.write("rita_R_squared:"+str(calculate_r_square(rita_test_output.detach(),y_test).item())+"\n")
                R_squared_list.append(calculate_r_square(rita_test_output.detach(),y_test))
        f.write("\n"+"max_R_squared:"+str(max(R_squared_list).item())+"\n")
        f.close()