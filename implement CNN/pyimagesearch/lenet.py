#kien truc thiet ke Lenet (Yann-Lecun)
# (conv + relu + pool) * 2 +FC + relu + FC + softmax
#cnn use pytorch
#implement lenet,training use Pytorch
from torch.nn import Module,Conv2d,Linear,MaxPool2d,ReLU,LogSoftmax
from torch import flatten

class LeNet(Module):
    def __init__(self,numChannels,classes):
        super(LeNet,self).__init__()#chua biet cach khoi tao class

        self.conv1 = Conv2d(in_channels = numChannels,out_channels = 20,kernel_size = (5,5)) #thieu inchannel,outchannel
        self.relu1 = ReLU() #cai dat relu
        self.maxpool1 = MaxPool2d(kernel_size=(2,2),stride=(2,2)) #cai dat pooling

        self.conv2 = Conv2d(in_channels = 20,out_channels = 50,kernel_size = (5,5))#sai in_channels va out_channel
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.fc1 = Linear(in_features = 800,out_features=500)#sai infeatures,outfeatures
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500,out_features=classes)
        self.logsoftmax = LogSoftmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = flatten(x,1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logsoftmax(x)
        return output




