import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils
import math
from torch.autograd import Variable

def calc_speedup(growthRate,nDenseBlocks,t_0,how_scale):
    # Height*Width at each stage
    HW = [32**2, 16**2, 8**2]

    # FLOPs of first layer
    c = [3* (2*growthRate)*HW[0]*9]
    # num channels
    n = 2
    for i in range(3):
        for j in range(nDenseBlocks):
            # Calc flops for this layer
            c.append(n*(4*growthRate*growthRate)*HW[i] + 4*9*growthRate*growthRate*HW[i])
            n +=1
        n = math.floor(n*0.5)
    
    # Total computational cost for training run without freezeout
    C = 2*sum(c)

    # Computational Cost with FreezeOut
    C_f = sum(c)+sum([c_i*scale_fn[how_scale](
                    (t_0 + (1 - t_0) * float(index) / len(c) ))
                    for index,c_i in enumerate(c)])

    
    if how_scale=='linear':
        return 1.3*(1-float(C_f)/C)
    else:
        return 1-float(C_f)/C


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate,layer_index):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

        # If the layer is still being trained
        self.active=True
        
        # The index of this layer relative to the overall net
        self.layer_index=layer_index
        
    def forward(self, x):
    
        # If we're not training this layer, set to eval mode so that we use
        # running batchnorm stats (both for time-saving and to avoid updating
        # said stats).
        if not self.active:
            self.eval()
            
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        
        # If we're not active, return a detached output to prevent backprop.
        if self.active: 
            return out
        else:
            return out.detach()

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, layer_index):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        
        # Current Layer Index
        self.layer_index = layer_index
        # If the layer is being trained or not
        self.active = True
            
    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(self.bn1(F.relu(x)))
        out = torch.cat((x, out), 1)
        if self.active:
            return out
        else:
            return out.detach()

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, layer_index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

        # If the layer is being trained or not
        self.active = True
        
        # The layer index relative to the rest of the net
        self.layer_index = layer_index

    def forward(self, x):
    
        # If we're not training this layer, set to eval mode so that we use
        # running batchnorm stats (both for time-saving and to avoid updating
        # said stats).
        if not self.active:
            self.eval()
        out = self.conv1(self.bn1(F.relu(x)))
        out = F.avg_pool2d(out, 2)
        
        # If we're not active, return a detached output to prevent backprop.
        if self.active:
            return out
        else:
            return out.detach()

# growthRate=12, depth=10
class DenseNet(nn.Module):
    def __init__(self, growthRate=4, depth=16, nClasses=10, epochs=2, t_0=0.8, scale_lr=True, how_scale = 'cubic',const_time=False,reduction=0.5, bottleneck=True):
        super(DenseNet, self).__init__()
        
        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time
        
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
            
        # Calculate the speedup
        speedup = calc_speedup(growthRate,nDenseBlocks,t_0,how_scale)
        print('Estimated speedup is '+str((np.round(100*speedup)))+'%.')
        
        # Optionally scale the epochs based on the speedup so we train for
        # the same approximate wall-clock time.
        if self.const_time:
            self.epochs /= 1-speedup    
        
        
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.conv1.layer_index = 0
        self.conv1.active=True
        self.layer_index = 1
        
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels,self.layer_index)
        self.layer_index += 1

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels, self.layer_index)
        self.layer_index += 1

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        
        # Set bn and fc layers to active, permanently. Have them share a layer
        # index with the last conv layer.
        self.bn1.active=True
        self.fc.active=True
        self.bn1.layer_index = self.layer_index
        self.fc.layer_index = self.layer_index
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            

            # Set the layerwise scaling and annealing parameters
            if hasattr(m,'active'):
                m.lr_ratio = scale_fn[self.how_scale](self.t_0 + (1 - self.t_0) * float(m.layer_index) / self.layer_index)
                m.max_j = self.epochs * 1000 * m.lr_ratio
                
                # Optionally scale the learning rates to have the same total
                # distance traveled (modulo the gradients).
                m.lr = 1e-1 / m.lr_ratio if self.scale_lr else 1e-1
                
        # Optimizer
        self.optim = optim.SGD([{'params':m.parameters(), 'lr':m.lr, 'layer_index':m.layer_index} for m in self.modules() if hasattr(m,'active')],  
                         nesterov=True,momentum=0.9, weight_decay=1e-4)
        # Iteration Counter            
        self.j = 0  

        # A simple dummy variable that indicates we are using an iteration-wise
        # annealing scheme as opposed to epoch-wise. 
        self.lr_sched = {'itr':0}
                             
    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, self.layer_index))                
            else:
                layers.append(SingleLayer(nChannels, growthRate, self.layer_index))
            nChannels += growthRate
            self.layer_index += 1
        return nn.Sequential(*layers)

    def update_lr(self):
    
        # Loop over all modules
        for m in self.modules():
        
            # If a module is active:
            if hasattr(m,'active') and m.active:
            
                # If we've passed this layer's freezing point, deactivate it.
                if self.j > m.max_j: 
                    m.active = False
                    
                    # Also make sure we remove all this layer from the optimizer
                    for i,group in enumerate(self.optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            self.optim.param_groups.remove(group)
                
                # If not, update the LR
                else:
                    for i,group in enumerate(self.optim.param_groups):
                        if group['layer_index']==m.layer_index:
                            self.optim.param_groups[i]['lr'] = (0.05/m.lr_ratio)*(1+np.cos(np.pi*self.j/m.max_j))\
                                                              if self.scale_lr else 0.05 * (1+np.cos(np.pi*self.j/m.max_j))
        
        # Update the iteration counter
        self.j += 1
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 7))
        out = F.log_softmax(self.fc(out))
        return out
