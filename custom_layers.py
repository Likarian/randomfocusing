import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_RF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 division_rate):
        super(Linear_RF, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.division_rate = division_rate

        self.Linear = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):

        if self.training:
            rand_idx = torch.arange(x.size(1))
            rand_idx = torch.randperm(rand_idx.numel())
            division_idx = int(x.size(1)*self.division_rate)
            rand_array = rand_idx[:division_idx]

            division_mask = torch.zeros( x.shape ).cuda()
            division_mask[:, rand_array] = 1
            
            x_throwed = x * division_mask
            x_saved = x * (1-division_mask)

            output_saved = self.Linear(x_saved)
            output_throwed = self.Linear(x_throwed)
            return output_saved / (1-self.division_rate) + output_throwed / self.division_rate

        else:
            output = self.Linear(x)
            return output

class Conv2d_RF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=True,
                 division_rate=0.0):
        super(Conv2d_RF, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.division_rate = division_rate

        self.Conv = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                bias=bias)

    def forward(self, x):

        if self.training:
            rand_idx = torch.arange(x.size(1))
            rand_idx = torch.randperm(rand_idx.numel())
            division_idx = int(x.size(1)*self.division_rate)
            rand_array = rand_idx[:division_idx]

            division_mask = torch.zeros( x.shape ).cuda()
            division_mask[:, rand_array, :, :] = 1
            
            x_throwed = x * division_mask
            x_saved = x * (1-division_mask)

            output_saved = self.Conv(x_saved)
            output_throwed = self.Conv(x_throwed)

            return output_saved / (1-self.division_rate) + output_throwed / self.division_rate

        else:
            output = self.Conv(x)
            return output

class Linear_RFwJSD(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 division_rate):
        super(Linear_RFwJSD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.division_rate = division_rate

        self.Linear = nn.Linear(self.in_channels, self.out_channels)


    def cal_js_div(self, p, q):
        with torch.no_grad():
            log_p = F.log_softmax(p, dim=1)
            log_q = F.log_softmax(q, dim=1)

            mixture = ( log_p + log_q ) / 2
            
            pm = F.kl_div(mixture, log_p, reduction = 'batchmean', log_target = True)
            qm = F.kl_div(mixture, log_q, reduction = 'batchmean', log_target = True)

            js_div = (pm + qm) / 2
            return js_div


    def forward(self, x):

        if self.training:
            rand_idx = torch.arange(x.size(1))
            rand_idx = torch.randperm(rand_idx.numel())
            division_idx = int(x.size(1)*self.division_rate)
            rand_array = rand_idx[:division_idx]

            division_mask = torch.zeros( x.shape ).cuda()
            division_mask[:, rand_array] = 1
            
            x_throwed = x * division_mask
            x_saved = x * (1-division_mask)

            output_saved = self.Linear(x_saved)
            output_throwed = self.Linear(x_throwed)

            JS_div_outputs = self.cal_js_div( output_saved, output_throwed )

            return output_saved / ( (1+JS_div_outputs) * (1-self.division_rate) ) + output_throwed * self.division_rate * JS_div_outputs / (1+JS_div_outputs)

        else:
            output = self.Linear(x)
            return output

class Conv2d_RFwJSD(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias=True,
                 division_rate=0.0):
        super(Conv2d_RFwJSD, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.division_rate = division_rate

        self.Conv = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                bias=bias)

    def cal_js_div(self, p, q):
        with torch.no_grad():
            log_p = F.log_softmax(p.view(p.size(0), -1), dim=1)
            log_q = F.log_softmax(q.view(q.size(0), -1), dim=1)

            mixture = ( log_p + log_q ) / 2
            
            pm = F.kl_div(mixture, log_p, reduction = 'batchmean', log_target = True)
            qm = F.kl_div(mixture, log_q, reduction = 'batchmean', log_target = True)

            js_div = (pm + qm) / 2
            return js_div


    def forward(self, x):

        if self.training:
            rand_idx = torch.arange(x.size(1))
            rand_idx = torch.randperm(rand_idx.numel())
            division_idx = int(x.size(1)*self.division_rate)
            rand_array = rand_idx[:division_idx]

            division_mask = torch.zeros( x.shape ).cuda()
            division_mask[:, rand_array, :, :] = 1
            
            x_throwed = x * division_mask
            x_saved = x * (1-division_mask)

            output_saved = self.Conv(x_saved)
            output_throwed = self.Conv(x_throwed)

            JS_div_outputs = self.cal_js_div( output_saved, output_throwed )

            return output_saved / ( (1+JS_div_outputs) * (1-self.division_rate) ) + output_throwed * self.division_rate * JS_div_outputs / (1+JS_div_outputs)

        else:
            output = self.Conv(x)
            return output

