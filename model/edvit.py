import torch
import torch.nn as nn
from .edvit_ed import EDViTED
from .edvit_vae import EDViTVAE
from torchvision import transforms

class EDViT(nn.Module):

    def __init__(self, config, ed, vae, net, fp16):
        super(EDViT, self).__init__()
        self.net = net
        self.fp16 = fp16
        if self.net=='ed':
            try:
                self.model_ed = EDViTED(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))

                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)

                self.model_ed.eval()
                if self.fp16:
                    self.model_ed.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
        elif self.net=='vae':
            try:
                self.model_vae = EDViTVAE(config)
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))

                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                    
                self.model_vae.eval()
                if self.fp16:
                    self.model_vae.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{vae}.pth file not found.")
        elif self.net=='both':
            try:
                self.model_ed = EDViTED(config)
                self.model_vae = EDViTVAE(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))
                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)
                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                self.model_ed.eval()
                self.model_vae.eval()
                if self.fp16:
                    self.model_ed.half()
                    self.model_vae.half()
            except FileNotFoundError as e:
                raise Exception(f"Error: Model weights file not found.")
        else:
            raise Exception(f"Error: Model variant not found.")


    def forward(self, x):
        if self.net == 'ed' :
            x = self.model_ed(x)
        elif self.net == 'vae':
            x,_ = self.model_vae(x)
        elif self.net == 'both':
            x1 = self.model_ed(x)
            x2,_ = self.model_vae(x)
            x =  torch.cat((x1, x2), dim=0) #(x1+x2)/2 #
        return x
