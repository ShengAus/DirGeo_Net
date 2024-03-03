import torch
from models.base_model import BaseModel
from models.networks import define_net
from utils.utils import fa_calc, sum_SOPM, ratio_map

class DirGeoModel(BaseModel): # HADTI
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add model-specific options.
        """

        return parser


    def __init__(self, opt):
        """
        Initialize this mask Inpaint class.
        """
        BaseModel.__init__(self, opt)

        self.model_names = ['U3D']
        self.loss_names = ['l1']
        
        self.net_U3D = define_net(net_name = opt.net, input_nc = opt.input_nc, output_nc = opt.output_nc
        , init_type = opt.init_type, init_gain = opt.init_gain
        , gpu_ids = opt.gpu_ids)

        self.twweight = torch.tensor([1e3,1e6,1e6,1e3,1e6,1e3]).view(-1,1,1,1).to(self.device)
        if self.isTrain:
            self.l1loss = torch.nn.L1Loss(reduction='none')
            self.optimizer_sr = torch.optim.Adam(self.net_U3D.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_sr)

        self.relu = torch.nn.ReLU()

    def set_input(self, input):
        """
        Read the data of input from dataloader then
        """
        self.dwi = input['dwi'].to(self.device) # (N,C,W,H,D)

        grad = input['grad'].unsqueeze(0).repeat(self.dwi.shape[0],1,1) # (N, 7, 3)
        self.grad = grad.to(self.device).type(self.dwi.dtype) # (N,7,3)
        # print(self.grad.shape)

        if self.isTrain:
            self.gt_dti = input['gt_dti'].to(self.device)
            # print(self.gt_dti.shape)
            self.wm_mask = input['wm_mask'].to(self.device)

    def forward(self):
        """
        Run forward pass
        """
        # self.input = self.dwi
        self.sr = self.net_U3D(self.dwi, self.grad)

        return self.sr

    def backward_sr(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.wm_mask = self.wm_mask.squeeze(1)
        # print(self.wm_mask.shape)
        self.sr = self.sr.permute(0,2,3,4,1)
        # print(self.sr.shape)
        self.gt_dti = self.gt_dti.permute(0,2,3,4,1)

        pm3 = ratio_map(self.sr[self.wm_mask==1], self.gt_dti[self.wm_mask==1], relu=self.relu)
        print('Volume Invariant:', pm3)

        l1_part1 = 0
        l1_part1 = 1e6*torch.mean(self.l1loss(self.sr[self.wm_mask==1], self.gt_dti[self.wm_mask==1]))
        print('part1', l1_part1)

        l1_part2 = 0
        fa_pred = fa_calc(self.sr[self.wm_mask==1])
        fa_gt = fa_calc(self.gt_dti[self.wm_mask==1])
        lamb1 = 10*self.current_epoch
        l1_part2 = lamb1*torch.mean(pm3*self.l1loss(fa_pred,fa_gt))
        print('part2' , l1_part2)

        l1_part3 = 0
        S_pred = sum_SOPM(self.sr[self.wm_mask==1])
        S_gt = sum_SOPM(self.gt_dti[self.wm_mask==1])
        l1_part3 = 1e6*torch.mean(pm3*self.l1loss(S_pred, S_gt))
        print('part3', l1_part3)

        self.loss_l1 = l1_part1 + l1_part3 + l1_part2 
        self.loss_l1.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        # update optimizer of the inpainting network
        self.optimizer_sr.zero_grad()
        self.backward_sr()
        self.optimizer_sr.step()



    




    

        
    
