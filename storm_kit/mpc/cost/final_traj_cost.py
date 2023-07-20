#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.nn as nn
# import torch.nn.functional as F

from .gaussian_projection import GaussianProjection

class FinalTrajCost(nn.Module):
    def __init__(self, weight=None, vec_weight=None, gaussian_params={}, device=torch.device('cpu'), float_dtype=torch.float32, **kwargs):
        super(FinalTrajCost, self).__init__()
        self.device = device
        self.float_dtype = float_dtype
        self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)
        if(vec_weight is not None):
            self.vec_weight = torch.as_tensor(vec_weight, device=device, dtype=float_dtype).unsqueeze(0)
        else:
            self.vec_weight = 1.0
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
    
    def forward(self, disp_vec, dist_type="l2", beta=1.0, RETURN_GOAL_DIST=False):
        inp_device = disp_vec.device
        # we might use the vec_weight to set 0,0,0,0,0,0,1.0 to keep the cost of last point only
        disp_vec = self.vec_weight * disp_vec.to(self.device)
        # only consider the last point of every sampled trajectory wrt horizon goal point
        disp_vec[:, :-1, :] = 0.0
        if dist_type == 'l2':
            dist = torch.norm(disp_vec, p=2, dim=-1,keepdim=False)
        elif dist_type == 'squared_l2':

            dist = (torch.sum(torch.square(disp_vec), dim=-1,keepdim=False))
        elif dist_type == 'l1':
            dist = torch.norm(disp_vec, p=1, dim=-1,keepdim=False)
        elif dist_type == 'smooth_l1':
            l1_dist = torch.norm(disp_vec, p=1, dim=-1)
            dist = None
            raise NotImplementedError

        cost = self.weight * self.proj_gaussian(dist)

        if(RETURN_GOAL_DIST):
            return cost.to(inp_device), dist.to(inp_device)
        return cost.to(inp_device)


