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

class StraightCost(nn.Module):
    """ Straight cost 
    """
    def __init__(self, ndofs, weight=None, gaussian_params={}, tensor_args={'device':"cpu", 'dtype':torch.float32}, **kwargs):
        super(StraightCost, self).__init__()
        self.ndofs = ndofs
        self.tensor_args = tensor_args
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']
        self.weight = torch.as_tensor(weight, device=self.device, dtype=self.dtype)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

    

    def forward(self, ee_pos_batch, ee_goal_pos, jac_batch,state_batch):
        
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_goal_pos = ee_goal_pos.to(device=self.device,
                                     dtype=self.dtype)
        jac_batch = jac_batch.to(self.device)
        
        goal_vector = ee_goal_pos - ee_pos_batch
        goal_vector_norm = torch.norm(goal_vector, p=2, dim=-1, keepdim=True)
        goal_vector_normal = goal_vector/goal_vector_norm
        goal_vector_normal_no_nan = torch.where(torch.isnan(goal_vector_normal), torch.zeros_like(goal_vector_normal), goal_vector_normal)

        qdot = state_batch[:,:,self.ndofs:2*self.ndofs]

        eef_vector = torch.matmul(jac_batch,qdot.unsqueeze(-1)).squeeze(-1)
        eef_vector_norm = torch.norm(eef_vector, p=2, dim=-1, keepdim=True)
        eef_vector_normal = eef_vector/eef_vector_norm
        eef_vector_normal_no_nan = torch.where(torch.isnan(eef_vector_normal), torch.zeros_like(eef_vector_normal), eef_vector_normal)
        # import pdb; pdb.set_trace()
        dot_product = (goal_vector_normal_no_nan * eef_vector_normal_no_nan).sum(dim=-1, keepdim=True).squeeze(-1)
        cost = self.weight * self.proj_gaussian(1.0 - dot_product)

        return cost.to(inp_device)


