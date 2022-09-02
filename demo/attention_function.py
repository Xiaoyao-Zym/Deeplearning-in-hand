import torch
import math
from torch import nn
from d2l import torch as d2l
#掩蔽softmax操作
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作
    任何超出有效长度的位置都被掩蔽并置为0。
    X----输入张量
    valid_lens---有效长度
    """
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape=X.shape
        if valid_lens.dim()==1:
            valid_lens=torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens=valid_lens.reshape(-1)
         # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X=d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=1e6 )
        return nn.functional.softmax(X.reshape(shape), dim=-1)

masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))