import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.d_out=d_out
        self.W_k=nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_q=nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_v=nn.Linear(d_in, d_out,bias=qkv_bias)
        self.out=nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length),diagonal=1)
        )

    def forward(self,x):
        b,num_tokens,d_in=x.shape
        keys=self.W_k(x).view(b,num_tokens,self.num_heads,self.d_head).transpose(1,2)
        queries=self.W_q(x).view(b,num_tokens,self.num_heads,self.d_head).transpose(1,2)
        values=self.W_v(x).view(b,num_tokens,self.num_heads,self.d_head).transpose(1,2)

        attention_scores=queries @ keys.transpose(2,3)
        maskbool=self.mask.bool()[:num_tokens,:num_tokens]
        attention_scores.masked_fill_(maskbool,-torch.inf)

        attention_weights=torch.softmax(attention_scores/keys.shape[-1]**0.5,dim=-1)
        attention_weights=self.dropout(attention_weights)
        context=attention_weights @ values
        context=context.transpose(1,2).contiguous().view(b,num_tokens,self.d_out)#contiguous()：让数据连续。
        return self.out(context)
