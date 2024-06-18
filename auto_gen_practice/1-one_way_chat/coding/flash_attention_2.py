# filename: flash_attention_2.py
import torch
import torch.nn as nn

class FlashAttention2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FlashAttention2, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias, **factory_kwargs)
        self.v = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv, **factory_kwargs)
        self.proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qk.weight)
        if self.qk.bias is not None:
            nn.init.constant_(self.qk.bias, 0.)
        nn.init.xavier_uniform_(self.v.weight)
        if self.v.bias is not None:
            nn.init.constant_(self.v.bias, 0.)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        bsz, q_len, _ = query.size()
        _, k_len, _ = key.size()
        head_dim = self.embed_dim // self.num_heads

        qk = self.qk(query).view(bsz, q_len, self.num_heads, 2 * head_dim)
        q, k = qk.chunk(2, dim=-1)
        v = self.v(value).view(bsz, k_len, self.num_heads, head_dim)

        q = q.transpose(1, 2).contiguous().view(bsz * self.num_heads, q_len, head_dim)
        k = k.transpose(1, 2).contiguous().view(bsz * self.num_heads, k_len, head_dim)
        v = v.transpose(1, 2).contiguous().view(bsz * self.num_heads, k_len, head_dim)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "Expect key_padding_mask shape to be (batch_size, seq_len), but got {}".format(key_padding_mask.shape)
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, k_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, k_len)

        attn_output, attn_output_weights = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0,
            need_weights=need_weights, key_padding_mask=key_padding_mask
        )
        attn_output = attn_output.view(bsz, self.num_heads, q_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)

        attn_output = self.proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)
        return attn_output, attn_output_weights
