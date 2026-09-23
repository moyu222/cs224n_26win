"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter


# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int # 是模型允许的最大 token 数。
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        # TODO, complete
        # ===== 以下为新增实现：因果多头自注意力 =====
        """Apply masked (causal) multi-head self-attention.

        For each head, the attention score from token i to token j is
        ``(q_i @ k_j) / sqrt(D_a)``，其中 ``D_a = d_attention = D / H``
        是单个 attention head 的维度。Scores for future tokens (j > i)
        are masked to -infinity before softmax, making their probability zero.
        """
        _, seq_len, _ = x.shape

        # 中文理解：每个 token 都会生成 Q（“我想找什么”）、K（“我能提供什么”）
        # 和 V（“真正传递的信息”）。Q 与 K 的点积决定应该从哪些 token 取 V。
        # D = H * D_a，其中 D_a 对应代码里的 self.d_attention。
        # Each projection is [B, T, D].
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split D into H heads:
        # [B, T, H * D_a] -> [B, H, T, D_a].
        q = rearrange(q, "b t (h d) -> b h t d", d=self.d_attention)
        k = rearrange(k, "b t (h d) -> b h t d", d=self.d_attention)
        v = rearrange(v, "b t (h d) -> b h t d", d=self.d_attention)

        # [B, H, T, D_a] @ [B, H, D_a, T] -> [B, H, T, T].
        # Scaling by sqrt(D_a) prevents softmax from becoming too sharp.
        # 中文：transpose(-2, -1) 把 K 的最后两维变成 [D_a, T]，这样矩阵
        # 乘法后第 (i, j) 项正好是第 i 个 query 与第 j 个 key 的内积。
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_attention)

        # 中文：下三角 mask 保证位置 i 看不到 i 后面的词；否则训练时模型会
        # “偷看答案”。masked_fill 后 softmax(-inf) 恰好为 0。
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_probs = softmax(scores, dim=-1)  # [B, H, T, T]

        # Weighted value sum: [B, H, T, T] @ [B, H, T, D_a].
        # 中文：这里不是挑一个 token，而是按 attention 概率对所有允许的 V 做加权平均。
        attended = attention_probs @ v  # [B, H, T, D_a]

        # Join heads and project: [B, H, T, D_a] -> [B, T, D] -> [B, T, D].
        attended = rearrange(attended, "b h t d -> b t (h d)")
        output = self.W_o(attended)

        return output



class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(self, x: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:

        # Position-wise feed-forward network: [B, T, D] -> [B, T, 4D]
        # -> GELU -> [B, T, D].  It does not mix different token positions.
        # 中文：MLP 相当于每个位置各自经过同一套两层非线性网络；token 之间的交流
        # 已由 attention 完成，MLP 负责变换/提炼每个位置收集到的信息。
        h1 = self.fc1(x)
        h2 = self.gelu(h1)
        return self.fc2(h2)
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        # TODO complete
        # ===== 以下为新增实现：Pre-LN 残差解码器块 =====
        """Run one GPT-2 pre-layer-normalized transformer block.

        h = x + Attention(LN_1(x))
        y = h + MLP(LN_2(h))

        Every residual tensor has shape [B, T, D].
        """
        # GPT-2 normalizes the input before each sub-layer (Pre-LN), then
        # adds the sub-layer output back through the residual connection.
        # 中文：残差连接的“+ x”让原信息可以直接绕过子层，也使深层网络的梯度更稳定。
        # 这里一定是 Attention(LN(x))，而不是 Attention(x) + LN(x)。
        attention_input = self.pre_layer_norm(x)  # [B, T, D]
        attention_output = self.attention(attention_input)
        hidden = x + attention_output  # [B, T, D]

        mlp_input = self.post_layer_norm(hidden)      # [B, T, D]
        return hidden + self.mlp(mlp_input)           # [B, T, D]


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        # 残差add 前的矩阵参数，减小
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(self, x: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        # TODO, complete
        # ===== 以下为新增实现：embedding、transformer backbone 与语言模型头 =====
        """Map token ids [B, T] to vocabulary logits [B, T, V]."""
        _, seq_len = x.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length "
                f"{self.config.context_length}."
            )

        # E_token[x[b, t]] is [B, T, D].  Position vectors [T, D] broadcast
        # across B: h_0[b, t] = E_token[x[b, t]] + E_position[t].
        # 中文：纯 attention 不知道词序，所以要把“第 t 个位置”的可学习向量加到
        # token embedding 上；同一个 batch 内每条样本使用相同的位置编号 0...T-1。
        
        token_embeddings = self.embeddings(x)  # [B, T, D]

        position_ids = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embeddings(position_ids)  # [T, D]
        hidden = token_embeddings + position_embeddings  # [B, T, D]

        for block in self.backbone:
            hidden = block(hidden)  # [B, T, D]

        # Final vocabulary projection: [B, T, D] -> [B, T, V].
        hidden = self.final_layer_norm(hidden)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad() # 装饰器，表示下面的 generate() 函数执行时不记录梯度。
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:
        # TODO, complete
        # ===== 以下为新增实现：贪心（argmax）生成 =====
        """Greedily append ``num_new_tokens`` next-token predictions.

        ``argmax(logits[:, -1, :])`` gives the most likely next token.  The
        full generated history is retained, but forward only sees the most
        recent context window when the sequence becomes long.
        """
        for _ in range(num_new_tokens):
            # 若历史超过 context_length，只保留最新窗口；B 不受限制。
            context = x[:, -self.config.context_length:]
            logits = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            x = torch.cat((x, next_token), dim=1)                      # [B, T + 1]
        return x


    def get_loss_on_batch(self, input_ids: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, ""]:
        # TODO, complete
        # ===== 以下为新增实现：next-token cross-entropy loss =====
        """Compute mean next-token cross entropy.

        For ids [x_0, ..., x_(T-1)], logits at t predict target x_(t+1):
        logits[:, :-1, :] has shape [B, T-1, V], while labels input_ids[:, 1:]
        have shape [B, T-1].  The loss is -mean(log_softmax(logits)[target]).
        """
        if input_ids.shape[1] < 2:
            raise ValueError("Need at least two tokens to compute next-token loss.")

        logits = self(input_ids)  # [B, T, V]
        # 中文例子：输入 ["我", "爱", "学习"] 时，位置 0 的 logits 应预测“爱”，
        # 位置 1 的 logits 应预测“学习”，因此 logits 与标签要分别去掉最后/第一个位置。
        prediction_logits = logits[:, :-1, :]  # [B, T - 1, V]
        target_tokens = input_ids[:, 1:]       # [B, T - 1]

        vocab_size = prediction_logits.shape[-1]
        return F.cross_entropy(
            prediction_logits.reshape(-1, vocab_size),
            target_tokens.reshape(-1),
        )


    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # Uncomment this if you are not logged in
    # huggingface_hub.login()
    
    model = Transformer.from_pretrained()
