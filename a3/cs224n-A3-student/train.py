"""A simple training loop for our transformer model.

训练主线（每个 batch 重复一次）：
1. 取一小批 token 序列；2. 模型预测每个位置的下一个 token；
3. 用 cross-entropy 得到 loss；4. 反向传播得到梯度；5. 优化器更新参数。
"""
import warnings
warnings.filterwarnings("ignore")
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Optional
from jaxtyping import Int
from torch import Tensor
import torch
import matplotlib.pyplot as plt

from model_solution import Transformer, ModelConfig


# 训练和数据必须在同一设备上。优先使用 NVIDIA GPU，其次是 Mac 的 MPS；
# 若均不可用则退回 CPU（更慢，但逻辑完全相同）。
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using Mac MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


def get_chunked_tinystories(
    chunk_size: int,
) -> Int[Tensor, "num_chunks chunk_size"]:
    """下载 TinyStories，并整理成形状为 [num_chunks, chunk_size] 的 token 块。

    模型不能直接读字符串；GPT-2 tokenizer 先把文本转成词表中的整数 token id。
    每一行 chunk 是一条长度固定的训练序列，之后会被切成 batch。
    """

    # 使用 GPT-2 自己的 tokenizer，确保 token id 与模型 vocab_size=50257 对应。
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 从 Hugging Face 下载 TinyStories 的训练集。
    train_dataset = load_dataset("roneneldan/TinyStories")["train"]

    # 教学演示仅使用前 1%，否则下载和训练时间会比较长。
    train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.01)))

    # chunks 最终是二维 Python 列表；每个元素都是一个长度为 chunk_size 的 token 序列。
    chunks: List[List[int]] = []
    current_chunk: List[int] = []
    for row in tqdm(train_dataset, desc="Tokenizing dataset"):
        document: str = row["text"]
        # truncation 保证单篇故事至多取 chunk_size 个 token，避免单篇文本过长。
        tokens: List[int] = tokenizer(document, truncation=True, max_length=chunk_size).input_ids

        # 把多篇短故事连续放入当前序列；凑满 chunk_size 后保存一条训练样本。
        current_chunk.extend(tokens)
        if len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size])
            # 未装入本 chunk 的 token 留给下一条，避免无谓丢弃语料。
            current_chunk = current_chunk[chunk_size:]

    # 训练代码假设每条样本长度一致，故在这里做一次断言检查。
    assert all(len(chunk) == chunk_size for chunk in chunks)

    return torch.tensor(chunks, dtype=torch.long)


def plot_results(
    losses: List[float],
    grad_norms: List[float],
    save_path: str,
) -> None:
    """把训练过程中的 loss 与梯度范数保存为图片，便于观察是否稳定收敛。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel - Loss curve
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Right panel - Gradient norm
    ax2.plot(grad_norms)
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Grad Norm')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(
    learning_rate: float,
    gradient_clipping: Optional[float],
    model_config: ModelConfig,
    batch_size: int,
    max_steps: Optional[int] = None,
) -> None:
    """训练一个随机初始化的 Transformer。

    形状流：dataset [N, T] -> batch [B, T] -> logits [B, T, V] -> loss 标量。
    其中 N 是总训练块数，B 是 batch_size，T 是 context_length，V 是词表大小。

    """

    if gradient_clipping is None:
        # inf 表示不裁剪；仍会记录真实梯度范数，方便诊断梯度爆炸。
        gradient_clipping = float("inf")

    # 每条训练样本的 token 数和模型最大上下文窗口一致。
    chunk_size: int = model_config.context_length
    cached_dataset_path: str = f"./datasets/tinystories_10pct_chunk_size_{chunk_size}.pt"
    os.makedirs(os.path.dirname(cached_dataset_path), exist_ok=True)
    
    if os.path.exists(cached_dataset_path):
        # 缓存的是已经 tokenized 的 Tensor；二次运行不必重新下载、切分与 tokenizer 编码。
        dataset = torch.load(cached_dataset_path)
    else:
        dataset: Int[Tensor, "num_chunks chunk_size"] = get_chunked_tinystories(chunk_size)
        torch.save(dataset, cached_dataset_path)


    # 新模型的参数先按 Transformer._init_weights() 随机初始化，然后整体移动到 device。
    model = Transformer(model_config).to(device)

    # AdamW 根据每个参数的梯度更新参数。learning_rate 控制每一步更新的幅度。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    num_chunks: int = dataset.shape[0]

    losses: List[float] = []
    grad_norms: List[float] = []
    num_steps_completed: int = 0

    if max_steps is not None:
        tqdm_max_steps = min(max_steps, num_chunks // batch_size)
    else:
        tqdm_max_steps = num_chunks // batch_size

    # range 步长是 batch_size；第 i 次循环取 dataset[i : i + batch_size]。
    for i in tqdm(range(0, num_chunks, batch_size), desc="Training", total=tqdm_max_steps):

        if max_steps is not None and num_steps_completed >= max_steps:
            break

        if num_steps_completed % 10 == 0 and num_steps_completed > 0:
            # 每 10 个 batch 画一次图，观察 loss 是否下降、梯度是否异常大。
            plot_results(losses, grad_norms, save_path=f"./losses_and_grad_norms.png")

        # batch 是整数 token id，而不是 embedding： [B, T]。
        # .to(device) 必须与 model 所在设备一致，才可进行矩阵运算。
        batch: Int[Tensor, "batch_size chunk_size"] = dataset[i:i+batch_size].to(device)

        # PyTorch 默认会累积梯度；每个新 batch 前清零，避免把上一个 batch 的梯度混进来。
        optimizer.zero_grad()

        # 1) Forward：模型内部得到 logits，并用“预测下一个 token”的交叉熵得到一个标量 loss。
        loss = model.get_loss_on_batch(batch)

        # 2) Backward：链式法则计算 d(loss)/d(parameter)，结果写进每个 parameter.grad。
        loss.backward()

        # 3) 梯度裁剪：若总梯度范数超过阈值，就按比例缩小，防止更新一步过大。
        # clip_grad_norm_ 返回的是裁剪前范数；no_grad 表示该诊断步骤不进入计算图。
        with torch.no_grad():
            grad_norm: float = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping).item()
            grad_norms.append(grad_norm)

        # 4) Update：AdamW 使用 parameter.grad 更新所有可训练参数。
        optimizer.step()

        # .item() 把 GPU/CPU 上的 0 维 Tensor 转成普通 Python float，供画图使用。
        losses.append(loss.item())

        num_steps_completed += 1


    # 训练结束后保存最终曲线。
    plot_results(losses, grad_norms, save_path="./losses_and_grad_norms.png")
    print(f"Final loss after {num_steps_completed} steps: {losses[-1]:.4f}")



if __name__ == "__main__":

    # 一个很小的 GPT 配置，适合快速熟悉训练流程；不是完整 GPT-2 small。
    tiny_model_config = ModelConfig(
        d_model=33,
        n_heads=3,
        n_layers=3,
        context_length=512,
        vocab_size=50257,
    )

    train(
        learning_rate=1e-5,
        gradient_clipping=1,
        model_config = tiny_model_config,
        batch_size=16,
        max_steps=100,
    )
