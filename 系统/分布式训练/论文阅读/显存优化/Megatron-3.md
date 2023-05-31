# Reducing Activation Recomputation in Large Transformer Models

这篇论文来自MLSys 2023，英伟达Megatron-LM团队，实际2022年早些时候就已经放在arxiv上了。主要目标是减少流水线并行中每个microbatch的中间激活值的内存。

## Introduction

![image-20230529222851955](images/image-20230529222851955.png)

如图，A100的显存是80G，显然是放不下的。

之前的做法一般是只保留切分的partition边缘的激活值，其余的全部丢弃，在反向传播的时候重新计算这些激活值，作者团队在他们的训练中测得这样做有额外的30%-40%开销。这篇论文提出一种新的方法，节省内存，并且 have no, or very low, impact on compute efficiency。
## Transformer Architecture

<img src="images/image-20230529225139896.png" alt="image-20230529225139896" style="zoom: 67%;" />

<img src="images/image-20230529225452248.png" alt="image-20230529225452248" style="zoom: 67%;" />

