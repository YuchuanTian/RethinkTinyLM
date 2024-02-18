# Rethinking Optimization and Architecture for Tiny Language Models

<p align="left">
<a href="https://arxiv.org/abs/2402.02791" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2402.02791-b31b1b.svg?style=flat" /></a>
</p>

This is the official implementation of [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/pdf/2402.02791.pdf), an empirical investigation about how to construct powerful language models.

Four strategies are proposed to improve performance:

- 🎯 Compact Tokenizer: efficient coverage of corpus;
- 🔍 Architecture Tweak: better depth and width tradeoffs;
- 🎁 Parameter Inheritance: powerful knowledge from larger LLMs;
- 🔥 Multiple-Round Training: memory reinforcement of tiny models.

<p align="center">
<img src="fig/improve.png" width="700">
</p>
Based on the observations above, PanGu-π-1B Pro and PanGu-π-1.5B Pro are trained on 1.6T multilingual corpora. Model configurations are shown as follows:

<p align="center">
<img src="fig/configure.png" width="500">
</p>

## Benchmark Results

<p align="center">
<img src="fig/results.png" width="900">
</p>

## Training

This repository is modified from the [InternEvo](https://github.com/InternLM/InternEvo) training framework.

Here are the steps to organize the codes:

1. Clone the [InternEvo](https://github.com/InternLM/InternEvo) repository and configure the runtime environment.
2. Copy the configuration files `configs/LLM1B.py` to the `InternEvo/configs/` directory.
3. Copy the start script `start_finetune.py` to the `InternEvo` root directory.

You can follow the guide of InternEvo to pretrain data and  train models (https://github.com/InternLM/InternEvo/blob/develop/doc/en/usage.md).  

The model's depth, width, and expanding rate can by easily adjusted in the config.

#### Compact Tokenizer

The compact tokenizer is constructed by removing low-frequency vocabularies. To prune tokenizer, you can follow these steps:

1. Counting the frequency of tokens cached by the original big tokenizer.
2. Firstly add the special tokens,  and then add the tokens with the highest word frequency to the new tokenizer.

#### Parameter Inheritance

To pretrain by inheriting parameter from a large model, you can use the following command:

```shell
python start_finetune.py --config ./configs/LLM1B.py
```

Note that `MODEL_ONLY_FOLDER` is the model's checkpoint pruned from a large model.

If you want to train from scratch, you need the set `load_given_ckpt=False` in the config.

#### Multiple-Round Training

To extract a certain proportion of challenging examples from the last epoch, you can utilize the following steps:

1. Compute the batch-wise loss $L=\{l_1,l_2,\cdots,l_N\}$ using the pre-trained frozen model from the previous epoch, where $N$ represents the total number of batches. For instance, a dataset containing 150B tokens might yield approximately 75000 batches when utilizing a batch size of 2M.
2. Calculate the sampling probability $p_i = \exp(l_i) \bigg/ {\sum \limits_{j=1}^N \exp(l_j)}$. 
3. Sample $N_0$ batches out of $N$ according to the sampling probability $\boldsymbol{p}$, i.e., `filtered = torch.multinomial(p, N_0, replacement=False)`
4. Concatenate all the filtered batches to create the training dataset for the subsequent epoch.

## Inference

Convert the model weight to Hugging Face format using the script `tools/transformers/convert2hf.py`.

```shell
python tools/transformers/convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer tokenizer_path/
```

Then the model can be inferred with Hugging Face.

## Acknowledgements

- [InternLM/InternEvo](https://github.com/InternLM/InternEvo)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [google/sentencepiece](https://github.com/google/sentencepiece)
- [open-compass/opencompass](https://github.com/open-compass/opencompass)
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

## Citation

```
@article{tang2024rethinking,
  title={Rethinking Optimization and Architecture for Tiny Language Models},
  author={Tang, Yehui and Liu, Fangcheng and Ni, Yunsheng and Tian, Yuchuan and Bai, Zheyuan and Hu, Yi-Qi and Liu, Sichao and Jui, Shangling and Han, Kai and Wang, Yunhe},
  journal={arXiv preprint arXiv:2402.02791},
  year={2024}
}

@article{wang2023pangu,
  title={PanGu-$$\backslash$pi $: Enhancing Language Model Architectures via Nonlinearity Compensation},
  author={Wang, Yunhe and Chen, Hanting and Tang, Yehui and Guo, Tianyu and Han, Kai and Nie, Ying and Wang, Xutao and Hu, Hailin and Bai, Zheyuan and Wang, Yun and others},
  journal={arXiv preprint arXiv:2312.17276},
  year={2023}
}
```
