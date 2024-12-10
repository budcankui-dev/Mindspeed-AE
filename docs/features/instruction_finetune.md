# MindSpeed-LLM 单样本指令微调

## 指令微调简介

指令微调（Instruction Fine-Tuning）是一种使预训练模型能够理解和执行自然语言指令的微调方法，广泛应用于大规模语言模型。通过在多任务、多样化的数据集上进行微调，指令微调使得模型在应对各种任务时更加灵活并更具泛化能力。

## 指令微调的原理

指令微调的核心思想是通过多任务和多样化指令数据集，训练模型以理解和执行数据集中不同任务的指令。具体来说，指令微调首先收集多个不同任务的数据集，并将每个任务转换为指令形式的输入，帮助大模型在多样化任务上提升泛化能力。该过程不仅包括直接的监督训练，还可结合人类反馈进行进一步优化。指令微调的基本原理如下：

![指令微调原理](../../sources/images/instruction_finetune/General_pipline_of_instruction_tuning.png)

**MindSpeed-LLM支持的指令微调，方案与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>统一，在微调效果保持一致的前提下，MindSpeed-LLM可以表现出优异性能**

## 【与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>微调Loss对比】

<table>
<tr>
<th rowspan="1">Llama2-7b模型与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>微调5个epoch后的loss对比图
<p align="center"> <img src="../../sources/images/instruction_finetune/tune_llama2_7b_MindSpeed-LLM_DeepSpeed_compare.png" height="270px" width="500px"> </p>
</th>
<th>
<th rowspan="1">Qwen-7b模型与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>微调4个epoch后的loss对比图
<p align="center"> <img src="../../sources/images/instruction_finetune/tune_qwen_7b_MindSpeed-LLM_DeepSpeed_compare.png" height="270px" width="500px"> </p>
</tr>
</table>



## 【Llama2-7b微调后对话效果对比】

<table>
    <tr>
        <th rowspan="1">Question</th>
        <th rowspan="1">MindSpeed-LLM</th>
        <th colspan="1"><a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a></th>
    </tr>
    <tr>
        <th>What is the product of 6 and 2?</th>
        <th>The product of 6 and 2 is 12.</th>
        <th>The product of 6 and 2 is 12.</th>
    </tr>
    <tr>
        <th>Look up the boiling point of water.</th>
        <th>The boiling point of water is 100°C (212°F)..</th>
        <th>The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit).</th>
    </tr>
    <tr>
        <th>Generate a poem that expresses joy.</th>
        <th>I open my arms and accept life's challenge, 


Embracing all of life's uncertainty. 

A future of promise awaits,

One that I can find in my fate. 

A journey full of hope and discovery,

A journey that will never end in disappointment. 

For I have found the strength to keep going,

Researching and learning, ever knowing. 

The future is mine, and mine alone,

A future that I have to own,

And I will never be alone,

For I will have the courage to face all of my fears,

And the joy of life will be mine forever.</th>
        <th>I am so happy, so thankful and blessed,

To have a heart that's full of cheer,

My soul is on fire, my spirit runs free,

My love is warm, my courage is high,

My world is bright and my outlook is clear,

My heart is light, my spirit runs free.

I take each day with a positive outlook,

My energy is so high, I can't be beat,

My joy is real, my emotions run deep,

My soul is full and my heart does soep.

I am thankful for everything I have,

My life is precious and my values ​​true,

My hope is high and my spirit runs free,

My soul is full and my heart does soep.</th>
    </tr>
</table>

## 指令微调示例

在`1x8`的集群配置下，使用`Atlas 900 A2 PODc`进行全参数微调。**以LLaMA2-7B模型在TP8PP1切分后生成的权重作为输入，进行指令微调示例。**

`Alpaca`数据预处理部分详见[**Alpaca风格数据的说明文档**](alpaca_dataset.md)。

`ShareGPT`数据预处理部分详见[**ShareGPT风格数据的说明文档**](sharegpt_dataset.md)。

**接下来将以Alpaca数据集作为输入，进行全参数微调示例。**

### 启动脚本

使用LLaMA2-7B模型目录下的<a href="../../examples/legacy/llama2/tune_llama2_7b_full_ptd.sh">微调脚本</a>。

#### 填写相关路径

`DATA_PATH`：指定数据预处理后的保存路径。

`TOKENIZER_MODEL`：指定模型的分词器路径（例如`tokenizer.model`）。

`CKPT_LOAD_DIR`：指向权重转换后保存的路径。

`CKPT_SAVE_DIR`：指向用户指定的微调后权重保存路径（例如指定保存到`ckpt`文件夹下）。

因此，根据之前的示例，路径应填写如下：
```bash
DATA_PATH="./finetune_dataset/alpaca"
CKPT_SAVE_DIR="./ckpt/"
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/"
CKPT_LOAD_DIR="./model_weights/llama-2-7b-legacy/"
```
#### 指令微调相关参数

【--is-instruction-dataset】

用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../mindspeed_llm/tasks/preprocess/templates.py)文件内查看。

【--variable-seq-lengths】

支持以动态的序列长度进行微调，默认padding到`8`的整数倍，可以通过`--pad-to-multiple-of`参数来修改padding的倍数。

#### 运行脚本

```bash
bash examples/legacy/llama2/tune_llama2_7b_full_ptd.sh
```

## 参考文献

[Zhang, S., Dong, L., Li, X., Zhang, S., Sun, X., Wang, S., Li, J., Hu, R., Zhang, T., Wu, F., & Wang, G. (2023). *Instruction Tuning for Large Language Models: A Survey*.](https://arxiv.org/pdf/2308.10792v5)
