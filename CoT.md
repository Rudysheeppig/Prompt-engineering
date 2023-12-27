# 自我一致性（CoT）
introduction about Prompt Engineering


Self-Consistency Improves Chain of Thought Reasoning in Language Models.、

实验假设：复杂推理问题中，从问题到唯一正确答案会存在许多不同的解法，即推理方式

核心方案：self-consistency 解码策略代替贪婪解码, 先采样多个不同的推理路径【重复请求多次】, 然后选择最一致的答案

实验结论：GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).

论文贡献
self-consistency解码策略假设复杂推理任务一般可以通过多个推理路径获得正确答案，从解码器中抽样生成多样化的推理路径集合，选择一致性最高的输出结果作为最终答案，降低了贪婪解码方式的单次采样的随机性
self-consistency不需要训练额外训练或者辅助模型，类似于在单个语言模型上工作的自集成方法
self-consistency 结合PaLM-540B 或 GPT-3，算术推理任务中都获得最新的sota水平，GSM8K(Cobbe 等人，2021 年+17.9% )()、SVAMP(Patel 等人，2021 年+11.0%)、AQuA(Ling 等人，2017 年+12.2%)，以及 StrategyQA 等常识性推理任务(Geva 等人，2021 年+6.4%)和 ARC 挑战(Clark 等人，2018 年+3.9%)
self-consistency 在抽样策略和提示缺陷场景下都具有很强的鲁棒性，在一般NLP任务中也能获得性能提升
核心方法
Step1: 思维链提示
Step2: 对语言模型进行多次采样, 生成多个推理路径
Step3: 对不同推理路径生成结果基于投票策略选择最一致的答案输出

The self-consistency method contains three steps
实验部分
1 评估数据

Arithmetic reasoning -> Math Word Problem Repository，AQUA-RAT，GSM8K，SVAMP
Commonsense reasoning -> CommonsenseQA, StrategyQA, AI2 Reasoning Challenge (ARC)
Symbolic Reasoning -> last letter concatenation, Coinflip
2 baseline

UL2 -> 20B, encoder-decoder, UL2-encoder-decoder-GitHub
GPT-3 -> 175B, decoder-only, code-davinci001 | code-davinci-002 https://openai.com/api/
LaMDA -> 137B, decoder-only
PaLM -> 540B, decoder-only
Few-shot prompt, arithmetic reasoning -> 8个 人工手写示例, commonsense reasoning task -> 4~7示例，从人工标注的推理链提示的训练集随机采样

3 采样策略

UL2-20B and LaMDA-137B， T = 0.5， top-k (k = 40) tokens
PaLM-540B we applied T = 0.7, k = 40
GPT-3，T = 0.7，without top-k
4 实验结果

Self-consistency 在算术推理, 常识推理，符号推理中都表现高于一般的Cot效果，在ood的数据上[Letter(4)、Coinflip(4)]仍然保持效果增益

Arithmetic reasoning accuracy by self-consistency

Commonsense and symbolic reasoning accuracy by self-consistency

Self-consistency (blue) significantly improves accuracy over CoT-prompting with greedy decoding (orange)
Self-consistency 降低了CoT在一般NLP任务中的效果影响, 甚至还带来了显著的提升

Compare Standard/CoT prompting with self-consistency on common NLP tasks (PaLM540B).
比Sample-and-Rank策略的效果也高出不少[sample-and-rank策略指先采样，然后对结果进行排序]

Self-consistency significantly outperforms sample-and-rank with the same # of samples.
Beam Search 产生的多样性较差, self-consistency基于beam-search方式采样效果反而变差，验证了多样性是影响self-consistency的因素

Compare self-consistency with beam search decoding on the UL2-20B model.
prompt集成方式测试, 也不如直接的随机采样的self-consistency效果

Self-consistency outperforms prompt-order and multiple-prompt ensembles on LaMDA137B
鲁棒性测试1: Self-Consistency在采样策略，参数以及模型规模都显著高于greedy decode
鲁棒性测试2: Self-Consistency 应用在有缺陷的prompt上的效果也能带来显著提升，高于正确的prompt效果
鲁棒性测试3: Self-Consistency对于方程式的推理路径 和 Zero-shot CoT 也可以带来直接的提升
其他发现: Self-Consistency的一致性率和模型准确率呈正相关, consistency-rate可以用来评估模型的不确定性

GSM8K accuracy. (Left) Self-consistency is robust to various sampling strategies

论文点评
Self-consistency 可以用来解决LLM输出的不稳定性，一般我们在测试case时候也会习惯性跑测多次来确认效果，取平均结果作为测试结论，这种评测方式就是类似Self-consistency

SC策略符合人类解决问题的直觉，做一道数学题，难免会存在多种不同的解决方案，然后基于投票策略选择最终结果输出，但是SC只关注了推理后最终的答案，没有关注推理路径本身，之前有研究表明过，LLM有时会给出正确的答案，看似合理实则错误的推理过程，这个论文没有讨论，应该看看推理链正确 + 答案正确的准确率，答案可能很容易评判，推理过程有没有可以衡量的方法呢？有什么准则指令可以约束推理过程，保证正确性呢？如果可以做到，应该也就近似解决了模型幻觉输出的问题....
