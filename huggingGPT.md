参考：
https://zhuanlan.zhihu.com/p/619896296
https://blog.csdn.net/youcans/article/details/129927345


**相关信息
原文链接 https://arxiv.org/pdf/2303.17580.pdf

## 摘要
解决不同领域和模态的AI任务是迈向人工智能的关键一步。虽然现在有大量的AI模型可以用于解决不同的领域和模态的问题，但是它们不能解决复杂的AI问题。由于大模型(LLM)在语言理解、生成、交互和推理上展现出很强的能力，所以作者认为LLM可以充当一个控制器的作用来管理现有的AI模型以解决复杂的AI任务，并且语言可以成为一个通用的接口来启动AI处理这些任务。基于这个想法，作者提出HuggingGPT，一个框架用于连接不同的AI模型来解决AI任务。

具体的步骤是：

任务规划：使用ChatGPT来获取用户请求
模型选择：根据Hugging Face中的函数描述选择模型，并用选中的模型执行AI任务
任务执行：使用第2步选择的模型执行的任务，总结成回答返回给ChatGPT
回答生成：使用ChatGPT融合所有模型的推理，生成回答返回给用户
通过ChatGPT的强语言能力和Hugging Face丰富的模型库，HuggingGPT可以解决大部分复杂的AI任务，为走向真正的人工智能奠定基石。

### 介绍
大语言模型(LLM)，比如ChatGPT由于其强大的能力吸引了学术界和工业界的广泛关注。LLM的出现也催生了很多研究课题（比如in-context learning, instruction learning, chain-of-thought prompting）。

尽管表现出色，LLM还是存在一些缺陷：

受限于输入和输出格式，当前的LLM缺少处理复杂信息，比如视觉、语音的能力。
现实任务通常由多个子任务构成，需要不同模型的调度和合作。
在一些任务上，虽然LLM能够做到zero-shot或者few-shot，但是能力还是不如专家。
针对以上问题，作者认为：可以将ChatGPT作为不同模型之间的桥梁，为不同的任务调度不同的模型。具体而言，就是将每个模型用语言的形式表示他们的功能，即把模型描述融入到prompts中。这里就引发了另一个问题：哪里去找高质量的模型描述？作者团队发现一些ML社区有高质量的模型描述，比如GitHub，HuggingFace，Azure等等。

因此，作者将ChatGPT和HuggingFace连接在一起，提出HuggingGPT，HuggingGPT的使用分为四步，在摘要中已经给出。






基于此，HuggingGPT融合了HuggingFce中成百上千的模型和GPT，可以解决24种任务，包括文本分类、对象检测、语义分割、图像生成、问答、文本语音转换和文本视频转换。

#### 主要贡献
提出inter-model cooperation protocol（模型间合作协议）来充分利用LLM和专家模型。LLM作为大脑进行规划和做决策，小模型作为执行者完成特定任务，提出通用AI的新设计。
打造HuggingGPT解决通用AI任务，给用户提供多模态和可靠的回答服务。
做了大量实验
相关工作
### LLM
近几年，ChatGPT对NLP领域进行了重大变革。

一些LLM中的领域被重新激活：

chain-of-thought prompting (CoT) ：通过设置许多情景来prompt LM生成解决问题的过程，比如添加简单的prompt，“Let's think step by step”，也可以得到性能的提升。
Instruction tuning：收集和转变传统的NLP任务数据集为instructions，并且在instruction datasets上微调大模型来提高在未知任务上的生成能力。通过Instruction Tuning的方法，Flan T5和Flan-UL2仅仅使用100B的参数，就超过了有650B参数的PaLM。此外，InstructGPT和ChatGPT都用了强化学习的方法，导致语言理解和生成能力的提高。
### LLM能力的进步
为了让LLM的能力超于文本生成，现有的研究主要分为两个流派：

统一多模态模型：比如BLIP-2，Kosmos-1
与额外的工具融合：比如Toolformer
与上述工作相比，HuggingGPT有如下优势：

使用LLM作为将请求路由到专家模型的接口，充分利用LLM的语言理解能力
HuggingGPT不限于视觉任务，能处理任何模态和领域的任务
HuggingGPT可以快速便捷地添加新的"专家"
## HuggingGPT
HuggingGPT是以ChatGPT为控制器，专家模型为执行者的一个合作系统，其工作流为：任务规划、模型选择、任务执行和回答生成。下面是每个步骤的详细解读。

### 任务规划
HuggingGPT在接收到request后，将其解构为一个结构化任务的序列，并且需要识别这些任务之间的依赖关系和执行顺序。为了让LM做高效的任务规划，HuggingGPT在设计中使用specification-based instruction和demonstration-based parsing。

Specification-based Instruction
Task specification提供一个统一的模板来允许LLM进行任务解析。HuggingGPT提供四个槽，分别是task type，task ID，task dependencies和task arguments：






task id：任务的唯一标识符，用于依赖任务和生成资源的引用
task type：不同的任务类型，可以在上表中看到
task dependencies：该任务执行前的任务，该任务只会在所有的预任务(pre-requisite)执行后再执行
task arguments：任务执行需要的arguments，在上表中可以看到
Demonstration-based Parsing
每个demonstration是一组在任务规划上的输入和输出，输入是用户的请求，输出是期望的任务序列。进一步而言，这些demonstration包含解析任务之间的依赖，有效帮助HuggingGPT理解任务之间的逻辑关系，并且决定执行顺序和资源依赖。

### 模型选择
在解析出任务列表后，HuggingGPT接下来需要将任务和模型进行匹配。为了完成这个任务，首先需要获取专家模型的描述，然后使用in-context task-model assignment来动态选择模型。

Model Description：没什么好说的，就是HuggingFace社区中，模型发布者提供的模型描述。
In-Context Task-Model Assignment：通过在prompt种加入用户query和解析后的任务，HuggingGPT能够选择最适合这个任务的模型。但是因为字数限制，不可能把所有的模型信息添加在prompt种，所以作者先基于task type过滤模型，并且对剩下的模型根据下载量排序，然后选择top-K个模型作为HuggingGPT的候选模型。
### 任务执行
模型一旦选定，下一个动作就是执行任务。为了加速和计算的稳定性，HuggingGPT在混合推理终端上运行这些模型。通过将这些任务参数作为输入，模型计算推理结果，然后把它们发回给LLM。为了进一步提高推理效率，没有资源以来的模型将被并行。这意味着多个满足prerequisite dependencies的模型可以同时启动。

### Hybrid Endpoint
一个理想的场景是我们只使用Hugging Face的推理终端，但是在很多情况下，我们不得不部署本地的推理终端，比如在一些特定模型的推理终端不存在的情况，推理将是耗时的，网络权限是受限的。为了使得系统平稳且有效，HuggingGPT拉取并且运行一些常用且耗时的本地模型。本地的推理终端涵盖较少的模型但是快速，所以比Hugging Face上的推理终端有更高的优先级。

### Resource Dependency
虽然HuggingGPT能够在任务规划阶段安排任务顺序，但是在任务执行阶段有效管理资源依赖还是具有挑战性的。原因是HuggingGPT不能在任务规划阶段指定未来发生的任务。为了解决这个问题，我们使用一个唯一的符号<resource>来管理资源依赖。具体而言，HuggingGPT用<resource>-task_id来标识预备任务产生的资源。在任务规划阶段，如果有任务依赖于任务task_id，HuggingGPT就会把这个符号放在任务参数的相关资源子域中。在任务执行阶段，HuggingGPT动态地用这个预备任务产生的资源替代这个symbol。这个方法使得HuggingGPT能高效处理任务执行过程中的资源依赖性。

### 回答生成
在任务执行结束后，HuggingGPT进行回答生成阶段。在这个阶段，HuggingGPT融合过去三个阶段的答案到一个精简的summary中，包含规划的任务列表、任务选中的模型和模型的推理结果。

其中最重要的是推理结果，以结构化的格式发送给LLM，LLM再生成response返回给user requests。

实验
设置
LLM

gpt-3.5-turbo和text-davinci-003，设置temperature=0，logit_bias=0.2，prompt设计如下图所示。






接着是一些可视化的结果，这里不做阐述。

## 不足
效率
在每个回合中，HuggingGPT需要至少一次和LLM的交互。这些交互很大程度上增加了response的延迟，并且导致用户体验的下降。

最大文本限制
受限于LLM能接受的最大文本数量，HuggingGPT也面临着最大文本长度限制的问题。作者通过只追踪任务规划阶段的对话文本来减少文本。

稳定性
LM也许会不听指令
专家模型也许会不受控**