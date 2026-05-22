1.不允许修改的内容: bm25 检索以及 eval相关逻辑,此外其他的 agent_loop.py agent.py env.py run_serial.py 允许大面积重构甚至删除

2.重构后的代码架构如下:
1.agent.py 
- 只负责 agent 类的定义，接受 system prompt,user prompt, tool list,max_turn等参数,注意所有 agent 都需要共享这个 agent 结构包括 eval ,subagent 等,不同 agent 的功能可以根据上下文和工具来完全决定，不要分开多个类来写，然后用一个 yaml 来控制不同 agent 的配置
- 需要定义chat和 chat_with_tool_retry 函数，chat 直接接受 user prompt 回答(请求的时候带上 agent 实例化时的 system prompt和 tool),chat with retry 调用 chat 函数但是在外层封装重试机制，注意重试机制需要在格式错误/没有调用 tool时使用提醒 prompt 引导模型重试，重试成功后重试部分的上下文应该移除避免上下文污染
- 需要定义 run,即 agent 的循环逻辑不再由外部决定而是由类内部自己处理,run 终止条件有两个: 
1.达到 max turn 
2.agent 调用了 self.end_tool (end_tool是 agent 初始化时从 init 传入的某个工具,agent 通过 end_tool 提交最终回答)
run 内部调用一个 condense 方法对上下文进行压缩,该 condense 不需要作为一个独立的 agent,但是需要预先给每个 agent 配置好 condense prompt, 然后通过 chat_with_tool_retry 直接调用 condense ,condense 请求提供的 tool 为 submit_condense,condense 之前 需要对agent 的 thinking block 和工具调用格式进行预处理避免 condense 模型被这些特殊格式影响
- 需要定义轨迹收集逻辑,每个 agent 实例独立收集自己 run 的轨迹,注意处理路径名称避免覆盖，轨迹的格式需要和当前代码保持一致

2.tool_docs.py 专门放置 tool 文档
3.tool_func.py 专门放置 tool 函数

4.整个系统有一个 main agent 作为入口 ，从该 agent的 run 函数开始,随后通过 agent 调用 tool 作为驱动,tool 里面可能包含调用别的 agent 的逻辑 , 然后 别的 agnet 将处理结果通过 tool 返回，即 agent间的通信完全通过 tool 实现

5.需要支持的 agent 类型和 tool 工具
**所有 agent 都需要通过 chat_with_tool_retry 来确保工具调用正常
- main agent 提供 search/submit answer/call_searchs_agent  执行的 workflow 为 main agent 先根据问题调用 search ,search 会返回一些候选样本，然后 main agent 根据 问题约束调用 sub agent 去验证候选集合不断缩小范围找到目标,end_tool 为 submit answer

- search 工具参考当前实现,提供参数控制是否使用 subagent 总结(这个参数放到 yaml agent 配置那里,作为 tool 的控制参数)，但是注意该 subagent 也应该通过 agent.py 里的类实例化,改 subagent 无需任何工具，但是需要通过 submit_summary 工具返回 submit_summary是该 agent 的 end_tool,

- call_subagent 参数为一个问题列表由 main agent 提问 [question1,question2,question3,...],所有 search agent async 异步执行，去掉所有并发数限制 后期改为从整个端口层面设置转发限制并发即可

- call_agent  发起的 search agent 提供 search,get_document工具, 以及 submit_answer工具,search agent 的 work flow 和 prompt 参考当前的 main agent,但是一些关于没有答案的强调可能需要去掉因为 main agent 问的问题可能是没有答案的

- verify agent 提供 search get_document工具

**整体要求** 
- 终端打印日志必须详细，格式参考当前的代码,层次分明，信息具体，要求能从终端日志完全了解当前的进展
- 所有 agent 的 system prompt 以下内容结尾,**IMPORTANT** using xxx tool is the **ONLY** way to submit your answer
- 所有 agent 上下文长度，输出长度,压缩 token 线在 yaml 里配置,是否 thinking 也在 yaml 里配置,取消原来代码的 think 重试,只保留工具重试,所有上下文长度统计量必须为 token 而不是 char 
- prompt 绝对禁止数据泄露, prompt 需要详细写出 agent 的 workflow,特殊 token 对应的符号绝对不能出现在 prompt 里面，比如think block
- 多余代码直接删除
- 需要 async 异步执行
- 每个 agent 用一个独立的 yaml 文件配置 初始化需要的所有参数