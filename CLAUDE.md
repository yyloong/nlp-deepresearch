# Deep Research Agent 开发规范

## Prompt 设计原则

### 1. 工具调用强制规则
所有要求 agent 通过 tool 来回答的 prompt，**必须**：
- 明确声明 tool call 是**唯一**的响应方式（"You MUST call X tool"、"plain text is ignored"）
- 提供**重试机制**：tool 调用失败时 inject nudge message 后重试（最多 2 次）
- 重试 nudge 要**具体**指出缺少什么参数（"Missing field: key_thoughts"），不要泛泛说 "try again"

### 2. 绝对禁止数据泄露
prompt 中**绝对不能**出现：
- BrowseComp 数据集相关实体名、线索词、答案片段
- 任何可能出现在真实问题中的领域词汇（如 author/book/publisher/company/married/botanist/settlements 等）
- 示例必须使用**完全虚构**的场景（如 wizard/Elf King/Dragon Taming Cup/recipe/ingredient）

### 3. 禁止内部标记泄露
以下标记**绝对不能**出现在 model-facing prompt 中：
- `<think>`, `</think>` — 模型的内部思考格式
- `[reasoning]`, `[/reasoning]` — 压缩时的内部替换标记
- `[PROGRESS SUMMARY]` — 旧的压缩格式标记

### 4. Condense prompt 规范
- 让 condense 模型使用 tool（`submit_condensed_summary`）输出
- condense 模型**不禁用 thinking**（让它思考才能压缩得好）
- 用 `tool_summary` 参数让模型自己总结工具调用，不做机械截断
- condense 输出要**从模型视角**提供有用信息（facts found, insights gained），不只是"做了什么"
- 提供重试：参数缺失时 nudge，最多 2 次

### 5. 主 Agent 搜索 prompt 规范
- few-shot example 展示完整搜索链（用虚构场景）
- 第一搜必须选最稀有/独特的词，不要泛泛描述
- 2-3 词 per query，不超过 5 词
- 链式推进：entity from result + next clue
- 结尾规则（近因效应）

### 6. Verify Agent prompt 规范
- suggestions 不给具体搜索词（避免数据泄露）
- 引导方向：证据不足→查某方面，约束不符→换角度
- Stage 1 用 PASS/SURRENDER 分类，不是 ANSWER
- keywords: "not found", "not mentioned", "not available", "no evidence" 等是 SURRENDER

### 7. Sub-Agent prompt 规范
- 读文档提取相关信息，用 `submit_information` 提交
- 提供重试

## 轨迹分析流程

每次测试后必须按以下流程分析：

1. **根据答案推导标准搜索轨迹**：用 gold answer 反推需要搜索的关键词和文档，用 `BrowseCompBM25Searcher` 手动验证能否检索到目标文档
2. **对比模型实际轨迹**：提取模型每一步的 search/get_document/submit_answer，与标准轨迹对比
3. **定位偏差点**：找出模型从哪一步开始偏离标准轨迹，分析原因（query 太长/太泛/没链式推进/子 agent 信息丢失等）
4.**分析模型的 thinking block**,这是体现模型为什么出现某些意料之外行为的**最好方法**，通过 thinking block 可以分析 prompt 的改进方向
5. **针对性修改 prompt**：只改偏差点对应的 prompt 部分，不要大改
6. **多样性测试**：每次修改后用**不同样本**测试，不要重复同一条
7. 检查子 agent 提取质量：对比文档原文和子 agent 输出，看是否遗漏关键信息

## 代码修改后必须检查

1. `python -c "import py_compile; py_compile.compile(...)"` 编译检查
2. 运行至少 1 个 query 测试
3. 检查轨迹文件 `trajectories/<qid>.json` 格式正确、无重复消息

## 日志规范

- **任何工具执行异常必须打印详细错误信息**（包括 traceback），不能 silent fail
- 子 agent 的输入（prompt + document）和输出（提取结果）必须打印，方便对比真正需要的信息
- condense 前后的 token 数、消息数必须打印
- 不要依赖反复跑代码来看 bug，日志要详尽到能直接定位问题
