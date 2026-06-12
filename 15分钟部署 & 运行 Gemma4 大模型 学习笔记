## 📋 目录
- [一、任务概述](#一任务概述)
- [二、前置环境校验](#二前置环境校验)
- [三、Gemma4 模型完整部署流程](#三gemma4-模型完整部署流程)
  - [步骤1：切换国内pip镜像源](#步骤1切换国内pip镜像源)
  - [步骤2：安装 ModelScope 魔搭社区](#步骤2安装-modelscope-魔搭社区)
  - [步骤3：下载 Gemma4 模型](#步骤3下载-gemma4-模型)
  - [步骤4：升级安装 vLLM 推理框架](#步骤4升级安装-vllm-推理框架)
  - [步骤5：启动 vLLM 模型服务](#步骤5启动-vllm-模型服务)
  - [步骤6：新建终端对话测试](#步骤6新建终端对话测试)
  - [步骤7：关闭服务释放资源](#步骤7关闭服务释放资源)
- [四、常见问题排查](#四常见问题排查)
- [五、核心概念科普](#五核心概念科普)
- [六、学习心得](#六学习心得)

---

## 一、任务概述
本次任务基于 **AMD ROCm 云平台**，从零完成 Google 开源大模型 **Gemma4-E4B-it** 的部署、下载、推理服务搭建与对话测试。全程使用命令行操作，借助 `ModelScope` 国内镜像加速下载、`vLLM` 高性能推理框架运行模型，最终验证模型可正常交互，为后续模型微调做准备。

官方环境入口：[AMD 平台登录](https://developer.amd.com.cn/login?source=ifF119ybS)

---

## 二、前置环境校验
部署模型前必须校验 **AMD GPU、ROCm、PyTorch** 环境可用性，确保硬件与驱动正常。

### 1. 查看 AMD GPU 硬件信息
1. 在 Jupyter Lab 界面点击 `+` → 选择 `Terminal` 打开终端
2. 执行硬件检测命令：
```bash
amd-smi
```
**正常输出说明**：
- 展示 GPU 型号、显存占用、显卡温度、功耗、进程等信息
- 示例关键字段：`AMD Radeon Graphics`、显存 `49136 MB`、温度 `32~33℃`
- 无报错、GPU 利用率正常 → **GPU 硬件可用**

### 2. 校验 PyTorch & ROCm 兼容性
执行以下命令检测深度学习环境：
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```
✅ **环境合格判定标准**
1. `ROCm available: True`（ROCm 驱动正常启用）
2. `Device: AMD Radeon Graphics`（成功识别 AMD 显卡）
3. PyTorch 版本正常（示例版本：`2.9.1+gitff65f5b`）

> 💡 提示：以上两项检测全部通过，才可进入后续模型部署环节。

---

## 三、Gemma4 模型完整部署流程
### 步骤1：切换国内pip镜像源
为解决海外依赖下载慢、超时问题，将 pip 切换为**腾讯云镜像**：
```bash
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/
```
运行成功标识：终端输出 `Writing to /root/.config/pip/pip.conf`。

### 步骤2：安装 ModelScope 魔搭社区
`ModelScope`（阿里达摩院）是国内主流 AI 模型社区，用于高速下载开源大模型：
```bash
pip install modelscope
```
运行成功标识：终端输出 `Successfully installed modelscope-1.37.1`。

### 步骤3：下载 Gemma4 模型
使用 `modelscope` 拉取 `google/gemma-4-E4B-it` 模型，指定本地存储路径 `./models`：
```bash
modelscope download --model google/gemma-4-E4B-it --cache_dir "./models"
```
#### ⚠️ 重要注意事项
1. 模型总大小约 **15G**，预计耗时 8 分钟左右，耐心等待
2. 进度条卡住、未显示 100% 属于正常现象，**不要强制终止**
3. 下载成功标识：终端出现 `Successfully Downloaded from model google/gemma-4-E4B-it`

#### 校验模型文件完整性
下载完成后，查看模型目录文件：
```bash
ls -lh ./models/google/gemma-4-E4B-it/
```
目录存在 `model.safetensors`（核心模型权重文件）即代表文件完整。

### 步骤4：升级安装 vLLM 推理框架
`vLLM` 是高性能大模型推理框架，AMD 环境需特殊配置安装，先卸载冲突依赖再重装：
```bash
# 卸载原有 torchvision
uv pip uninstall torchvision

# 重装 vLLM、torchvision（搭配阿里云镜像 + ROCm 专属源）
uv pip install vllm torchvision \
  --no-cache \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://wheels.vllm.ai/rocm/ \
  -U
```
等待约 2 分钟，依赖全部安装完成即可。

### 步骤5：启动 vLLM 模型服务
启动 Gemma4 在线推理服务，对外提供接口：
```bash
vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it
```
✅ 服务启动成功标识：终端输出 `Application startup complete.`

> 🚫 关键提醒：
> 该终端会被服务独占，**禁止关闭窗口、禁止按下 `Ctrl+C`**，否则服务直接终止。

### 步骤6：新建终端对话测试
原终端运行服务，需**新建 Terminal** 作为交互终端：
1. Jupyter Lab 再次点击 `+` → 打开新 `Terminal`
2. 执行对话连接命令，对接本地 8000 端口服务：
```bash
vllm chat --url http://localhost:8000/v1 --model gemma-4-E4B-it
```
3. 交互测试：输入对话内容，例如：
```
你是谁，你能做什么
```
模型正常返回文本回答 → **部署&推理全部成功**。

### 步骤7：关闭服务释放资源
后续需进行模型微调，需释放 GPU 显存与进程，分两步关闭：
1. **交互终端**：按下 `Ctrl + C` 退出聊天界面
2. **服务终端**：按下 `Ctrl + C` 终止 vLLM 服务
3. 关闭成功标识：终端输出 `Shutting down`

---

## 四、常见问题排查
整理部署过程高频报错与解决方案，按场景分类：

| 问题现象 | 解决方案 |
|---------|---------|
| `vllm serve` 启动速度极慢 | 首次启动会加载模型、编译内核，等待 3~5 分钟，日志持续输出则无需中断 |
| 提示 **显存不足 OOM** | 限制模型上下文长度，降低显存占用：<br>`vllm serve ./models/google/gemma-4-E4B-it/ --served-model-name gemma-4-E4B-it --max-model-len 8192`<br>仍不足则改为 `--max-model-len 4096` |
| `modelscope download` 命令不存在 | 1. 检查安装：`pip show modelscope`<br>2. 重装：`pip install -U modelscope` |
| `vllm chat` 连接失败 | 1. 检查原终端 vLLM 服务是否启动完成（必须出现 `Application startup complete`）<br>2. 确认端口 `8000` 未被占用 |

---

## 五、核心概念科普
结合本次实操，梳理大模型领域高频基础概念：

### 1. 大模型底层逻辑
传统软件依靠人工编写固定规则；**大语言模型**通过学习海量文本，自主总结语言规律，核心逻辑为：**根据上文预测下一个字符**，不断拼接生成完整句子、段落，从而实现对话、写作、推理等能力。

### 2. Gemma4 模型介绍
- 出品方：Google DeepMind，开源大模型家族，与 Gemini 3 同源技术
- 开源协议：Apache 2.0，**免费下载、免费商用**
- 版本划分：E2B、E4B、26B、31B，本次使用 **E4B（40 亿参数）**，单张显卡即可运行
- 能力：支持多步推理、代码生成、多模态、140+ 语言，综合性能强劲

### 3. 实操工具名词解释
1. **参数/权重**：模型的核心本体（`model.safetensors`），决定模型智能程度，单位 `B`（十亿参数）
2. **推理（Inference）**：使用已训练好的模型进行对话、生成内容（本次聊天环节）
3. **部署（Deploy）**：将本地模型搭建为在线服务，供外部调用
4. **ModelScope**：国内 AI 模型仓库，解决海外模型下载慢问题
5. **vLLM**：高性能推理框架，优化运算效率，提升模型响应速度

### 4. 整体流程链路
`模型文件（静态权重）` → `ModelScope 下载` → `vLLM 加载启动服务` → `终端远程对话（推理）`

---

## 六、学习心得
### 1. 实操总结
本次完整走完了开源大模型 **环境校验 → 依赖安装 → 模型下载 → 服务部署 → 交互测试** 的全流程，对 AMD ROCm 生态、大模型部署链路有了直观认知。整个流程以命令行为主，步骤标准化，只要严格按照指令执行，新手也可在 15 分钟内完成部署。

### 2. 踩坑反思
1. 模型下载阶段容易因进度条卡住误判为失败，需耐心等待，以官方成功提示为准；
2. vLLM 服务终端不可随意中断，否则需要重新启动服务，浪费时间；
3. 国内环境务必优先切换镜像源，能大幅提升依赖与模型下载速度。

### 3. 学习收获
1. 掌握了 AMD 云环境下 GPU 检测、ROCm 环境校验的基础命令；
2. 理解了 `ModelScope`、`vLLM` 两大工具的定位与使用场景；
3. 厘清了「模型、权重、推理、部署」等行业基础概念，打破了对大模型的技术壁垒认知；
4. 为后续 **Gemma4 模型微调** 任务铺垫了环境与理论基础。

### 4. 后续规划
熟悉 Gemma4 基础部署后，下一步将学习模型微调技术，基于自有数据对模型进行个性化优化，探索开源大模型的二次开发能力。
