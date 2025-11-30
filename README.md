# LM Lens (LM透镜)

一个用于可视化大语言模型内部工作机制和模型修复过程的交互式工具。

## ✨ 主要功能

### 核心功能
* 选择模型，输入提示，运行推理
* 浏览贡献图
    * 选择token来构建图
    * 调整贡献阈值
* 选择任何block后任何token的表示
* 查看表示到输出词汇表的投影，查看哪些tokens被前一个block提升/抑制
* 可点击的元素：
  * 边：显示注意力头的详细信息
  * 头：当选择边时，可以看到这个头在提升/抑制什么
  * FFN块：图上的小方块
  * 神经元：当选择FFN块时

### 🆕 模型修复可视化功能
* **多种修复方法**：
  - `mint`: MINT baseline方法，使用神经元编辑和正交投影
  - `me-sgd`: 基于SGD的迭代修复
  - `me-iter`: 迭代修复，使用语义引导（推荐）
  - `me-batch`: 批量修复模式
* **修复过程可视化**：
  - 修复前后对比
  - Token级别准确率指标
  - 编辑的神经元信息（MINT方法）
  - 修复统计信息

## 🚀 部署到 Streamlit Cloud

### 详细部署步骤（Step-by-Step）

#### 步骤 1: 准备 GitHub 仓库

1. **初始化 Git 仓库**（如果还没有）
   ```bash
   git init
   git add .
   git commit -m "Initial commit: LM Lens project"
   ```

2. **在 GitHub 上创建新仓库**
   - 访问 https://github.com/new
   - 填写仓库名称（如 `lm-lens`）
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"
   - 点击 "Create repository"

3. **连接本地仓库到 GitHub**
   ```bash
   git remote add origin https://github.com/你的用户名/lm-lens.git
   git branch -M main
   git push -u origin main
   ```

#### 步骤 2: 部署到 Streamlit Cloud

1. **访问 Streamlit Cloud**
   - 打开浏览器，访问 https://share.streamlit.io
   - 点击右上角的 "Sign in" 按钮

2. **使用 GitHub 登录**
   - 点击 "Continue with GitHub"
   - 授权 Streamlit 访问你的 GitHub 账户
   - 如果提示，选择允许访问仓库的权限

3. **创建新应用**
   - 登录后，点击 "New app" 按钮
   - 或者点击右上角的 "+ New app"

4. **配置应用设置**
   - **Repository**: 从下拉菜单中选择你的仓库（如 `你的用户名/lm-lens`）
   - **Branch**: 选择 `main`（或你的主分支）
   - **Main file path**: 输入 `app.py`
   - **App URL**（可选）: 可以自定义 URL，如 `lm-lens`
   - 点击 "Deploy" 按钮

5. **等待构建完成**
   - Streamlit 会自动检测并执行 `.streamlit/packages.sh` 构建前端组件
   - 然后安装 Python 依赖并启动应用
   - 构建过程通常需要 3-8 分钟（包含前端构建）
   - 可以在构建日志中查看进度：
     - 前端构建：`🔨 开始构建前端组件...`
     - Python 依赖安装
     - 应用启动

6. **访问应用**
   - 构建完成后，你会看到 "Your app is live!" 消息
   - 应用 URL 格式：`https://你的应用名.streamlit.app`
   - 点击 URL 或 "View app" 按钮访问应用

#### 步骤 3: 验证部署

1. **检查应用是否正常运行**
   - 访问应用 URL
   - 确认页面正常加载
   - 测试基本功能（选择模型、输入文本等）

2. **查看日志**（如有问题）
   - 在 Streamlit Cloud 控制台中点击 "Manage app"
   - 查看 "Logs" 标签页
   - 检查是否有错误信息

#### 步骤 4: 更新应用

每次更新代码后，只需推送到 GitHub：

```bash
git add .
git commit -m "Update: 描述你的更改"
git push origin main
```

Streamlit Cloud 会自动检测到更改并重新部署应用（通常需要 1-2 分钟）。

### 重要提示

⚠️ **前端组件构建**
- 项目包含 React 前端组件，需要在部署前构建
- **方法 1（推荐）**：在本地构建并提交构建产物
  ```bash
  cd lm_lens/components/frontend
  npm install
  npm run build
  git add lm_lens/components/frontend/build/
  git commit -m "Build frontend components"
  git push
  ```
- **方法 2**：Streamlit Cloud 会自动执行 `.streamlit/packages.sh` 脚本构建前端
  - 如果构建失败，请检查构建日志中的错误信息
  - 确保 Node.js 和 npm 在 Streamlit Cloud 环境中可用

### 优势
- ✅ 完全免费
- ✅ 自动构建前端组件
- ✅ 自动更新（Git push 后自动部署）
- ✅ 无需服务器维护

### 注意事项

1. **资源限制**
   - 免费版有 CPU/内存限制
   - 建议使用较小的模型（如 Qwen/Qwen2.5-Coder-0.5B, Qwen/Qwen2.5-Coder-1.5B）
   - 大模型可能导致内存不足或响应缓慢

2. **模型下载**
   - 首次加载模型需要从 Hugging Face 下载
   - 下载时间取决于模型大小和网络速度
   - 建议在配置文件中只包含小模型

3. **构建时间**
   - 首次部署可能需要 5-10 分钟
   - 后续更新通常只需 1-3 分钟

4. **依赖安装**
   - 确保 `requirements.txt` 包含所有必需的依赖
   - Streamlit 会自动安装这些依赖

5. **配置文件**
   - 应用使用 `config/docker_hosting.json` 作为配置文件
   - 确保该文件存在且格式正确

### 故障排除

**问题 1: 构建失败**
- 检查 `requirements.txt` 是否包含所有依赖
- 查看构建日志中的错误信息
- 确保 `app.py` 文件存在且路径正确

**问题 2: 应用无法启动**
- 检查 `config/docker_hosting.json` 是否存在
- 查看应用日志中的错误信息
- 确认所有 Python 导入路径正确

**问题 3: 模型加载失败**
- 检查模型名称是否正确
- 确认网络连接正常（需要访问 Hugging Face）
- 尝试使用更小的模型

**问题 4: 内存不足**
- 在配置文件中移除大模型
- 只保留小模型（如 Qwen/Qwen2.5-Coder-0.5B, Qwen/Qwen2.5-Coder-1.5B）
- 考虑升级到付费计划（如果需要使用大模型）

## 🎯 使用修复功能

### 在UI中使用

1. 部署后访问应用，在侧边栏展开 "Model Repair" 面板
2. 勾选 "Enable repair"
3. 选择修复方法（推荐使用 `me-iter`）
4. 设置训练轮数（默认10）
5. 输入源提示和目标输出
6. 点击 "Run Repair" 运行修复
7. 查看主界面显示的修复结果

### 示例

```
Source prompt: def hello():
Target output: def hello():\n    print('Hello, world!')
Approach: me-iter
Epochs: 10
```

## 📁 项目结构

```
lm_lens/
├── components/                   # 前端组件
│   └── frontend/                 # React 前端应用
├── models/                       # 模型定义
│   ├── transparent_lm.py         # 透明LM接口
│   └── tlens_model.py           # TransformerLens 模型实现
├── function/                     # 功能模块（包含修复等功能）
│   ├── core/                     # 核心修复实现
│   │   ├── star_core.py         # STAR核心实现
│   │   └── mint_core.py         # MINT实现
│   ├── config/                   # 配置模块
│   │   ├── configure.py         # 配置类
│   │   └── shared.py            # 共享配置
│   ├── utils/                    # 工具函数
│   │   ├── attribution.py       # 归因分析
│   │   ├── loader.py            # 数据加载
│   │   ├── optimizer.py         # 优化器
│   │   └── utilities.py         # 通用工具
│   └── repair_interface.py      # 修复接口
├── routes/                       # 路由定义
│   ├── contributions.py         # 贡献计算
│   ├── graph.py                 # 图构建
│   └── graph_node.py           # 图节点
└── server/                       # 服务器代码
    ├── app.py                   # Streamlit主应用
    ├── graph_selection.py       # 图选择
    ├── monitor.py               # 系统监控
    ├── styles.py                # 样式定义
    └── utils.py                 # 服务器工具
├── config/                       # 配置文件
│   └── docker_hosting.json      # Streamlit部署配置
├── app.py                        # Streamlit Cloud 入口
├── Dockerfile.streamlit-cloud    # Docker 配置（可选）
├── requirements.txt              # Python 依赖
├── setup.py                     # 安装配置
└── sample_input.txt            # 示例输入文件
```

## 🔧 配置

配置文件位于 `config/docker_hosting.json`：

```json
{
  "allow_loading_dataset_files": false,
  "max_user_string_length": 100,
  "preloaded_dataset_filename": "sample_input.txt",
  "debug": false,
  "demo_mode": true,
  "models": {
    "Qwen/Qwen2.5-Coder-0.5B": null,
    "Qwen/Qwen2.5-Coder-1.5B": null,
    "Qwen/Qwen2.5-Coder-3B": null
  },
  "default_model": "Qwen/Qwen2.5-Coder-0.5B"
}
```

## 📚 文档

- [修复模块文档](./lm_lens/function/README.md) - 修复功能详细说明

## 📝 许可证

本代码基于 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可证提供。

## 🙏 致谢

- LM Transparency Tool - 原始可视化工具
- STAR (Semantic Targeting for Analytical Repair) - 模型修复方法
