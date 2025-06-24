
# Python Requirement Generator

这是一个轻量级工具项目，用于自动生成 Python 项目的 `requirements.txt`，仅包含代码中实际使用（import）的 pip 包。

---

## 🔧 功能说明

- 使用 [`pipreqs`](https://github.com/bndr/pipreqs) 扫描 Python 源码中的 `import` 语句
- 自动生成精简版本的 `requirements.txt`（适合部署、发布等场景）
- 可配合 Git 初始化脚本快速搭建项目结构

---

## 📦 使用方法

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <project-dir>
```

### 2. 初始化 Git（可选）

如果你使用 `setup_git_project.sh`：

```bash
bash setup_git_project.sh
```

### 3. 生成 requirements.txt

```bash
bash generate_requirements.sh
```

脚本会自动安装 `pipreqs`（如果未安装），并在当前目录下生成或覆盖 `requirements.txt`。

---

## 📁 项目结构

```
├── generate_requirements.sh     # 使用 pipreqs 生成精简 requirements.txt
├── setup_git_project.sh         # （可选）快速初始化 Git 项目结构
├── README.md                    # 项目说明
```

---

## ✅ 依赖

- Python 3.x
- pip
- pipreqs（脚本会自动安装）

---

## 💡 示例

```bash
$ bash generate_requirements.sh
📦 pipreqs 未安装，正在安装...
🔍 正在使用 pipreqs 生成 requirements.txt...
✅ requirements.txt 已生成！
```

---

## 📜 License

MIT License
```

---

如需根据你实际项目内容补充说明（比如具体用在哪类项目、是否支持 Conda 等），可以告诉我我来继续优化~