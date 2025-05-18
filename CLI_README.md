# AI小说自动生成器命令行工具

这是一个基于AI的小说自动生成器命令行工具，可以一键生成包含多个章节的完整小说。

## 功能特点

- 全自动生成小说架构
- 自动创建章节目录
- 逐章生成小说内容
- 支持导入自定义知识文件
- 可配置小说主题、类型、章节数和字数

## 使用方法

增强版CLI工具采用子命令（subcommand）结构，以支持更精细化的操作。

### 基本结构

```bash
python cli.py <command> [options]
```

### 可用命令

*   `create`: 创建新的小说项目并初始化结构。
*   `outline`: 生成指定卷或整个小说的大纲。
*   `generate`: 开始生成指定卷或整个小说的内容。
*   `export`: 导出指定卷或整个小说的内容为不同格式。
*   `status`: 查看指定卷或整个小说的生成状态和进度。
*   `pause`: 暂停正在进行的生成任务。
*   `resume`: 恢复已暂停的生成任务。
*   `generate-architecture`: 生成小说架构。
*   `generate-blueprint`: 生成章节蓝图。

### 命令详情与参数

#### `create`

创建新的小说项目。

```bash
python cli.py create --output <path> --name <novel_name> [options]
```

**参数:**

*   `--output, -o` (必需): 输出文件夹路径。
*   `--name, -n` (必需): 小说项目名称。
*   `--total-words, -tw` (可选): 目标总字数（用于智能分卷规划）。 默认值：无（如果未指定，可能需要通过其他方式确定或使用内部默认）
*   `--volumes, -v` (可选): 指定分卷数量。 默认值：无（如果未指定，可能需要通过总字数智能计算或使用内部默认）
*   `--topic, -t` (可选): 小说主题。
*   `--genre, -g` (可选): 小说类型。
*   `--guidance` (可选): 用户指导内容。
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `outline`

生成指定卷或整个小说的大纲。

```bash
python cli.py outline --output <path> [--volume <volume_number>] [options]
```

**参数:**

*   `--output, -o` (必需): 小说项目输出文件夹路径。
*   `--volume, -v` (可选): 指定生成大纲的卷号。 默认行为：如果省略，则生成所有卷的大纲。
*   `--chapters, -c` (可选): 指定当前卷的章节数量（仅在指定 `--volume` 时有效）。 默认值：无（如果未指定，可能需要通过卷的总字数或内部默认确定）
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `generate`

开始生成指定卷或整个小说内容。

```bash
python cli.py generate --output <path> [--volume <volume_number>] [options]
```

**参数:**

*   `--output, -o` (必需): 小说项目输出文件夹路径。
*   `--volume, -v` (可选): 指定生成内容的卷号。 默认行为：如果省略，则按顺序生成所有卷。
*   `--no-finalize` (可选): 不执行最终润色。 默认值：`False`
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `export`

导出指定卷或整个小说内容为不同格式。

```bash
python cli.py export --output <path> --format <format> [--volume <volume_number>] [options]
```

**参数:**

*   `--output, -o` (必需): 小说项目输出文件夹路径。
*   `--format, -f` (必需): 导出格式（例如: `txt`, `epub`）。
*   `--volume, -v` (可选): 指定导出的卷号。 默认行为：如果省略，则导出整个小说。
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `status`

查看指定卷或整个小说的生成状态和进度。

```bash
python cli.py status --output <path> [--volume <volume_number>] [options]
```

**参数:**

*   `--output, -o` (必需): 小说项目输出文件夹路径。
*   `--volume, -v` (可选): 指定查看状态的卷号。 默认行为：如果省略，则查看整个小说的汇总状态。
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `pause`

暂停正在进行的生成任务。

```bash
python cli.py pause --output <path> [options]
```

**参数:**

*   `--output, -o` (必需): 小说项目输出文件夹路径。
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `resume`

恢复已暂停的生成任务。

```bash
python cli.py resume --output <path> [options]
```

**参数:**

*   `--output, -o` (必需): 小说项目输出文件夹路径。
*   `--config` (可选): 配置文件路径。 默认值：`config.json`

#### `generate-architecture`

生成小说架构文件。

```bash
python cli.py generate-architecture --config <path> --output <directory>
```

**参数:**

*   `--config, -c` (必需): 配置文件路径。该文件应包含LLM配置、小说参数等信息。
*   `--output, -o` (必需): 输出小说架构文件的目录路径。

**示例:**

```bash
python cli.py generate-architecture --config ./config.json --output ./my_novel_project
```

#### `generate-blueprint`

根据小说架构生成章节蓝图。

```bash
python cli.py generate-blueprint --config <path> --architecture <path> --chapter <number|range|all> --output <directory>
```

**参数:**

*   `--config, -c` (必需): 配置文件路径。该文件应包含LLM配置等信息。
*   `--architecture, -a` (必需): 小说架构文件的路径。
*   `--chapter, -ch` (必需): 指定要生成蓝图的章节。可以是单个章节号（例如 `5`），一个章节范围（例如 `1-10`），或者 `all` 表示生成所有章节的蓝图。
*   `--output, -o` (必需): 输出章节蓝图文件的目录路径。

**示例:**

```bash
# 生成第5章的蓝图
python cli.py generate-blueprint --config ./config.json --architecture ./my_novel_project/Novel Architecture.txt --chapter 5 --output ./my_novel_project

# 生成第1到第10章的蓝图
python cli.py generate-blueprint --config ./config.json --architecture ./my_novel_project/Novel Architecture.txt --chapter 1-10 --output ./my_novel_project

# 生成所有章节的蓝图
python cli.py generate-blueprint --config ./config.json --architecture ./my_novel_project/Novel Architecture.txt --chapter all --output ./my_novel_project
```

### 配置文件格式说明

CLI工具支持使用配置文件 (`config.json`) 来管理生成参数，避免命令行过长。配置文件采用JSON格式，包含以下主要部分：

```json
{
  "llm_config": { ... },
  "output_config": { ... },
  "novel_params": { ... },
  "generation_config": { ... },
  "knowledge_config": { ... },
  "backup_config": { ... }
}
```

命令行参数会覆盖配置文件中的相应设置。

以下是 `config.json` 中主要部分的详细说明和关键配置项：

#### `llm_config`

配置大型语言模型（LLM）相关的参数。

*   `api_key` (字符串): **必需**。LLM服务提供商的API密钥。
*   `model_name` (字符串): 可选。使用的LLM模型名称，例如 `"gpt-4o"`, `"claude-3-opus-20240229"` 等。默认值：无（通常需要指定或依赖于内部配置）。
*   `temperature` (数字): 可选。控制生成文本的随机性，值越高越随机。通常在 0.1 到 1.0 之间。默认值：无（依赖于LLM服务提供商的默认设置）。
*   `max_tokens` (整数): 可选。限制LLM生成文本的最大长度（以token计）。默认值：无（依赖于LLM服务提供商的默认设置）。
*   *其他可能的配置项，例如 `base_url`, `api_version` 等，取决于具体的LLM适配器实现。*

#### `output_config`

配置小说项目输出相关的参数。

*   `output_dir` (字符串): **必需**。小说项目输出的根目录路径。
*   `novel_name` (字符串): **必需**。小说项目的名称。
*   `total_words` (整数): 可选。目标总字数。用于规划小说整体规模。默认值：无（如果未指定，可能需要通过其他方式确定或使用内部默认）。
*   `volumes` (整数): 可选。指定小说的分卷数量。默认值：无（如果未指定，可能需要通过总字数智能计算或使用内部默认）。
*   *其他可能的配置项，例如是否覆盖现有文件等。*

#### `novel_params`

配置小说本身的创作参数，用于指导AI生成内容。

*   `topic` (字符串): 可选。小说的主题或核心概念。示例值：`"末世求生"`。
*   `genre` (字符串): 可选。小说的类型，例如 `"科幻"`, `"奇幻"`, `"武侠"` 等。示例值：`"科幻"`。
*   `guidance` (字符串): 可选。用户提供的额外创作指导、背景设定、关键情节或角色描述等。示例值：`"主角是一个天赋异禀但性格内向的少年，在魔法学院中经历成长"`。
*   *其他可能的小说参数，例如主角设定、世界观设定、重要配角等，取决于具体的生成逻辑。*

#### `generation_config`

配置小说生成流程的参数，控制生成过程的细节。

*   `chapters_per_volume` (整数): 可选。每卷的目标章节数量。默认值：无（如果未指定，可能需要通过卷的总字数或内部默认确定）。示例值：`200`。
*   `words_per_chapter` (整数): 可选。每章的目标字数。默认值：无（如果未指定，可能需要通过卷的总字数或内部默认确定）。示例值：`5000`。
*   `finalize_chapters` (布尔值): 可选。是否在生成章节草稿后进行最终润色。默认值：`true`。示例值：`true`。
*   *其他可能的生成流程控制参数，例如并发生成章节数量、重试次数等。*

#### `knowledge_config`

配置知识库相关的参数，用于为AI提供额外的背景知识或设定。

*   `knowledge_file` (字符串): 可选。外部知识文件的路径。该文件可以包含世界观、人物关系、重要事件等信息。示例值：`"./my_knowledge.txt"`。
*   *其他可能的知识库配置项，例如知识库类型（文本文件、向量数据库等）、向量数据库连接信息等。*

#### `backup_config`

配置自动备份相关的参数，用于保存生成过程中的进度和内容。

*   `auto_backup` (布尔值): 可选。是否开启自动备份功能。默认值：`true`。示例值：`true`。
*   `backup_interval` (字符串): 可选。自动备份的时间间隔。支持的格式可能包括 `"Xs"` (秒), `"Xm"` (分钟), `"Xh"` (小时), `"Xd"` (天)。默认值：无（如果开启自动备份，通常会有一个默认间隔）。示例值：`"1h"`。
*   *其他可能的备份配置项，例如最大备份数量、备份路径等。*

### 常见使用场景示例

1.  **创建并生成一部新小说 (分步):**
    ```bash
    # 创建项目结构，指定总字数和分卷数
    python cli.py create --output ./my_epic_novel --name "史诗奇幻" --total-words 2000000 --volumes 10 --topic "英雄的崛起" --genre "史诗奇幻"

    # 生成第一卷的大纲 (假设每卷200章)
    python cli.py outline --output ./my_epic_novel --volume 1 --chapters 200

    # 开始生成第一卷的内容
    python cli.py generate --output ./my_epic_novel --volume 1

    # 查看生成进度
    python cli.py status --output ./my_epic_novel --volume 1

    # 导出第一卷为EPUB格式
    python cli.py export --output ./my_epic_novel --volume 1 --format epub

    # 继续生成第二卷...
    python cli.py generate --output ./my_epic_novel --volume 2
    ```

2.  **使用配置文件生成小说:**
    ```bash
    # 确保 config.json 中已配置好所有参数
    python cli.py generate --config ./path/to/your/config.json
    ```

3.  **仅生成大纲:**
    ```bash
    python cli.py outline --output ./my_short_story --chapters 5
    ```

4.  **导出整个小说:**
    ```bash
    python cli.py export --output ./my_completed_novel --format txt
    ```

### 故障排除指南

*   **API密钥错误:** 确保在 `config.json` 或通过环境变量正确设置了LLM的API密钥。检查密钥是否有效。
*   **配置验证失败:** 运行命令时，如果提示配置验证失败，请检查命令行参数和配置文件中的设置是否符合要求（例如，输出路径是否存在、字数和章节数是否为有效数字等）。参考错误信息进行修正。
*   **生成过程中断:** 如果生成过程意外中断，可以使用 `resume` 命令尝试恢复。系统会尝试从上次保存的进度继续。
*   **输出文件缺失或错误:** 检查 `--output` 路径是否正确，以及是否有写入权限。如果生成内容有问题，可能需要调整小说参数、指导内容或LLM配置。
*   **格式转换失败:** 确保已安装对应格式转换所需的依赖库。例如，导出EPUB可能需要特定的库。

## 输出文件

生成完成后，输出目录中会包含以下内容：

-   `[Novel Name]/`: 小说项目根目录
    -   `config.json`: 项目配置文件（包含生成时使用的最终配置）
    -   `Novel Architecture.txt`：小说架构文件
    -   `Chapter Blueprint.txt`：章节目录文件
    -   `volumes/`: 分卷目录
        -   `volume_1/`: 第一卷目录
            -   `outline.txt`: 第一卷大纲
            -   `chapters/`: 存放章节内容的目录
                -   `Chapter_1_draft.txt`：第1章草稿
                -   `Chapter_1_final.txt`：第1章润色后的最终版本
                -   以此类推...
        -   `volume_2/`: 第二卷目录
        -   ... 以此类推
    -   `exports/`: 导出文件目录
        -   `[novel_name].epub`: 导出的EPUB文件
        -   `[novel_name].txt`: 导出的TXT文件
        -   ... 以此类推
    -   `backups/`: 备份文件目录
        -   `[timestamp]_volume_[volume_number].zip`: 卷备份文件
        -   `[timestamp]_full_novel.zip`: 全本备份文件
        -   ... 以此类推

## 注意事项

1.  请确保在 `config.json` 或通过命令行参数中正确配置了API密钥和其他必要参数。
2.  生成过程可能需要较长时间，取决于总字数、分卷数、章节数和每章字数。
3.  知识文件和用户指导内容可以帮助AI生成更加符合特定背景设定的内容。
4.  暂停和恢复功能依赖于生成过程中的进度保存点。