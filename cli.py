import argparse
import os
import json
# 假设这些模块存在并具有必要的函数
import config_manager
import novel_generator.architecture
import novel_generator.blueprint

def main():
    parser = argparse.ArgumentParser(description="AI小说生成器命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate-architecture 命令
    parser_architecture = subparsers.add_parser("generate-architecture", help="生成小说整体架构")
    parser_architecture.add_argument("--config", required=True, help="配置文件的路径")
    parser_architecture.add_argument("--output", required=True, help="保存架构文件的目录")
    parser_architecture.set_defaults(func=generate_architecture_command)

    # generate-blueprint 命令
    parser_blueprint = subparsers.add_parser("generate-blueprint", help="生成章节蓝图")
    parser_blueprint.add_argument("--config", required=True, help="配置文件的路径")
    parser_blueprint.add_argument("--architecture", required=True, help="架构文件的路径")
    parser_blueprint.add_argument("--chapter", required=True, help="章节号、范围（例如，1-5）或 'all'")
    parser_blueprint.add_argument("--output", required=True, help="保存蓝图文件的目录")
    parser_blueprint.set_defaults(func=generate_blueprint_command)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

def generate_architecture_command(args):
    print(f"正在使用配置文件生成架构: {args.config}")
    try:
        # 加载配置
        config = config_manager.load_config(args.config) # 假设 load_config 存在

        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)

        # 调用架构生成函数
        # 假设 novel_generator.architecture 有类似 generate_architecture 的函数
        architecture_data = novel_generator.architecture.generate_architecture(config)

        # 保存架构数据
        output_path = os.path.join(args.output, "architecture.json") # 假设输出为 JSON 格式
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(architecture_data, f, indent=4, ensure_ascii=False)

        print(f"架构已保存到 {output_path}")

    except FileNotFoundError:
        print(f"错误：在 {args.config} 未找到配置文件")
    except Exception as e:
        print(f"生成架构时发生错误: {e}")


def generate_blueprint_command(args):
    print(f"正在为章节 {args.chapter} 生成蓝图，使用配置文件: {args.config} 和架构文件: {args.architecture}")
    try:
        # 加载配置
        config = config_manager.load_config(args.config) # 假设 load_config 存在

        # 读取架构文件
        with open(args.architecture, "r", encoding="utf-8") as f:
            architecture_data = json.load(f) # 假设架构文件为 JSON 格式

        # 解析章节输入
        chapters_to_generate = parse_chapter_input(args.chapter, architecture_data)

        if not chapters_to_generate:
            print(f"未找到有效的章节输入: {args.chapter}")
            return

        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)

        # 遍历并为指定章节生成蓝图
        for chapter_num in chapters_to_generate:
            print(f"正在为第 {chapter_num} 章生成蓝图...")
            # 假设 novel_generator.blueprint 有类似 generate_blueprint 的函数
            # 该函数可能需要配置、架构数据和特定章节号/数据
            blueprint_data = novel_generator.blueprint.generate_blueprint(config, architecture_data, chapter_num)

            # 保存蓝图数据
            output_path = os.path.join(args.output, f"blueprint_chapter_{chapter_num}.json") # 假设输出为 JSON 格式
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(blueprint_data, f, indent=4, ensure_ascii=False)

            print(f"第 {chapter_num} 章的蓝图已保存到 {output_path}")

    except FileNotFoundError:
        print(f"错误：未找到文件（{args.config} 或 {args.architecture}）")
    except json.JSONDecodeError:
        print(f"错误：无法解析架构文件 {args.architecture} 中的 JSON 数据")
    except Exception as e:
        print(f"生成蓝图时发生错误: {e}")

def parse_chapter_input(chapter_input, architecture_data):
    """解析章节输入字符串（'all'、'number' 或 'range'）"""
    chapters = []
    if chapter_input.lower() == "all":
        # 假设 architecture_data 具有允许遍历章节的结构
        # 例如，如果 architecture_data 是章节列表或具有 'chapters' 键
        # 这部分需要根据实际的 architecture_data 结构进行调整
        print("正在解析'所有'章节 - 注意：这需要了解 architecture_data 的结构。")
        # 占位符：在实际场景中，您需要从 architecture_data 中提取章节号
        # 现在，让我们假设 architecture_data 是一个带有 'chapters' 列表的字典，
        # 并且列表中的每个项目都有一个 'chapter_number' 键。
        if isinstance(architecture_data, dict) and 'chapters' in architecture_data:
             chapters = [c.get('chapter_number') for c in architecture_data['chapters'] if isinstance(c, dict) and 'chapter_number' in c]
             chapters = [c for c in chapters if isinstance(c, (int, str))] # 过滤掉 None 或无效类型
        else:
             print("警告：无法从 architecture_data 结构中确定章节。")
             # 如果架构结构未知，则进行回退或错误处理
             pass # 或抛出错误

    elif "-" in chapter_input:
        try:
            start, end = map(int, chapter_input.split("-"))
            chapters = list(range(start, end + 1))
        except ValueError:
            print(f"无效的章节范围格式: {chapter_input}。请使用 'start-end' 格式。")
    else:
        try:
            chapters = [int(chapter_input)]
        except ValueError:
            print(f"无效的章节号格式: {chapter_input}。请使用数字、范围（例如，1-5）或 'all'。")

    # 如果可能，过滤章节以确保它们存在于架构数据中
    # 这个过滤步骤在很大程度上取决于 architecture_data 的结构
    # 现在，我们将返回解析后的列表，假设生成函数会处理不存在的章节
    return chapters


if __name__ == "__main__":
    main()