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
    parser_architecture.add_argument("--config", default="config.json", help="配置文件的路径（默认：config.json）")
    parser_architecture.add_argument("--output", required=True, help="保存架构文件的目录")
    parser_architecture.add_argument("--topic", default="未命名小说", help="小说主题（默认：未命名小说）")
    parser_architecture.add_argument("--genre", default="奇幻", help="小说类型（默认：奇幻）")
    parser_architecture.add_argument("--chapters", type=int, default=20, help="章节数量（默认：20）")
    parser_architecture.add_argument("--words", type=int, default=100000, help="预计总字数（默认：100000）")
    parser_architecture.add_argument("--guidance", default="", help="用户指导说明（可选）")
    parser_architecture.add_argument("--temperature", type=float, default=0.7, help="生成温度（默认：0.7）")
    parser_architecture.add_argument("--max-tokens", type=int, default=2048, help="最大token数（默认：2048）")
    parser_architecture.add_argument("--timeout", type=int, default=600, help="超时时间（秒）（默认：600）")
    parser_architecture.set_defaults(func=generate_architecture_command)

    # generate-blueprint 命令
    parser_blueprint = subparsers.add_parser("generate-blueprint", help="生成章节蓝图")
    parser_blueprint.add_argument("--config", default="config.json", help="配置文件的路径（默认：config.json）")
    parser_blueprint.add_argument("--architecture", required=True, help="架构文件的路径")
    parser_blueprint.add_argument("--volume", required=True, help="卷号（1-N）或 'all'")
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
        # 检查配置文件是否存在
        if not os.path.exists(args.config):
            print(f"错误：配置文件 {args.config} 不存在")
            print("请确保在项目目录下存在 config.json 文件，或使用 --config 参数指定配置文件路径")
            return

        # 加载配置
        config = config_manager.load_config(args.config)

        # 获取 LLM 配置
        last_interface_format = config.get("last_interface_format", "OpenAI")
        llm_config = config.get("llm_configs", {}).get(last_interface_format, {})
        
        if not llm_config:
            raise KeyError(f"在 llm_configs 中未找到 {last_interface_format} 的配置")

        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)

        # 调用架构生成函数
        architecture_data = novel_generator.architecture.Novel_architecture_generate(
            interface_format=last_interface_format,
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
            llm_model=llm_config.get("model_name"),
            topic=args.topic,
            genre=args.genre,
            number_of_chapters=args.chapters,
            word_number=args.words,
            filepath=args.output,
            user_guidance=args.guidance,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout
        )

        # 保存架构数据到 JSON 文件
        output_path = os.path.join(args.output, "architecture.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(architecture_data, f, indent=4, ensure_ascii=False)

        print(f"架构数据已保存到 {output_path}")

    except FileNotFoundError:
        print(f"错误：在 {args.config} 未找到配置文件")
    except KeyError as e:
        print(f"错误：配置文件中缺少必要的参数: {e}")
        print("请确保 config.json 包含以下结构：")
        print("""
{
    "last_interface_format": "OpenAI",
    "llm_configs": {
        "OpenAI": {
            "api_key": "你的API密钥",
            "base_url": "API的基础URL",
            "model_name": "使用的模型名称"
        }
    }
}
        """)
    except Exception as e:
        print(f"生成架构时发生错误: {e}")


def generate_blueprint_command(args):
    print(f"正在生成蓝图，使用配置文件: {args.config} 和架构文件: {args.architecture}")
    try:
        # 检查配置文件是否存在
        if not os.path.exists(args.config):
            print(f"错误：配置文件 {args.config} 不存在")
            print("请确保在项目目录下存在 config.json 文件，或使用 --config 参数指定配置文件路径")
            return

        # 加载配置
        config = config_manager.load_config(args.config)

        # 获取 LLM 配置
        last_interface_format = config.get("last_interface_format", "OpenAI")
        llm_config = config.get("llm_configs", {}).get(last_interface_format, {})
        
        if not llm_config:
            raise KeyError(f"在 llm_configs 中未找到 {last_interface_format} 的配置")

        # 确保输出目录存在
        os.makedirs(args.output, exist_ok=True)

        # 解析卷号输入
        if args.volume.lower() == "all":
            # 如果是 "all"，则生成所有卷的蓝图
            # 这里需要从架构文件中获取卷数
            with open(args.architecture, "r", encoding="utf-8") as f:
                architecture_data = json.load(f)
            num_volumes = architecture_data.get("num_volumes", 1)
            
            for volume in range(1, num_volumes + 1):
                print(f"正在生成第 {volume} 卷的蓝图...")
                novel_generator.blueprint.Chapter_blueprint_generate(
                    interface_format=last_interface_format,
                    api_key=llm_config.get("api_key"),
                    base_url=llm_config.get("base_url"),
                    llm_model=llm_config.get("model_name"),
                    save_path=args.output,
                    user_guidance=config.get("other_params", {}).get("user_guidance", ""),
                    target_volume=volume,
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 4096),
                    timeout=llm_config.get("timeout", 600)
                )
        else:
            # 如果是具体卷号，则生成对应卷的蓝图
            try:
                volume = int(args.volume)
                if volume < 1:
                    raise ValueError("卷号必须大于0")
                
                print(f"正在生成第 {volume} 卷的蓝图...")
                novel_generator.blueprint.Chapter_blueprint_generate(
                    interface_format=last_interface_format,
                    api_key=llm_config.get("api_key"),
                    base_url=llm_config.get("base_url"),
                    llm_model=llm_config.get("model_name"),
                    save_path=args.output,
                    user_guidance=config.get("other_params", {}).get("user_guidance", ""),
                    target_volume=volume,
                    temperature=llm_config.get("temperature", 0.7),
                    max_tokens=llm_config.get("max_tokens", 4096),
                    timeout=llm_config.get("timeout", 600)
                )
            except ValueError as e:
                print(f"错误：无效的卷号格式: {args.volume}。请使用数字（大于0）或 'all'。")
                return

    except FileNotFoundError:
        print(f"错误：未找到文件（{args.config} 或 {args.architecture}）")
    except json.JSONDecodeError:
        print(f"错误：无法解析架构文件 {args.architecture} 中的 JSON 数据")
    except KeyError as e:
        print(f"错误：配置文件中缺少必要的参数: {e}")
        print("请确保 config.json 包含以下结构：")
        print("""
{
    "last_interface_format": "OpenAI",
    "llm_configs": {
        "OpenAI": {
            "api_key": "你的API密钥",
            "base_url": "API的基础URL",
            "model_name": "使用的模型名称"
        }
    }
}
        """)
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