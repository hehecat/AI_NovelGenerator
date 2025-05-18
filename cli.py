# cli.py
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import json

# --- 优先处理 sys.path ---
# 假设 cli.py 文件本身就位于项目的根目录下 (例如 AI_NovelGenerator/cli.py)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------------

# 现在可以安全导入其他自定义模块
from novel_generator import (
    Novel_architecture_generate,
    Chapter_blueprint_generate,
    generate_chapter_draft,
    finalize_chapter,
    import_knowledge_file,
    clear_vector_store,
    enrich_chapter_text
)
# 假设 config_manager.py 中有 load_config 和 save_config 函数
# 如果它们在 ConfigManager 类中，你需要实例化类来调用或将其设为静态方法/模块级函数
try:
    from config_manager import load_config as cm_load_config, save_config as cm_save_config
except ImportError:
    logging.error("错误：无法从 config_manager 导入 load_config 或 save_config。请确保该模块和函数存在。")
    # 定义备用函数以允许脚本至少能解析参数，但配置加载/保存会失败
    def cm_load_config(filepath): logging.warning(f"config_manager.load_config 未找到，无法加载 {filepath}"); return {}
    def cm_save_config(data, filepath): logging.warning(f"config_manager.save_config 未找到，无法保存到 {filepath}"); return False

from utils import read_file, save_string_to_txt, clear_file_content

CONFIG_FILE = os.path.join(project_root, "config.json") # 全局配置文件

def setup_logging():
    """配置应用程序的日志记录。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("CLI 日志系统已配置。")

def load_cli_config():
    """
    加载CLI配置。
    优先级：
    1. 从 config.json 文件中根据 last_interface_format 提取的特定接口配置。
    2. config.json 文件中的顶层配置值（如果有）。
    3. 硬编码的绝对默认值。
    """
    absolute_defaults = {
        "interface_format": "OpenAI",
        "api_key": "YOUR_ABSOLUTE_FALLBACK_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 600,
        "embedding_interface_format": "OpenAI",
        "embedding_api_key": "YOUR_EMBEDDING_FALLBACK_API_KEY",
        "embedding_url": "https://api.openai.com/v1",
        "embedding_model_name": "text-embedding-ada-002",
        "embedding_retrieval_k": 4,
        "filepath": os.path.join(project_root, "novel_projects_default"),
        # other_params 的默认值可以不在这里定义，因为它们是命令相关的
        # 如果有全局性的 other_params 默认值，可以加在这里
        "num_chapters_default": 100, # 用于 generate-architecture 的默认章节数
        "word_count_default": 300000, # 用于 generate-architecture 的默认总字数
        "chapter_word_count_default": 3000 # 用于 generate-chapter 的默认单章字数

    }

    config_from_file = cm_load_config(CONFIG_FILE)
    if not config_from_file:
        config_from_file = {}
        logging.info(f"未找到或无法加载全局配置文件 {CONFIG_FILE}，将使用内部默认值。")
    else:
        logging.info(f"已加载全局配置文件 {CONFIG_FILE}。")


    active_interface_format = config_from_file.get("last_interface_format", absolute_defaults["interface_format"])
    active_embedding_format = config_from_file.get("last_embedding_interface_format", absolute_defaults["embedding_interface_format"])

    llm_interface_specific_config = {}
    if "llm_configs" in config_from_file and isinstance(config_from_file["llm_configs"], dict):
        llm_interface_specific_config = config_from_file["llm_configs"].get(active_interface_format, {})
        if not llm_interface_specific_config:
            logging.warning(f"在 config.json 的 'llm_configs' 中未找到接口 '{active_interface_format}' 的特定配置。")

    embedding_interface_specific_config = {}
    if "embedding_configs" in config_from_file and isinstance(config_from_file["embedding_configs"], dict):
        embedding_interface_specific_config = config_from_file["embedding_configs"].get(active_embedding_format, {})
        if not embedding_interface_specific_config:
            logging.warning(f"在 config.json 的 'embedding_configs' 中未找到接口 '{active_embedding_format}' 的特定配置。")


    final_flat_config = {}

    # LLM 参数
    final_flat_config["interface_format"] = active_interface_format
    final_flat_config["api_key"] = llm_interface_specific_config.get("api_key", config_from_file.get("api_key", absolute_defaults["api_key"]))
    final_flat_config["base_url"] = llm_interface_specific_config.get("base_url", config_from_file.get("base_url", absolute_defaults["base_url"]))
    final_flat_config["model_name"] = llm_interface_specific_config.get("model_name", config_from_file.get("model_name", absolute_defaults["model_name"]))
    final_flat_config["temperature"] = llm_interface_specific_config.get("temperature", float(config_from_file.get("temperature", absolute_defaults["temperature"])))
    final_flat_config["max_tokens"] = llm_interface_specific_config.get("max_tokens", int(config_from_file.get("max_tokens", absolute_defaults["max_tokens"])))
    final_flat_config["timeout"] = llm_interface_specific_config.get("timeout", int(config_from_file.get("timeout", absolute_defaults["timeout"])))

    # Embedding 参数
    final_flat_config["embedding_interface_format"] = active_embedding_format
    final_flat_config["embedding_api_key"] = embedding_interface_specific_config.get("api_key", config_from_file.get("embedding_api_key", absolute_defaults["embedding_api_key"]))
    final_flat_config["embedding_url"] = embedding_interface_specific_config.get("base_url", config_from_file.get("embedding_url", absolute_defaults["embedding_url"]))
    final_flat_config["embedding_model_name"] = embedding_interface_specific_config.get("model_name", config_from_file.get("embedding_model_name", absolute_defaults["embedding_model_name"]))
    final_flat_config["embedding_retrieval_k"] = embedding_interface_specific_config.get("retrieval_k", int(config_from_file.get("embedding_retrieval_k", absolute_defaults["embedding_retrieval_k"])))

    # 其他通用参数
    final_flat_config["filepath"] = config_from_file.get("filepath", absolute_defaults["filepath"])
    final_flat_config["num_chapters_default"] = int(config_from_file.get("num_chapters_default", absolute_defaults["num_chapters_default"]))
    final_flat_config["word_count_default"] = int(config_from_file.get("word_count_default", absolute_defaults["word_count_default"]))
    final_flat_config["chapter_word_count_default"] = int(config_from_file.get("chapter_word_count_default", absolute_defaults["chapter_word_count_default"]))


    # 从 config.json 的 other_params 合并值 (如果存在且 argparse 参数名匹配)
    # argparse 的 default 会优先于这里的 setdefault
    if "other_params" in config_from_file and isinstance(config_from_file["other_params"], dict):
        for key, value in config_from_file["other_params"].items():
            final_flat_config.setdefault(key, value) # 只在 final_flat_config 中尚无此键时添加

    default_project_parent = final_flat_config.get("filepath")
    if default_project_parent and not os.path.exists(default_project_parent):
        try:
            os.makedirs(default_project_parent, exist_ok=True)
            logging.info(f"已创建/确认默认项目父目录: {default_project_parent}")
        except OSError as e:
            logging.warning(f"无法创建默认项目父目录 {default_project_parent}: {e}")
            fallback_path = os.path.join(project_root, "novel_project_fallback")
            final_flat_config["filepath"] = fallback_path
            os.makedirs(fallback_path, exist_ok=True)
            logging.info(f"使用备用项目父目录: {fallback_path}")

    logging.debug(f"最终加载的CLI配置 (用于argparse defaults): {final_flat_config}")
    return final_flat_config

def main():
    setup_logging()
    config_data = load_cli_config()

    parser = argparse.ArgumentParser(description="AI Novel Generator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- 配置命令 ---
    config_parser = subparsers.add_parser("config", help="Configure CLI settings (updates global config.json)")
    config_parser.add_argument("--show", action="store_true", help="Show current effective configuration")
    config_parser.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                               help="Set a configuration value in global config.json (e.g., api_key YOUR_KEY or llm_configs.OpenAI.api_key YOUR_KEY)")

    # --- 小说架构生成命令 ---
    arch_parser = subparsers.add_parser("generate-architecture", help="Generate novel architecture")
    arch_parser.add_argument("--filepath", "-fp", type=str, default=config_data.get("filepath"), help="Path to novel project files (e.g., ./my_novel_project)")
    arch_parser.add_argument("--topic", "-t", type=str, required=False, help="Topic of the novel (overrides config if set)") # 改为可选，依赖配置
    arch_parser.add_argument("--genre", "-g", type=str, required=False, help="Genre of the novel (overrides config if set)") # 改为可选
    arch_parser.add_argument("--num_chapters", "-nc", type=int, default=config_data.get("num_chapters_default"), help="Number of chapters for the novel outline")
    arch_parser.add_argument("--word_count", "-wc", type=int, default=config_data.get("word_count_default"), help="Approximate total word count for the novel outline")
    arch_parser.add_argument("--user_guidance", "-ug", type=str, default=config_data.get("user_guidance", ""), help="User guidance for generation")
    # LLM 参数
    arch_parser.add_argument("--interface_format", type=str, default=config_data.get("interface_format"))
    arch_parser.add_argument("--api_key", type=str, default=config_data.get("api_key"))
    arch_parser.add_argument("--base_url", type=str, default=config_data.get("base_url"))
    arch_parser.add_argument("--model_name", type=str, default=config_data.get("model_name"))
    arch_parser.add_argument("--temperature", type=float, default=config_data.get("temperature"))
    arch_parser.add_argument("--max_tokens", type=int, default=config_data.get("max_tokens"))
    arch_parser.add_argument("--timeout", type=int, default=config_data.get("timeout"))

    # --- 章节蓝图生成命令 ---
    blueprint_parser = subparsers.add_parser("generate-blueprint", help="Generate chapter blueprint for a specific volume")
    blueprint_parser.add_argument("--filepath", "-fp", type=str, default=config_data.get("filepath"))
    blueprint_parser.add_argument("--target-chapter", "-tc", type=int, default=1)
    blueprint_parser.add_argument("--novel-total-chapters", "-ntc", type=int, default=None)
    blueprint_parser.add_argument("--user_guidance", "-ug", type=str, default=config_data.get("user_guidance", ""))
    # LLM 参数
    blueprint_parser.add_argument("--interface_format", type=str, default=config_data.get("interface_format"))
    blueprint_parser.add_argument("--api_key", type=str, default=config_data.get("api_key"))
    blueprint_parser.add_argument("--base_url", type=str, default=config_data.get("base_url"))
    blueprint_parser.add_argument("--model_name", type=str, default=config_data.get("model_name"))
    blueprint_parser.add_argument("--temperature", type=float, default=config_data.get("temperature"))
    blueprint_parser.add_argument("--max_tokens", type=int, default=config_data.get("max_tokens"))
    blueprint_parser.add_argument("--timeout", type=int, default=config_data.get("timeout"))

    # --- 章节草稿生成命令 ---
    chapter_parser = subparsers.add_parser("generate-chapter", help="Generate chapter draft")
    chapter_parser.add_argument("--filepath", "-fp", type=str, default=config_data.get("filepath"))
    chapter_parser.add_argument("--chapter_num", "-cn", type=int, required=True)
    chapter_parser.add_argument("--word_count", "-wc", type=int, default=config_data.get("chapter_word_count_default"))
    chapter_parser.add_argument("--user_guidance", "-ug", type=str, default=config_data.get("user_guidance", ""))
    chapter_parser.add_argument("--characters_involved", type=str, default=config_data.get("characters_involved", ""))
    chapter_parser.add_argument("--key_items", type=str, default=config_data.get("key_items", ""))
    chapter_parser.add_argument("--scene_location", type=str, default=config_data.get("scene_location", ""))
    chapter_parser.add_argument("--time_constraint", type=str, default=config_data.get("time_constraint", ""))
    # LLM 参数
    chapter_parser.add_argument("--interface_format", type=str, default=config_data.get("interface_format"))
    chapter_parser.add_argument("--api_key", type=str, default=config_data.get("api_key"))
    chapter_parser.add_argument("--base_url", type=str, default=config_data.get("base_url"))
    chapter_parser.add_argument("--model_name", type=str, default=config_data.get("model_name"))
    chapter_parser.add_argument("--temperature", type=float, default=config_data.get("temperature"))
    chapter_parser.add_argument("--max_tokens", type=int, default=config_data.get("max_tokens"))
    chapter_parser.add_argument("--timeout", type=int, default=config_data.get("timeout"))
    # Embedding 参数
    chapter_parser.add_argument("--embedding_api_key", type=str, default=config_data.get("embedding_api_key"))
    chapter_parser.add_argument("--embedding_url", type=str, default=config_data.get("embedding_url"))
    chapter_parser.add_argument("--embedding_interface_format", type=str, default=config_data.get("embedding_interface_format"))
    chapter_parser.add_argument("--embedding_model_name", type=str, default=config_data.get("embedding_model_name"))
    chapter_parser.add_argument("--embedding_retrieval_k", type=int, default=config_data.get("embedding_retrieval_k"))

    # --- 知识库导入命令 ---
    knowledge_parser = subparsers.add_parser("import-knowledge", help="Import knowledge file into vector store")
    knowledge_parser.add_argument("--filepath", "-fp", type=str, default=config_data.get("filepath"))
    knowledge_parser.add_argument("--knowledge_file", "-kf", type=str, required=True)
    # Embedding 参数
    knowledge_parser.add_argument("--embedding_api_key", type=str, default=config_data.get("embedding_api_key"))
    knowledge_parser.add_argument("--embedding_url", type=str, default=config_data.get("embedding_url"))
    knowledge_parser.add_argument("--embedding_interface_format", type=str, default=config_data.get("embedding_interface_format"))
    knowledge_parser.add_argument("--embedding_model_name", type=str, default=config_data.get("embedding_model_name"))

    # --- 清空向量数据库命令 ---
    clear_vs_parser = subparsers.add_parser("clear-vectorstore", help="Clear the vector store for the project")
    clear_vs_parser.add_argument("--filepath", "-fp", type=str, default=config_data.get("filepath"))

    args = parser.parse_args()

    # --- 命令处理逻辑 ---
    if args.command == "config":
        if args.show:
            print("Current Effective CLI Configuration (from config.json merged with defaults):")
            # 重新加载一次，确保显示的是最新的，或者直接用 config_data
            # config_to_show = load_cli_config() # 如果希望总是显示最新合并的
            for key, value in config_data.items(): # 显示启动时加载和合并的配置
                print(f"  {key}: {value}")
        elif args.set:
            key_path, value_str = args.set
            
            # 尝试智能转换类型
            if value_str.lower() == 'true': value_to_set = True
            elif value_str.lower() == 'false': value_to_set = False
            elif value_str.replace('.', '', 1).isdigit(): # 检查是否为数字（整数或浮点数）
                if '.' in value_str: value_to_set = float(value_str)
                else: value_to_set = int(value_str)
            else: value_to_set = value_str

            # 更新磁盘上的 config.json
            current_disk_config = cm_load_config(CONFIG_FILE)
            if not current_disk_config: current_disk_config = {}

            # 处理嵌套键路径，例如 llm_configs.OpenAI.api_key
            keys = key_path.split('.')
            d = current_disk_config
            for k in keys[:-1]:
                d = d.setdefault(k, {}) # 如果路径不存在则创建字典
            if isinstance(d, dict):
                d[keys[-1]] = value_to_set
                if cm_save_config(current_disk_config, CONFIG_FILE):
                    logging.info(f"Configuration updated in {CONFIG_FILE}: {key_path} = {value_to_set}")
                    # 更新当前会话的 config_data (如果需要立即生效)
                    # 为了简单，这里不动态更新 config_data, 下次启动会加载新值
                else:
                    logging.error(f"Failed to save configuration to {CONFIG_FILE}")
            else:
                logging.error(f"无法设置配置：路径 '{'.'.join(keys[:-1])}' 不是一个字典.")

        else:
            config_parser.print_help()

    elif args.command == "generate-architecture":
        project_path = args.filepath
        if not project_path: # 如果 filepath 最终为空或None
            logging.error("错误：项目路径 (--filepath) 未指定。请在命令行或 config.json 中配置。")
            return
        if not os.path.exists(project_path):
            try:
                os.makedirs(project_path)
                logging.info(f"已创建项目目录: {project_path}")
            except OSError as e:
                logging.error(f"无法创建项目目录 {project_path}: {e}")
                return
        
        # 确保 topic 和 genre 有值
        topic_to_use = args.topic if args.topic is not None else config_data.get("topic", "")
        genre_to_use = args.genre if args.genre is not None else config_data.get("genre", "")
        if not topic_to_use or not genre_to_use:
            logging.error("错误：小说主题 (--topic) 和类型 (--genre) 是必需的。请在命令行或 config.json (other_params) 中提供。")
            arch_parser.print_help()
            return

        logging.info("CLI: Generating novel architecture...")
        try:
            Novel_architecture_generate(
                interface_format=args.interface_format, api_key=args.api_key, base_url=args.base_url,
                llm_model=args.model_name, topic=topic_to_use, genre=genre_to_use,
                number_of_chapters=args.num_chapters,
                word_number=args.word_count, # Novel_architecture_generate 接收的是总字数
                filepath=project_path,
                temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout,
                user_guidance=args.user_guidance
            )
            logging.info(f"Novel architecture generation initiated. Check files in {project_path}")
        except Exception as e:
            logging.error(f"生成小说架构时发生错误: {e}", exc_info=True)


    elif args.command == "generate-blueprint":
        project_path = args.filepath
        if not project_path:
            logging.error("错误：项目路径 (--filepath) 未指定。")
            return
        architecture_file = os.path.join(project_path, "Novel_architecture.txt")
        if not os.path.exists(architecture_file):
            logging.error(f"Novel architecture file not found at {architecture_file}. Please generate architecture first.")
            return
        logging.info(f"CLI: Generating chapter blueprint for volume containing chapter {args.target_chapter}...")
        try:
            Chapter_blueprint_generate(
                interface_format=args.interface_format, api_key=args.api_key, base_url=args.base_url,
                llm_model=args.model_name,
                save_path=project_path,
                target_chapter=args.target_chapter,
                novel_total_chapters_override=args.novel_total_chapters,
                temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout,
                user_guidance=args.user_guidance
            )
            logging.info(f"Chapter blueprint generation initiated. Check files in {project_path}")
        except Exception as e:
            logging.error(f"生成章节蓝图时发生错误: {e}", exc_info=True)

    elif args.command == "generate-chapter":
        project_path = args.filepath
        if not project_path:
            logging.error("错误：项目路径 (--filepath) 未指定。")
            return
        if not os.path.exists(project_path):
            logging.error(f"Project path {project_path} does not exist.")
            return
        chapters_dir = os.path.join(project_path, "章节内容")
        if not os.path.exists(chapters_dir):
            os.makedirs(chapters_dir)

        logging.info(f"CLI: Generating chapter {args.chapter_num} draft...")
        try:
            draft_content = generate_chapter_draft(
                interface_format=args.interface_format, api_key=args.api_key, base_url=args.base_url,
                model_name=args.model_name, filepath=project_path,
                chapter_number=args.chapter_num,
                word_count_target=args.word_count,
                temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout,
                user_guidance=args.user_guidance,
                characters_involved=args.characters_involved, key_items=args.key_items,
                scene_location=args.scene_location, time_constraint=args.time_constraint,
                embedding_api_key=args.embedding_api_key, embedding_url=args.embedding_url,
                embedding_interface_format=args.embedding_interface_format,
                embedding_model_name=args.embedding_model_name,
                embedding_retrieval_k=args.embedding_retrieval_k
            )
            if draft_content:
                draft_file_path = os.path.join(chapters_dir, f"chapter_{args.chapter_num}_draft.md")
                save_string_to_txt(draft_content, draft_file_path)
                logging.info(f"Chapter {args.chapter_num} draft saved to {draft_file_path}")
            else:
                logging.error(f"Failed to generate chapter {args.chapter_num} draft (empty content returned).")
        except Exception as e:
            logging.error(f"生成章节 {args.chapter_num} 草稿时发生错误: {e}", exc_info=True)


    elif args.command == "import-knowledge":
        project_path = args.filepath
        if not project_path:
            logging.error("错误：项目路径 (--filepath) 未指定。")
            return

        if os.path.isabs(args.knowledge_file):
            knowledge_file_full_path = args.knowledge_file
        else:
            # 优先尝试项目路径下的文件，其次是脚本同级目录下的 '事例' 文件夹
            path_in_project = os.path.join(project_path, args.knowledge_file)
            path_in_examples = os.path.join(project_root, "事例", args.knowledge_file)
            if os.path.exists(path_in_project):
                knowledge_file_full_path = path_in_project
            elif os.path.exists(path_in_examples):
                knowledge_file_full_path = path_in_examples
            else:
                logging.error(f"Knowledge file '{args.knowledge_file}' not found in project path '{project_path}' or in example directory '{os.path.join(project_root, '事例')}'.")
                return

        logging.info(f"CLI: Importing knowledge file {knowledge_file_full_path}...")
        try:
            import_knowledge_file(
                knowledge_file_path=knowledge_file_full_path,
                project_path=project_path,
                embedding_api_key=args.embedding_api_key,
                embedding_base_url=args.embedding_url,
                embedding_interface_format=args.embedding_interface_format,
                embedding_model_name=args.embedding_model_name
            )
            logging.info(f"Knowledge file import initiated for project at {project_path}.")
        except Exception as e:
            logging.error(f"导入知识文件时发生错误: {e}", exc_info=True)

    elif args.command == "clear-vectorstore":
        project_path = args.filepath
        if not project_path:
            logging.error("错误：项目路径 (--filepath) 未指定。")
            return
        logging.info(f"CLI: Clearing vector store for project at {project_path}...")
        try:
            clear_vector_store(project_path)
            logging.info(f"Vector store cleared for project at {project_path}.")
        except Exception as e:
            logging.error(f"清空向量数据库时发生错误: {e}", exc_info=True)

    else:
        parser.print_help()

if __name__ == "__main__":
    main();