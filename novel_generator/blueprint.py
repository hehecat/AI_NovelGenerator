#novel_generator/blueprint.py
# -*- coding: utf-8 -*-
"""
章节蓝图生成（Chapter_blueprint_generate 及辅助函数）
"""
import os
import re
import logging
from novel_generator.common import invoke_with_cleaning
from llm_adapters import create_llm_adapter
from prompt_definitions import chapter_blueprint_prompt, chunked_chapter_blueprint_prompt
from utils import read_file, clear_file_content, save_string_to_txt

def compute_chunk_size(number_of_chapters: int, max_tokens: int) -> int:
    """
    计算生成章节蓝图时的分块大小。
    
    基于每章约100 tokens的估算，结合max_tokens计算合适的分块大小。
    计算公式：chunk_size = (floor(max_tokens/100/10)*10) - 10
    
    Args:
        number_of_chapters (int): 需要生成的章节总数
        max_tokens (int): 模型的最大token限制
        
    Returns:
        int: 计算得到的分块大小，范围在[1, number_of_chapters]之间
    """
    tokens_per_chapter = 100.0
    ratio = max_tokens / tokens_per_chapter
    ratio_rounded_to_10 = int(ratio // 10) * 10
    chunk_size = ratio_rounded_to_10 - 10
    if chunk_size < 1:
        chunk_size = 1
    if chunk_size > number_of_chapters:
        chunk_size = number_of_chapters
    return chunk_size

def limit_chapter_blueprint(blueprint_text: str, limit_chapters: int = 100) -> str:
    """
    限制章节蓝图中的章节数量，只保留最近的limit_chapters章。
    
    用于避免prompt过长，同时保持上下文的连贯性。
    
    Args:
        blueprint_text (str): 完整的章节蓝图文本
        limit_chapters (int, optional): 要保留的最近章节数，默认为100
        
    Returns:
        str: 截取后的章节蓝图文本
    """
    pattern = r"(第\s*\d+\s*章.*?)(?=第\s*\d+\s*章|$)"
    chapters = re.findall(pattern, blueprint_text, flags=re.DOTALL)
    if not chapters:
        return blueprint_text
    if len(chapters) <= limit_chapters:
        return blueprint_text
    selected = chapters[-limit_chapters:]
    logging.info("已有章节{{selected}}")
    return "\n\n".join(selected).strip()

def parse_volume_architecture(architecture_text: str, target_volume: int) -> dict:
    """
    从架构文本中解析指定卷的所有信息。
    
    解析内容包括：
    - 卷号、标题
    - 章节范围（起始和结束章节）
    - 卷的核心情节概述
    - 主要角色发展弧光
    - 本卷结局状态/悬念
    - 关键情节里程碑列表
    
    Args:
        architecture_text (str): 完整的小说架构文本
        target_volume (int): 目标卷号
        
    Returns:
        dict: 包含卷信息的字典，如果未找到则返回None
    """
    # 初始化返回结果的变量
    volume_info = {
        "volume_number": None,
        "volume_title": "未找到卷标题",
        "start_chapter": None,
        "end_chapter": None,
        "volume_summary": f"未找到第 {target_volume} 卷的信息。",
        "character_arcs": "未能找到主要角色发展弧光。",
        "ending_cliffhanger": "未能找到本卷结局状态/悬念。",
        "milestones_list": []
    }

    # 主正则表达式：匹配卷信息
    volume_pattern_all = r"## 第\s*(\d+)\s*卷：\s*([^\n]+)\s*\n(?:[^\n]*?\n)*?-\s*\*\*(?:预估章节范围|Estimated Chapter Range)：\*\*\s*(\d+)\s*-\s*(\d+).*?(?=## 第\s*\d+\s*卷：|$)"

    for volume_match in re.finditer(volume_pattern_all, architecture_text, re.DOTALL | re.IGNORECASE):
        try:
            current_volume_number = int(volume_match.group(1))
            if current_volume_number != target_volume:
                continue

            current_volume_title = volume_match.group(2).strip()
            current_start_chapter = int(volume_match.group(3))
            current_end_chapter = int(volume_match.group(4))
        except (IndexError, ValueError) as e:
            logging.warning(f"解析卷基本信息（卷号、标题、章节范围）时出错，跳过此卷。错误：{e}")
            logging.debug(f"问题匹配内容片段：{volume_match.group(0)[:200]}...")
            continue

        volume_info["volume_number"] = current_volume_number
        volume_info["volume_title"] = current_volume_title
        volume_info["start_chapter"] = current_start_chapter
        volume_info["end_chapter"] = current_end_chapter

        # 获取整个匹配到的卷的文本内容
        volume_content = volume_match.group(0)

        # 提取核心情节概述
        summary_pattern = r"-\s*\*\*(本卷核心情节概述|Core Plot Summary)：\*\*\s*(.*?)(?=\n-\s*\*\*|\Z)"
        summary_match = re.search(summary_pattern, volume_content, re.DOTALL | re.IGNORECASE)
        volume_info["volume_summary"] = summary_match.group(2).strip() if summary_match else "未能找到本卷的核心情节概述。"

        # 提取主要角色发展弧光
        arcs_pattern = r"-\s*\*\*(主要角色发展弧光|Key Character Arcs)：\*\*\s*(.*?)(?=\n-\s*\*\*|\Z)"
        arcs_match = re.search(arcs_pattern, volume_content, re.DOTALL | re.IGNORECASE)
        volume_info["character_arcs"] = arcs_match.group(2).strip() if arcs_match else "未能找到主要角色发展弧光。"

        # 提取本卷结局状态/悬念
        cliffhanger_pattern = r"-\s*\*\*(本卷结局状态/悬念|Ending State / Cliffhanger)：\*\*\s*(.*?)(?=\n###|\Z)"
        cliffhanger_match = re.search(cliffhanger_pattern, volume_content, re.DOTALL | re.IGNORECASE)
        volume_info["ending_cliffhanger"] = cliffhanger_match.group(2).strip() if cliffhanger_match else "未能找到本卷结局状态/悬念。"

        # 提取关键情节里程碑
        milestones_pattern = r"###\s*(关键情节里程碑|Key Plot Milestones)\s*\n((?:-\s+[^\n]*(?:\n|$))+)"
        milestones_match = re.search(milestones_pattern, volume_content, re.DOTALL | re.IGNORECASE)
        if milestones_match:
            milestones_text = milestones_match.group(2).strip()
            volume_info["milestones_list"] = [
                line.strip().lstrip('-').strip()
                for line in milestones_text.split('\n')
                if line.strip().startswith('-')
            ]
        else:
            volume_info["milestones_list"] = []

        return volume_info

    return None

def generate_chapter_blueprint_chunk(
    llm_adapter,
    architecture_text: str,
    volume_info: dict,
    start_chapter: int,
    end_chapter: int,
    existing_blueprint: str,
    user_guidance: str,
    is_single_shot: bool = False
) -> str:
    """
    生成指定章节范围的蓝图。
    
    根据is_single_shot参数决定使用不同的提示词模板：
    - True: 使用chapter_blueprint_prompt一次性生成整个卷的蓝图
    - False: 使用chunked_chapter_blueprint_prompt分块生成蓝图
    
    Args:
        llm_adapter: LLM适配器实例，用于调用语言模型
        architecture_text (str): 完整的小说架构文本
        volume_info (dict): 卷信息字典，包含卷号、标题、摘要等信息
        start_chapter (int): 起始章节号
        end_chapter (int): 结束章节号
        existing_blueprint (str): 已有的蓝图内容，用于分块生成时提供上下文
        user_guidance (str): 用户提供的额外指导
        is_single_shot (bool, optional): 是否一次性生成整个卷的蓝图，默认为False
    
    Returns:
        str: 生成的蓝图文本，如果生成失败则返回空字符串
    """
    volume_number = volume_info["volume_number"]
    volume_chapter_count = volume_info["end_chapter"] - volume_info["start_chapter"] + 1
    formatted_milestones = "\n".join(volume_info["milestones_list"])
    
    if is_single_shot:
        prompt = chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            volume_summary=volume_info["volume_summary"],
            milestones=formatted_milestones,
            number_of_chapters=volume_chapter_count,
            start_chapter=start_chapter,
            end_chapter=end_chapter,
            user_guidance=user_guidance
        )
    else:
        limited_blueprint = limit_chapter_blueprint(existing_blueprint, 100)
        prompt = chunked_chapter_blueprint_prompt.format(
            # 全局信息
            user_guidance=user_guidance,
            novel_architecture=architecture_text,
            number_of_chapters=volume_chapter_count,

            # 当前卷的特定信息
            current_volume_number=volume_number,
            current_volume_title=volume_info["volume_title"],
            current_volume_summary=volume_info["volume_summary"],
            current_volume_character_arcs=volume_info["character_arcs"],
            current_volume_ending_cliffhanger=volume_info["ending_cliffhanger"],
            current_volume_milestones_formatted=formatted_milestones,

            # 上下文和任务范围
            chapter_list=limited_blueprint,
            n=start_chapter,
            m=end_chapter,
        )

    logging.info(f"正在生成章节 [{start_chapter}..{end_chapter}] 的蓝图...")
    result = invoke_with_cleaning(llm_adapter, prompt)
    if not result.strip():
        logging.warning(f"章节 [{start_chapter}..{end_chapter}] 的蓝图生成为空。")
        return ""
    return result.strip()

def Chapter_blueprint_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    save_path: str,
    user_guidance: str = "",
    target_volume: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 600
) -> None:
    """
    按卷生成章节蓝图的主函数。
    
    功能：
    1. 读取架构文件并解析指定卷的信息
    2. 检查是否已有部分生成的蓝图
    3. 根据章节数量决定生成策略：
       - 章节数 <= chunk_size：一次性生成
       - 章节数 > chunk_size：分块生成
    4. 将生成的蓝图保存到Novel_directory_vol{volume_number}.txt文件
    
    Args:
        interface_format (str): LLM接口格式（如"openai"）
        api_key (str): API密钥
        base_url (str): API基础URL
        llm_model (str): 使用的语言模型名称
        save_path (str): 保存路径
        user_guidance (str, optional): 用户提供的额外指导，默认为空字符串
        target_volume (int, optional): 目标卷号，默认为1
        temperature (float, optional): 生成温度参数，默认为0.7
        max_tokens (int, optional): 最大token限制，默认为4096
        timeout (int, optional): 超时时间（秒），默认为600
    """
    architecture_file_path = os.path.join(save_path, "Novel_architecture.txt")
    if not os.path.exists(architecture_file_path):
        logging.warning(f"架构文件未找到：{architecture_file_path}。请先生成架构。")
        return

    architecture_text = read_file(architecture_file_path).strip()
    if not architecture_text:
        logging.warning(f"架构文件为空：{architecture_file_path}。")
        return

    # 使用 parse_volume_architecture 函数解析卷信息
    volume_info = parse_volume_architecture(architecture_text, target_volume)
    if volume_info is None:
        logging.error(f"在架构文件中未找到第 {target_volume} 卷的信息。")
        return

    volume_number = volume_info["volume_number"]
    start_chapter = volume_info["start_chapter"]
    end_chapter = volume_info["end_chapter"]

    volume_chapter_count = end_chapter - start_chapter + 1
    if volume_chapter_count <= 0:
        logging.warning(f"卷 {volume_number} 的章节范围无效：第 {start_chapter} 章 - 第 {end_chapter} 章。")
        return

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    filename_dir = os.path.join(save_path, f"Novel_directory_vol{volume_number}.txt")
    if not os.path.exists(filename_dir):
        open(filename_dir, "w", encoding="utf-8").close()

    existing_blueprint = read_file(filename_dir).strip()
    chunk_size = compute_chunk_size(volume_chapter_count, max_tokens)
    logging.info(f"卷 {volume_number}: 章节范围 [{start_chapter}..{end_chapter}], 总章节数 = {volume_chapter_count}, 计算的分块大小 = {chunk_size}.")

    if existing_blueprint:
        logging.info(f"检测到卷 {volume_number} 的现有蓝图内容。将从该点继续分块生成。")
        pattern = r"第\s*(\d+)\s*章"
        existing_chapter_numbers = [int(x) for x in re.findall(pattern, existing_blueprint) if x.isdigit()]
        existing_chapter_numbers = [chap for chap in existing_chapter_numbers if start_chapter <= chap <= end_chapter]

        max_existing_chap = max(existing_chapter_numbers) if existing_chapter_numbers else start_chapter - 1
        logging.info(f"卷 {volume_number} 的现有蓝图显示已生成到第 {max_existing_chap} 章。")
        final_blueprint = existing_blueprint
        current_start_abs = max_existing_chap + 1

        if current_start_abs > end_chapter:
            logging.info(f"卷 {volume_number} 的蓝图已完成。")
            return

        while current_start_abs <= end_chapter:
            current_end_abs = min(current_start_abs + chunk_size - 1, end_chapter)
            chunk_result = generate_chapter_blueprint_chunk(
                llm_adapter,
                architecture_text,
                volume_info,
                current_start_abs,
                current_end_abs,
                final_blueprint,
                user_guidance
            )
            
            if not chunk_result:
                clear_file_content(filename_dir)
                save_string_to_txt(final_blueprint.strip(), filename_dir)
                return
                
            final_blueprint += "\n\n" + chunk_result
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            current_start_abs = current_end_abs + 1

        logging.info(f"卷 {volume_number} 的所有章节蓝图已生成完成（分块继续）。")
        return

    # 从头开始生成卷的蓝图
    if chunk_size >= volume_chapter_count:
        blueprint_text = generate_chapter_blueprint_chunk(
            llm_adapter,
            architecture_text,
            volume_info,
            start_chapter,
            end_chapter,
            "",
            user_guidance,
            is_single_shot=True
        )
        
        if not blueprint_text:
            logging.warning(f"卷 {volume_number} 的章节蓝图生成为空。")
            return

        clear_file_content(filename_dir)
        save_string_to_txt(blueprint_text, filename_dir)
        logging.info(f"Novel_directory_vol{volume_number}.txt（章节蓝图）已成功生成（一次性生成）。")
        return

    logging.info(f"将从头开始分块生成卷 {volume_number} 的章节蓝图。")
    final_blueprint = ""
    current_start_abs = start_chapter
    while current_start_abs <= end_chapter:
        current_end_abs = min(current_start_abs + chunk_size - 1, end_chapter)
        chunk_result = generate_chapter_blueprint_chunk(
            llm_adapter,
            architecture_text,
            volume_info,
            current_start_abs,
            current_end_abs,
            final_blueprint,
            user_guidance
        )
        
        if not chunk_result:
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            return
            
        if final_blueprint.strip():
            final_blueprint += "\n\n" + chunk_result
        else:
            final_blueprint = chunk_result
            
        clear_file_content(filename_dir)
        save_string_to_txt(final_blueprint.strip(), filename_dir)
        current_start_abs = current_end_abs + 1

    logging.info(f"Novel_directory_vol{volume_number}.txt（章节蓝图）已成功生成（分块生成）。")
