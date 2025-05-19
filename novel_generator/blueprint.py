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
    基于“每章约100 tokens”的粗略估算，
    再结合当前max_tokens，计算分块大小：
      chunk_size = (floor(max_tokens/100/10)*10) - 10
    并确保 chunk_size 不会小于1或大于实际章节数。
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
    从已有章节目录中只取最近的 limit_chapters 章，以避免 prompt 超长。
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

def parse_volume_architecture(architecture_text: str, volume_number: int) -> tuple:
    """
    从架构文本中解析指定卷的大纲和里程碑。
    返回 (volume_summary, milestones_list, start_chapter, end_chapter)
    """
    # Corrected regex to match the user's architecture format
    volume_pattern = rf"## 第 {volume_number} 卷：.*?预估章节范围：\*\* (\d+) - (\d+).*?(?=## 第 \d+ 卷：|$)"
    volume_match = re.search(volume_pattern, architecture_text, re.DOTALL)

    if not volume_match:
        logging.error(f"Volume {volume_number} not found in architecture file.")
        return None, None, None, None

    volume_content = volume_match.group(0)
    start_chapter = int(volume_match.group(1))
    end_chapter = int(volume_match.group(2))

    summary_pattern = r"### Volume Summary:\s*(.*?)\s*### Key Plot Milestones:"
    summary_match = re.search(summary_pattern, volume_content, re.DOTALL)
    volume_summary = summary_match.group(1).strip() if summary_match else "No summary provided."

    milestones_pattern = r"### Key Plot Milestones:\s*(.*?)(?=###|\Z)"
    milestones_match = re.search(milestones_pattern, volume_content, re.DOTALL)
    milestones_text = milestones_match.group(1).strip() if milestones_match else ""
    milestones_list = [line.strip() for line in milestones_text.split('\n') if line.strip().startswith('-')]

    return volume_summary, milestones_list, start_chapter, end_chapter


def Chapter_blueprint_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    save_path: str, # Added save_path
    user_guidance: str = "",
    target_chapter: int = 100, # Added target_chapter
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 600
) -> None:
    """
    按卷生成章节蓝图。
    读取指定的架构文件，解析包含 target_chapter 的卷的大纲和里程碑，并基于此生成该卷的章节蓝图。
    若对应卷的蓝图文件已存在且内容非空，则表示可能是之前的部分生成结果；
      解析其中已有的章节数，从下一个章节继续分块生成；
      对于已有章节目录，传入时仅保留最近100章目录，避免prompt过长。
    否则：
      - 若该卷章节数 <= chunk_size，直接一次性生成
      - 若该卷章节数 > chunk_size，进行分块生成
    生成完成后输出至 Novel_directory_vol{volume_number}.txt。
    """
    architecture_file_path = os.path.join(save_path, "Novel_architecture.txt")
    if not os.path.exists(architecture_file_path):
        logging.warning(f"Architecture file not found at {architecture_file_path}. Please generate architecture first.")
        return

    architecture_text = read_file(architecture_file_path).strip()
    if not architecture_text:
        logging.warning(f"Architecture file at {architecture_file_path} is empty.")
        return

   # 初始化返回结果的变量
    volume_number = None
    volume_title = "未找到卷标题" # 默认值
    start_chapter = None
    end_chapter = None
    volume_summary = f"目标章节 {target_chapter} 未在任何卷的范围内找到。" # 默认值
    character_arcs = "未能找到主要角色发展弧光。" # 默认值
    ending_cliffhanger = "未能找到本卷结局状态/悬念。" # 默认值
    milestones_list = []

    # --- 方案一：修改后的主正则表达式 ---
    # 捕获组：
    # group(1): 卷号
    # group(2): 卷标题
    # group(3): 起始章节
    # group(4): 结束章节
    volume_pattern_all = r"## 第\s*(\d+)\s*卷：\s*([^\n]+)\s*\n(?:[^\n]*?\n)*?-\s*\*\*(?:预估章节范围|Estimated Chapter Range)：\*\*\s*(\d+)\s*-\s*(\d+).*?(?=## 第\s*\d+\s*卷：|$)"

    for volume_match in re.finditer(volume_pattern_all, architecture_text, re.DOTALL | re.IGNORECASE):
        try:
            # --- 方案一：对应修改后的捕获组索引 ---
            current_volume_number = int(volume_match.group(1))
            current_volume_title = volume_match.group(2).strip()
            current_start_chapter = int(volume_match.group(3))
            current_end_chapter = int(volume_match.group(4))
        except (IndexError, ValueError) as e:
            print(f"警告：解析卷基本信息（卷号、标题、章节范围）时出错，跳过此卷。错误：{e}")
            print(f"问题匹配内容片段：{volume_match.group(0)[:200]}...") # 打印更长的片段帮助调试
            continue # 跳过这个格式不正确的卷

        if current_start_chapter <= target_chapter <= current_end_chapter:
            volume_number = current_volume_number
            volume_title = current_volume_title # 使用从主正则捕获的标题
            start_chapter = current_start_chapter
            end_chapter = current_end_chapter

            # 获取整个匹配到的卷的文本内容，用于提取内部细节
            volume_content = volume_match.group(0)

            # 提取核心情节概述
            summary_pattern = r"-\s*\*\*(本卷核心情节概述|Core Plot Summary)：\*\*\s*(.*?)(?=\n-\s*\*\*|\Z)"
            summary_match = re.search(summary_pattern, volume_content, re.DOTALL | re.IGNORECASE)
            volume_summary = summary_match.group(2).strip() if summary_match else "未能找到本卷的核心情节概述。"

            # 提取主要角色发展弧光
            arcs_pattern = r"-\s*\*\*(主要角色发展弧光|Key Character Arcs)：\*\*\s*(.*?)(?=\n-\s*\*\*|\Z)"
            arcs_match = re.search(arcs_pattern, volume_content, re.DOTALL | re.IGNORECASE)
            character_arcs = arcs_match.group(2).strip() if arcs_match else "未能找到主要角色发展弧光。"

            # 提取本卷结局状态/悬念
            cliffhanger_pattern = r"-\s*\*\*(本卷结局状态/悬念|Ending State / Cliffhanger)：\*\*\s*(.*?)(?=\n###|\Z)"
            cliffhanger_match = re.search(cliffhanger_pattern, volume_content, re.DOTALL | re.IGNORECASE)
            ending_cliffhanger = cliffhanger_match.group(2).strip() if cliffhanger_match else "未能找到本卷结局状态/悬念。"

            # 提取关键情节里程碑
            milestones_pattern = r"###\s*(关键情节里程碑|Key Plot Milestones)\s*\n((?:-\s+[^\n]*(?:\n|$))+)"
            milestones_match = re.search(milestones_pattern, volume_content, re.DOTALL | re.IGNORECASE)
            if milestones_match:
                milestones_text = milestones_match.group(2).strip()
                milestones_list = [
                    line.strip().lstrip('-').strip()
                    for line in milestones_text.split('\n')
                    if line.strip().startswith('-')
                ]
            else:
                milestones_list = []

            break # 找到了正确的卷，不需要继续迭代

    # 返回包含所有提取信息的字典
    # (注意，这里你之前的代码有一个 print 语句，我将其改为 return 语句，
    # 因为函数通常应该返回数据而不是直接打印，除非打印是其主要目的)
    print({
       "volume_number": volume_number,
        "volume_title": volume_title,
        "start_chapter": start_chapter,
        "end_chapter": end_chapter,
        "volume_summary": volume_summary,
        "character_arcs": character_arcs,
        "ending_cliffhanger": ending_cliffhanger,
        "milestones_list": milestones_list
    }) 

    if volume_number is None:
        logging.error(f"Could not find a volume containing chapter {target_chapter} in the architecture file.")
        return

    volume_chapter_count = end_chapter - start_chapter + 1
    if volume_chapter_count <= 0:
        logging.warning(f"Volume {volume_number} has an invalid chapter range: Chapter {start_chapter} - Chapter {end_chapter}.")
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
    logging.info(f"Volume {volume_number}: Chapter range [{start_chapter}..{end_chapter}], total chapters = {volume_chapter_count}, computed chunk_size = {chunk_size}.")

    # Format milestones for prompt
    formatted_milestones = "\n".join(milestones_list)

    if existing_blueprint:
        logging.info(f"Detected existing blueprint content for Volume {volume_number}. Will resume chunked generation from that point.")
        pattern = r"第\s*(\d+)\s*章"
        existing_chapter_numbers = re.findall(pattern, existing_blueprint)
        existing_chapter_numbers = [int(x) for x in existing_chapter_numbers if x.isdigit()]
        # Filter existing chapters to be within the current volume's range
        existing_chapter_numbers = [chap for chap in existing_chapter_numbers if start_chapter <= chap <= end_chapter]

        max_existing_chap = max(existing_chapter_numbers) if existing_chapter_numbers else start_chapter - 1
        logging.info(f"Existing blueprint for Volume {volume_number} indicates up to chapter {max_existing_chap} has been generated within this volume's range.")
        final_blueprint = existing_blueprint
        current_start_abs = max_existing_chap + 1

        if current_start_abs > end_chapter:
             logging.info(f"Volume {volume_number} blueprint already completed.")
             return

        while current_start_abs <= end_chapter:
            current_end_abs = min(current_start_abs + chunk_size - 1, end_chapter)

            limited_blueprint = limit_chapter_blueprint(final_blueprint, 100) # Still limit overall blueprint for context

            chunk_prompt = chunked_chapter_blueprint_prompt.format(

                # 全局信息
                user_guidance=user_guidance,
                novel_architecture=architecture_text, # 完整的小说架构
                number_of_chapters=volume_chapter_count, # 【重要】卷的总章节数

                # 当前卷的特定信息
                current_volume_number=volume_number,
                current_volume_title=volume_title,
                current_volume_summary=volume_summary,
                current_volume_character_arcs=character_arcs,
                current_volume_ending_cliffhanger=ending_cliffhanger,
                current_volume_milestones_formatted=formatted_milestones, # 确保与Prompt占位符一致

                # 上下文和任务范围
                chapter_list=limited_blueprint,
                n=current_start_abs,  # 当前块的全局起始章节号
                m=current_end_abs,    # 当前块的全局结束章节号
                
            )
            logging.info(f"Generating chapters2 [{current_start_abs}..{current_end_abs}] for Volume {volume_number} in a chunk...")
            chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
            if not chunk_result.strip():
                logging.warning(f"Chunk generation for chapters [{current_start_abs}..{current_end_abs}] is empty.")
                # If a chunk is empty, we should probably stop and save what we have
                clear_file_content(filename_dir)
                save_string_to_txt(final_blueprint.strip(), filename_dir)
                return
            final_blueprint += "\n\n" + chunk_result.strip()
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            current_start_abs = current_end_abs + 1

        logging.info(f"All chapters blueprint for Volume {volume_number} have been generated (resumed chunked).")
        return

    # Generate from scratch for the volume
    if chunk_size >= volume_chapter_count:
        prompt = chapter_blueprint_prompt.format(
            novel_architecture=architecture_text, # Keep full architecture?
            volume_summary=volume_summary, # Pass volume summary
            milestones=formatted_milestones, # Pass formatted milestones
            number_of_chapters=volume_chapter_count, # Total chapters in this volume
            start_chapter=start_chapter, # Pass starting chapter number
            end_chapter=end_chapter, # Pass ending chapter number
            user_guidance=user_guidance
        )
        blueprint_text = invoke_with_cleaning(llm_adapter, prompt)
        if not blueprint_text.strip():
            logging.warning(f"Chapter blueprint generation result for Volume {volume_number} is empty.")
            return

        clear_file_content(filename_dir)
        save_string_to_txt(blueprint_text, filename_dir)
        logging.info(f"Novel_directory_vol{volume_number}.txt (chapter blueprint) has been generated successfully (single-shot).")
        return

    logging.info(f"Will generate chapter blueprint for Volume {volume_number} in chunked mode from scratch.")
    final_blueprint = ""
    current_start_abs = start_chapter # Start from the volume's starting chapter
    while current_start_abs <= end_chapter:
        current_end_abs = min(current_start_abs + chunk_size - 1, end_chapter)

        limited_blueprint = limit_chapter_blueprint(final_blueprint, 100) # Still limit overall blueprint for context

        chunk_prompt = chunked_chapter_blueprint_prompt.format(

            # 全局信息
                user_guidance=user_guidance,
                novel_architecture=architecture_text, # 完整的小说架构
                number_of_chapters=volume_chapter_count, # 【重要】卷的总章节数

                # 当前卷的特定信息
                current_volume_number=volume_number,
                current_volume_title=volume_title,
                current_volume_summary=volume_summary,
                current_volume_character_arcs=character_arcs,
                current_volume_ending_cliffhanger=ending_cliffhanger,
                current_volume_milestones_formatted=formatted_milestones, # 确保与Prompt占位符一致

                # 上下文和任务范围
                chapter_list=limited_blueprint,
                n=current_start_abs,  # 当前块的全局起始章节号
                m=current_end_abs,    # 当前块的全局结束章节号
    
        )
        logging.info(f"Generating chapters1 [{current_start_abs}..{current_end_abs}] for Volume {volume_number} in a chunk...")
        chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
        if not chunk_result.strip():
            logging.warning(f"Chunk generation for chapters [{current_start_abs}..{current_end_abs}] is empty.")
            clear_file_content(filename_dir)
            save_string_to_txt(final_blueprint.strip(), filename_dir)
            return
        if final_blueprint.strip():
            final_blueprint += "\n\n" + chunk_result.strip()
        else:
            final_blueprint = chunk_result.strip()
        clear_file_content(filename_dir)
        save_string_to_txt(final_blueprint.strip(), filename_dir)
        current_start_abs = current_end_abs + 1

    logging.info(f"Novel_directory_vol{volume_number}.txt (chapter blueprint) has been generated successfully (chunked).")
