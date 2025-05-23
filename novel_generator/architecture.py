#novel_generator/architecture.py
# -*- coding: utf-8 -*-
"""
小说总体架构生成（Novel_architecture_generate 及相关辅助函数）
"""
import os
import json
import logging
import traceback
import time
from novel_generator.common import invoke_with_cleaning
from tqdm import tqdm
from llm_adapters import create_llm_adapter
from prompt_definitions import (
    core_seed_prompt,
    character_dynamics_prompt,
    world_building_prompt,
    plot_architecture_prompt,
    create_character_state_prompt
)
from utils import clear_file_content, save_string_to_txt

def load_partial_architecture_data(filepath: str) -> dict:
    """
    从 filepath 下的 partial_architecture.json 读取已有的阶段性数据。
    如果文件不存在或无法解析，返回空 dict。
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    if not os.path.exists(partial_file):
        return {}
    try:
        with open(partial_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.warning(f"Failed to load partial_architecture.json: {e}")
        return {}

def save_partial_architecture_data(filepath: str, data: dict):
    """
    将阶段性数据写入 partial_architecture.json。
    """
    partial_file = os.path.join(filepath, "partial_architecture.json")
    try:
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save partial_architecture.json: {e}")

def Novel_architecture_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    user_guidance: str = "",  # 新增参数
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 600
) -> dict:  # 修改返回类型为 dict
    """
    依次调用:
      1. core_seed_prompt
      2. character_dynamics_prompt
      3. world_building_prompt
      4. plot_architecture_prompt
    若在中间任何一步报错且重试多次失败，则将已经生成的内容写入 partial_architecture.json 并退出；
    下次调用时可从该步骤继续。
    最终输出 Novel_architecture.txt

    新增：
    - 在完成角色动力学设定后，依据该角色体系，使用 create_character_state_prompt 生成初始角色状态表，
      并存储到 character_state.txt，后续维护更新。
    """
    os.makedirs(filepath, exist_ok=True)
    partial_data = load_partial_architecture_data(filepath)
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    
    CHAPTERS_PER_VOLUME = 100 # Define chapters per volume
    # Assuming target_total_chapters is available as number_of_chapters
    num_volumes = (number_of_chapters + CHAPTERS_PER_VOLUME - 1) // CHAPTERS_PER_VOLUME

    # Total steps: Core Seed, Character Dynamics, Initial Character State, World Building, Volume Outline
    total_steps = 5 # Reverted total steps back to 5

    with tqdm(total=total_steps, desc="Generating Novel Architecture") as pbar:
        # Step1: 核心种子
        if "core_seed_result" not in partial_data:
            pbar.set_description("Generating Core Seed")
            logging.info("Step1: Generating core_seed_prompt (核心种子) ...")
            prompt_core = core_seed_prompt.format(
                topic=topic,
                genre=genre,
                number_of_chapters=number_of_chapters,
                word_number=word_number,
                user_guidance=user_guidance  # 修复：添加内容指导
            )
            cs_llm_start = time.time()
            core_seed_result = invoke_with_cleaning(llm_adapter, prompt_core)
            cs_llm_end = time.time()
            duration = cs_llm_end - cs_llm_start
            print(f"    [DEBUG] Core Seed LLM call duration: {duration:.2f} seconds.")
            if not core_seed_result.strip():
                logging.warning("core_seed_prompt generation failed and returned empty.")
                save_partial_architecture_data(filepath, partial_data)
                return
            partial_data["core_seed_result"] = core_seed_result
            save_partial_architecture_data(filepath, partial_data)
            pbar.update(1)
        else:
            logging.info("Step1 already done. Skipping...")
            pbar.update(1) # Update even if skipped
            
        # Step2: 角色动力学
        if "character_dynamics_result" not in partial_data:
            pbar.set_description("Generating Character Dynamics")
            logging.info("Step2: Generating character_dynamics_prompt ...")
            prompt_character = character_dynamics_prompt.format(
                core_seed=partial_data["core_seed_result"].strip(),
                user_guidance=user_guidance
            )
            cd_llm_start = time.time()
            character_dynamics_result = invoke_with_cleaning(llm_adapter, prompt_character)
            cd_llm_end = time.time()
            duration = cd_llm_end - cd_llm_start
            print(f"    [DEBUG] Character Dynamics LLM call duration: {duration:.2f} seconds.")
            if not character_dynamics_result.strip():
                logging.warning("character_dynamics_prompt generation failed.")
                save_partial_architecture_data(filepath, partial_data)
                return
            partial_data["character_dynamics_result"] = character_dynamics_result
            save_partial_architecture_data(filepath, partial_data)
            pbar.update(1)
        else:
            logging.info("Step2 already done. Skipping...")
            pbar.update(1) # Update even if skipped
            
        # 生成初始角色状态
        if "character_dynamics_result" in partial_data and "character_state_result" not in partial_data:
            pbar.set_description("Generating Initial Character State")
            logging.info("Generating initial character state from character dynamics ...")
            prompt_char_state_init = create_character_state_prompt.format(
                character_dynamics=partial_data["character_dynamics_result"].strip()
            )
            ics_llm_start = time.time()
            character_state_init = invoke_with_cleaning(llm_adapter, prompt_char_state_init)
            ics_llm_end = time.time()
            duration = ics_llm_end - ics_llm_start
            print(f"    [DEBUG] Initial Character State LLM call duration: {duration:.2f} seconds.")
            if not character_state_init.strip():
                logging.warning("create_character_state_prompt generation failed.")
                save_partial_architecture_data(filepath, partial_data)
                return
            partial_data["character_state_result"] = character_state_init
            character_state_file = os.path.join(filepath, "character_state.txt")
            clear_file_content(character_state_file)
            save_string_to_txt(character_state_init, character_state_file)
            save_partial_architecture_data(filepath, partial_data)
            logging.info("Initial character state created and saved.")
            pbar.update(1)
        elif "character_dynamics_result" in partial_data and "character_state_result" in partial_data:
             logging.info("Initial character state already done. Skipping...")
             pbar.update(1) # Update even if skipped
        else:
             # This case should ideally not happen if Step 2 is done, but handle for completeness
             logging.warning("Character dynamics not available to generate initial state. Skipping initial state generation.")
             pbar.update(1) # Still update to keep progress bar accurate

        # Step3: 世界观
        if "world_building_result" not in partial_data:
            pbar.set_description("Generating World Building")
            logging.info("Step3: Generating world_building_prompt ...")
            prompt_world = world_building_prompt.format(
                core_seed=partial_data["core_seed_result"].strip(),
                user_guidance=user_guidance  # 修复：添加用户指导
            )
            wb_llm_start = time.time()
            world_building_result = invoke_with_cleaning(llm_adapter, prompt_world)
            wb_llm_end = time.time()
            duration = wb_llm_end - wb_llm_start
            print(f"    [DEBUG] World Building LLM call duration: {duration:.2f} seconds.")
            if not world_building_result.strip():
                logging.warning("world_building_prompt generation failed.")
                save_partial_architecture_data(filepath, partial_data)
                return
            partial_data["world_building_result"] = world_building_result
            save_partial_architecture_data(filepath, partial_data)
            pbar.update(1)
        else:
            logging.info("Step3 already done. Skipping...")
            pbar.update(1) # Update even if skipped
            
        # Step4: 生成分卷大纲
        if "volume_outline_result" not in partial_data:
            pbar.set_description("Generating Volume Outline")
            logging.info("Step4: Generating volume outline ...")
            # Define the prompt string directly for now
            volume_outline_prompt_str = plot_architecture_prompt
            prompt_volume_outline = volume_outline_prompt_str.format(
                num_volumes=num_volumes,
                chapters_per_volume=CHAPTERS_PER_VOLUME,
                core_seed=partial_data["core_seed_result"].strip(),
                character_dynamics=partial_data["character_dynamics_result"].strip(),
                world_building=partial_data["world_building_result"].strip(),
                user_guidance=user_guidance
            )
            vo_llm_start = time.time()
            volume_outline_result = invoke_with_cleaning(llm_adapter, prompt_volume_outline)
            vo_llm_end = time.time()
            duration = vo_llm_end - vo_llm_start
            print("volume_outline_result",volume_outline_result)
            if not volume_outline_result.strip():
                logging.warning("volume outline generation failed.")
                save_partial_architecture_data(filepath, partial_data)
                return
            partial_data["volume_outline_result"] = volume_outline_result
            save_partial_architecture_data(filepath, partial_data)
            pbar.update(1)
        else:
            logging.info("Step4 already done. Skipping...")
            pbar.update(1) # Update even if skipped

       

    core_seed_result = partial_data["core_seed_result"]
    character_dynamics_result = partial_data["character_dynamics_result"]
    world_building_result = partial_data["world_building_result"]
    # plot_arch_result = partial_data["plot_arch_result"] # Removed
    volume_outline_result = partial_data["volume_outline_result"] # Added

    save_final_start = time.time()
    final_content = (
        "#=== 0) 小说设定 ===\n"
        f"主题：{topic},类型：{genre},篇幅：约{num_volumes}卷 {number_of_chapters}章（每章{word_number}字）\n\n"
        "#=== 1) 核心种子 ===\n"
        f"{core_seed_result}\n\n"
        "#=== 2) 角色动力学 ===\n"
        f"{character_dynamics_result}\n\n"
        "#=== 3) 世界观 ===\n"
        f"{world_building_result}\n\n"
        "#=== 4) 分卷大纲 ===\n" 
        f"{volume_outline_result}\n"
    )

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    clear_file_content(arch_file)
    save_string_to_txt(final_content, arch_file)
    save_final_end = time.time()
    save_final_duration = save_final_end - save_final_start
    print(f"    [DEBUG] Saving Novel_architecture.txt duration: {save_final_duration:.2f} seconds.")
    logging.info("Novel_architecture.txt has been generated successfully.")

    partial_arch_file = os.path.join(filepath, "partial_architecture.json")
    if os.path.exists(partial_arch_file):
        os.remove(partial_arch_file)
        logging.info("partial_architecture.json removed (all steps completed).")

    # 返回生成的数据
    return {
        "topic": topic,
        "genre": genre,
        "num_volumes": num_volumes,
        "number_of_chapters": number_of_chapters,
        "word_number": word_number,
        "core_seed": core_seed_result,
        "character_dynamics": character_dynamics_result,
        "world_building": world_building_result,
        "volume_outline": volume_outline_result
    }
