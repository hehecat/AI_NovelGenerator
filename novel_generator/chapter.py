# novel_generator/chapter.py
# -*- coding: utf-8 -*-
"""
章节草稿生成及获取历史章节文本、当前章节摘要等
"""
import os
import json
import logging
import re  # 添加re模块导入
from llm_adapters import create_llm_adapter
from prompt_definitions import (
    first_chapter_draft_prompt, 
    next_chapter_draft_prompt, 
    summarize_recent_chapters_prompt,
    knowledge_filter_prompt,
    knowledge_search_prompt
)
from chapter_directory_parser import get_chapter_info_from_blueprint
from novel_generator.common import invoke_with_cleaning
from utils import read_file, clear_file_content, save_string_to_txt
from novel_generator.vectorstore_utils import (
    get_relevant_context_from_vector_store,
    load_vector_store  # 添加导入
)

def get_last_n_chapters_text(chapters_dir: str, current_chapter_num: int, n: int = 3) -> list:
    """
    获取最近n章的文本内容。
    
    从指定目录中读取当前章节之前的n章内容，用于生成新章节时提供上下文。
    
    Args:
        chapters_dir (str): 章节文件所在的目录路径
        current_chapter_num (int): 当前章节号
        n (int, optional): 需要获取的最近章节数，默认为3
        
    Returns:
        list: 包含最近n章文本内容的列表，如果某章不存在则对应位置为空字符串
    """
    texts = []
    start_chap = max(1, current_chapter_num - n)
    for c in range(start_chap, current_chapter_num):
        chap_file = os.path.join(chapters_dir, f"chapter_{c}.txt")
        if os.path.exists(chap_file):
            text = read_file(chap_file).strip()
            texts.append(text)
        else:
            texts.append("")
    return texts

def summarize_recent_chapters(
    interface_format: str,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    chapters_text_list: list,
    novel_number: int,
    chapter_info: dict,
    next_chapter_info: dict,
    timeout: int = 600
) -> str:
    """
    根据最近章节内容生成当前章节的精准摘要。
    
    使用LLM模型分析最近章节的内容，生成一个简洁的摘要，用于指导新章节的生成。
    如果生成失败，则返回空字符串。
    
    Args:
        interface_format (str): LLM接口格式（如"openai"）
        api_key (str): API密钥
        base_url (str): API基础URL
        model_name (str): 使用的语言模型名称
        temperature (float): 生成温度参数
        max_tokens (int): 最大token限制
        chapters_text_list (list): 最近章节的文本内容列表
        novel_number (int): 当前章节号
        chapter_info (dict): 当前章节的信息字典
        next_chapter_info (dict): 下一章节的信息字典
        timeout (int, optional): 超时时间（秒），默认为600
        
    Returns:
        str: 生成的章节摘要，如果生成失败则返回空字符串
    """
    try:
        combined_text = "\n".join(chapters_text_list).strip()
        if not combined_text:
            return ""
            
        # 限制组合文本长度
        max_combined_length = 4000
        if len(combined_text) > max_combined_length:
            combined_text = combined_text[-max_combined_length:]
            
        llm_adapter = create_llm_adapter(
            interface_format=interface_format,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        # 确保所有参数都有默认值
        chapter_info = chapter_info or {}
        next_chapter_info = next_chapter_info or {}
        
        prompt = summarize_recent_chapters_prompt.format(
            combined_text=combined_text,
            novel_number=novel_number,
            chapter_title=chapter_info.get("chapter_title", "未命名"),
            chapter_role=chapter_info.get("chapter_role", "常规章节"),
            chapter_purpose=chapter_info.get("chapter_purpose", "内容推进"),
            suspense_level=chapter_info.get("suspense_level", "中等"),
            foreshadowing=chapter_info.get("foreshadowing", "无"),
            plot_twist_level=chapter_info.get("plot_twist_level", "★☆☆☆☆"),
            chapter_summary=chapter_info.get("chapter_summary", ""),
            next_chapter_number=novel_number + 1,
            next_chapter_title=next_chapter_info.get("chapter_title", "（未命名）"),
            next_chapter_role=next_chapter_info.get("chapter_role", "过渡章节"),
            next_chapter_purpose=next_chapter_info.get("chapter_purpose", "承上启下"),
            next_chapter_summary=next_chapter_info.get("chapter_summary", "衔接过渡内容"),
            next_chapter_suspense_level=next_chapter_info.get("suspense_level", "中等"),
            next_chapter_foreshadowing=next_chapter_info.get("foreshadowing", "无特殊伏笔"),
            next_chapter_plot_twist_level=next_chapter_info.get("plot_twist_level", "★☆☆☆☆")
        )
        
        response_text = invoke_with_cleaning(llm_adapter, prompt)
        summary = extract_summary_from_response(response_text)
        
        if not summary:
            logging.warning("Failed to extract summary, using full response")
            return response_text[:2000]  # 限制长度
            
        return summary[:2000]  # 限制摘要长度
        
    except Exception as e:
        logging.error(f"Error in summarize_recent_chapters: {str(e)}")
        return ""

def extract_summary_from_response(response_text: str) -> str:
    """
    从LLM响应文本中提取摘要部分。
    
    通过识别特定的摘要标记来提取摘要内容。如果找不到标记，
    则返回整个响应文本。
    
    Args:
        response_text (str): LLM的响应文本
        
    Returns:
        str: 提取出的摘要文本，如果提取失败则返回原文本
    """
    if not response_text:
        return ""
        
    # 查找摘要标记
    summary_markers = [
        "当前章节摘要:", 
        "章节摘要:",
        "摘要:",
        "本章摘要:"
    ]
    
    for marker in summary_markers:
        if (marker in response_text):
            parts = response_text.split(marker, 1)
            if len(parts) > 1:
                return parts[1].strip()
    
    return response_text.strip()

def format_chapter_info(chapter_info: dict) -> str:
    """
    将章节信息字典格式化为易读的文本格式。
    
    将包含章节各种信息的字典转换为结构化的文本格式，
    包括章节编号、标题、定位、核心作用等信息。
    
    Args:
        chapter_info (dict): 包含章节信息的字典
        
    Returns:
        str: 格式化后的章节信息文本
    """
    template = """
章节编号：第{number}章
章节标题：《{title}》
章节定位：{role}
核心作用：{purpose}
主要人物：{characters}
关键道具：{items}
场景地点：{location}
伏笔设计：{foreshadow}
悬念密度：{suspense}
转折程度：{twist}
章节简述：{summary}
"""
    return template.format(
        number=chapter_info.get('chapter_number', '未知'),
        title=chapter_info.get('chapter_title', '未知'),
        role=chapter_info.get('chapter_role', '未知'),
        purpose=chapter_info.get('chapter_purpose', '未知'),
        characters=chapter_info.get('characters_involved', '未指定'),
        items=chapter_info.get('key_items', '未指定'),
        location=chapter_info.get('scene_location', '未指定'),
        foreshadow=chapter_info.get('foreshadowing', '无'),
        suspense=chapter_info.get('suspense_level', '一般'),
        twist=chapter_info.get('plot_twist_level', '★☆☆☆☆'),
        summary=chapter_info.get('chapter_summary', '未提供')
    )

def parse_search_keywords(response_text: str) -> list:
    """
    解析新版关键词格式。
    
    从响应文本中提取关键词组，每组关键词由"·"分隔。
    例如：'科技公司·数据泄露\n地下实验室·基因编辑'
    
    Args:
        response_text (str): 包含关键词的响应文本
        
    Returns:
        list: 最多5组关键词的列表，每组关键词中的"·"被替换为空格
    """
    return [
        line.strip().replace('·', ' ')
        for line in response_text.strip().split('\n')
        if '·' in line
    ][:5]  # 最多取5组

def apply_content_rules(texts: list, novel_number: int) -> list:
    """
    应用内容处理规则。
    
    根据文本内容与当前章节的时间距离，应用不同的处理规则：
    - 距离≤2章：跳过内容
    - 距离3-5章：需要修改≥40%
    - 距离>5章：可以引用核心内容
    - 非章节内容：优先使用
    
    Args:
        texts (list): 待处理的文本列表
        novel_number (int): 当前章节号
        
    Returns:
        list: 处理后的文本列表，每个文本前添加处理标记
    """
    processed = []
    for text in texts:
        if re.search(r'第[\d]+章', text) or re.search(r'chapter_[\d]+', text):
            chap_nums = list(map(int, re.findall(r'\d+', text)))
            recent_chap = max(chap_nums) if chap_nums else 0
            time_distance = novel_number - recent_chap
            
            if time_distance <= 2:
                processed.append(f"[SKIP] 跳过近章内容：{text[:120]}...")
            elif 3 <= time_distance <= 5:
                processed.append(f"[MOD40%] {text}（需修改≥40%）")
            else:
                processed.append(f"[OK] {text}（可引用核心）")
        else:
            processed.append(f"[PRIOR] {text}（优先使用）")
    return processed

def is_foreshadowing_content(text: str, chapter_info: dict) -> bool:
    """
    轻量级判断内容是否是伏笔或需要呼应的情节
    
    Args:
        text: 待判断的文本内容
        chapter_info: 当前章节信息
    """
    # 1. 特征提取
    features = {
        # 情节特征
        "plot_features": {
            "has_question": bool(re.search(r'[？?]', text)),  # 问句
            "has_unknown": bool(re.search(r'[不未]知|神秘|奇怪|疑惑', text)),  # 未知描述
            "has_future": bool(re.search(r'[将可能或许]|以后|未来|之后', text)),  # 未来暗示
            "has_plot_hook": bool(re.search(r'[但然而却]|不过|可是', text)),  # 情节钩子
        },
        
        # 角色特征
        "character_features": {
            "has_new_character": bool(re.search(r'[一]个.*[人者]|.*[出现遇到]', text)),  # 新角色
            "has_character_hint": bool(re.search(r'[眼神表情]中|似乎|好像|仿佛', text)),  # 角色暗示
            "has_character_background": bool(re.search(r'[身份背景]|来历|过去|曾经', text)),  # 角色背景
        },
        
        # 道具特征
        "item_features": {
            "has_new_item": bool(re.search(r'[一]个.*[物品道具]|.*[发现获得]', text)),  # 新道具
            "has_item_hint": bool(re.search(r'[特殊奇怪]|不同寻常|不一般', text)),  # 道具暗示
            "has_item_effect": bool(re.search(r'[效果作用]|能力|力量', text)),  # 道具效果
        }
    }
    
    # 2. 规则判断
    def check_plot_foreshadowing(features: dict) -> bool:
        """检查情节伏笔"""
        plot = features["plot_features"]
        return (plot["has_question"] and plot["has_unknown"]) or \
               (plot["has_future"] and plot["has_plot_hook"])
    
    def check_character_foreshadowing(features: dict) -> bool:
        """检查角色伏笔"""
        char = features["character_features"]
        return (char["has_new_character"] and char["has_character_hint"]) or \
               (char["has_new_character"] and char["has_character_background"])
    
    def check_item_foreshadowing(features: dict) -> bool:
        """检查道具伏笔"""
        item = features["item_features"]
        return (item["has_new_item"] and item["has_item_hint"]) or \
               (item["has_new_item"] and item["has_item_effect"])
    
    # 3. 与当前章节伏笔设计的相关性检查
    def check_chapter_relevance(text: str, chapter_info: dict) -> bool:
        """检查与当前章节伏笔设计的相关性"""
        if not chapter_info.get("foreshadowing"):
            return False
        keywords = set(chapter_info["foreshadowing"].split())
        return any(keyword in text for keyword in keywords)
    
    # 4. 主判断逻辑
    try:
        # 检查各类伏笔
        is_plot_foreshadowing = check_plot_foreshadowing(features)
        is_character_foreshadowing = check_character_foreshadowing(features)
        is_item_foreshadowing = check_item_foreshadowing(features)
        
        # 检查与当前章节的相关性
        is_chapter_relevant = check_chapter_relevance(text, chapter_info)
        
        # 综合判断
        is_foreshadowing = is_plot_foreshadowing or \
                          is_character_foreshadowing or \
                          is_item_foreshadowing or \
                          is_chapter_relevant
        
        # 记录判断结果
        if is_foreshadowing:
            logging.debug(f"检测到伏笔内容：{text[:50]}...")
            logging.debug(f"特征分析：{features}")
        
        return is_foreshadowing
        
    except Exception as e:
        logging.error(f"伏笔判断出错：{str(e)}")
        return False

def apply_knowledge_rules(contexts: list, chapter_num: int, chapter_info: dict) -> list:
    """
    应用知识库使用规则。
    
    根据文本内容类型和与当前章节的时间距离，应用不同的处理规则：
    - 历史章节内容：
      * 距离≤3章：跳过
      * 距离>3章：需要30%以上改写
    - 伏笔内容：优先使用
    - 第三方知识：优先处理
    
    Args:
        contexts (list): 待处理的上下文列表
        chapter_num (int): 当前章节号
        chapter_info (dict): 当前章节信息
        
    Returns:
        list: 处理后的上下文列表，每个文本前添加处理标记
    """
    processed = []
    for text in contexts:
        # 1. 检查是否是伏笔内容
        if is_foreshadowing_content(text, chapter_info):
            processed.append(f"[伏笔呼应] {text} (优先使用)")
            continue
            
        # 2. 检查是否是历史章节内容
        if "第" in text and "章" in text:
            # 提取章节号判断时间远近
            chap_nums = [int(s) for s in text.split() if s.isdigit()]
            recent_chap = max(chap_nums) if chap_nums else 0
            time_distance = chapter_num - recent_chap
            
            if time_distance <= 3:  # 近三章内容
                processed.append(f"[历史章节限制] 跳过近期内容: {text[:50]}...")
            else:
                # 允许引用但需要转换
                processed.append(f"[历史参考] {text} (需进行30%以上改写)")
        else:
            # 第三方知识优先处理
            processed.append(f"[外部知识] {text}")
    return processed

def get_filtered_knowledge_context(
    api_key: str,
    base_url: str,
    model_name: str,
    interface_format: str,
    embedding_adapter,
    filepath: str,
    chapter_info: dict,
    retrieved_texts: list,
    max_tokens: int = 2048,
    timeout: int = 600
) -> str:
    """
    优化后的知识过滤处理。
    
    对检索到的知识内容进行过滤和处理，确保内容与当前章节相关且合适。
    处理步骤：
    1. 应用知识规则处理检索文本
    2. 限制文本长度
    3. 使用LLM进行内容过滤
    
    Args:
        api_key (str): API密钥
        base_url (str): API基础URL
        model_name (str): 使用的语言模型名称
        interface_format (str): LLM接口格式
        embedding_adapter: 向量嵌入适配器
        filepath (str): 文件路径
        chapter_info (dict): 当前章节信息
        retrieved_texts (list): 检索到的文本列表
        max_tokens (int, optional): 最大token限制，默认为2048
        timeout (int, optional): 超时时间（秒），默认为600
        
    Returns:
        str: 过滤后的知识内容，如果处理失败则返回错误提示
    """
    if not retrieved_texts:
        return "（无相关知识库内容）"

    try:
        # 应用知识规则
        processed_texts = apply_knowledge_rules(
            retrieved_texts, 
            chapter_info.get('chapter_number', 0),
            chapter_info
        )
        
        # 限制文本长度并格式化
        formatted_texts = []
        max_text_length = 600
        for i, text in enumerate(processed_texts, 1):
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            formatted_texts.append(f"[预处理结果{i}]\n{text}")

        # 使用格式化函数处理章节信息
        formatted_chapter_info = (
            f"当前章节定位：{chapter_info.get('chapter_role', '')}\n"
            f"核心目标：{chapter_info.get('chapter_purpose', '')}\n"
            f"关键要素：{chapter_info.get('characters_involved', '')} | "
            f"{chapter_info.get('key_items', '')} | "
            f"{chapter_info.get('scene_location', '')}"
        )

        # 构造提示词
        prompt = knowledge_filter_prompt.format(
            chapter_info=formatted_chapter_info,
            retrieved_texts="\n\n".join(formatted_texts) if formatted_texts else "（无检索结果）"
        )
        
        # 调用LLM进行过滤
        llm_adapter = create_llm_adapter(
            interface_format=interface_format,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        filtered_content = invoke_with_cleaning(llm_adapter, prompt)
        return filtered_content if filtered_content else "（知识内容过滤失败）"
        
    except Exception as e:
        logging.error(f"知识过滤过程出错：{str(e)}")
        return "（内容过滤过程出错）"

def build_chapter_prompt(
    api_key: str,
    base_url: str,
    model_name: str,
    filepath: str,
    novel_number: int,
    word_number: int,
    temperature: float,
    user_guidance: str,
    characters_involved: str,
    key_items: str,
    scene_location: str,
    time_constraint: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    embedding_retrieval_k: int = 2,
    interface_format: str = "openai",
    max_tokens: int = 2048,
    timeout: int = 600
) -> str:
    """
    构造当前章节的请求提示词。
    
    功能：
    1. 读取基础文件（架构、目录、全局摘要、角色状态）
    2. 获取当前章节和下一章节的信息
    3. 获取前文内容和摘要
    4. 进行知识库检索和处理
    5. 构造完整的提示词
    
    Args:
        api_key (str): API密钥
        base_url (str): API基础URL
        model_name (str): 使用的语言模型名称
        filepath (str): 文件路径
        novel_number (int): 当前章节号
        word_number (int): 目标字数
        temperature (float): 生成温度参数
        user_guidance (str): 用户指导
        characters_involved (str): 涉及的角色
        key_items (str): 关键道具
        scene_location (str): 场景地点
        time_constraint (str): 时间约束
        embedding_api_key (str): 向量嵌入API密钥
        embedding_url (str): 向量嵌入API URL
        embedding_interface_format (str): 向量嵌入接口格式
        embedding_model_name (str): 向量嵌入模型名称
        embedding_retrieval_k (int, optional): 检索数量，默认为2
        interface_format (str, optional): LLM接口格式，默认为"openai"
        max_tokens (int, optional): 最大token限制，默认为2048
        timeout (int, optional): 超时时间（秒），默认为600
        
    Returns:
        str: 构造好的提示词文本
    """
    # 读取基础文件
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    novel_architecture_text = read_file(arch_file)
    directory_file = os.path.join(filepath, "Novel_directory.txt")
    blueprint_text = read_file(directory_file)
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    global_summary_text = read_file(global_summary_file)
    character_state_file = os.path.join(filepath, "character_state.txt")
    character_state_text = read_file(character_state_file)
    
    # 获取章节信息
    chapter_info = get_chapter_info_from_blueprint(blueprint_text, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_role = chapter_info["chapter_role"]
    chapter_purpose = chapter_info["chapter_purpose"]
    suspense_level = chapter_info["suspense_level"]
    foreshadowing = chapter_info["foreshadowing"]
    plot_twist_level = chapter_info["plot_twist_level"]
    chapter_summary = chapter_info["chapter_summary"]

    # 获取下一章节信息
    next_chapter_number = novel_number + 1
    next_chapter_info = get_chapter_info_from_blueprint(blueprint_text, next_chapter_number)
    next_chapter_title = next_chapter_info.get("chapter_title", "（未命名）")
    next_chapter_role = next_chapter_info.get("chapter_role", "过渡章节")
    next_chapter_purpose = next_chapter_info.get("chapter_purpose", "承上启下")
    next_chapter_suspense = next_chapter_info.get("suspense_level", "中等")
    next_chapter_foreshadow = next_chapter_info.get("foreshadowing", "无特殊伏笔")
    next_chapter_twist = next_chapter_info.get("plot_twist_level", "★☆☆☆☆")
    next_chapter_summary = next_chapter_info.get("chapter_summary", "衔接过渡内容")

    # 创建章节目录
    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    # 第一章特殊处理
    if novel_number == 1:
        return first_chapter_draft_prompt.format(
            novel_number=novel_number,
            word_number=word_number,
            chapter_title=chapter_title,
            chapter_role=chapter_role,
            chapter_purpose=chapter_purpose,
            suspense_level=suspense_level,
            foreshadowing=foreshadowing,
            plot_twist_level=plot_twist_level,
            chapter_summary=chapter_summary,
            characters_involved=characters_involved,
            key_items=key_items,
            scene_location=scene_location,
            time_constraint=time_constraint,
            user_guidance=user_guidance,
            novel_setting=novel_architecture_text
        )

    # 获取前文内容和摘要
    recent_texts = get_last_n_chapters_text(chapters_dir, novel_number, n=3)
    
    try:
        logging.info("Attempting to generate summary")
        short_summary = summarize_recent_chapters(
            interface_format=interface_format,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            chapters_text_list=recent_texts,
            novel_number=novel_number,
            chapter_info=chapter_info,
            next_chapter_info=next_chapter_info,
            timeout=timeout
        )
        logging.info("Summary generated successfully")
    except Exception as e:
        logging.error(f"Error in summarize_recent_chapters: {str(e)}")
        short_summary = "（摘要生成失败）"

    # 获取前一章结尾
    previous_excerpt = ""
    for text in reversed(recent_texts):
        if text.strip():
            previous_excerpt = text[-800:] if len(text) > 800 else text
            break

    # 知识库检索和处理
    try:
        # 生成检索关键词
        llm_adapter = create_llm_adapter(
            interface_format=interface_format,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        search_prompt = knowledge_search_prompt.format(
            chapter_number=novel_number,
            chapter_title=chapter_title,
            characters_involved=characters_involved,
            key_items=key_items,
            scene_location=scene_location,
            chapter_role=chapter_role,
            chapter_purpose=chapter_purpose,
            foreshadowing=foreshadowing,
            short_summary=short_summary,
            user_guidance=user_guidance,
            time_constraint=time_constraint
        )
        
        search_response = invoke_with_cleaning(llm_adapter, search_prompt)
        keyword_groups = parse_search_keywords(search_response)

        # 执行向量检索
        all_contexts = []
        from embedding_adapters import create_embedding_adapter
        embedding_adapter = create_embedding_adapter(
            embedding_interface_format,
            embedding_api_key,
            embedding_url,
            embedding_model_name
        )
        
        store = load_vector_store(embedding_adapter, filepath)
        if store:
            collection_size = store._collection.count()
            actual_k = min(embedding_retrieval_k, max(1, collection_size))
            
            for group in keyword_groups:
                context = get_relevant_context_from_vector_store(
                    embedding_adapter=embedding_adapter,
                    query=group,
                    filepath=filepath,
                    k=actual_k
                )
                if context:
                    if any(kw in group.lower() for kw in ["技法", "手法", "模板"]):
                        all_contexts.append(f"[TECHNIQUE] {context}")
                    elif any(kw in group.lower() for kw in ["设定", "技术", "世界观"]):
                        all_contexts.append(f"[SETTING] {context}")
                    else:
                        all_contexts.append(f"[GENERAL] {context}")

        # 应用内容规则
        processed_contexts = apply_content_rules(all_contexts, novel_number)
        
        # 执行知识过滤
        chapter_info_for_filter = {
            "chapter_number": novel_number,
            "chapter_title": chapter_title,
            "chapter_role": chapter_role,
            "chapter_purpose": chapter_purpose,
            "characters_involved": characters_involved,
            "key_items": key_items,
            "scene_location": scene_location,
            "foreshadowing": foreshadowing,  # 修复拼写错误
            "suspense_level": suspense_level,
            "plot_twist_level": plot_twist_level,
            "chapter_summary": chapter_summary,
            "time_constraint": time_constraint
        }
        
        filtered_context = get_filtered_knowledge_context(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            interface_format=interface_format,
            embedding_adapter=embedding_adapter,
            filepath=filepath,
            chapter_info=chapter_info_for_filter,
            retrieved_texts=processed_contexts,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
    except Exception as e:
        logging.error(f"知识处理流程异常：{str(e)}")
        filtered_context = "（知识库处理失败）"

    # 返回最终提示词
    return next_chapter_draft_prompt.format(
        user_guidance=user_guidance if user_guidance else "无特殊指导",
        global_summary=global_summary_text,
        previous_chapter_excerpt=previous_excerpt,
        character_state=character_state_text,
        short_summary=short_summary,
        novel_number=novel_number,
        chapter_title=chapter_title,
        chapter_role=chapter_role,
        chapter_purpose=chapter_purpose,
        suspense_level=suspense_level,
        foreshadowing=foreshadowing,
        plot_twist_level=plot_twist_level,
        chapter_summary=chapter_summary,
        word_number=word_number,
        characters_involved=characters_involved,
        key_items=key_items,
        scene_location=scene_location,
        time_constraint=time_constraint,
        next_chapter_number=next_chapter_number,
        next_chapter_title=next_chapter_title,
        next_chapter_role=next_chapter_role,
        next_chapter_purpose=next_chapter_purpose,
        next_chapter_suspense_level=next_chapter_suspense,
        next_chapter_foreshadowing=next_chapter_foreshadow,
        next_chapter_plot_twist_level=next_chapter_twist,
        next_chapter_summary=next_chapter_summary,
        filtered_context=filtered_context
    )

def generate_chapter_draft(
    api_key: str,
    base_url: str,
    model_name: str, 
    filepath: str,
    novel_number: int,
    word_number: int,
    temperature: float,
    user_guidance: str,
    characters_involved: str,
    key_items: str,
    scene_location: str,
    time_constraint: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    embedding_retrieval_k: int = 2,
    interface_format: str = "openai",
    max_tokens: int = 2048,
    timeout: int = 600,
    custom_prompt_text: str = None
) -> str:
    """
    生成章节草稿。
    
    功能：
    1. 构造提示词（使用自定义提示词或通过build_chapter_prompt生成）
    2. 调用LLM生成章节内容
    3. 保存生成的章节到文件
    
    Args:
        api_key (str): API密钥
        base_url (str): API基础URL
        model_name (str): 使用的语言模型名称
        filepath (str): 文件路径
        novel_number (int): 当前章节号
        word_number (int): 目标字数
        temperature (float): 生成温度参数
        user_guidance (str): 用户指导
        characters_involved (str): 涉及的角色
        key_items (str): 关键道具
        scene_location (str): 场景地点
        time_constraint (str): 时间约束
        embedding_api_key (str): 向量嵌入API密钥
        embedding_url (str): 向量嵌入API URL
        embedding_interface_format (str): 向量嵌入接口格式
        embedding_model_name (str): 向量嵌入模型名称
        embedding_retrieval_k (int, optional): 检索数量，默认为2
        interface_format (str, optional): LLM接口格式，默认为"openai"
        max_tokens (int, optional): 最大token限制，默认为2048
        timeout (int, optional): 超时时间（秒），默认为600
        custom_prompt_text (str, optional): 自定义提示词，默认为None
        
    Returns:
        str: 生成的章节内容
    """
    if custom_prompt_text is None:
        prompt_text = build_chapter_prompt(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            filepath=filepath,
            novel_number=novel_number,
            word_number=word_number,
            temperature=temperature,
            user_guidance=user_guidance,
            characters_involved=characters_involved,
            key_items=key_items,
            scene_location=scene_location,
            time_constraint=time_constraint,
            embedding_api_key=embedding_api_key,
            embedding_url=embedding_url,
            embedding_interface_format=embedding_interface_format,
            embedding_model_name=embedding_model_name,
            embedding_retrieval_k=embedding_retrieval_k,
            interface_format=interface_format,
            max_tokens=max_tokens,
            timeout=timeout
        )
    else:
        prompt_text = custom_prompt_text

    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    chapter_content = invoke_with_cleaning(llm_adapter, prompt_text)
    if not chapter_content.strip():
        logging.warning("Generated chapter draft is empty.")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)
    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")
    return chapter_content
