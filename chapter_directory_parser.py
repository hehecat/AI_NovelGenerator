# chapter_blueprint_parser.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Optional
from functools import lru_cache

# 全局缓存
_blueprint_cache: Dict[str, Dict[int, dict]] = {}

def clear_blueprint_cache():
    """清除蓝图缓存"""
    global _blueprint_cache
    _blueprint_cache.clear()

@lru_cache(maxsize=32)
def parse_chapter_blueprint(blueprint_text: str) -> List[dict]:
    """
    解析整份章节蓝图文本，返回一个列表，每个元素是一个 dict。
    使用LRU缓存优化重复解析。
    
    Returns:
        List[dict]: 包含所有章节信息的列表，每个章节信息包含：
            - chapter_number: int
            - chapter_title: str
            - chapter_role: str       # 本章定位
            - chapter_purpose: str    # 核心作用
            - suspense_level: str     # 核心悬念
            - emotion_tone: str       # 情感基调
            - foreshadowing: str      # 伏笔操作
            - plot_twist_level: str   # 认知颠覆
            - chapter_summary: str    # 本章梗概
    """
    # 先按空行进行分块，以免多章之间混淆
    chunks = re.split(r'\n\s*\n', blueprint_text.strip())
    results = []

    # 兼容是否使用方括号包裹章节标题
    # 例如：
    #   第1章 - 紫极光下的预兆
    # 或
    #   第1章 - [紫极光下的预兆]
    chapter_number_pattern = re.compile(r'^第\s*(\d+)\s*章\s*-\s*\[?(.*?)\]?$')

    role_pattern = re.compile(r'^本章定位：\s*\[?(.*)\]?$')
    purpose_pattern = re.compile(r'^核心作用：\s*\[?(.*)\]?$')
    suspense_pattern = re.compile(r'^核心悬念：\s*\[?(.*)\]?$')
    emotion_pattern = re.compile(r'^情感基调：\s*\[?(.*)\]?$')
    foreshadow_pattern = re.compile(r'^伏笔操作：\s*\[?(.*)\]?$')
    twist_pattern = re.compile(r'^认知颠覆：\s*\[?(.*)\]?$')
    summary_pattern = re.compile(r'^本章梗概.*?：\s*\[?(.*)\]?$')

    for chunk in chunks:
        lines = chunk.strip().splitlines()
        if not lines:
            continue

        chapter_info = {}
        
        # 解析章节标题和编号
        first_line = lines[0].strip()
        chapter_match = chapter_number_pattern.match(first_line)
        if not chapter_match:
            continue
            
        chapter_info["chapter_number"] = int(chapter_match.group(1))
        chapter_info["chapter_title"] = chapter_match.group(2).strip()
        
        # 解析其他信息
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            if role_pattern.match(line):
                chapter_info["chapter_role"] = role_pattern.match(line).group(1).strip()
            elif purpose_pattern.match(line):
                chapter_info["chapter_purpose"] = purpose_pattern.match(line).group(1).strip()
            elif suspense_pattern.match(line):
                chapter_info["suspense_level"] = suspense_pattern.match(line).group(1).strip()
            elif emotion_pattern.match(line):
                chapter_info["emotion_tone"] = emotion_pattern.match(line).group(1).strip()
            elif foreshadow_pattern.match(line):
                chapter_info["foreshadowing"] = foreshadow_pattern.match(line).group(1).strip()
            elif twist_pattern.match(line):
                chapter_info["plot_twist_level"] = twist_pattern.match(line).group(1).strip()
            elif summary_pattern.match(line):
                chapter_info["chapter_summary"] = summary_pattern.match(line).group(1).strip()
        
        results.append(chapter_info)
    
    return results


def get_chapter_info_from_blueprint(blueprint_text: str, target_chapter_number: int) -> dict:
    print('blueprint_text', blueprint_text)
    """
    在已经加载好的章节蓝图文本中，找到对应章号的结构化信息，返回一个 dict。
    使用缓存优化重复查询。
    
    Args:
        blueprint_text (str): 章节蓝图文本
        target_chapter_number (int): 目标章节号
        
    Returns:
        dict: 章节信息字典，包含以下字段：
            - chapter_number: 章节编号
            - chapter_title: 章节标题
            - chapter_role: 章节定位
            - chapter_purpose: 核心作用
            - suspense_level: 核心悬念
            - emotion_tone: 情感基调
            - foreshadowing: 伏笔操作
            - plot_twist_level: 认知颠覆
            - chapter_summary: 章节梗概
    """
    global _blueprint_cache
    
    # 检查缓存
    if blueprint_text in _blueprint_cache:
        chapters_dict = _blueprint_cache[blueprint_text]
        if target_chapter_number in chapters_dict:
            return chapters_dict[target_chapter_number]
    else:
        # 解析并缓存所有章节信息
        all_chapters = parse_chapter_blueprint(blueprint_text)
        chapters_dict = {ch["chapter_number"]: ch for ch in all_chapters}
        _blueprint_cache[blueprint_text] = chapters_dict
        
        # 检查目标章节
        if target_chapter_number in chapters_dict:
            return chapters_dict[target_chapter_number]
    
    # 如果找不到目标章节，尝试从最近的章节推断
    if chapters_dict:
        # 找到最接近的章节
        closest_chapter = None
        min_diff = float('inf')
        
        for chapter_num, chapter_info in chapters_dict.items():
            diff = abs(chapter_num - target_chapter_number)
            if diff < min_diff:
                min_diff = diff
                closest_chapter = chapter_info
        
        if closest_chapter:
            # 基于最近的章节创建新章节信息
            new_chapter = closest_chapter.copy()
            new_chapter["chapter_number"] = target_chapter_number
            new_chapter["chapter_title"] = f"第{target_chapter_number}章"
            new_chapter["chapter_summary"] = f"基于第{closest_chapter['chapter_number']}章的内容继续发展"
            return new_chapter
    
    # 如果没有任何章节信息，返回基本结构
    return {
        "chapter_number": target_chapter_number,
        "chapter_title": f"第{target_chapter_number}章",
        "chapter_role": "常规章节",
        "chapter_purpose": "内容推进",
        "suspense_level": "中等",
        "emotion_tone": "平稳",
        "foreshadowing": "无特殊伏笔",
        "plot_twist_level": "★☆☆☆☆",
        "chapter_summary": "常规内容推进"
    }
