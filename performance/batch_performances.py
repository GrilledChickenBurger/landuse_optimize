#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理性能评估文档的Pipeline
该模块提供了一个pipeline来处理指定timestamp下的所有性能评估文档
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path


def get_performance_path_list(save_folder: str, timestamp: str) -> List[str]:
    """
    根据时间戳查找所有需要处理的性能评估文档路径\n
    Parameters:
    -------
        save_folder: str, 结果保存文件夹路径
        timestamp: str, 时间戳，例如 "20251127_164711"
        
    Returns:
    -------
        performance_path_list: List[str], 所有性能评估文档目录的路径列表
    """
    base_path = Path(save_folder)
    timestamp_path = base_path / timestamp
    if not timestamp_path.exists():
        raise FileNotFoundError(f"找不到时间戳目录: {timestamp_path}")
    
    performance_path_list = []
    
    # 遍历timestamp目录下的所有子目录
    for index_dir in timestamp_path.iterdir():
        if index_dir.is_dir():
            # 查找performance目录
            for item in index_dir.iterdir():
                if item.is_dir() and item.name.startswith("performance_"):
                    performance_path_list.append(str(item.resolve()))
                    
    return performance_path_list


def load_performance_data(performance_dir: str) -> Dict[str, Any]:
    """
    加载单个性能评估文档的数据\n
    Parameters:
    -------
        performance_dir: str, 性能评估文档目录路径, 例如 "results/SA/20251127_164711/10_1_5/performance_1"
        
    Returns:
    -------
        performance_data: Dict[str, Any], 性能评估数据
    """
    score_file = Path(performance_dir) / "performance_score.json"
    if not score_file.exists():
        raise FileNotFoundError(f"找不到性能评估文件: {score_file}")
    
    with open(score_file, 'r', encoding='utf-8') as f:
        performance_data = json.load(f)
        # 提取index目录名
        path_parts = performance_dir.split(os.sep)
        if len(path_parts) >= 2:
            performance_data["index"] = path_parts[-2]
        else:
            performance_data["index"] = "unknown"
        return performance_data


def aggregate_performance_data(performance_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    聚合性能评估数据，计算各指标的平均提升情况
    
    Args:
        performance_data_list (List[Dict[str, Any]]): 性能评估数据列表，每个元素是一个index的性能数据
        
    Returns:
        Dict[str, Any]: 聚合后的数据，包括各指标的平均提升情况
    """
    if not performance_data_list:
        return {"total_documents": 0, "documents": [], "average_improvements": {}}
    
    # 初始化累加器
    total_tgt_plans = {"food": 0.0, "carbon": 0.0, "water": 0.0, "culture": 0.0}
    total_improvements = {"food": 0.0, "carbon": 0.0, "water": 0.0, "culture": 0.0}
    document_count = len(performance_data_list)
    
    # 处理每个文档的数据
    processed_documents = []
    for doc_data in performance_data_list:
        # 提取ES部分的数据
        es_data = doc_data.get("ES", {})
        
        # 创建存储该文档指标的字典
        doc_metrics = {}
        
        # 处理每个指标
        for metric in ["food", "carbon", "water", "culture"]:
            if metric in es_data:
                metric_data = es_data[metric]
                tgt_plan = metric_data.get("tgt_plan", 0)
                best_plan = metric_data.get("best_plan", 0)
                improvement = best_plan - tgt_plan
                
                # 累加到总计中
                total_tgt_plans[metric] += tgt_plan
                total_improvements[metric] += improvement
                
                # 存储该文档的指标数据
                doc_metrics[metric] = {
                    "tgt_plan": tgt_plan,
                    "best_plan": best_plan,
                    "improvement": improvement
                }
        
        # 将处理后的文档数据添加到列表中
        processed_doc = doc_data.copy()
        processed_doc["metrics"] = doc_metrics
        processed_documents.append(processed_doc)
    
    # 计算平均提升情况
    average_improvements = {}
    for metric in ["food", "carbon", "water", "culture"]:
        if total_tgt_plans[metric] != 0:
            # 平均提升情况 = 所有差值之和 / 所有tgt_plan之和
            average_improvements[metric] = total_improvements[metric] / total_tgt_plans[metric]
        else:
            average_improvements[metric] = 0.0
    
    return {
        "total_documents": document_count,
        "documents": processed_documents,
        "average_improvements": average_improvements,
        "total_improvements": total_improvements,
        "total_tgt_plans": total_tgt_plans
    }


def generate_summary_report(aggregated_data: Dict[str, Any], output_path: str) -> None:
    """
    生成汇总报告
    
    Args:
        aggregated_data (Dict[str, Any]): 聚合后的性能评估数据
        output_path (str): 输出报告路径
    """
    # 这里只是一个占位函数，实际实现会生成具体的报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
    
    print(f"汇总报告已生成: {output_path}")


def batch_performance_pipeline(timestamp: str, output_dir: str = None) -> None:
    """
    批量处理性能评估文档的主pipeline
    
    Args:
        timestamp (str): 时间戳，用于定位需要处理的文档
        output_dir (str, optional): 输出目录路径，默认为None
    """
    print(f"开始处理时间戳 '{timestamp}' 的性能评估文档...")
    
    # 步骤1: 查找所有需要处理的性能评估文档
    print("步骤1: 查找性能评估文档...")
    performance_dirs = get_performance_path_list("results/SA", timestamp)
    print(f"找到 {len(performance_dirs)} 个性能评估文档目录")
    
    # 步骤2: 加载所有性能评估数据
    print("步骤2: 加载性能评估数据...")
    performance_data_list = []
    for perf_dir in performance_dirs:
        try:
            data = load_performance_data(perf_dir)
            # 添加来源信息
            data["_source"] = perf_dir
            performance_data_list.append(data)
            print(f"  成功加载: {perf_dir}")
        except Exception as e:
            print(f"  加载失败: {perf_dir}, 错误: {e}")
    
    # 步骤3: 聚合性能评估数据
    print("步骤3: 聚合性能评估数据...")
    aggregated_data = aggregate_performance_data(performance_data_list)
    
    # 步骤4: 生成汇总报告
    print("步骤4: 生成汇总报告...")
    if output_dir is None:
        output_dir = f"results/SA/{timestamp}"
    
    output_path = Path(output_dir) / f"batch_performance_summary_{timestamp}.json"
    generate_summary_report(aggregated_data, str(output_path))
    
    print("批量处理完成!")


if __name__ == "__main__":
    # 示例用法
    # batch_performance_pipeline("20251127_164711")
    pass