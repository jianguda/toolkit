"""
Streamlit Cloud 入口文件
用于 Streamlit Community Cloud 部署
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入主应用
from lm_lens.server.app import App, LmViewerConfig
from transformers import HfArgumentParser
import streamlit as st

if __name__ == "__main__":
    # 使用Streamlit部署配置
    config_file = os.getenv("CONFIG_FILE", "config/docker_hosting.json")
    
    try:
        parser = HfArgumentParser([LmViewerConfig])
        config = parser.parse_json_file(config_file)[0]
    except Exception as e:
        st.error(f"配置加载失败: {e}")
        # 使用默认配置
        config = LmViewerConfig()
    
    app = App(config)
    app.run()

