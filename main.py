# main.py
# -*- coding: utf-8 -*-
import customtkinter as ctk
from ui import NovelGeneratorGUI
import logging

def setup_logging():
    """配置应用程序的日志记录。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("日志系统已配置。") # 这条日志会根据上面的配置显示出来

def main():
    setup_logging() 
    app = ctk.CTk()
    logging.info("NovelGenerator GUI 应用程序启动...") 
    gui = NovelGeneratorGUI(app)
    app.mainloop()

if __name__ == "__main__":
    main()
