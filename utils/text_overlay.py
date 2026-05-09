# utils/text_overlay.py

import datetime
from typing import Tuple

import numpy as np
from PySide6.QtGui import QImage, QPainter, QColor, QFont
from PySide6.QtCore import Qt

class TextOverlayGen:
    """Generate a text image overlay for given size"""
    
    def __init__(self, width: int = 200, height: int = 30, 
                 bg_rgb: Tuple[int] = (0, 0, 0), bg_alpha: int = 0,
                 text_rgb: Tuple[int] = (255, 255, 255), text_alpha: int = 255,
                 font_family: str = "Arial", font_size: int = 12, font_bold: bool = True,   
                ): 
        self.width = width
        self.height = height
        self.bg_rgb = bg_rgb
        self.bg_alpha = bg_alpha
        self.text_rgb = text_rgb
        self.text_alpha = text_alpha
        self.font = QFont(font_family, font_size, QFont.Weight.Bold if font_bold else QFont.Weight.Normal)

        self.image = QImage(self.width, self.height, QImage.Format_RGBA8888)
        
    def generate(self, text: str = None) -> memoryview:
        """Generate image for given text, if None, timestamp by default.
        :param text: The text to generate image for.
        :return: memoryview for the QImage in RGBA format.
        """
        if text is None:
            text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        img = self.image
        img.fill(QColor(*self.bg_rgb, self.bg_alpha))
        
        painter = QPainter(img)

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        painter.setPen(QColor(*self.text_rgb, self.text_alpha))
        painter.setFont(self.font)
        painter.drawText(img.rect(), Qt.AlignCenter, text)
        painter.end()

        ptr = img.constBits()
        return memoryview(ptr)