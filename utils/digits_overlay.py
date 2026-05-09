# utils/digits_overlay.py

import numpy as np

GLYPHS = {
    '0': [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    '1': ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    '2': [" ### ", "#   #", "    #", "  ## ", " #   ", "#    ", "#####"],
    '3': ["#####", "    #", "   # ", "  ## ", "    #", "#   #", " ### "],
    '4': ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    '5': ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    '6': ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],
    '7': ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", "#    "],
    '8': [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    '9': [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
    ':': ["     ", "  #  ", "     ", "     ", "  #  ", "     ", "     "],
    '-': ["     ", "     ", "     ", " ### ", "     ", "     ", "     "],
    ' ': ["     ", "     ", "     ", "     ", "     ", "     ", "     "],
}

class FastDigitsOverlay:
    def __init__(self, x: int=0, y: int=0, scale: int=2):
        """
        :param x: X coordinate in pixel (upper-left)
        :param y: Y coordinate in pixel (upper-left)
        :param scale: scale factor. 1 ~ 80 us; 2 ~ 140 us; 3 ~ 220 us
        """
        self.x, self.y = x, y
        self.scale = scale
        self.last_time_str = ""
        
        self.char_masks = self._preprocess_glyphs()
        
        # Cache last text
        self.cached_border_mask = None
        self.cached_text_mask = None
        self.h, self.w = 0, 0

    def _preprocess_glyphs(self):
        """Generate bordered glyphs"""
        masks = {}
        for char, rows in GLYPHS.items():
            # Str list To 0/1 Matrix
            m = np.array([[1 if c == '#' else 0 for c in row] for row in rows], dtype=np.uint8)
            
            if self.scale > 1:
                m = np.repeat(np.repeat(m, self.scale, axis=0), self.scale, axis=1)
            
            h, w = m.shape
            # Extend to 1px border.
            border = np.zeros((h + 2, w + 2), dtype=np.uint8)
            for dy in range(3):
                for dx in range(3):
                    border[dy:dy+h, dx:dx+w] |= m
            
            # Inner text part
            text = np.zeros((h + 2, w + 2), dtype=np.uint8)
            text[1:-1, 1:-1] = m
            
            masks[char] = (border.astype(bool), text.astype(bool))
        return masks

    def __call__(self, frame: np.ndarray, text: str):
        """Apply Text Overlay to Frame inplace"""
        if text != self.last_time_str:
            self._update_cache(text)
            self.last_time_str = text

        roi = frame[self.y : self.y + self.h, self.x : self.x + self.w]
        
        roi[self.cached_border_mask] = 0
        roi[self.cached_text_mask] = 255

    def _update_cache(self, text):
        """Create mask for text"""
        all_borders = []
        all_texts = []
        
        for char in text:
            b, t = self.char_masks.get(char, self.char_masks[' '])
            all_borders.append(b)
            all_texts.append(t)
            
        self.cached_border_mask = np.concatenate(all_borders, axis=1)
        self.cached_text_mask = np.concatenate(all_texts, axis=1)
        self.h, self.w = self.cached_border_mask.shape