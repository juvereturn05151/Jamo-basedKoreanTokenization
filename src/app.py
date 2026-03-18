"""
File Name:    app.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import tkinter as tk
from tkinter import ttk


class JamoBPEApp:
    def __init__(self, root, tokenizer, full_tokenizer):
        self.root = root
        self.root.title("Jamo BPE Tokenizer App")
        self.root.geometry("1000x650")

        self.tokenizer = tokenizer
        self.full_tokenizer = full_tokenizer

        self._build_ui()

    def _build_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Input section
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill="x", pady=(0, 10))

        label = ttk.Label(input_frame, text="Enter Korean text:")
        label.pack(anchor="w")

        self.input_text = tk.Text(input_frame, height=4, font=("Arial", 14))
        self.input_text.pack(fill="x", pady=5)
        self.input_text.insert("1.0", "안녕하세요 여러분")

        execute_button = ttk.Button(
            input_frame,
            text="Execute",
            command=self.execute_tokenization
        )
        execute_button.pack(anchor="e", pady=5)

        # Output section
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill="both", expand=True)

        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # Left panel: Tokens
        tokens_frame = ttk.LabelFrame(output_frame, text="Jamo-based Tokens", padding=8)
        tokens_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.tokens_output = tk.Text(
            tokens_frame,
            wrap="word",
            font=("Consolas", 12)
        )
        self.tokens_output.pack(fill="both", expand=True)

        # Right panel: Full Tokens
        full_tokens_frame = ttk.LabelFrame(output_frame, text="Full Tokens", padding=8)
        full_tokens_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.full_tokens_output = tk.Text(
            full_tokens_frame,
            wrap="word",
            font=("Consolas", 12)
        )
        self.full_tokens_output.pack(fill="both", expand=True)

    def execute_tokenization(self):
        text = self.input_text.get("1.0", "end").strip()

        if not text:
            self._set_output(self.tokens_output, "No input provided.")
            self._set_output(self.full_tokens_output, "No input provided.")
            return

        tokens = self.tokenizer.encode(text)
        full_tokens = self.full_tokenizer.encode(text)

        self._set_output(self.tokens_output, f"Tokens:\n{tokens}")
        self._set_output(self.full_tokens_output, f"Full Tokens:\n{full_tokens}")

    @staticmethod
    def _set_output(widget, content):
        widget.delete("1.0", "end")
        widget.insert("1.0", content)