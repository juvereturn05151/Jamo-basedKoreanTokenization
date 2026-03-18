"""
File Name:    app.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import threading
import tkinter as tk
from tkinter import ttk


class JamoBPEApp:
    def __init__(self, root, tokenizer, full_tokenizer, input_dir, output_dir):
        self.root = root
        self.root.title("Jamo BPE Tokenizer App")
        self.root.geometry("1000x650")

        self.tokenizer = tokenizer
        self.full_tokenizer = full_tokenizer
        self.input_dir = input_dir
        self.output_dir = output_dir

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill="x", pady=(0, 10))

        label = ttk.Label(input_frame, text="Enter Korean text:")
        label.pack(anchor="w")

        self.input_text = tk.Text(input_frame, height=4, font=("Arial", 14))
        self.input_text.pack(fill="x", pady=5)
        self.input_text.insert("1.0", "안녕하세요 여러분")

        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x", pady=5)

        self.status_label = ttk.Label(button_frame, text="Ready")
        self.status_label.pack(side="left")

        self.retrain_button = ttk.Button(
            button_frame,
            text="Retrain Tokenizers",
            command=self.start_retrain_thread
        )
        self.retrain_button.pack(side="right", padx=(5, 0))

        self.execute_button = ttk.Button(
            button_frame,
            text="Execute",
            command=self.execute_tokenization
        )
        self.execute_button.pack(side="right")

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill="both", expand=True)

        output_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(1, weight=1)
        output_frame.rowconfigure(0, weight=1)

        tokens_frame = ttk.LabelFrame(output_frame, text="Tokens", padding=8)
        tokens_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.tokens_output = tk.Text(
            tokens_frame,
            wrap="word",
            font=("Consolas", 12)
        )
        self.tokens_output.pack(fill="both", expand=True)

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

        try:
            tokens = self.tokenizer.encode(text)
            full_tokens = self.full_tokenizer.encode(text)

            self._set_output(self.tokens_output, f"Tokens:\n{tokens}")
            self._set_output(self.full_tokens_output, f"Full Tokens:\n{full_tokens}")
            self.status_label.config(text="Tokenization complete")
        except Exception as e:
            self.status_label.config(text=f"Tokenization failed: {e}")

    def start_retrain_thread(self):
        self._set_buttons_enabled(False)
        self.status_label.config(text="Retraining tokenizers...")

        thread = threading.Thread(target=self._retrain_tokenizers_worker, daemon=True)
        thread.start()

    def _retrain_tokenizers_worker(self):
        try:
            raw_texts = self.tokenizer.read_txt_files(self.input_dir)
            preprocessed_texts = self.tokenizer.preprocess(raw_texts)

            self.tokenizer.train(
                preprocessed_texts,
                vocab_size=2000,
                min_frequency=2,
            )
            self.full_tokenizer.train(
                preprocessed_texts,
                vocab_size=2000,
                min_frequency=2,
            )

            self.tokenizer.save(self.output_dir / "jamo", preprocessed_texts)
            self.full_tokenizer.save(self.output_dir / "full", preprocessed_texts)

            self.root.after(0, self._on_retrain_success)
        except Exception as e:
            self.root.after(0, self._on_retrain_failure, str(e))

    def _on_retrain_success(self):
        self.status_label.config(text="Retraining complete")
        self._set_buttons_enabled(True)

    def _on_retrain_failure(self, error_message):
        self.status_label.config(text=f"Retraining failed: {error_message}")
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.execute_button.config(state=state)
        self.retrain_button.config(state=state)

    @staticmethod
    def _set_output(widget, content):
        widget.delete("1.0", "end")
        widget.insert("1.0", content)