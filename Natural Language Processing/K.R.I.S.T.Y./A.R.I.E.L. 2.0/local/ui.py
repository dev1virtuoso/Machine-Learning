import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk

from client import ask

class ArielChat(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A.R.I.E.L. 2.0 – Local Client")
        self.geometry("600x400")

        self.chat_area = ScrolledText(self, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_area.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            self.input_frame, textvariable=self.input_var
        )
        self.input_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.input_entry.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(
            self.input_frame, text="Send", command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def send_message(self, event=None):
        text = self.input_var.get().strip()
        if not text:
            return
        self._append_chat("You", text)
        self.input_var.set("")
        try:
            response = ask(text)
            if response.get("status") == "success":
                out = response["ariel_output"]
                self._append_chat("A.R.I.E.L.", out)
            else:
                self._append_chat(
                    "A.R.I.E.L.", f"[ERROR] {response.get('message')}"
                )
        except Exception as exc:
            self._append_chat(
                "A.R.I.E.L.", f"[FAIL] {exc}"
            )

    def _append_chat(self, speaker: str, msg: str):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{speaker}: {msg}\n\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def on_close(self):
        self.destroy()

if __name__ == "__main__":
    app = ArielChat()
    app.mainloop()
