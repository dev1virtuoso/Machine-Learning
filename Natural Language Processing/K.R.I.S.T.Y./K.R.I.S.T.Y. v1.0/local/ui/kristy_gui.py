import tkinter as tk
from tkinter import scrolledtext
from core.kristy import KristyGraphMath
from core.prepare import build_database
import os

class KristyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K.R.I.S.T.Y. v1.0 – Desktop Client")
        self.root.geometry("600x450")

        self.chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled')
        self.chat.pack(expand=True, fill='both', padx=5, pady=5)

        self.entry = tk.Entry(root)
        self.entry.pack(fill='x', padx=5, pady=5)
        self.entry.bind("<Return>", self.send)

        db_path = "./data/knowledge_base.db"
        if not os.path.exists(db_path):
            build_database()
        self.engine = KristyGraphMath(db_path)

        self.append_text("K.R.I.S.T.Y. v1.0 – Ready to answer queries.")

    def append_text(self, text, user=True):
        self.chat.configure(state='normal')
        prefix = "You: " if user else "K.R.I.S.T.Y.: "
        self.chat.insert(tk.END, f"{prefix}{text}\n")
        self.chat.configure(state='disabled')
        self.chat.see(tk.END)

    def send(self, event=None):
        query = self.entry.get().strip()
        if not query:
            return
        self.append_text(query, user=True)
        self.entry.delete(0, tk.END)
        response = self.engine.handle_query(query)
        self.append_text(response, user=False)

def run():
    root = tk.Tk()
    app = KristyApp(root)
    root.mainloop()
