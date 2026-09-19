import tkinter as tk
from tkinter import scrolledtext
from .main import abc_respond

class ABCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("A.B.C. v1.0 – Offline Chat")
        self.root.geometry("500x400")

        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(expand=True, fill='both', padx=5, pady=5)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(root, textvariable=self.input_var)
        self.input_entry.pack(fill='x', padx=5, pady=5)
        self.input_entry.bind("<Return>", self.on_enter)

        self.send_button = tk.Button(root, text="Send", command=self.on_enter)
        self.send_button.pack(pady=5)

        self.display_message("A.B.C.: Hi! I'm A.B.C. v1.0. How can I help you today?")

    def display_message(self, msg):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, msg + "\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)

    def on_enter(self, event=None):
        user_text = self.input_var.get().strip()
        if not user_text:
            return
        self.display_message(f"You: {user_text}")
        self.input_var.set("")
        reply = abc_respond(user_text)
        self.display_message(f"A.B.C.: {reply}")
        if user_text.lower() in ["bye", "goodbye", "exit", "quit", "88"]:
            self.root.after(1000, self.root.destroy)

def run_gui():
    root = tk.Tk()
    app = ABCGUI(root)
    root.mainloop()
