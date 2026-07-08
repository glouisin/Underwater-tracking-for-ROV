#!/usr/bin/env python3
"""
selector_list_mac.py — Liste temps réel des objets suivis + sélection.

Reçoit du ROV (UDP) la liste des objets actuellement suivis et l'affiche en
temps réel. Cliquer sur une ligne verrouille cet objet : le ROV n'asservit
alors le PID que sur lui. Le bouton « Désélectionner » relâche la cible.

Protocole :
  ROV  -> Mac (LIST_PORT)   : "id,x,y,z,conf;id,x,y,z,conf;..."  (vide si aucun objet)
  Mac  -> ROV (SELECT_PORT) : "select,<id>"   ou   "clear"

Dépendances : uniquement la bibliothèque standard (Tkinter).
Si Tkinter manque sur le Mac :  brew install python-tk
"""

import socket
import threading
import tkinter as tk
from tkinter import ttk

# -------------------- Réseau --------------------
ROV_IP      = "192.168.5.54"   # destinataire des commandes de sélection
SELECT_PORT = 17002            # Mac -> ROV (sélection)
LIST_PORT   = 17003            # ROV -> Mac (liste des objets)

# -------------------- Socket d'émission des sélections --------------------
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_command(msg: str):
    send_sock.sendto(msg.encode(), (ROV_IP, SELECT_PORT))

# -------------------- État partagé (thread réseau -> thread Tk) --------------------
# On NE touche JAMAIS aux widgets Tk depuis le thread réseau : il se contente
# de déposer la dernière liste reçue ici, sous verrou. L'affichage est rafraîchi
# par la boucle Tk via root.after().
_latest = []                   # liste de tuples (id, x, y, z, conf)
_lock   = threading.Lock()

def receiver():
    """Thread d'écoute UDP : parse les paquets et stocke la dernière liste."""
    global _latest
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", LIST_PORT))
    while True:
        data, _ = s.recvfrom(8192)
        txt = data.decode(errors="ignore").strip()
        objs = []
        if txt:
            for part in txt.split(";"):
                f = part.split(",")
                if len(f) == 5:
                    try:
                        objs.append((int(f[0]), int(f[1]), int(f[2]),
                                     float(f[3]), float(f[4])))
                    except ValueError:
                        pass
        with _lock:
            _latest = objs

# -------------------- Interface --------------------
class App:
    def __init__(self, root):
        self.root = root
        self.selected_id = None
        root.title("ROV — objets suivis")
        root.geometry("470x380")

        cols = ("id", "x", "y", "z", "conf")
        widths = (50, 75, 75, 90, 70)
        self.tree = ttk.Treeview(root, columns=cols, show="headings", height=13)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="center")
        self.tree.tag_configure("sel", background="#ffe08a")   # surlignage cible
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.tree.bind("<ButtonRelease-1>", self.on_click)

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=8, pady=(0, 8))
        self.status = tk.Label(bar, text="Aucune cible", anchor="w")
        self.status.pack(side="left")
        tk.Button(bar, text="Désélectionner", command=self.clear).pack(side="right")

        self.refresh()

    def on_click(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        tid = int(sel[0])                  # l'iid de chaque ligne = str(id)
        self.selected_id = tid
        send_command(f"select,{tid}")
        self.status.config(text=f"Cible verrouillée : #{tid}")
        self.retag()

    def clear(self):
        self.selected_id = None
        send_command("clear")
        self.status.config(text="Aucune cible")
        for iid in self.tree.selection():
            self.tree.selection_remove(iid)
        self.retag()

    def retag(self):
        """Applique le surlignage à la ligne sélectionnée."""
        target = str(self.selected_id)
        for iid in self.tree.get_children():
            self.tree.item(iid, tags=("sel",) if iid == target else ())

    def refresh(self):
        with _lock:
            objs = list(_latest)
        present = {str(o[0]) for o in objs}

        # Supprime les objets disparus
        for iid in self.tree.get_children():
            if iid not in present:
                self.tree.delete(iid)

        # Ajoute / met à jour en place (pas de scintillement, sélection conservée)
        for (tid, x, y, z, conf) in objs:
            iid  = str(tid)
            vals = (tid, x, y, f"{z:.1f}", f"{conf:.2f}")
            if self.tree.exists(iid):
                self.tree.item(iid, values=vals)
            else:
                self.tree.insert("", "end", iid=iid, values=vals)

        self.retag()

        # Statut si la cible verrouillée n'est plus visible
        if self.selected_id is not None and str(self.selected_id) not in present:
            self.status.config(text=f"Cible #{self.selected_id} perdue…")

        self.root.after(100, self.refresh)   # ~10 Hz, suffisant pour l'œil

if __name__ == "__main__":
    threading.Thread(target=receiver, daemon=True).start()
    root = tk.Tk()
    App(root)
    root.mainloop()