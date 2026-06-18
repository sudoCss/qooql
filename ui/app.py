# ui/app.py
import json
import os
import sqlite3
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import requests

from config import API_PORTS, DB_CONFIG, RES_DIR

DATA_LOADER_API_URL = f"http://127.0.0.1:{API_PORTS['DATA_LOADER']}"
REPRESENTATION_API_URL = f"http://127.0.0.1:{API_PORTS['REPRESENTATION']}"
SEARCH_API_URL = f"http://127.0.0.1:{API_PORTS['SEARCH']}"


class PotatoSafeIRApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Information Retrieval System - Complete Client")
        self.geometry("900x800")

        self.bg_main = "#1a1a24"
        self.bg_card = "#242432"
        self.bg_input = "#2d2d3d"
        self.fg_main = "#ffffff"
        self.fg_muted = "#9ca3af"
        self.accent_blue = "#3b82f6"
        self.accent_green = "#10b981"

        self.configure(bg=self.bg_main)
        self.suggestion_timer = None

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()
        self.create_widgets()

    def _configure_styles(self):
        self.style.configure(".", background=self.bg_main, foreground=self.fg_main)
        self.style.configure(
            "TLabel",
            font=("Segoe UI", 10, "bold"),
            background=self.bg_main,
            foreground=self.fg_muted,
        )
        self.style.configure("TFrame", background=self.bg_main)
        self.style.configure(
            "TCombobox",
            fieldbackground=self.bg_input,
            background=self.bg_card,
            foreground=self.fg_main,
            arrowcolor=self.fg_main,
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.bg_input)],
            foreground=[("readonly", self.fg_main)],
        )
        self.style.configure(
            "TButton",
            font=("Segoe UI", 9, "bold"),
            background=self.accent_blue,
            foreground=self.fg_main,
            borderwidth=0,
        )
        self.style.map(
            "TButton", background=[("active", "#2563eb"), ("disabled", "#4b5563")]
        )
        self.style.configure(
            "TLabelframe",
            background=self.bg_main,
            foreground=self.accent_blue,
            bordercolor=self.bg_card,
        )
        self.style.configure(
            "TLabelframe.Label",
            background=self.bg_main,
            foreground=self.accent_blue,
            font=("Segoe UI", 9, "bold"),
        )
        self.style.configure("Card.TFrame", background=self.bg_card, relief="flat")

    def create_widgets(self):
        # --- NEW SECTION: Dataset Ingestion Control Panel ---
        admin_frame = ttk.LabelFrame(
            self,
            text=" Dataset Lifecycle Control Panel (Run before searching) ",
            padding="12",
        )
        admin_frame.pack(fill=tk.X, padx=20, pady=10)

        dataset_names = (
            "cranfield (1,400 docs . ~500 KB)",  #
            "beir/nfcorpus (3,633 docs . ~3 MB)",  #
            "beir/scifact (5,183 docs . ~3 MB)",  #
            "vaswani (11,429 docs . ~2 MB)",  #
            "beir/quora (523,000 docs . ~41 MB)",  #
            "lotte/lifestyle/dev (268,893 docs . ~3.5 GB (the whole pre /dev))",  #
        )

        self.admin_dataset_var = tk.StringVar(value=dataset_names[0])
        admin_combo = ttk.Combobox(
            admin_frame, textvariable=self.admin_dataset_var, state="readonly", width=25
        )
        admin_combo["values"] = dataset_names
        admin_combo.grid(row=0, column=0, padx=5, pady=5)

        btn_ingest = tk.Button(
            admin_frame,
            text="1. Fetch, Preprocess & Ingest",
            command=self.trigger_ingestion,
            bg="#f59e0b",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
        )
        btn_ingest.grid(row=0, column=1, padx=5)

        btn_matrix = tk.Button(
            admin_frame,
            text="2. Training",
            command=self.trigger_matrices,
            bg="#8b5cf6",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
        )
        btn_matrix.grid(row=0, column=2, padx=5)

        btn_show_data = tk.Button(
            admin_frame,
            text="3. Show Testing Results",
            command=self.trigger_show_data_popup,
            bg="#ec4899",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=10,
        )
        btn_show_data.grid(row=0, column=3, padx=5)

        # --- Search Query Operations Interface Frame ---
        top_frame = ttk.Frame(self, padding="15")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Target Search Dataset:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.dataset_var = tk.StringVar(value=dataset_names[0])
        dataset_combo = ttk.Combobox(
            top_frame, textvariable=self.dataset_var, state="readonly", width=25
        )
        dataset_combo["values"] = dataset_names
        dataset_combo.grid(row=1, column=0, padx=5, pady=5)

        ttk.Label(top_frame, text="Engine Mode:").grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )
        self.model_var = tk.StringVar(value="hybrid")
        model_combo = ttk.Combobox(
            top_frame, textvariable=self.model_var, state="readonly", width=20
        )
        model_combo["values"] = ("hybrid", "hybrid_serial", "bm25", "bert", "tfidf")
        model_combo.grid(row=1, column=1, padx=5, pady=5)
        model_combo.bind("<<ComboboxSelected>>", self.toggle_parameter_views)

        self.param_frame = ttk.LabelFrame(self, text=" Model Parameters ", padding="10")
        self.param_frame.pack(fill=tk.X, padx=20, pady=5)

        self.lbl_k1 = ttk.Label(self.param_frame, text="BM25 k1:")
        self.lbl_k1.grid(row=0, column=0, padx=5, sticky=tk.W)
        self.ent_k1 = tk.Entry(
            self.param_frame,
            width=8,
            bg=self.bg_input,
            fg=self.fg_main,
            insertbackground="white",
            relief="flat",
            bd=3,
        )
        self.ent_k1.insert(0, "1.6")
        self.ent_k1.grid(row=0, column=1, padx=5)

        self.lbl_b = ttk.Label(self.param_frame, text="BM25 b:")
        self.lbl_b.grid(row=0, column=2, padx=5, sticky=tk.W)
        self.ent_b = tk.Entry(
            self.param_frame,
            width=8,
            bg=self.bg_input,
            fg=self.fg_main,
            insertbackground="white",
            relief="flat",
            bd=3,
        )
        self.ent_b.insert(0, "0.75")
        self.ent_b.grid(row=0, column=3, padx=5)

        self.lbl_weight = ttk.Label(self.param_frame, text="BM25 Weight (0.80):")
        self.lbl_weight.grid(row=0, column=4, padx=15, sticky=tk.W)
        self.slider_weight = ttk.Scale(
            self.param_frame,
            from_=0.0,
            to=1.0,
            value=0.8,
            command=self.update_slider_label,
        )
        self.slider_weight.grid(row=0, column=5, padx=5, sticky=tk.EW)

        query_frame = ttk.Frame(self, padding="15")
        query_frame.pack(fill=tk.X)

        ttk.Label(query_frame, text="Enter Search Query:").pack(anchor=tk.W, padx=5)
        input_row = ttk.Frame(query_frame)
        input_row.pack(fill=tk.X, pady=5)

        self.query_var = tk.StringVar()
        self.query_entry = tk.Entry(
            input_row,
            textvariable=self.query_var,
            bg=self.bg_input,
            fg=self.fg_main,
            insertbackground="white",
            font=("Segoe UI", 11),
            relief="flat",
            bd=5,
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=5)
        self.query_entry.bind("<KeyRelease>", self.handle_query_key_release)
        self.query_entry.bind("<Return>", lambda e: self.start_search_thread())

        self.btn_search = ttk.Button(
            input_row, text="SEARCH", command=self.start_search_thread
        )
        self.btn_search.pack(side=tk.RIGHT, padx=5, ipady=4)

        features_frame = ttk.Frame(query_frame)
        features_frame.pack(fill=tk.X, pady=2)

        self.faiss_var = tk.BooleanVar(value=False)
        self.chk_faiss = tk.Checkbutton(
            features_frame,
            text="Enable FAISS (Vector store)",
            variable=self.faiss_var,
            bg=self.bg_main,
            fg=self.fg_muted,
            selectcolor=self.bg_main,
            activebackground=self.bg_main,
            activeforeground=self.fg_main,
        )
        self.chk_faiss.pack(side=tk.LEFT, padx=5)

        self.suggestion_box = tk.Listbox(
            self,
            bg=self.bg_card,
            fg=self.fg_main,
            selectbackground=self.accent_blue,
            selectforeground=self.fg_main,
            highlightthickness=0,
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.suggestion_box.bind("<<ListboxSelect>>", self.select_suggestion_item)

        self.status_label = tk.Label(
            self,
            text="Ready",
            font=("Segoe UI", 9, "italic"),
            bg=self.bg_main,
            fg=self.fg_muted,
        )
        self.status_label.pack(anchor=tk.W, padx=20)

        results_container = ttk.Frame(self, padding="15")
        results_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            results_container, bg=self.bg_main, highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            results_container, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.configure(style="TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(
                self.canvas.find_withtag("all")[0], width=e.width
            ),
        )

        self.toggle_parameter_views()

    # --- Ingestion Button Directives handlers ---
    def trigger_ingestion(self):
        ds = self.admin_dataset_var.get().split(" ")[0]
        threading.Thread(
            target=self._api_post_alert,
            args=(
                f"{DATA_LOADER_API_URL}/load-dataset/",
                {"dataset_name": ds},
                "Ingestion initiated in background pipeline! Check console logs.",
            ),
            daemon=True,
        ).start()

    def trigger_matrices(self):
        ds = self.admin_dataset_var.get().split(" ")[0]
        threading.Thread(
            target=self._api_post_alert,
            args=(
                f"{REPRESENTATION_API_URL}/build-representations/",
                {"dataset_name": ds},
                "Matrix indexing started in backend! This might take a minute.",
            ),
            daemon=True,
        ).start()

    def trigger_show_data_popup(self):
        # 1. Parse the raw dataset name from the combobox (e.g., "beir/quora")
        raw_selection = self.admin_dataset_var.get()
        dataset_name = raw_selection.split(" ")[0]

        # 2. Derive the target JSON filename used by the script
        json_filename = f"{dataset_name.replace('/', '_')}.json"
        json_file_full_path = os.path.join(RES_DIR, json_filename)

        # 3. Read and validate the file
        if not os.path.exists(json_file_full_path):
            messagebox.showerror(
                "Data File Not Found",
                f"Could not find the evaluation records file:\n'{json_file_full_path}'\n\n"
                f"Please run your script or baseline evaluation pipeline for '{dataset_name}' first!",
            )
            return

        try:
            with open(json_file_full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror(
                "Error Reading JSON", f"Failed to load or parse JSON file:\n{str(e)}"
            )
            return

        # 4. Construct the Popup Window (Toplevel Window)
        popup = tk.Toplevel(self)
        popup.title(f"Performance & Evaluation Reports — {dataset_name}")
        popup.geometry("750x750")
        popup.configure(bg=self.bg_main)
        popup.transient(self)  # Keeps popup on top of main window
        popup.grab_set()  # Makes window modal

        # Container with Padding
        main_container = ttk.Frame(popup, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Title Label
        title_lbl = tk.Label(
            main_container,
            text=f"📊 METRICS REPORT: {dataset_name.upper()}",
            font=("Segoe UI", 14, "bold"),
            bg=self.bg_main,
            fg=self.accent_blue,
        )
        title_lbl.pack(anchor=tk.W, pady=(0, 5))

        # Subtitle Metatdata
        timestamp = data.get("timestamp", "N/A")
        total_q = data.get("total_queries_evaluated", 0)
        meta_lbl = tk.Label(
            main_container,
            text=f"Evaluated on: {timestamp}  |  Total Test Queries: {total_q}",
            font=("Segoe UI", 9, "italic"),
            bg=self.bg_main,
            fg=self.fg_muted,
        )
        meta_lbl.pack(anchor=tk.W, pady=(0, 15))

        # Create a beautiful Notebook Tab View system
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # ---- TAB 1: OVERVIEW & ACCURACY ----
        tab1 = ttk.Frame(notebook, padding="10")
        notebook.add(tab1, text=" Global Metrics Summary ")

        # --- SECTION A: MODEL METRICS TABLE ---
        metrics_frame = ttk.LabelFrame(
            tab1,
            text=" Information Retrieval Accuracy Metrics ",
            padding="10",
        )
        metrics_frame.pack(fill=tk.X, pady=(0, 15))

        self.style.configure(
            "Popup.Treeview",
            background=self.bg_card,
            fieldbackground=self.bg_card,
            foreground=self.fg_main,
            rowheight=28,
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Popup.Treeview.Heading",
            background=self.bg_input,
            foreground=self.accent_blue,
            font=("Segoe UI", 10, "bold"),
        )

        cols = ("Model", "MAP", "Precision@10", "Recall", "nDCG")
        tree = ttk.Treeview(
            metrics_frame,
            columns=cols,
            show="headings",
            style="Popup.Treeview",
            height=6,
        )

        tree.column("Model", width=130, anchor=tk.CENTER)
        for col in cols[1:]:
            tree.column(col, width=100, anchor=tk.CENTER)

        for col in cols:
            tree.heading(col, text=col)

        model_data = data.get("model_performance_metrics", {})
        for model_name, metrics in model_data.items():
            if metrics:  # check if metrics are not empty
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        model_name.upper(),
                        f"{metrics.get('MAP', 0.0):.4f}",
                        f"{metrics.get('Precision@10', 0.0):.4f}",
                        f"{metrics.get('Recall', 0.0):.4f}",
                        f"{metrics.get('nDCG', 0.0):.4f}",
                    ),
                )
        tree.pack(fill=tk.X, pady=5)

        # Efficiency Summary Aggregates Card
        efficiency_frame = ttk.LabelFrame(
            tab1,
            text=" Performance Aggregates Summary ",
            padding="15",
        )
        efficiency_frame.pack(fill=tk.BOTH, expand=True)

        eff_data = data.get("search_efficiency_comparison", {})
        status = eff_data.get("status", "unknown")

        if status == "completed":
            card_inner = ttk.Frame(efficiency_frame, style="Card.TFrame", padding="15")
            card_inner.pack(fill=tk.BOTH, expand=True)

            def create_metric_block(parent, row, col, label_text, val_text, color):
                lbl = tk.Label(
                    parent,
                    text=label_text,
                    font=("Segoe UI", 9, "bold"),
                    bg=self.bg_card,
                    fg=self.fg_muted,
                )
                lbl.grid(row=row, column=col, padx=20, pady=(5, 2), sticky=tk.W)
                val = tk.Label(
                    parent,
                    text=val_text,
                    font=("Segoe UI", 13, "bold"),
                    bg=self.bg_card,
                    fg=color,
                )
                val.grid(row=row + 1, column=col, padx=20, pady=(0, 5), sticky=tk.W)

            faiss_t = eff_data.get("average_faiss_time_seconds", 0.0)
            manual_t = eff_data.get("average_manual_time_seconds", 0.0)
            create_metric_block(
                card_inner,
                0,
                0,
                "Avg FAISS Response (API)",
                f"{faiss_t:.4f} sec",
                self.accent_green,
            )
            create_metric_block(
                card_inner,
                0,
                1,
                "Avg Manual Response (Brute)",
                f"{manual_t:.4f} sec",
                "#ef4444",
            )

            speedup = eff_data.get("speedup_factor", 1.0)
            precision_match = eff_data.get("average_precision_match_at_5", 1.0) * 100.0
            create_metric_block(
                card_inner,
                2,
                0,
                "Speedup Multiplier Factor",
                f"{speedup:.2f}x Faster",
                "#f59e0b",
            )
            create_metric_block(
                card_inner,
                2,
                1,
                "Index Match Precision @5",
                f"{precision_match:.1f}%",
                "#8b5cf6",
            )
        else:
            lbl_fallback = tk.Label(
                efficiency_frame,
                text=f"Benchmark summary skipped or unavailable. Status: {status}",
                font=("Segoe UI", 10, "italic"),
                bg=self.bg_main,
                fg=self.fg_muted,
            )
            lbl_fallback.pack(expand=True, pady=20)

        # ---- TAB 2: DETAILED QUERY BREAKDOWN ----
        tab2 = ttk.Frame(notebook, padding="10")
        notebook.add(tab2, text=" Detailed Results per Query ")

        det_frame = ttk.LabelFrame(
            tab2,
            text=" Query-by-Query Benchmarking Metric Values ",
            padding="10",
        )
        det_frame.pack(fill=tk.BOTH, expand=True)

        detailed_queries = eff_data.get("detailed_queries", [])

        if status == "completed" and detailed_queries:
            # Treeview table with Scrollbar for handling query logs safely
            det_cols = (
                "Query Text Sample",
                "FAISS Time (s)",
                "Manual Time (s)",
                "Speedup Factor",
                "Precision Match @5",
            )

            tree_scroll = ttk.Scrollbar(det_frame)
            tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            det_tree = ttk.Treeview(
                det_frame,
                columns=det_cols,
                show="headings",
                style="Popup.Treeview",
                yscrollcommand=tree_scroll.set,
            )
            tree_scroll.config(command=det_tree.yview)

            det_tree.column("Query Text Sample", width=250, anchor=tk.W)
            det_tree.column("FAISS Time (s)", width=110, anchor=tk.CENTER)
            det_tree.column("Manual Time (s)", width=110, anchor=tk.CENTER)
            det_tree.column("Speedup Factor", width=110, anchor=tk.CENTER)
            det_tree.column("Precision Match @5", width=120, anchor=tk.CENTER)

            for col in det_cols:
                det_tree.heading(col, text=col)

            # Insert all details per query logs
            for item in detailed_queries:
                f_time = item.get("faiss_time", 0.0)
                m_time = item.get("manual_time", 0.0)
                spd = item.get("speedup", 0.0)
                pm5 = item.get("precision_match_at_5", 0.0)

                det_tree.insert(
                    "",
                    tk.END,
                    values=(
                        item.get("query_text", "Unknown"),
                        f"{f_time:.4f}s",
                        f"{m_time:.4f}s",
                        f"{spd:.2f}x",
                        f"{pm5 * 100.0:.0f}%",
                    ),
                )
            det_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        else:
            lbl_fallback_det = tk.Label(
                det_frame,
                text="No query-by-query breakdown record log found for this dataset.\nMake sure you have run the full benchmarking script.",
                font=("Segoe UI", 10, "italic"),
                bg=self.bg_main,
                fg=self.fg_muted,
                justify=tk.CENTER,
            )
            lbl_fallback_det.pack(expand=True, pady=20)

        # Close Window Button
        btn_close = tk.Button(
            main_container,
            text="Dismiss Report",
            command=popup.destroy,
            bg=self.bg_input,
            fg=self.fg_main,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=20,
            pady=5,
            cursor="hand2",
        )
        btn_close.pack(anchor=tk.E, pady=(5, 0))

    def _api_post_alert(self, url, payload, success_msg):
        try:
            r = requests.post(url, json=payload, timeout=5)
            if r.status_code in (200, 202):
                messagebox.showinfo("Pipeline Action", success_msg)
            else:
                messagebox.showerror("Error", f"Backend rejected request: {r.text}")
        except Exception as e:
            messagebox.showerror("Network Error", f"Cannot connect to module: {e}")

    def update_slider_label(self, val):
        self.lbl_weight.config(text=f"BM25 Weight ({float(val):.2f}):")

    def toggle_parameter_views(self, event=None):
        mode = self.model_var.get()
        if mode in ("bm25", "hybrid", "hybrid_serial"):
            self.lbl_k1.grid()
            self.ent_k1.grid()
            self.lbl_b.grid()
            self.ent_b.grid()
        else:
            self.lbl_k1.grid_remove()
            self.ent_k1.grid_remove()
            self.lbl_b.grid_remove()
            self.ent_b.grid_remove()
        if mode == "hybrid":
            self.lbl_weight.grid()
            self.slider_weight.grid()
        else:
            self.lbl_weight.grid_remove()
            self.slider_weight.grid_remove()
        if mode == "hybrid" or mode == "bert":
            self.chk_faiss.pack(side=tk.LEFT, padx=5)
        else:
            self.chk_faiss.pack_forget()

    def handle_query_key_release(self, event):
        if event.keysym in ("Up", "Down", "Return", "Escape"):
            return
        if self.suggestion_timer:
            self.after_cancel(self.suggestion_timer)
        prefix = self.query_var.get().strip()
        if len(prefix) < 2:
            self.suggestion_box.place_forget()
            return
        self.suggestion_timer = self.after(250, self.fetch_suggestions, prefix)

    def fetch_suggestions(self, prefix):
        try:
            r = requests.get(
                f"{SEARCH_API_URL}/suggest/",
                params={
                    "dataset_name": self.dataset_var.get().split(" ")[0],
                    "prefix": prefix,
                },
                timeout=1,
            )
            if r.status_code == 200:
                self.render_suggestions(r.json())
        except Exception:
            pass

    def render_suggestions(self, suggestions):
        if not suggestions:
            self.suggestion_box.place_forget()
            return
        self.suggestion_box.delete(0, tk.END)
        for item in suggestions:
            self.suggestion_box.insert(tk.END, item)
        x = self.query_entry.winfo_x() + self.query_entry.master.winfo_x()
        y = (
            self.query_entry.winfo_y()
            + self.query_entry.master.winfo_y()
            + self.query_entry.winfo_height()
            + 275
        )
        self.suggestion_box.place(
            x=x, y=y, width=self.query_entry.winfo_width(), height=len(suggestions) * 18
        )
        self.suggestion_box.lift()

    def select_suggestion_item(self, event):
        if not self.suggestion_box.curselection():
            return
        self.query_var.set(
            self.suggestion_box.get(self.suggestion_box.curselection()[0])
        )
        self.suggestion_box.place_forget()

    def start_search_thread(self):
        self.suggestion_box.place_forget()
        query = self.query_var.get().strip()
        if not query:
            return
        self.btn_search.config(state=tk.DISABLED)
        self.status_label.config(
            text="Querying backend matrix models...", fg=self.accent_blue
        )
        for child in self.scrollable_frame.winfo_children():
            child.destroy()
        threading.Thread(target=self.execute_search, args=(query,), daemon=True).start()

    def execute_search(self, query):
        payload = {
            "query": query,
            "dataset_name": self.dataset_var.get().split(" ")[0],
            "model_type": self.model_var.get(),
            "k1": float(self.ent_k1.get() or 1.6),
            "b": float(self.ent_b.get() or 0.75),
            "use_faiss": self.faiss_var.get(),
            "hybrid_bm25_weight": float(self.slider_weight.get()),
            "top_k": 10,
        }
        try:
            response = requests.post(
                f"{SEARCH_API_URL}/search/", json=payload, timeout=15
            )
            if response.status_code != 200:
                raise Exception(response.json().get("detail", "Server error"))
            search_results = response.json().get("results", [])

            if not search_results:
                self.after(0, self.finalize_ui_state, "No matches found.", [])
                return

            doc_ids_scores = {res["doc_id"]: res["score"] for res in search_results}
            conn = sqlite3.connect(DB_CONFIG.get("database", "ir_system.db"))
            cursor = conn.cursor()
            placeholders = ", ".join(["?"] * len(doc_ids_scores))
            cursor.execute(
                f"SELECT doc_id, original_text FROM documents WHERE doc_id IN ({placeholders}) AND dataset = ?",
                list(doc_ids_scores.keys()) + [payload["dataset_name"]],
            )
            docs_from_db = cursor.fetchall()
            conn.close()

            final_results = [
                {
                    "doc_id": r[0],
                    "original_text": r[1],
                    "score": doc_ids_scores.get(r[0], 0.0),
                }
                for r in docs_from_db
            ]
            final_results.sort(key=lambda x: x["score"], reverse=True)
            self.after(
                0,
                self.finalize_ui_state,
                f"Done. Found {len(final_results)} items.",
                final_results,
            )
        except Exception as e:
            self.after(0, self.finalize_ui_state, f"Failure: {str(e)}", [], True)

    def finalize_ui_state(self, status_msg, results, is_error=False):
        self.btn_search.config(state=tk.NORMAL)
        self.status_label.config(
            text=status_msg, fg="#ef4444" if is_error else self.accent_green
        )
        if is_error or not results:
            return

        for doc in results:
            card = ttk.Frame(self.scrollable_frame, style="Card.TFrame", padding="12")
            card.pack(fill=tk.X, padx=5, pady=5)
            h = ttk.Frame(card, style="Card.TFrame")
            h.pack(fill=tk.X)
            tk.Label(
                h,
                text=f"Doc ID: {doc['doc_id']}",
                font=("Segoe UI", 10, "bold"),
                bg=self.bg_card,
                fg="#60a5fa",
            ).pack(side=tk.LEFT)
            tk.Label(
                h,
                text=f" Score: {doc['score']:.4f} ",
                font=("Segoe UI", 9, "bold"),
                bg=self.accent_blue,
                fg=self.fg_main,
            ).pack(side=tk.RIGHT)
            tk.Message(
                card,
                text=doc["original_text"],
                font=("Segoe UI", 10),
                bg=self.bg_card,
                fg="#e4e4e7",
                aspect=500,
                justify=tk.RIGHT
                if self._contains_arabic(doc["original_text"])
                else tk.LEFT,
            ).pack(fill=tk.X, pady=4)

    def _contains_arabic(self, text):
        return any("\u0600" <= char <= "\u06ff" for char in text)


if __name__ == "__main__":
    app = PotatoSafeIRApp()
    app.mainloop()
