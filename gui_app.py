import ezdxf
from ezdxf import path
from ezdxf.math import Matrix44
import math
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely.affinity import rotate
from PIL import Image
import json

# ▼▼▼ 追加した魔法のコード ▼▼▼
import multiprocessing.dummy as mp_dummy
import complete_nesting_algorithm_Version2_Version3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 裏の作業員を「別プロセス」から「別スレッド」に切り替えることで、GUIが全てのログを受信できるようになります！
complete_nesting_algorithm_Version2_Version3.mp.Pool = mp_dummy.Pool
# ▲▲▲ ここまで ▲▲▲

from complete_nesting_algorithm_Version2_Version3 import Part, NestingAlgorithm
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import multiprocessing
import sys
import re


# ===== データ構造 =====
class OriginalShape:
    def __init__(self, entities, polygon):
        self.entities = entities
        self.polygon = polygon
        self.centroid = polygon.centroid


# ===== 1. オートジョイン付きの図形抽出 =====
def extract_original_shapes_from_dxf(filepath: str):
    try:
        doc = ezdxf.readfile(filepath)
    except Exception as e:
        return []

    msp = doc.modelspace()
    entities_data = []

    for entity in msp:
        try:
            if entity.dxftype() in ['TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'INSERT']:
                continue
            p = path.make_path(entity)
            vertices = list(p.flattening(distance=0.5))
            if len(vertices) < 2:
                continue
            points = [(v.x, v.y) for v in vertices]
            line = LineString(points)
            entities_data.append({'entity': entity, 'line': line})
        except:
            pass

    if not entities_data: return []

    buffered_lines = [data['line'].buffer(0.2) for data in entities_data]
    union_islands = unary_union(buffered_lines)

    if union_islands.geom_type == 'Polygon':
        islands = [union_islands]
    elif union_islands.geom_type == 'MultiPolygon':
        islands = list(union_islands.geoms)
    else:
        islands = []

    islands = sorted(islands, key=lambda x: x.area, reverse=True)
    final_islands = []

    for island in islands:
        exterior_poly = Polygon(island.exterior)
        contained = False
        for final_isl in final_islands:
            if final_isl.contains(exterior_poly.centroid):
                contained = True
                break
        if not contained:
            final_islands.append(exterior_poly)

    groups = {i: [] for i in range(len(final_islands))}

    for data in entities_data:
        line = data['line']
        best_idx = -1
        for i, final_isl in enumerate(final_islands):
            if final_isl.contains(line.centroid) or final_isl.distance(line) < 0.5:
                best_idx = i
                break
        if best_idx != -1:
            groups[best_idx].append(data['entity'])

    shapes = []
    for i, entities in groups.items():
        if not entities: continue
        calc_poly = final_islands[i].buffer(-0.1).simplify(0.5, preserve_topology=True)
        if calc_poly.is_empty or calc_poly.area < 1:
            calc_poly = final_islands[i]
        if calc_poly.is_valid and calc_poly.area > 0:
            shapes.append(OriginalShape(entities, calc_poly))

    return shapes


# ===== 2. 図形の出力 =====
# ===== 2. 図形の出力 =====
def export_to_dxf_with_originals(placed_parts, original_shapes_dict, output_filepath: str, sheet_width: float,
                                 sheet_height: float, alignment: str): # ← alignment を追加
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    frame_points = [(0, 0), (sheet_width, 0), (sheet_width, sheet_height), (0, sheet_height)]
    msp.add_lwpolyline(frame_points, close=True, dxfattribs={'color': 1})

    placed_entities = [] # ★配置したエンティティを記憶するリスト

    for part in placed_parts:
        shape = original_shapes_dict[part.id]
        cx = shape.centroid.x
        cy = shape.centroid.y
        angle_deg = part.rotation

        rotated_poly = rotate(shape.polygon, angle_deg, origin='centroid')
        minx_rot, miny_rot, _, _ = rotated_poly.bounds

        target_x, target_y = part.position
        dx = target_x - minx_rot
        dy = target_y - miny_rot

        m1 = Matrix44.translate(-cx, -cy, 0)
        m2 = Matrix44.z_rotate(math.radians(angle_deg))
        m3 = Matrix44.translate(cx, cy, 0)
        m4 = Matrix44.translate(dx, dy, 0)
        transform_matrix = m1 @ m2 @ m3 @ m4

        for entity in shape.entities:
            new_entity = entity.copy()
            new_entity.transform(transform_matrix)
            new_entity.dxf.color = 3
            msp.add_entity(new_entity)
            placed_entities.append(new_entity) # リストに追加

    # ▼▼▼ 最後に、指定に合わせてエンティティ全体をミラー反転 ▼▼▼
    if alignment == 'top_left':
        # Y軸で反転させて、上に持ち上げる
        m_total = Matrix44.scale(1, -1, 1) @ Matrix44.translate(0, sheet_height, 0)
        for ent in placed_entities:
            ent.transform(m_total)
    elif alignment == 'bottom_right':
        # X軸で反転させて、右に寄せる
        m_total = Matrix44.scale(-1, 1, 1) @ Matrix44.translate(sheet_width, 0, 0)
        for ent in placed_entities:
            ent.transform(m_total)
    elif alignment == 'top_right':
        # XY両方で反転させて、右上に寄せる
        m_total = Matrix44.scale(-1, -1, 1) @ Matrix44.translate(sheet_width, sheet_height, 0)
        for ent in placed_entities:
            ent.transform(m_total)
    # ▲▲▲

    doc.saveas(output_filepath)


# ===== ターミナルの出力を横取りするクラス =====
class PrintLogger:
    def __init__(self, log_callback, progress_callback):
        self.terminal = sys.stdout
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def write(self, message):
        # ターミナル（self.terminal）が存在する場合のみ書き込む
        if self.terminal is not None:
            try:
                self.terminal.write(message)
            except:
                pass

        # GUIのログボックスへの書き込みはターミナルの有無に関わらず実行
        if message.strip():
            self.log_callback(message)
            self.progress_callback(message)

    def flush(self):
        self.terminal.flush()


# ===== GUIアプリケーション本体 =====
class NestingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DXF Auto Nesting Tool")
        self.geometry("1100x750")
        ctk.set_appearance_mode("dark")

        self.file_entries = {}

        self.logger = PrintLogger(self.log, self.parse_progress)
        sys.stdout = self.logger
        sys.stderr = self.logger


        self.grid_columnconfigure(0, weight=1, minsize=400)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.left_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.left_panel, text="DXF 自動ネスティング", font=("Arial", 22, "bold"))
        self.title_label.pack(pady=(0, 10), anchor="w")

        self.file_btn = ctk.CTkButton(self.left_panel, text="📁 DXFファイルを選択 (複数可)", command=self.select_files)
        self.file_btn.pack(pady=5, fill="x")

        self.file_list_frame = ctk.CTkScrollableFrame(self.left_panel, height=150)
        self.file_list_frame.pack(pady=5, fill="x")

        self.settings_frame = ctk.CTkFrame(self.left_panel)
        self.settings_frame.pack(pady=10, fill="x", ipadx=10, ipady=10)

        ctk.CTkLabel(self.settings_frame, text="【材料サイズ設定】", font=("Arial", 14, "bold")).grid(row=0, column=0,
                                                                                                    columnspan=4,
                                                                                                    pady=(5, 10),
                                                                                                    sticky="w", padx=10)

        ctk.CTkLabel(self.settings_frame, text="幅(W):").grid(row=1, column=0, padx=5, sticky="e")
        self.entry_width = ctk.CTkEntry(self.settings_frame, width=70)
        self.entry_width.insert(0, "900")
        self.entry_width.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(self.settings_frame, text="高さ(H):").grid(row=1, column=2, padx=5, sticky="e")
        self.entry_height = ctk.CTkEntry(self.settings_frame, width=70)
        self.entry_height.insert(0, "600")
        self.entry_height.grid(row=1, column=3, padx=5, pady=5)

        ctk.CTkLabel(self.settings_frame, text="隙間(mm):").grid(row=2, column=0, padx=5, sticky="e")
        self.entry_margin = ctk.CTkEntry(self.settings_frame, width=70)
        self.entry_margin.insert(0, "2.0")
        self.entry_margin.grid(row=2, column=1, padx=5, pady=5)

        self.rotation_var = ctk.BooleanVar(value=True)
        self.chk_rotation = ctk.CTkCheckBox(self.settings_frame, text="回転を許可", variable=self.rotation_var)
        self.chk_rotation.grid(row=2, column=2, columnspan=2, pady=5, sticky="w", padx=5)


        ctk.CTkLabel(self.settings_frame, text="形状優先:").grid(row=3, column=0, padx=5, pady=(5, 0), sticky="e")

        self.priority_var = ctk.StringVar(value="none")

        self.priority_subframe = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.priority_subframe.grid(row=3, column=1, columnspan=3, sticky="w", pady=(5, 0))

        ctk.CTkRadioButton(self.priority_subframe, text="面積 (標準)", variable=self.priority_var, value="none").pack(
            side="left", padx=(0, 10))
        ctk.CTkRadioButton(self.priority_subframe, text="高さ (平べったく)", variable=self.priority_var,
                           value="height").pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(self.priority_subframe, text="幅 (縦長に)", variable=self.priority_var, value="width").pack(
            side="left")

        ctk.CTkLabel(self.settings_frame, text="配置基準:").grid(row=4, column=0, padx=5, pady=(5, 0), sticky="e")
        self.alignment_var = ctk.StringVar(value="bottom_left")
        self.align_subframe = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.align_subframe.grid(row=4, column=1, columnspan=3, sticky="w", pady=(5, 0))
        ctk.CTkRadioButton(self.align_subframe, text="左下 (標準)", variable=self.alignment_var,
                           value="bottom_left").pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(self.align_subframe, text="左上 (レーザー用)", variable=self.alignment_var,
                           value="top_left").pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(self.align_subframe, text="右下", variable=self.alignment_var, value="bottom_right").pack(
            side="left", padx=(0, 10))
        ctk.CTkRadioButton(self.align_subframe, text="右上", variable=self.alignment_var, value="top_right").pack(
            side="left")

        ctk.CTkLabel(self.settings_frame, text="計算モード:").grid(row=5, column=0, padx=5, pady=(5, 0), sticky="ne")
        self.accuracy_var = ctk.StringVar(value="normal")
        self.acc_subframe = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.acc_subframe.grid(row=5, column=1, columnspan=3, sticky="w", pady=(5, 0))

        # --- 1段目 ---
        ctk.CTkRadioButton(self.acc_subframe, text="精密", variable=self.accuracy_var, value="high").grid(row=0,
                                                                                                          column=0,
                                                                                                          padx=(0, 10),
                                                                                                          pady=2,
                                                                                                          sticky="w")
        ctk.CTkRadioButton(self.acc_subframe, text="標準", variable=self.accuracy_var, value="normal").grid(row=0,
                                                                                                            column=1,
                                                                                                            padx=(0,
                                                                                                                  10),
                                                                                                            pady=2,
                                                                                                            sticky="w")
        ctk.CTkRadioButton(self.acc_subframe, text="速さ", variable=self.accuracy_var, value="fast").grid(row=0,
                                                                                                          column=2,
                                                                                                          padx=(0, 10),
                                                                                                          pady=2,
                                                                                                          sticky="w")

        # --- 2段目 ---
        ctk.CTkRadioButton(self.acc_subframe, text="合体オフ(非推奨)", variable=self.accuracy_var, value="none").grid(
            row=1, column=0, padx=(0, 10), pady=(5, 2), sticky="w")
        ctk.CTkRadioButton(self.acc_subframe, text="カスタム", variable=self.accuracy_var, value="custom").grid(row=1,
                                                                                                                column=1,
                                                                                                                padx=(0,
                                                                                                                      5),
                                                                                                                pady=(5,
                                                                                                                      2),
                                                                                                                sticky="w")

        self.custom_btn = ctk.CTkButton(self.acc_subframe, text="⚙️", width=30, command=self.open_custom_settings)
        self.custom_btn.grid(row=1, column=2, pady=(5, 2), sticky="w")

        # カスタム設定の読み込み
        self.custom_config = self.load_custom_config()


        self.run_btn = ctk.CTkButton(self.left_panel, text="▶ 保存先を決めて実行", command=self.start_nesting,
                                     fg_color="#28a745", hover_color="#218838", font=("Arial", 16, "bold"), height=40)
        self.run_btn.pack(pady=20, fill="x")

        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # タブビューを作成
        self.tab_view = ctk.CTkTabview(self.right_panel)
        self.tab_view.pack(padx=10, pady=(10, 5), fill="both", expand=True)  # 余白を少し調整

        self.tab_log = self.tab_view.add("ログ")
        self.tab_preview = self.tab_view.add("プレビュー")

        # --- ▼▼▼ ここを修正 ▼▼▼ ---
        # プレビュータブの中身をスクロール可能にする
        # self.tab_preview 自体をスクロールフレームにするのではなく、
        # その中にスクロールフレームを配置します。
        self.preview_frame = ctk.CTkFrame(self.tab_preview, fg_color="transparent")
        self.preview_frame.pack(fill="both", expand=True)
        self.canvas_widget = None
        self.toolbar_widget = None

        self.preview_label = ctk.CTkLabel(self.preview_frame, text="計算完了後にここに配置結果が表示されます")
        self.preview_label.pack(padx=5, pady=5, expand=True, anchor="center")
        # --- ▲▲▲ ここまで修正 ▲▲▲ ---

        # ログタブの中身
        self.log_box = ctk.CTkTextbox(self.tab_log, state="disabled", wrap="word", font=("Consolas", 12))
        self.log_box.pack(padx=5, pady=5, fill="both", expand=True)

        # プログレスバー
        self.progress_bar = ctk.CTkProgressBar(self.right_panel, mode="determinate")
        self.progress_bar.pack(padx=10, pady=(0, 15), fill="x")
        self.progress_bar.set(0)

    def log(self, message):
        self.after(0, self._append_log, message)

    def _append_log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def parse_progress(self, message):
        if "NFPの事前計算" in message:
            self.after(0, self._set_progress, "determinate", 0.0)

        match_nfp = re.search(r"NFP計算進捗:\s*([\d\.]+)%", message)
        if match_nfp:
            val = float(match_nfp.group(1)) / 100.0
            self.after(0, self._set_progress, "determinate", val)

        if "プロセスで並列実行を開始" in message:
            self.after(0, self._set_progress, "indeterminate", None)

        if "全試行完了" in message:
            self.after(0, self._set_progress, "determinate", 1.0)

    def _set_progress(self, mode, val=None):
        if mode == "determinate":
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            if val is not None:
                self.progress_bar.set(val)
        else:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()

    def select_files(self):
        initial_dir = os.path.expanduser("~")
        filepaths = filedialog.askopenfilenames(initialdir=initial_dir, filetypes=[("DXF Files", "*.dxf")])

        if filepaths:
            # ▼ 毎回クリアせず、新しく選んだファイルだけを「追加」する
            for filepath in filepaths:
                if filepath in self.file_entries:
                    continue  # 既に追加されているファイルは無視

                filename = os.path.basename(filepath)
                row_frame = ctk.CTkFrame(self.file_list_frame, fg_color="transparent")
                row_frame.pack(fill="x", pady=2)

                lbl = ctk.CTkLabel(row_frame, text=filename, width=150, anchor="w")
                lbl.pack(side="left", padx=5)

                # ▼▼▼ 新規追加: 個別削除ボタンの機能 ▼▼▼
                def delete_file(fp=filepath, row=row_frame):
                    row.destroy()
                    if fp in self.file_entries:
                        del self.file_entries[fp]

                del_btn = ctk.CTkButton(row_frame, text="×", width=25, height=25, fg_color="#dc3545",
                                        hover_color="#c82333", command=delete_file)
                del_btn.pack(side="right", padx=5)
                # ▲▲▲

                qty_entry = ctk.CTkEntry(row_frame, width=40)
                qty_entry.insert(0, "1")
                qty_entry.pack(side="right", padx=5)

                ctk.CTkLabel(row_frame, text="個数:").pack(side="right")
                self.file_entries[filepath] = qty_entry

            print(f"ファイルリストを更新しました。現在 {len(self.file_entries)} 件セットされています。")

    # ▼▼▼ 新規追加: カスタムパラメータの保存と読み込み ▼▼▼
    def load_custom_config(self):
        import json
        try:
            with open('nesting_custom_config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            # ファイルが無い場合（初回起動時）の初期値
            return {
                "simp_val": 1.0, "step_div": 8, "angle_step": 15,
                "pop_size": 60, "gens": 20, "patience": 10
            }

    def open_custom_settings(self):
        top = ctk.CTkToplevel(self)
        top.title("カスタムパラメータ設定")
        top.geometry("340x380")
        top.transient(self)

        entries = {}
        labels = {
            "simp_val": "輪郭の粗さ (例:1.0~5.0):",
            "step_div": "衝突判定の間引き (例:4~12):",
            "angle_step": "回転角度の刻み (例:15~90):",
            "pop_size": "GA 個体数 (例:40~120):",
            "gens": "GA 最大世代数 (例:10~50):",
            "patience": "GA 諦める世代数 (例:3~15):"
        }

        row = 0
        for key, label_text in labels.items():
            ctk.CTkLabel(top, text=label_text).grid(row=row, column=0, padx=10, pady=10, sticky="e")
            e = ctk.CTkEntry(top, width=80)
            e.insert(0, str(self.custom_config.get(key, "")))
            e.grid(row=row, column=1, padx=10, pady=10)
            entries[key] = e
            row += 1

        def save_and_close():
            import json
            try:
                for key in entries:
                    # simp_valは小数、それ以外は整数として保存
                    self.custom_config[key] = float(entries[key].get()) if "simp" in key else int(entries[key].get())
                with open('nesting_custom_config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.custom_config, f)
                self.accuracy_var.set("custom")  # 自動でカスタムモードを選択状態にする
                top.destroy()
            except ValueError:
                messagebox.showerror("エラー", "数値を正しく入力してください", parent=top)

        ctk.CTkButton(top, text="保存して閉じる", command=save_and_close).grid(row=row, column=0, columnspan=2, pady=20)





    def start_nesting(self):
        if not self.file_entries:
            messagebox.showerror("エラー", "DXFファイルを選択してください！")
            return

        try:
            sheet_w = float(self.entry_width.get())
            sheet_h = float(self.entry_height.get())
            margin = float(self.entry_margin.get())
        except ValueError:
            messagebox.showerror("エラー", "数値入力欄に誤りがあります！")
            return

        file_quantities = []
        for filepath, entry in self.file_entries.items():
            try:
                qty = int(entry.get())
                if qty > 0:
                    file_quantities.append((filepath, qty))
            except ValueError:
                messagebox.showerror("エラー", "個数は整数で入力してください！")
                return

        if not file_quantities:
            return

        initial_dir = os.path.expanduser("~/Desktop")
        save_path = filedialog.asksaveasfilename(
            title="結果の保存先を指定してください",
            initialdir=initial_dir,
            initialfile="ネスティング結果.dxf",
            defaultextension=".dxf",
            filetypes=[("DXF Files", "*.dxf")]
        )

        if not save_path:
            print("保存先の選択がキャンセルされました。")
            return

        self.run_btn.configure(state="disabled", text="⏳ 処理中...")
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.progress_bar.set(0)

        allow_rot = self.rotation_var.get()
        priority_val = self.priority_var.get()
        alignment_val = self.alignment_var.get()
        accuracy_val = self.accuracy_var.get()
        custom_config = self.custom_config

        threading.Thread(target=self.run_nesting_logic,
                         args=(file_quantities, sheet_w, sheet_h, margin, allow_rot, priority_val, alignment_val,
                               accuracy_val, custom_config, save_path),
                         daemon=True).start()

    def run_nesting_logic(self, file_quantities, sheet_width, sheet_height, margin, allow_rotation, priority, alignment, accuracy,
                          custom_config, save_path):

        try:
            parts = []
            original_shapes_dict = {}
            part_counter = 1

            print("=== ネスティング処理を開始 ===")
            print("1. ファイルから図形データを抽出・結合しています...")
            for filepath, qty in file_quantities:
                shapes = extract_original_shapes_from_dxf(filepath)
                if not shapes:
                    print(f"⚠️ 警告: {os.path.basename(filepath)} から図形を抽出できませんでした。")
                    continue

                for _ in range(qty):
                    for shape in shapes:
                        part_id = f"DXF_{part_counter}"
                        points = list(shape.polygon.exterior.coords)[:-1]
                        parts.append(Part(part_id, points))
                        original_shapes_dict[part_id] = shape
                        part_counter += 1

            if not parts:
                self.show_result("エラー", "配置できる図形がありませんでした。処理を中止します。")
                return

            print(f"合計 {len(parts)} 個の部品を配置します。")
            # ... (中略) ...
            print(f"2. パズルの最適解を計算中... (優先モード: {priority} / 配置基準: {alignment})")

            nester = NestingAlgorithm(parts, sheet_width, sheet_height, safety_margin=margin,
                                      allow_rotation=allow_rotation, priority=priority, alignment=alignment,
                                      accuracy=accuracy, custom_config=custom_config)
            result_packer, _ = nester.run()

            # ▼▼▼ AIの計算が終わった後、DXF出力の直前にキャンバスを反転させる ▼▼▼
            if alignment == 'top_left':
                result_packer.flip_to_top_left()
            elif alignment == 'bottom_right':
                result_packer.flip_to_bottom_right()
            elif alignment == 'top_right':
                result_packer.flip_to_top_right()
            # ▲▲▲

            print("\n3. 計算完了！高品質DXFファイルを生成しています...")
            export_to_dxf_with_originals(result_packer.placed_parts, original_shapes_dict, save_path, sheet_width,
                                         sheet_height, alignment)  # ← 一番最後に alignment を追加！

            print(f"🎉 成功！ファイルを出力しました。\n保存先: {save_path}")

            rot_text = "回転あり" if allow_rotation else "回転なし"
            plot_title = f"Result ({sheet_width} x {sheet_height} / M:{margin} / {rot_text})"

            preview_img_path = os.path.join(os.path.dirname(save_path), "preview_nesting.png")

            # ▼▼▼ 画像ではなく、Figure(グラフ本体)を取得する。ダークモードもON ▼▼▼
            fig = result_packer.visualize(save_path=preview_img_path, show_plot=False,
                                          title=plot_title, return_fig=True, dark_mode=True,
                                          fixed_figsize=(6, 4))

            # ▼▼▼ 新規追加：最終結果が枠に収まっているかチェックしてメッセージを変える ▼▼▼
            final_h = result_packer.get_bin_height()
            final_w = result_packer.get_bin_width_used()

            if final_h > sheet_height or final_w > sheet_width:
                msg_title = "完了 (※はみ出し警告)"
                msg_body = f"指定された材料サイズ({sheet_width}x{sheet_height})に収まりきらず、はみ出た状態で出力されました。\n部品を減らすか、材料サイズを大きくしてください。"
            else:
                msg_title = "完了"
                msg_body = "ネスティングが完了しました！"

            self.show_result(msg_title, msg_body, is_success=True, result_packer=result_packer,
                             plot_title=plot_title, fig=fig)


        except Exception as e:
            # ...
            print(f"\n❌ エラー発生: {e}")
            self.show_result("エラー", f"予期せぬエラー: {e}", is_success=False)

    def show_result(self, title, message, is_success=False, result_packer=None, plot_title="", fig=None):
        self.after(0, self._reset_ui, title, message, is_success, result_packer, plot_title, fig)

    def _reset_ui(self, title, message, is_success, result_packer, plot_title, fig):
        self.progress_bar.stop()
        self.run_btn.configure(state="normal", text="▶ 保存先を決めて実行")

        if not is_success:
            messagebox.showerror(title, message)
        else:
            # ▼▼▼ MatplotlibのグラフをGUIに直接埋め込む ▼▼▼
            if fig:
                # 古いキャンバスがあれば削除
                if self.canvas_widget:
                    self.canvas_widget.destroy()
                if self.toolbar_widget:
                    self.toolbar_widget.destroy()

                # 新しいキャンバスを作成して配置
                canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
                canvas.draw()

                # CADらしく、ズームや移動ができるツールバーを追加！
                self.toolbar_widget = ctk.CTkFrame(self.preview_frame, height=40)
                self.toolbar_widget.pack(side="bottom", fill="x", pady=(5, 0))
                toolbar = NavigationToolbar2Tk(canvas, self.toolbar_widget)
                toolbar.update()

                self.canvas_widget = canvas.get_tk_widget()
                self.canvas_widget.pack(side="top", fill="both", expand=True)

                # 自動的に「プレビュー」タブに切り替える
                self.tab_view.set("プレビュー")

            messagebox.showinfo(title, message)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = NestingApp()
    app.mainloop()