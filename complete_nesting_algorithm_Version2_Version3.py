import math
import random
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'  # または 'Meiryo', 'Yu Gothic' など
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.affinity import rotate, translate
from typing import List, Tuple, Dict, Any, Union, Optional
import copy
import time
import pyclipper
import multiprocessing as mp
import shutil
import os


class Part:
    """部品を表すクラス（改良版）"""
    def __init__(self, id: str, points: List[Tuple[float, float]], holes=None):
        self.id = id
        self.points = points
        self.holes = holes
        self.polygon = Polygon(points, self.holes)
        self.rotation = 0  # 現在の回転角度
        self.position = (0, 0)  # 現在の配置位置
        
    def get_area(self) -> float:
        """部品の面積を取得"""
        return self.polygon.area
        
    def rotate_to(self, angle: float) -> 'Part':
        """指定角度に回転させた新しい部品を返す"""
        rotated_polygon = rotate(self.polygon, angle, origin='centroid')
        part_copy = copy.deepcopy(self)
        part_copy.polygon = rotated_polygon
        part_copy.rotation = angle
        # 回転後のポイントも更新
        part_copy.points = list(rotated_polygon.exterior.coords)[:-1]  # 最後の点は最初と同じなので除外
        return part_copy
    
    def translate_to(self, position: Tuple[float, float]) -> 'Part':
        """指定位置に移動させた新しい部品を返す（左下隅を基準）"""
        # 現在のバウンディングボックスを取得
        minx, miny, _, _ = self.polygon.bounds
        
        # 左下隅から目的位置までの差分を計算
        dx = position[0] - minx
        dy = position[1] - miny
        
        translated_polygon = translate(self.polygon, dx, dy)
        part_copy = copy.deepcopy(self)
        part_copy.polygon = translated_polygon
        part_copy.position = position
        # 移動後のポイントも更新
        part_copy.points = list(translated_polygon.exterior.coords)[:-1]
        return part_copy
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """部品のバウンディングボックスを取得 (minx, miny, maxx, maxy)"""
        return self.polygon.bounds
    
    def __str__(self) -> str:
        return f"Part {self.id} at position {self.position}, rotation {self.rotation}°"

class NFP:
    """No-Fit Polygonを表すクラス（pyclipperを使用した改良版）"""

    def __init__(self, stationary_part: Part, orbiting_part: Part, margin: float = 0.0):
        self.stationary_part = stationary_part
        self.orbiting_part = orbiting_part
        self.margin = margin  # ★ マージンを保存
        self.nfp = self._calculate_nfp()

    def _calculate_nfp(self) -> Union[Polygon, MultiPolygon]:
        """pyclipperを使用してNFPを計算する（マージン対応）"""
        # ▼▼▼ バッファ処理を追加 ▼▼▼
        # マージンが指定されている場合、固定部品を少しだけ大きくする
        if self.margin > 0:
            buffered_stationary_poly = self.stationary_part.polygon.buffer(self.margin)
            stationary_points = list(buffered_stationary_poly.exterior.coords)
        else:
            stationary_points = self.stationary_part.points

        # 移動部品を180度回転させてミラーリング
        orbiting_points = self.orbiting_part.points
        orbiting_centroid = self.orbiting_part.polygon.centroid
        mirrored_orbiting_points = []

        for point in orbiting_points:
            # 点を中心に対して180度回転（ミラーリング）
            mx = orbiting_centroid.x - (point[0] - orbiting_centroid.x)
            my = orbiting_centroid.y - (point[1] - orbiting_centroid.y)
            mirrored_orbiting_points.append((mx, my))

        # pyclipperで処理するために点を整数にスケーリング
        scale = 100000  # 高い精度のための大きなスケール値

        # 点をスケーリングして整数に変換
        scaled_stationary = [(int(p[0] * scale), int(p[1] * scale)) for p in stationary_points]
        scaled_mirrored = [(int(p[0] * scale), int(p[1] * scale)) for p in mirrored_orbiting_points]

        # Minkowskiの和を計算
        try:
            minkowski_sum = pyclipper.MinkowskiSum(scaled_stationary, scaled_mirrored, True)

            # 結果をスケールバックして元のサイズに戻す
            result_polygon_points = []

            # 外側の輪郭と穴を考慮する
            for path in minkowski_sum:
                rescaled_path = [(p[0] / scale, p[1] / scale) for p in path]
                result_polygon_points.append(rescaled_path)

            # 結果のポリゴンを作成
            if not result_polygon_points:
                # 空の結果の場合（エラー状態）
                print(f"警告: NFPの計算結果が空です ({self.stationary_part.id} と {self.orbiting_part.id})")
                return Polygon()

            # 最初の輪郭は外側の輪郭、残りは穴
            exterior = result_polygon_points[0]
            holes = result_polygon_points[1:] if len(result_polygon_points) > 1 else []

            try:
                # Shapely Polygonを生成（穴も考慮）
                if holes:
                    return Polygon(exterior, holes)
                else:
                    return Polygon(exterior)
            except Exception as e:
                print(f"NFP生成エラー: {e}")
                return Polygon(exterior)  # エラー時は穴なしで作成
        except Exception as e:
            print(f"Minkowski和の計算エラー: {e}")
            # エラー時はバックアップとして凸包を使用
            combined_points = stationary_points + mirrored_orbiting_points
            return Polygon(combined_points).convex_hull
    
    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """指定された位置がNFPの外側（配置可能）かどうかを確認"""
        try:
            # NFPが無効な場合は配置不可
            if self.nfp.is_empty or not self.nfp.is_valid:
                return False
                
            # ポイントがNFPの内側にあるかチェック
            return not self.nfp.contains(Point(position))
        except Exception as e:
            print(f"位置チェックエラー: {e}")
            return False  # エラー時は安全のため配置不可

class NFPCalculator:
    """NFPを事前計算して保存するクラス（並列処理対応）"""
    def __init__(self, parts: List[Part], angle_step: int = 20, margin: float = 0.0):
        self.parts = parts
        self.angle_step = angle_step
        self.nfp_cache = {}  # キャッシュ: (stationary_id, orbiting_id, angle) -> NFP
        self.margin = margin  # ★ マージンを保存

    def precompute_nfps(self) -> None:
        """すべての部品ペアと角度のNFPを事前計算"""
        print("NFPを事前計算中...")
        total_computations = len(self.parts) * (len(self.parts) - 1) * (360 // self.angle_step)
        completed = 0
        
        for i, part1 in enumerate(self.parts):
            for j, part2 in enumerate(self.parts):
                if i != j:  # 異なる部品間のみ計算
                    for angle in range(0, 360, self.angle_step):
                        rotated_part2 = part2.rotate_to(angle)
                        nfp = NFP(part1, rotated_part2, margin=self.margin)
                        self.nfp_cache[(part1.id, part2.id, angle)] = nfp
                        
                        completed += 1
                        if completed % 10 == 0:  # 10回ごとに進捗を表示
                            progress = (completed / total_computations) * 100
                            print(f"NFP計算進捗: {progress:.1f}% ({completed}/{total_computations})")
                        
    def get_nfp(self, stationary_id: str, orbiting_id: str, angle: float) -> NFP:
        """キャッシュからNFPを取得。ない場合は近いものを使用または計算"""
        # 角度を最も近い事前計算済みの角度に丸める
        closest_angle = round(angle / self.angle_step) * self.angle_step
        closest_angle = closest_angle % 360
        
        key = (stationary_id, orbiting_id, closest_angle)
        
        if key in self.nfp_cache:
            return self.nfp_cache[key]
        else:
            # キャッシュにない場合は新たに計算
            stationary_part = next(p for p in self.parts if p.id == stationary_id)
            orbiting_part = next(p for p in self.parts if p.id == orbiting_id)
            rotated_orbiting = orbiting_part.rotate_to(angle)
            nfp = NFP(stationary_part, rotated_orbiting, margin=self.margin)
            nfp = NFP(stationary_part, rotated_orbiting)
            self.nfp_cache[key] = nfp
            return nfp

    def calculate_ifp(self, container_polygon: Polygon, part: Part) -> Union[Polygon, MultiPolygon]:
        """Inner-Fit Polygonを計算（両方を原点基準に揃える最終確定版）"""
        if not container_polygon.is_valid or not part.polygon.is_valid:
            return Polygon()

        scale = 100000.0

        # --- 1. コンテナを原点中心に移動 ---
        container_centroid = container_polygon.centroid
        # translate関数を使ってコンテナをその重心が(0,0)に来るように移動
        origin_centered_container = translate(container_polygon, xoff=-container_centroid.x, yoff=-container_centroid.y)

        # --- 2. 部品を準備し、原点中心に移動 ---
        mirrored_poly = rotate(part.polygon, 180, origin='centroid')
        mirrored_centroid = mirrored_poly.centroid
        # ミラーリングした部品も、その重心が(0,0)に来るように移動
        origin_centered_mirrored = translate(mirrored_poly, xoff=-mirrored_centroid.x, yoff=-mirrored_centroid.y)

        # --- 3. Pyclipper用にデータを変換 ---
        scaled_container = [(int(p[0] * scale), int(p[1] * scale)) for p in origin_centered_container.exterior.coords]
        scaled_mirrored = [(int(p[0] * scale), int(p[1] * scale)) for p in origin_centered_mirrored.exterior.coords]

        # --- 4. ミンコフスキー差を計算 ---
        ifp_paths = pyclipper.MinkowskiDiff(scaled_container, scaled_mirrored)
        if not ifp_paths:
            return Polygon()

        # --- 5. 結果をShapelyポリゴンに戻す ---
        rescaled_path = [(p[0] / scale, p[1] / scale) for p in ifp_paths[0]]
        if len(rescaled_path) < 3:
            return Polygon()

        # この時点でのIFPは原点を中心に計算されている
        origin_centered_ifp = Polygon(rescaled_path)

        # --- 6. 最後に、IFPをコンテナの元の中心位置に移動させて戻す ---
        final_ifp = translate(origin_centered_ifp, xoff=container_centroid.x, yoff=container_centroid.y)

        if not final_ifp.is_valid:
            final_ifp = final_ifp.buffer(0)

        return final_ifp


class ImprovedBinPacking:
    """改良版ビンパッキングクラス（安定版・空き領域管理機能付き）"""

    def __init__(self, bin_width: float, bin_height: float, frame_output_dir: str = None):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.placed_parts: List[Part] = []

        # 初期の空き領域は全体のビン（十分な高さを仮定）
        initial_space = Polygon([(0, 0), (self.bin_width, 0), (self.bin_width, self.bin_height), (0, self.bin_height)])
        self.free_spaces: List[Polygon] = [initial_space]
        # --- ▼▼▼ ここから追加 ▼▼▼ ---
        self.frame_output_dir = frame_output_dir
        self.frame_count = 0
        if self.frame_output_dir and not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        # --- ▲▲▲ ここまで追加 ▲▲▲ ---

    def update_free_spaces(self, placed_part: Part) -> None:
        """部品配置後に空き領域を更新する"""
        new_free_spaces = []
        for space in self.free_spaces:
            # 新しい部品と空き領域との差分を計算
            if not space.is_valid or not placed_part.polygon.is_valid:
                continue
            difference = space.difference(placed_part.polygon)

            if difference.is_empty:
                continue

            if difference.geom_type == 'MultiPolygon':
                for poly in difference.geoms:
                    if poly.area > 1.0:  # 小さすぎる領域は無視
                        new_free_spaces.append(poly)
            elif difference.geom_type == 'Polygon':
                if difference.area > 1.0:
                    new_free_spaces.append(difference)
        self.free_spaces = new_free_spaces

    def _generate_candidate_positions(self, part: Part) -> List[Tuple[float, float]]:
        """配置候補位置を生成（接触を回避する最終確定版）"""
        if not self.placed_parts:
            return [(0, 0)]

        candidates = set()
        epsilon = 0.01  # 接触を回避するための非常に小さな隙間

        max_height = 0
        for p in self.placed_parts:
            minx, miny, maxx, maxy = p.get_bounds()

            # 1. 右隣に配置する候補（わずかに隙間を空ける）
            candidates.add((maxx + epsilon, miny))

            # 2. 真上に配置する候補（わずかに隙間を空ける）
            candidates.add((minx, maxy + epsilon))

            max_height = max(max_height, maxy)

        # 3. ビンの左上（新しい行）から配置する候補
        candidates.add((0, max_height + epsilon))

        return list(candidates)

    def find_optimal_position(self, part: Part, nfp_calculator: 'NFPCalculator', priority: str = 'none') -> Tuple[float, float]:
        """最適配置位置を見つける（IFP候補生成 + intersects衝突判定による最終安定版・マージン対応）"""
        if not self.placed_parts:
            return (0, 0)

        # 1. 候補点を生成 (この部分は変更なし)
        candidate_positions = self._generate_candidate_positions(part)
        part_minx, part_miny, _, _ = part.get_bounds()
        part_centroid = part.polygon.centroid
        offset_x = part_minx - part_centroid.x
        offset_y = part_miny - part_centroid.y

        for space in self.free_spaces:
            ifp = nfp_calculator.calculate_ifp(space, part)
            if ifp.is_empty or not ifp.is_valid:
                continue

            polygons = [ifp] if ifp.geom_type == 'Polygon' else list(ifp.geoms)
            for poly in polygons:
                if poly.is_empty or not hasattr(poly, 'exterior') or not poly.exterior:
                    continue
                for v_x, v_y in list(poly.exterior.coords):
                    candidate_positions.append((v_x + offset_x, v_y + offset_y))

        # --- ▼▼▼ ここから修正 ▼▼▼ ---
        # 2. 各候補点を検証 (マージンを考慮)
        valid_positions = []

        # NFPCalculatorからマージン値を取得
        margin = nfp_calculator.margin

        for pos in candidate_positions:
            try:
                test_part = part.translate_to(pos)
                test_bounds = test_part.get_bounds()

                if (test_bounds[0] < -margin or
                        test_bounds[2] > self.bin_width + margin or
                        test_bounds[1] < -margin or
                        test_bounds[3] > self.bin_height + margin):  # ← この行(Y軸上限のチェック)を追加
                    continue

                # --- 衝突判定を確実な intersects() にします（マージン対応版） ---
                collision = False
                for placed in self.placed_parts:
                    # バウンディングボックスチェックはそのまま
                    placed_bounds = placed.get_bounds()
                    if (test_bounds[2] < placed_bounds[0] - margin or
                            test_bounds[0] > placed_bounds[2] + margin or
                            test_bounds[3] < placed_bounds[1] - margin or
                            test_bounds[1] > placed_bounds[3] + margin):
                        continue

                    # ★ 配置済み部品をマージン分だけ大きくして（buffer）、衝突判定を行う
                    if placed.polygon.buffer(margin).intersects(test_part.polygon):
                        collision = True
                        break

                if not collision:
                    # 基本は原点への距離の二乗（面積最小化）
                    # 緩い制約として、抑え込みたい方向に重み(1.5)をかけてペナルティを増やす
                    if priority == 'height':
                        metric = pos[0] ** 2 + (pos[1] * 1.5) ** 2
                    elif priority == 'width':
                        metric = (pos[0] * 1.5) ** 2 + pos[1] ** 2
                    else:
                        metric = pos[0] ** 2 + pos[1] ** 2

                    valid_positions.append((pos, metric))
            except Exception as e:
                print(f"候補位置検証エラー: {e}")
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---

        if not valid_positions:
            # 配置可能な位置が見つからなかった場合、現在の最高高さの上に配置
            max_height = 0
            if self.placed_parts:
                max_height = max(p.get_bounds()[3] for p in self.placed_parts)
            # Y方向にもマージンを追加
            return (0, max_height + margin)

        return min(valid_positions, key=lambda x: x[1])[0]
        # """最適配置位置を見つける（IFP候補生成 + NFP衝突判定による最終版）""""""NFPを使用して最適配置位置を見つける（基準点問題を修正した最終版）"""
        # if not self.placed_parts:
        #     return (0, 0)
        # 
        # candidate_positions = self._generate_candidate_positions(part)
        # valid_positions = []
        # 
        # for pos in candidate_positions:
        #     test_part = part.translate_to(pos)
        #     minx, miny, maxx, _ = test_part.get_bounds()
        #     
        #     if miny < -0.1 or minx < -0.1 or maxx > self.bin_width + 0.1:
        #         continue
        # 
        #     collision = False
        #     for placed_part in self.placed_parts:
        #         nfp = nfp_calculator.get_nfp(placed_part.id, part.id, part.rotation)
        # 
        #         # --- ▼▼▼ この基準点補正ロジックが重要 ▼▼▼ ---
        #         part_bounds = part.get_bounds()
        #         centroid_offset_x = part.polygon.centroid.x - part_bounds[0]
        #         centroid_offset_y = part.polygon.centroid.y - part_bounds[1]
        #         candidate_centroid_pos = (pos[0] + centroid_offset_x, pos[1] + centroid_offset_y)
        #         placed_part_pos = placed_part.position
        #         relative_pos = (candidate_centroid_pos[0] - placed_part_pos[0],
        #                         candidate_centroid_pos[1] - placed_part_pos[1])
        # 
        #         if not nfp.is_valid_position(relative_pos):
        #             collision = True
        #             break
        #         # --- ▲▲▲ ---
        # 
        #     if not collision:
        #         metric = pos[1] * 1000 + pos[0]
        #         valid_positions.append((pos, metric))
        # 
        # if not valid_positions:
        #     max_height = 0
        #     if self.placed_parts:
        #         max_height = max(p.get_bounds()[3] for p in self.placed_parts)
        #     return (0, max_height + 0.1)

        return min(valid_positions, key=lambda x: x[1])[0]

    def place_part(self, part: Part, position: Tuple[float, float]) -> None:
        """部品を指定位置に配置し、空き領域を更新"""
        placed_part = part.translate_to(position)
        self.placed_parts.append(placed_part)
        self.update_free_spaces(placed_part)

        # --- ▼▼▼ ここから追加 ▼▼▼ ---
        # フレームを保存する設定が有効な場合、可視化して保存する
        if self.frame_output_dir:
            frame_path = os.path.join(self.frame_output_dir, f"frame_{self.frame_count:04d}.png")
            self.visualize(save_path=frame_path)
            self.frame_count += 1
        # --- ▲▲▲ ここまで追加 ▲▲▲ ---

    def get_bin_height(self) -> float:
        if not self.placed_parts: return 0
        return max(p.get_bounds()[3] for p in self.placed_parts)

    def get_bin_width_used(self) -> float:
        """現在配置されている部品の最大幅（Xの最大値）を取得"""
        if not self.placed_parts: return 0
        return max(p.get_bounds()[2] for p in self.placed_parts)

    def get_utilization(self) -> float:
        total_area = sum(part.get_area() for part in self.placed_parts)
        bin_area = self.bin_width * self.get_bin_height()
        return total_area / bin_area if bin_area > 0 else 0

    def visualize(self, save_path: str = None, show_plot: bool = True, title: str = None,
                  # ▼▼▼ 新しい引数を追加 ▼▼▼
                  fixed_figsize: tuple = None, fixed_ylim: float = None):
        """配置結果を可視化（タイトル追加、プロット管理強化）"""
        final_height = self.get_bin_height()
        if final_height == 0:
            if save_path: # 保存先が指定されている場合は空のプロットを閉じる
                plt.close('all')
            return

        if fixed_figsize:
            # 動画フレーム生成時は、指定された固定サイズを使用
            fig, ax = plt.subplots(figsize=fixed_figsize)
        else:
            # 通常の可視化では、これまで通り動的にサイズを計算
            fig_width = 8
            fig_height = (final_height / self.bin_width) * fig_width if self.bin_width > 0 else 6
            fig_height = max(fig_height, 6)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # 各部品を描画
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, part in enumerate(self.placed_parts):
            # ポリゴンを閉じるために、最初の点を末尾に追加
            points_to_draw = part.points + [part.points[0]]
            x, y = zip(*points_to_draw)

            ax.fill(x, y, alpha=0.8, color=colors[i % 20],
                    label=f"Part {part.id} ({part.rotation}°)")

            # 部品IDをポリゴンの代表点に表示
            try:
                rep_point = part.polygon.representative_point()
                ax.text(rep_point.x, rep_point.y, part.id,
                        ha='center', va='center', fontsize=8, color='white',
                        bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.4))
            except Exception:
                # 代表点が計算できない場合でもエラーにしない
                pass

        # 描画範囲とアスペクト比を厳密に設定
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, self.bin_width)
        if fixed_ylim:
            ax.set_ylim(0, fixed_ylim)
        else:
            # ▼ final_height から self.bin_height に変更して、枠の表示を固定する
            ax.set_ylim(0, self.bin_height)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'ネスティング結果 (利用率: {self.get_utilization():.2%})')

        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

        # tight_layout() は、このようなプロットでは自動調整が失敗することがあるため使用しない

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

            # ▼▼▼ 表示/非表示の切り替えを追加 ▼▼▼
        if show_plot:
            plt.show()

            # フレーム保存時はメモリを解放するため、プロットを閉じる
        plt.close(fig)

    # ▼▼▼ 以下の3つのメソッドを ImprovedBinPacking クラスに追加 ▼▼▼

    def verify_placement(self) -> bool:
        """部品の重なりがないか検証する"""
        self.nfp_discrepancy_log = []
        for i, part1 in enumerate(self.placed_parts):
            for j, part2 in enumerate(self.placed_parts[i + 1:], i + 1):
                if part1.polygon.intersects(part2.polygon):
                    intersection = part1.polygon.intersection(part2.polygon)
                    if isinstance(intersection, (Polygon, MultiPolygon)) and intersection.area > 0.01:
                        discrepancy = {
                            "part1": part1, "part2": part2,
                            "intersection_area": intersection.area
                        }
                        self.nfp_discrepancy_log.append(discrepancy)

        return len(self.nfp_discrepancy_log) == 0

    def run_diagnostics(self, nfp_calculator: 'NFPCalculator') -> None:
        """診断を実行し、NFPと実際の配置の不一致を検出"""
        if not self.verify_placement():
            print("\n" + "=" * 10 + " NFPと実際の配置の不一致を検出 " + "=" * 10)
            for i, discrepancy in enumerate(self.nfp_discrepancy_log):
                part1 = discrepancy["part1"]
                part2 = discrepancy["part2"]
                print(f"\n[不一致 {i + 1}]")
                print(
                    f"  部品 {part1.id} と 部品 {part2.id} が面積 {discrepancy['intersection_area']:.2f} で重なっています。")

                # NFPの判定をチェック
                nfp = nfp_calculator.get_nfp(part1.id, part2.id, part2.rotation)

                # 正しい相対位置を計算
                part_bounds = part2.get_bounds()
                centroid_offset_x = part2.polygon.centroid.x - part_bounds[0]
                centroid_offset_y = part2.polygon.centroid.y - part_bounds[1]
                candidate_centroid_pos = (
                part2.position[0] + centroid_offset_x, part2.position[1] + centroid_offset_y)
                placed_part_pos = part1.position
                relative_pos = (candidate_centroid_pos[0] - placed_part_pos[0],
                                candidate_centroid_pos[1] - placed_part_pos[1])

                is_valid_by_nfp = nfp.is_valid_position(relative_pos)
                print(
                    f"  NFPの判定: {'衝突しない (有効)' if is_valid_by_nfp else '衝突する (無効)'} -> 判定が甘い（バグ）")

                self._visualize_discrepancy(part1, part2, nfp, i)

    def _visualize_discrepancy(self, part1: Part, part2: Part, nfp: 'NFP', index: int) -> None:
        """NFPと実際の部品配置の不一致を視覚化"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # NFPはpart1の原点を基準に描画
        p1_origin_nfp = translate(nfp.nfp, xoff=part1.position[0], yoff=part1.position[1])
        if hasattr(p1_origin_nfp, 'exterior'):
            x_nfp, y_nfp = p1_origin_nfp.exterior.xy
            ax.plot(x_nfp, y_nfp, 'g--', linewidth=2, label='NFP境界 (進入禁止エリア)')

        # 部品を描画
        p1_points = part1.points + [part1.points[0]]
        x1, y1 = zip(*p1_points)
        ax.fill(x1, y1, alpha=0.5, color='blue', label=f"Part {part1.id} (固定)")

        p2_points = part2.points + [part2.points[0]]
        x2, y2 = zip(*p2_points)
        ax.fill(x2, y2, alpha=0.5, color='red', label=f"Part {part2.id} (移動)")

        # 重なり部分を強調
        intersection = part1.polygon.intersection(part2.polygon)
        if not intersection.is_empty:
            if intersection.geom_type == 'Polygon':
                x_int, y_int = intersection.exterior.xy
                ax.fill(x_int, y_int, alpha=0.9, color='yellow', label='重なり')

        # 部品2の基準点（重心）に印
        p2_centroid = part2.polygon.centroid
        ax.plot(p2_centroid.x, p2_centroid.y, 'ro', markersize=8, label=f'{part2.id} の重心')

        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title(f"NFP不一致レポート: {part1.id} vs {part2.id}")

        filename = f"discrepancy_{part1.id}_{part2.id}_{index}.png"
        plt.savefig(filename)
        print(f"  不一致の視覚化を {filename} に保存しました。")
        plt.close(fig)

    # ▲▲▲ ここまでを追加 ▲▲▲

class GeneticAlgorithm:
    """遺伝的アルゴリズムによる最適順序と回転角度の探索"""
    def __init__(self, parts: List[Part], bin_width: float, bin_height: float, # ← bin_height 追加
                 population_size: int = 40, generations: int = 60,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.8,
                 angle_step: int = 15, priority: str = 'none'):
        self.parts = parts
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.angle_step = angle_step
        self.priority = priority
        self.nfp_calculator = None
        
    def initialize_population(self) -> List[List[Tuple[str, float]]]:
        """初期集団をランダムに生成"""
        population = []
        
        for _ in range(self.population_size):
            # 部品IDと角度のリストをランダム化
            individual = []
            parts_ids = [part.id for part in self.parts]
            random.shuffle(parts_ids)
            
            for part_id in parts_ids:
                # ランダムな回転角度（angle_stepの倍数）
                angle = random.randrange(0, 360, self.angle_step)
                individual.append((part_id, angle))
                
            population.append(individual)
            
        return population

    def evaluate_individual(self, individual: List[Tuple[str, float]]) -> float:
        """個体の適応度を評価（面積ベース ＋ 緩い形状制約）"""
        bin_packer = ImprovedBinPacking(self.bin_width, self.bin_height)

        for part_id, angle in individual:
            part = next(p for p in self.parts if p.id == part_id)
            rotated_part = part.rotate_to(angle)

            # priority を渡す
            position = bin_packer.find_optimal_position(rotated_part, self.nfp_calculator, self.priority)
            bin_packer.place_part(rotated_part, position)

        bin_height = bin_packer.get_bin_height()
        bin_width_used = bin_packer.get_bin_width_used()

        # ▼▼▼ 面積を基本としつつ、優先したい辺に1.1乗の緩いペナルティをかける ▼▼▼
        if self.priority == 'height':
            score = bin_width_used * (bin_height ** 1.1)
        elif self.priority == 'width':
            score = (bin_width_used ** 1.1) * bin_height
        else:
            score = bin_width_used * bin_height

        return 1.0 / (score + 1)
        
    def selection(self, population: List[Any], fitness_values: List[float]) -> List[Any]:
        """トーナメント選択"""
        selected = []
        
        for _ in range(len(population)):
            # ランダムに3つの個体を選び、最も適応度の高い方を選択
            candidates = random.sample(range(len(population)), k=min(3, len(population)))
            best_idx = max(candidates, key=lambda idx: fitness_values[idx])
            selected.append(copy.deepcopy(population[best_idx]))
                
        return selected
        
    def crossover(self, parent1: List[Any], parent2: List[Any]) -> Tuple[List[Any], List[Any]]:
        """順序交叉（Order Crossover）"""
        if random.random() > self.crossover_rate or len(parent1) < 2:
            return parent1, parent2
            
        size = len(parent1)
        
        # 交叉区間をランダムに選択
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        # 交叉を実行
        def create_child(p1, p2):
            # p1から交叉区間を取得
            mid_section = p1[start:end+1]
            
            # p2の順序に基づいて残りの部分を埋める
            remaining = [item for item in p2 if item[0] not in [m[0] for m in mid_section]]
            
            child = remaining[:start] + mid_section + remaining[start:]
            return child
            
        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
        
        return child1, child2
        
    def mutate(self, individual: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """突然変異（順序の入れ替えと角度の変更）"""
        # 順序の入れ替え
        if random.random() < self.mutation_rate:
            i = random.randrange(len(individual))
            j = random.randrange(len(individual))
            individual[i], individual[j] = individual[j], individual[i]
            
        # 角度の変更
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                part_id, _ = individual[i]
                new_angle = random.randrange(0, 360, self.angle_step)
                individual[i] = (part_id, new_angle)
            
        return individual

    def run(self) -> Tuple[List[Tuple[str, float]], float]:
        """遺伝的アルゴリズムを実行（収束判定付き）"""

        population = self.initialize_population()
        best_individual = None
        best_fitness = 0
        best_history = []


        # ▼▼▼ 収束判定用の変数を追加 ▼▼▼
        patience = 20  # 20世代改善がなければ収束したとみなす
        generations_without_improvement = 0
        # ▲▲▲ 追加ここまで ▲▲▲

        for generation in range(self.generations):
            print(f"世代 {generation + 1}/{self.generations} 処理中...")

            fitness_values = [self.evaluate_individual(ind) for ind in population]

            # ▼▼▼ 収束判定のロジックを追加 ▼▼▼
            current_best_fitness_in_gen = max(fitness_values)

            if current_best_fitness_in_gen > best_fitness:
                # 最良解が更新された場合
                best_fitness = current_best_fitness_in_gen
                max_idx = fitness_values.index(best_fitness)
                best_individual = copy.deepcopy(population[max_idx])
                generations_without_improvement = 0  # カウンターをリセット
                print(f"  新しい最良解を発見: 適応度 {best_fitness:.6f}")
            else:
                # 最良解が更新されなかった場合
                generations_without_improvement += 1  # カウンターを増やす

            best_history.append(best_fitness)


            # カウンターが我慢の限界に達したらループを抜ける
            if generations_without_improvement >= patience:
                print(f"\n{patience}世代にわたり改善が見られなかったため、収束したと判断し早期終了します。")
                break
            # ▲▲▲ 判定ロジックここまで ▲▲▲

            # ... (選択、交叉、突然変異のロジックは変更なし) ...
            selected = self.selection(population, fitness_values)
            next_population = []
            max_idx = fitness_values.index(max(fitness_values))  # エリート選択のため再取得
            next_population.append(copy.deepcopy(population[max_idx]))

            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)

            population = next_population

        # ... (適応度の進化をプロットする部分は変更なし) ...

        return best_individual, best_fitness

        # ▼▼▼ 動画フレーム生成専用のメソッドを新しく追加 ▼▼▼
    def run_and_save_frames(self, frame_output_dir: str):
        """動画生成のため、1世代ごとにフレームを保存しながらGAを実行する"""
        population = self.initialize_population()
        patience = 20
        generations_without_improvement = 0
        best_fitness = 0

        for generation in range(self.generations):
            print(f"  - 動画生成中: 世代 {generation + 1}/{self.generations}")

            fitness_values = [self.evaluate_individual(ind) for ind in population]

            # --- フレーム保存ロジック ---
            packer = ImprovedBinPacking(self.bin_width, self.bin_height)
            best_ind_in_gen = population[fitness_values.index(max(fitness_values))]
            for part_id, angle in best_ind_in_gen:
                part = next(p for p in self.parts if p.id == part_id)
                rotated_part = part.rotate_to(angle)
                position = packer.find_optimal_position(rotated_part, self.nfp_calculator)
                packer.place_part(rotated_part, position)

            frame_path = os.path.join(frame_output_dir, f"frame_{generation:04d}.png")
            title = f"Generation: {generation + 1}, Height: {packer.get_bin_height():.2f}"
            packer.visualize(save_path=frame_path, show_plot=False, title=title)
            # --- フレーム保存ここまで ---

            # GAの進化ロジック
            current_best_fitness = max(fitness_values)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= patience:
                break

            selected = self.selection(population, fitness_values)
            next_population = []
            max_idx = fitness_values.index(max(fitness_values))
            next_population.append(copy.deepcopy(population[max_idx]))
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                next_population.append(self.mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self.mutate(child2))
            population = next_population


class LocalSearch:
    """ローカル探索による角度の微調整（改良版）"""
    def __init__(self, parts: List[Part], bin_width: float, bin_height: float, angle_range: int = 10, angle_step: int = 2): # ← bin_height 追加
        self.parts = parts
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.angle_range = angle_range  # 微調整する角度範囲
        self.angle_step = angle_step    # 微調整の角度ステップ

    def optimize_angles(self, solution: List[Tuple[str, float]], nfp_calculator: 'NFPCalculator', priority: str = 'none') -> List[Tuple[str, float]]:
        """各部品の角度を微調整して最適化（nfp_calculatorを正しく受け取る修正版）"""
        print("ローカル探索による角度の微調整を開始...")
        best_solution = copy.deepcopy(solution)
        # _evaluate_solutionの呼び出しにnfp_calculatorを渡す
        best_height = self._evaluate_solution(best_solution, nfp_calculator, priority)

        improved = True
        iteration = 0
        max_iterations = 3

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            print(f"ローカル探索 反復 {iteration}/{max_iterations}")

            for i in range(len(solution)):
                part_id, angle = best_solution[i]
                best_angle = angle
                current_best_height = best_height

                angles_to_try = [
                    (angle + delta) % 360
                    for delta in range(-self.angle_range, self.angle_range + 1, self.angle_step)
                    if delta != 0
                ]

                for new_angle in angles_to_try:
                    new_solution = copy.deepcopy(best_solution)
                    new_solution[i] = (part_id, new_angle)

                    # _evaluate_solutionの呼び出しにnfp_calculatorを渡す
                    new_height = self._evaluate_solution(new_solution, nfp_calculator, priority='height')

                    if new_height < current_best_height:
                        current_best_height = new_height
                        best_angle = new_angle
                        improved = True
                        print(f"  改善: 部品 {part_id} の角度を {angle}° → {best_angle}° に変更 (面積スコア: {current_best_height:.2f})")
                if best_angle != angle:
                    best_solution[i] = (part_id, best_angle)
                    best_height = current_best_height

        print(f"角度の微調整完了 (最終高さ: {best_height:.2f})")
        return best_solution

    def _evaluate_solution(self, solution: List[Tuple[str, float]], nfp_calculator: 'NFPCalculator',
                           priority: str = 'none') -> float:
        """解の評価"""
        bin_packer = ImprovedBinPacking(self.bin_width, self.bin_height)

        for part_id, angle in solution:
            part = next(p for p in self.parts if p.id == part_id)
            rotated_part = part.rotate_to(angle)

            # priority を渡す
            position = bin_packer.find_optimal_position(rotated_part, nfp_calculator, priority)
            bin_packer.place_part(rotated_part, position)

        bin_height = bin_packer.get_bin_height()
        bin_width_used = bin_packer.get_bin_width_used()

        # ▼▼▼ こちらも同じく1.1乗のペナルティ ▼▼▼
        if priority == 'height':
            return bin_width_used * (bin_height ** 1.1)
        elif priority == 'width':
            return (bin_width_used ** 1.1) * bin_height
        else:
            return bin_width_used * bin_height


# マルチプロセシング用の関数をクラス外に移動
def calculate_nfp_task(task, parts_dict):
    """NFP計算タスクを処理する関数（マルチプロセス用）"""
    part1_id, part2_id, angle = task
    
    # 部品の取得
    part1 = parts_dict[part1_id]
    part2 = parts_dict[part2_id]
    
    # NFP計算
    rotated_part2 = part2.rotate_to(angle)
    nfp = NFP(part1, rotated_part2)
    
    return (part1_id, part2_id, angle), nfp


class NestingAlgorithm:
    """ネスティングアルゴリズムのメインクラス（IFP統合版）"""

    # ▼▼▼ safety_margin引数を追加 ▼▼▼
    def __init__(self, parts: List[Part], bin_width: float, bin_height: float, safety_margin: float = 0.0, allow_rotation: bool = True, priority: str = 'none'):
        self.parts = parts
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.safety_margin = safety_margin
        # ▼▼▼ NFPCalculatorにマージンを渡す ▼▼▼
        self.nfp_calculator = NFPCalculator(parts, angle_step=15, margin=safety_margin)
        self.allow_rotation = allow_rotation
        self.priority = priority

    def run(self) -> ImprovedBinPacking:
        """複数の探索戦略を並列実行し、最良の結果を返す"""
        start_time = time.time()

        nfp_calculator = NFPCalculator(self.parts, angle_step=15, margin=self.safety_margin)
        print("ステップ0: NFPの事前計算...")
        nfp_calculator.precompute_nfps()

        strategies = [
            {"name": "バランス型", "ga_params": {"population_size": 80, "generations": 5, "mutation_rate": 0.15}},
            {"name": "広域探索型", "ga_params": {"population_size": 80, "generations": 5, "mutation_rate": 0.35}},
            {"name": "深掘り探索型", "ga_params": {"population_size": 80, "generations": 5, "mutation_rate": 0.05}}
        ]

        # 並列処理の引数を準備
        trial_args = [(self.parts, self.bin_width, self.bin_height, s, nfp_calculator, i + 1, self.allow_rotation, self.priority) for i, s in enumerate(strategies)]
        # CPUのコア数（最大3）で並列実行
        num_processes = min(mp.cpu_count(), len(strategies))
        print(f"\n{num_processes}個のプロセスで並列実行を開始...")
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_single_trial, trial_args)

        # 最良の結果を選択
        best_result_packer, best_solution = min(results, key=lambda r: r[0].get_bin_height())
        best_height = best_result_packer.get_bin_height()

        final_time = time.time()
        utilization = best_result_packer.get_utilization()

        # ▼▼▼ 診断機能の呼び出し ▼▼▼
        print("\n" + "=" * 15 + " 最終配置の検証と診断を開始 " + "=" * 15)
        best_result_packer.run_diagnostics(self.nfp_calculator)
        # ▲▲▲

        print("\n" + "=" * 20 + " 全試行完了 " + "=" * 20)
        print(f"最終結果:")
        print(f"  最終高さ: {best_height:.2f}")
        print(f"  材料利用率: {utilization:.2%}")
        print(f"  総実行時間: {final_time - start_time:.2f}秒")

        return best_result_packer, best_solution

def create_airfoil_part(id: str = "Airfoil", chord: float = 50, n_points: int = 50) -> Part:
    # NACA 2412パラメータ
    m = 0.04  # 最大キャンバー
    p = 0.4   # 最大キャンバー位置
    t = 0.12  # 厚み
    points = []

    for i in range(n_points + 1):
        x = chord * i / n_points
        xc = x / chord
        # 厚み分布
        yt = 5 * t * chord * (
            0.2969 * math.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 +
            0.2843 * xc**3 - 0.1015 * xc**4
        )
        # キャンバー線
        if xc < p:
            yc = m * chord * (2 * p * xc - xc**2) / (p**2)
            dyc_dx = 2 * m / (p**2) * (p - xc)
        else:
            yc = m * chord * ((1 - 2 * p) + 2 * p * xc - xc**2) / ((1 - p)**2)
            dyc_dx = 2 * m / ((1 - p)**2) * (p - xc)
        theta = math.atan(dyc_dx)
        # 上面
        xu = x - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)
        points.append((xu, yu))
    # 下面（逆順で追加）
    for i in reversed(range(n_points + 1)):
        x = chord * i / n_points
        xc = x / chord
        yt = 5 * t * chord * (
            0.2969 * math.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 +
            0.2843 * xc**3 - 0.1015 * xc**4
        )
        if xc < p:
            yc = m * chord * (2 * p * xc - xc**2) / (p**2)
            dyc_dx = 2 * m / (p**2) * (p - xc)
        else:
            yc = m * chord * ((1 - 2 * p) + 2 * p * xc - xc**2) / ((1 - p)**2)
            dyc_dx = 2 * m / ((1 - p)**2) * (p - xc)
        theta = math.atan(dyc_dx)
        # 下面
        xl = x + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)
        points.append((xl, yl))

    return Part(id, points)

# 既存の create_sample_parts に翼型部品を追加
# サンプル部品の作成
def create_sample_parts() -> List[Part]:
    """基本的なサンプル部品を作成"""
    parts = []
    
    # 長方形の部品
    rect1 = Part("R1", [(0, 0), (40, 0), (40, 20), (0, 20)])
    rect2 = Part("R2", [(0, 0), (30, 0), (30, 25), (0, 25)])
    rect3 = Part("R3", [(0, 0), (50, 0), (50, 15), (0, 15)])
    
    # L字型の部品
    l_shape = Part("L1", [(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)])
    
    # T字型の部品
    t_shape = Part("T1", [(0, 10), (10, 10), (10, 0), (20, 0), (20, 10), (30, 10), (30, 20), (0, 20)])
    
    # 不規則な多角形
    poly1 = Part("P1", [(0, 0), (20, 0), (25, 10), (15, 20), (0, 15)])
    
    # U字型の部品（凹多角形）
    u_shape = Part("U1", [(0, 0), (30, 0), (30, 25), (25, 25), (25, 5), (5, 5), (5, 25), (0, 25)])
    
    parts = [rect1, rect2, rect3, l_shape, t_shape, poly1, u_shape]
    return parts


def create_extended_parts() -> List[Part]:
    """より多くのサンプル部品を作成"""
    parts = []

    # 長方形の部品（異なるサイズ）
    rect_sizes = [
        ("R1", 40, 20), ("R2", 30, 25), ("R3", 50, 15),
        ("R4", 25, 35), ("R5", 45, 10)
    ]
    for id, width, height in rect_sizes:
        parts.append(Part(id, [(0, 0), (width, 0), (width, height), (0, height)]))

    # L字型の部品（異なるサイズ）
    l_shapes = [
        ("L1", 30, 10, 30),
        ("L2", 40, 15, 25),
        ("L3", 20, 8, 35)
    ]
    for id, width, leg_width, leg_height in l_shapes:
        parts.append(Part(id, [
            (0, 0), (width, 0), (width, leg_width), (leg_width, leg_width),
            (leg_width, leg_height), (0, leg_height)
        ]))

    # T字型の部品
    t_shape = Part("T1", [(0, 10), (10, 10), (10, 0), (20, 0), (20, 10), (30, 10), (30, 20), (0, 20)])

    # 不規則な多角形
    poly1 = Part("P1", [(0, 0), (20, 0), (25, 10), (15, 20), (0, 15)])
    poly2 = Part("P2", [(0, 0), (15, 5), (30, 0), (25, 15), (30, 30), (15, 25), (0, 30), (5, 15)])

    # U字型の部品（凹多角形）
    u_shape = Part("U1", [(0, 0), (30, 0), (30, 25), (25, 25), (25, 5), (5, 5), (5, 25), (0, 25)])

    # H字型の部品（凹多角形）
    h_shape = Part("H1", [
        (0, 0), (10, 0), (10, 10), (20, 10), (20, 0), (30, 0),
        (30, 30), (20, 30), (20, 20), (10, 20), (10, 30), (0, 30)
    ])

    parts.extend([t_shape, poly1, poly2, u_shape, h_shape])
    airfoil = create_airfoil_part("Airfoil1", chord=50, n_points=60)
    print(airfoil.points)
    parts.append(airfoil)

    return parts


def run_single_trial(trial_args):
    """1回のネスティング試行を実行するトップレベル関数（並列処理用）"""
    parts, bin_width, bin_height, strategy, nfp_calculator, trial_num, allow_rotation, priority = trial_args

    print(f"\n--- 試行 {trial_num} (戦略: {strategy['name']}) を開始 ---")

    if not allow_rotation:
        ga_angle_step = 360  # 360度にすることで、0度しか選ばれなくする（実質回転なし）
        ls_angle_range = 0  # 最後の微調整も0度にする
        ls_angle_step = 1
    else:
        ga_angle_step = strategy['ga_params'].get('angle_step', 15)
        ls_angle_range = 10
        ls_angle_step = 2

    params = strategy['ga_params'].copy()
    params.pop('angle_step', None)

    ga = GeneticAlgorithm(parts, bin_width, bin_height, angle_step=ga_angle_step, priority=priority, **params)
    ga.nfp_calculator = nfp_calculator
    best_solution, _ = ga.run()

    ls = LocalSearch(parts, bin_width, bin_height, angle_range=ls_angle_range, angle_step=ls_angle_step)
    refined_solution = ls.optimize_angles(best_solution, nfp_calculator, priority=priority)

    # 3. 最終配置
    packer = ImprovedBinPacking(bin_width, bin_height)
    for part_id, angle in refined_solution:
        part = next(p for p in parts if p.id == part_id)
        rotated_part = part.rotate_to(angle)
        position = packer.find_optimal_position(rotated_part, nfp_calculator, priority=priority)
        packer.place_part(rotated_part, position)

    height = packer.get_bin_height()
    print(f"--- 試行 {trial_num} 結果: 高さ {height:.2f} ---")

    return packer, refined_solution

# メイン関数
def main():
    """メイン関数"""
    print("高度なネスティングアルゴリズムを開始...")

    parts = create_sample_parts()
    bin_width = 100
    bin_height = 100

    # ネスティングアルゴリズムの実行
    nester = NestingAlgorithm(parts, bin_width, bin_height, safety_margin=2.0, allow_rotation=True)
    result_packer = nester.run()
    
    # 結果の可視化
    result_packer.visualize()
    print("\nプログラム終了")


if __name__ == "__main__":
    main()