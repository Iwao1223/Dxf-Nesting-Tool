import ezdxf
from ezdxf import path
import math
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize
import matplotlib.pyplot as plt


# 既存のネスティングエンジンから必要なクラスをインポート
from complete_nesting_algorithm_Version2_Version3 import Part, NestingAlgorithm


def extract_polygons_from_dxf(filepath: str):
    try:
        doc = ezdxf.readfile(filepath)
    except Exception as e:
        print(f"DXFファイルの読み込みに失敗しました ({filepath}): {e}")
        return []

    msp = doc.modelspace()
    polygons = []
    lines = []

    for entity in msp:
        try:
            if entity.dxftype() in ['TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'INSERT']:
                continue

            points = []
            is_closed_flag = False

            # 1. ポリライン（連続した線）の場合：元の綺麗な直線をそのまま取得！
            if entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0], p[1]) for p in entity.get_points('xy')]
                is_closed_flag = entity.closed

            # 2. 単純な直線の場合：始点と終点をそのまま取得！
            elif entity.dxftype() == 'LINE':
                start = (entity.dxf.start.x, entity.dxf.start.y)
                end = (entity.dxf.end.x, entity.dxf.end.y)
                points = [start, end]
                is_closed_flag = False

            # 3. スプライン曲線や円などの場合：ここで初めて細かく分割する
            elif entity.dxftype() in ['SPLINE', 'ARC', 'CIRCLE', 'ELLIPSE']:
                p = path.make_path(entity)
                vertices = list(p.flattening(distance=0.1))
                if len(vertices) >= 2:
                    points = [(v.x, v.y) for v in vertices]
                    if entity.dxftype() == 'SPLINE':
                        is_closed_flag = entity.closed
                    else:
                        is_closed_flag = True  # 円などは閉じている

            # 4. その他のマイナーな図形
            else:
                p = path.make_path(entity)
                vertices = list(p.flattening(distance=0.1))
                if len(vertices) >= 2:
                    points = [(v.x, v.y) for v in vertices]
                    is_closed_flag = getattr(p, 'is_closed', False)

            if len(points) < 2:
                continue

            # 始点と終点の距離チェック（0.5mm以内のズレは「閉じた図形」とみなす）
            is_closed_dist = math.dist(points[0], points[-1]) < 0.5 if len(points) > 1 else False

            if is_closed_flag or is_closed_dist:
                # 隙間が空いている場合は、始点を無理やり最後に追加して完全に閉じる
                if points[0] != points[-1]:
                    points.append(points[0])

                if len(points) >= 3:
                    poly = Polygon(points)

                    # 線がねじれて不正な図形になった場合、自動修復する
                    if not poly.is_valid:
                        poly = poly.buffer(0)

                    if poly.is_valid and poly.area > 0:
                        polygons.append(poly)
            else:
                # 閉じていないバラバラの線分として保存
                for i in range(len(points) - 1):
                    lines.append(LineString([points[i], points[i + 1]]))

        except Exception as e:
            pass

    # バラバラの線をパズルのように繋ぎ合わせて面（ポリゴン）化する
    if lines:
        for poly in polygonize(lines):
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)

    return polygons


def export_to_dxf(placed_parts, output_filepath: str, sheet_width: float, sheet_height: float):
    """
    配置済みのパーツリストを新しいDXFファイルとして保存する
    """
    # 新しいDXFドキュメントを作成 (R2010フォーマットが安定しています)
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # 1. 材料の枠を描画（赤色にして分かりやすくします）
    # color=1 は赤色
    frame_points = [(0, 0), (sheet_width, 0), (sheet_width, sheet_height), (0, sheet_height)]
    msp.add_lwpolyline(frame_points, close=True, dxfattribs={'color': 1})

    ## 2. 各パーツを描画
    for part in placed_parts:
        msp.add_lwpolyline(part.points, close=True, dxfattribs={'color': 3})
        # centroid = part.polygon.centroid
        # msp.add_text(part.id, dxfattribs={
        #     'height': max(2.0, sheet_height * 0.02),
        #     'color': 7
        # }).set_placement((centroid.x, centroid.y), align=TextEntityAlignment.MIDDLE_CENTER) # ← ★文字列から変更

    # ファイルを保存
    doc.saveas(output_filepath)
    print(f"\n★ 配置結果をDXFファイルに保存しました: {output_filepath}")

def run_dxf_nesting(dxf_filepath: str, sheet_width: float, sheet_height: float):
    print(f"--- DXF読み込み: {dxf_filepath} ---")

    # 1. DXFからポリゴンを抽出
    shapely_polygons = extract_polygons_from_dxf(dxf_filepath)

    if not shapely_polygons:
        print("エラー: DXFから閉じた図形（ポリゴン）を検出できませんでした。")
        return

    print(f"{len(shapely_polygons)} 個のパーツを抽出しました。ネスティングを開始します...")

    # 2. 抽出したポリゴンを、既存エンジンの `Part` クラスに変換
    parts = []
    for i, poly in enumerate(shapely_polygons):
        # ShapelyのPolygonから頂点座標のリストを取り出す（最後の点は重複するので除外）
        points = list(poly.exterior.coords)[:-1]
        parts.append(Part(f"DXF_{i + 1}", points))

    # 3. 既存のネスティングエンジンを実行
    # ※ とりあえず現在の仕様（幅のみ指定、高さは無制限）でテストします
    safety_margin = 2.0  # 部品同士の隙間(mm)
    nester = NestingAlgorithm(parts, sheet_width, sheet_height, safety_margin=safety_margin)
    result_packer, _ = nester.run()

    output_dxf_name = "nested_result.dxf"  # 保存するファイル名
    export_to_dxf(result_packer.placed_parts, output_dxf_name, sheet_width, sheet_height)

    # 4. 結果のプレビュー表示
    print("ネスティングが完了しました。プレビューを表示します。")
    result_packer.visualize(title=f"DXF Nesting Result ({sheet_width} x {sheet_height})")


if __name__ == "__main__":
    TARGET_DXF = "test_shapes.dxf"

    # ▼ 幅と高さを両方指定する
    SHEET_WIDTH = 900.0
    SHEET_HEIGHT = 600.0

    run_dxf_nesting(TARGET_DXF, SHEET_WIDTH, SHEET_HEIGHT)