import geopandas as gpd

base_path = "d:/urbanbus_rl_project/data/raw/daegu/shapes/extracted/2025-09-03"
layers = ["bs_20250903.shp", "link_20250903.shp", "node_20250903.shp"]

for layer in layers:
    print(f"=== Inspecting {layer} ===")
    try:
        # DBF 인코딩 고려하여 cp949 적용 (대중교통 시스템 한글 데이터 표준 호환성)
        gdf = gpd.read_file(f"{base_path}/{layer}", encoding="cp949")
        print(f"Row count: {len(gdf)}")
        print(f"Geometry type: {list(gdf.geom_type.unique())}")
        print("Columns:")
        print(list(gdf.columns))
        print("Sample records (top 2):")
        print(gdf.head(2).drop(columns='geometry', errors='ignore').to_dict(orient='records'))
    except Exception as e:
        print(f"Error reading layer {layer}: {e}")
    print("\n" + "-"*50 + "\n")
