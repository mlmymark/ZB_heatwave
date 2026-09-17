# -*- coding: utf-8 -*-
"""
体感温度热浪事件识别：固定阈值与逐年百分位阈值，指定经纬度。

高温日须同时达到固定温度阈值与逐年百分位阈值（ARTT）。
该阈值分别使用每一年该像元全部有效日温度计算。
本脚本可独立运行；运行前须填写末尾用户参数中的路径、经度和纬度。
输入：同一目录中的逐日影像，文件名为 年月日.tif。
坐标：世界大地坐标系 WGS84 的十进制度，经度在前、纬度在后。
空间处理：读取包含目标坐标的像元，仅使用第一波段的单像元窗口，不插值。
温度单位：输入值须已为摄氏度，不执行缩放或偏移转换。
事件规则：达到阈值包含等于，逐年识别，跨年连续高温段在年界拆分。
连续性：严格按日历连续；任何缺失日期文件均中断事件。
缺测规则：缺测值、非有限值与无效掩膜中断事件；未观测日不自动补值。
输出：逗号分隔事件表（CSV），每行一次事件，保留标准字段
start_date（开始日期）、end_date（结束日期）、duration_days（持续天数）。
依赖：数值计算库 NumPy 与地理空间数据抽象库 GDAL 的 Python 绑定。
"""
import calendar
import csv
import datetime
import math
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()


def validate_parameters(start_year, end_year, temp_threshold, duration_threshold,
                        longitude, latitude, method, percentile):
    """校验坐标、年份和阈值；未填写目标坐标时中止运行，避免误用示例地点。"""
    if longitude is None or latitude is None:
        raise ValueError("请先在脚本末尾填写 LONGITUDE（经度）和 LATITUDE（纬度）。")
    if not (math.isfinite(longitude) and -180 <= longitude <= 180):
        raise ValueError("LONGITUDE 必须是 -180 至 180 之间的有限数值。")
    if not (math.isfinite(latitude) and -90 <= latitude <= 90):
        raise ValueError("LATITUDE 必须是 -90 至 90 之间的有限数值。")
    if not (isinstance(start_year, int) and isinstance(end_year, int)
            and 1 <= start_year <= end_year <= 9999):
        raise ValueError("年份必须是整数，且 1 <= START_YEAR <= END_YEAR <= 9999。")
    if not math.isfinite(temp_threshold):
        raise ValueError("TEMP_THRESHOLD 必须是有限数值。")
    if not isinstance(duration_threshold, int) or duration_threshold < 1:
        raise ValueError("DURATION_THRESHOLD 必须是正整数。")
    if method not in ("absolute", "relative"):
        raise ValueError("METHOD 只能是 'absolute' 或 'relative'。")
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("百分位数必须位于 0 至 100 之间。")


def list_daily_tifs(input_folder, start_year=None, end_year=None):
    """收集以八位年月日命名的影像，检查日期和重复文件，按日历日期排序。"""
    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{folder}")
    records = []
    seen_dates = set()
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
            continue
        try:
            if len(path.stem) != 8 or not path.stem.isdigit():
                raise ValueError
            date = datetime.datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(f"TIFF 文件名必须为 YYYYMMDD.tif：{path.name}") from exc
        if start_year is not None and date.year < start_year:
            continue
        if end_year is not None and date.year > end_year:
            continue
        if date in seen_dates:
            raise ValueError(f"存在重复日期 {date}，请检查 .tif/.tiff 文件：{path}")
        seen_dates.add(date)
        records.append((date, path))
    if not records:
        raise FileNotFoundError("未找到符合年份范围的逐日 TIFF 文件。")
    return sorted(records, key=lambda item: item[0])


def spatial_reference(projection):
    """解析影像投影信息，并固定为经度在前、纬度在后的坐标轴顺序。"""
    if not projection:
        raise ValueError("TIFF 缺少坐标系，无法可靠地用经纬度定位像元。")
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(projection) != 0:
        raise ValueError("无法解析 TIFF 的坐标系。")
    # GDAL 3 默认尊重 EPSG 轴顺序；这里明确输入始终为经度、纬度。
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def locate_pixel(dataset, longitude, latitude):
    """将输入经纬度转换到影像坐标系，读取包含该点的像元，不做空间插值。"""
    projection = dataset.GetProjection()
    target_srs = spatial_reference(projection)
    source_srs = osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)
    if hasattr(source_srs, "SetAxisMappingStrategy"):
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    x, y, _ = transform.TransformPoint(float(longitude), float(latitude))
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("经纬度转换失败，请检查输入坐标及 TIFF 坐标系。")
    geo = dataset.GetGeoTransform(can_return_null=True)
    if geo is None:
        raise ValueError("TIFF 缺少仿射变换参数，无法定位像元。")
    inverse = gdal.InvGeoTransform(geo)
    if inverse is None:
        raise ValueError("TIFF 仿射变换不可逆。")
    pixel_x, pixel_y = gdal.ApplyGeoTransform(inverse, x, y)
    col, row = math.floor(pixel_x), math.floor(pixel_y)
    if not (0 <= row < dataset.RasterYSize and 0 <= col < dataset.RasterXSize):
        raise ValueError(
            f"指定经纬度 ({longitude}, {latitude}) 不在栅格范围内；"
            f"计算所得行列为 ({row}, {col})。"
        )
    center_x, center_y = gdal.ApplyGeoTransform(geo, col + 0.5, row + 0.5)
    center_lon, center_lat, _ = osr.CoordinateTransformation(
        target_srs, source_srs
    ).TransformPoint(center_x, center_y)
    return {
        "row": row, "col": col,
        "rows": dataset.RasterYSize, "cols": dataset.RasterXSize,
        "projection": projection, "geo": geo,
        "center_longitude": center_lon, "center_latitude": center_lat,
    }


def check_same_grid(dataset, reference, path):
    """逐日影像须与首幅影像同网格，避免读取到不同空间位置。"""
    if (dataset.RasterYSize, dataset.RasterXSize) != (
            reference["rows"], reference["cols"]):
        raise ValueError(f"栅格行列数与首幅影像不一致：{path}")
    geo = dataset.GetGeoTransform(can_return_null=True)
    if geo is None or not np.allclose(geo, reference["geo"], rtol=0, atol=1e-9):
        raise ValueError(f"栅格仿射变换与首幅影像不一致：{path}")
    projection = dataset.GetProjection()
    if projection != reference["projection"]:
        if not spatial_reference(projection).IsSame(
                spatial_reference(reference["projection"])):
            raise ValueError(f"栅格坐标系与首幅影像不一致：{path}")


def read_point_series(input_folder, longitude, latitude,
                      start_year=None, end_year=None):
    """逐日读取单像元窗口，返回单精度温度序列、日期列表与像元定位信息。"""
    records = list_daily_tifs(input_folder, start_year, end_year)
    dates = [date for date, _ in records]
    series = np.full(len(records), np.nan, dtype=np.float32)
    reference = None
    for index, (date, path) in enumerate(records):
        dataset = band = mask_band = None
        try:
            dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
            if dataset is None or dataset.RasterCount < 1:
                raise RuntimeError(f"无法读取影像第一波段：{path}")
            if reference is None:
                reference = locate_pixel(dataset, longitude, latitude)
                print(
                    f"选中像元：row={reference['row']}, col={reference['col']}（从 0 开始）；"
                    f"中心经纬度=({reference['center_longitude']:.8f}, "
                    f"{reference['center_latitude']:.8f})"
                )
            else:
                check_same_grid(dataset, reference, path)
            band = dataset.GetRasterBand(1)
            row, col = reference["row"], reference["col"]
            values = band.ReadAsArray(col, row, 1, 1)
            if values is None or values.shape != (1, 1):
                raise RuntimeError(f"像元读取失败：{path}")
            # 温度以单精度浮点数读取；每幅影像分别使用自身的缺测值。
            value = np.float32(values[0, 0])
            nodata = band.GetNoDataValue()
            valid = bool(np.isfinite(value))
            if nodata is not None and value == np.float32(nodata):
                valid = False
            if valid and band.GetMaskFlags() != gdal.GMF_ALL_VALID:
                mask_band = band.GetMaskBand()
                mask = mask_band.ReadAsArray(col, row, 1, 1)
                if mask is None:
                    raise RuntimeError(f"有效数据掩膜读取失败：{path}")
                valid = bool(mask[0, 0] != 0)
            if valid:
                series[index] = value
        finally:
            mask_band = band = dataset = None
        if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(records):
            print(f"读取进度：{index + 1}/{len(records)}，日期 {date}")
    missing_days = sum(max(0, (right - left).days - 1)
                       for left, right in zip(dates, dates[1:]))
    if missing_days:
        print(f"提示：首末影像日期之间缺少 {missing_days} 天文件，事件不会跨缺失日期连接。")
    return series, dates, reference


def group_year_indices(dates, start_year, end_year):
    """按年份汇总日期索引；跨年连续高温段分别在各年内计算。"""
    groups = {}
    for index, date in enumerate(dates):
        if start_year <= date.year <= end_year:
            groups.setdefault(date.year, []).append(index)
    return groups


def find_heatwave_events(series, dates, thresholds, duration_threshold):
    """按日历连续性识别事件；低温、缺测、无有效阈值均中断，起止日均计入。"""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError("温度序列必须是一维，且与日期列表等长。")
    if any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError("日期必须严格递增，不能重复。")
    if not isinstance(duration_threshold, int) or duration_threshold < 1:
        raise ValueError("最小持续天数必须是正整数。")
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if threshold_values.ndim == 0:
        threshold_values = np.full(values.shape, float(threshold_values))
    if threshold_values.shape != values.shape:
        raise ValueError("阈值必须是标量，或与温度序列等长的一维数组。")
    hot = (np.isfinite(values) & np.isfinite(threshold_values)
           & (values >= threshold_values))
    events = []
    start = end = None

    def finish_run():
        """结束当前连续高温段，仅保存达到最短持续时间的事件。"""
        if start is not None:
            duration = (end - start).days + 1
            if duration >= duration_threshold:
                events.append({"start_date": start, "end_date": end,
                               "duration_days": duration})

    for date, is_hot in zip(dates, hot):
        if is_hot:
            if start is None:
                start = end = date
            elif (date - end).days == 1:
                end = date
            else:
                finish_run()
                start = end = date
        else:
            finish_run()
            start = end = None
    finish_run()
    return events


def write_events_csv(output_folder, events, longitude, latitude,
                     start_year, end_year, label):
    """每行写出一次事件，日期格式为四位年-两位月-两位日，编码兼容电子表格软件。"""
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (
        f"heatwave_{label}_lon{float(longitude)}_lat{float(latitude)}_"
        f"{start_year}-{end_year}.csv"
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_date", "end_date", "duration_days"])
        for event in events:
            writer.writerow([event["start_date"].isoformat(),
                             event["end_date"].isoformat(),
                             event["duration_days"]])
    print(f"共输出 {len(events)} 次热浪事件：{path}")
    return path


def run_analysis(input_folder, output_folder, start_year, end_year,
                 temp_threshold, duration_threshold, longitude, latitude,
                 threshold_kind, method="absolute", percentile=90):
    """读取目标像元温度序列，识别事件并写出结果表。"""
    validate_parameters(start_year, end_year, temp_threshold, duration_threshold,
                        longitude, latitude, method, percentile)
    # 气候逐日阈值须使用目录内全部年份计算，不能先按输出年份截取。
    use_all_years = threshold_kind == "CRTT" and method == "relative"
    series, dates, _ = read_point_series(
        input_folder, longitude, latitude,
        None if use_all_years else start_year,
        None if use_all_years else end_year,
    )
    selected = [i for i, date in enumerate(dates)
                if start_year <= date.year <= end_year]
    if not selected:
        raise ValueError("指定输出年份内没有任何逐日 TIFF。")
    n_valid = int(np.count_nonzero(np.isfinite(series[selected])))
    print(f"输出年份内：{len(selected)} 个日文件，其中 {n_valid} 天为有效像元值。")
    if n_valid == 0:
        print("提示：该位置在输出年份内全部为缺测；将输出仅含表头的 CSV。")
    events = detect_events(series, dates, start_year, end_year, temp_threshold,
                           duration_threshold, method=method, percentile=percentile)
    label = "absolute" if method == "absolute" else f"{threshold_kind}_P{percentile:g}"
    label += f"_AT{temp_threshold:g}_D{duration_threshold}"
    return write_events_csv(output_folder, events, longitude, latitude,
                            start_year, end_year, label)


def detect_events(series, dates, start_year, end_year, temp_threshold,
                  duration_threshold, method='absolute', percentile=90):
    """逐年识别事件；组合阈值模式须同时达到当年百分位阈值和固定温度阈值。"""
    if method not in ('absolute', 'relative'):
        raise ValueError("METHOD 必须为 'absolute' 或 'relative'。")
    if method == 'relative' and (not np.isfinite(percentile) or not 0 <= percentile <= 100):
        raise ValueError("PERCENTILE 必须在 0—100 之间。")
    values = np.asarray(series)
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError("时间序列必须是一维数组，且与日期数量一致。")
    year_to_indices = group_year_indices(dates, start_year, end_year)
    events = []
    for year in range(start_year, end_year + 1):
        indices = year_to_indices.get(year, [])
        if not indices:
            print(f"警告: {year} 年没有数据，跳过。")
            continue
        year_dates = [dates[index] for index in indices]
        year_series = values[indices]
        threshold = float(temp_threshold)
        if method == 'relative':
            # 当年全部有效日值参与，包括闰年 2 月 29 日；不做跨年或同月日分组。
            samples = year_series[np.isfinite(year_series)]
            if samples.size:
                # 不将阈值降为单精度，保留百分位计算函数返回的精度。
                artt = np.percentile(samples, percentile)
                threshold = max(float(artt), float(temp_threshold))
            else:
                threshold = np.nan
        year_events = find_heatwave_events(year_series, year_dates, threshold,
                                          duration_threshold)
        events.extend(year_events)
        print(f"  {year} 年：识别到 {len(year_events)} 次热浪事件。")
    return events


def main(input_folder, output_folder, start_year, end_year,
         temp_threshold, duration_threshold, longitude, latitude,
         method="relative", percentile=80):
    """运行一种百分位方案并返回事件表路径；固定阈值模式不使用相对阈值。"""
    return run_analysis(input_folder, output_folder, start_year, end_year,
                        temp_threshold, duration_threshold, longitude, latitude,
                        threshold_kind="ARTT", method=method, percentile=percentile)


if __name__ == "__main__":
    # ========== 用户参数：运行前填写路径、经度和纬度 ==========
    INPUT_FOLDER = r""  # 填写存放逐日体感温度影像的文件夹
    OUTPUT_FOLDER = r""  # 填写事件表输出文件夹；重复运行会覆盖同名结果
    LONGITUDE = None  # 填写目标经度（WGS84 十进制度，东经为正）
    LATITUDE = None  # 填写目标纬度（WGS84 十进制度，北纬为正）
    START_YEAR = 1980  # 需要输出的起始年份（包含该年）
    END_YEAR = 2024  # 需要输出的终止年份（包含该年）
    TEMP_THRESHOLD = 29.0  # 固定温度阈值（摄氏度），达到或超过即满足该判据
    DURATION_THRESHOLD = 3  # 最少连续高温日历日数，包含开始和结束当天
    METHOD = "relative"  # "absolute" 为固定阈值；"relative" 为固定与相对阈值同时满足
    PERCENTILES = (80, 85)  # 分别计算第 80、85 百分位方案；单组可写 (80,)
    # ============================================
    if METHOD == "absolute":
        main(INPUT_FOLDER, OUTPUT_FOLDER, START_YEAR, END_YEAR,
             TEMP_THRESHOLD, DURATION_THRESHOLD, LONGITUDE, LATITUDE,
             method=METHOD)
    else:
        for percentile in PERCENTILES:
            main(INPUT_FOLDER, OUTPUT_FOLDER, START_YEAR, END_YEAR,
                 TEMP_THRESHOLD, DURATION_THRESHOLD, LONGITUDE, LATITUDE,
                 method=METHOD, percentile=percentile)
