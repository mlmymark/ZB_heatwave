# -*- coding: utf-8 -*-
"""
地表温度热浪事件识别：固定阈值，指定经纬度。

高温日须达到固定温度阈值；默认45 摄氏度，最短持续时间为 3 天。
本脚本可独立运行；运行前须填写末尾用户参数中的路径、经度和纬度。
输入：按年份分目录存放的逐日影像，结构为 年份/年月日.tif。
坐标：世界大地坐标系 WGS84 的十进制度，经度在前、纬度在后。
空间处理：读取包含目标坐标的像元，仅使用第一波段的单像元窗口，不插值。
温度单位：默认按 原始值 × 0.1 转为摄氏度；可设置缩放系数和偏移量。
事件规则：达到阈值包含等于，逐年识别，跨年连续高温段在年界拆分。
连续性：默认普通缺日中断；缺失闰日允许连接，但不计入持续天数。
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

THRESHOLD_KIND = "absolute"  # 阈值类型：固定温度阈值
DEFAULT_METHOD = "absolute"  # 默认判定模式


def validate_year_range(start_year, end_year, label, allow_none=False):
    """校验起止年份，可按需允许其中一端不设置年份限制。"""
    for year in (start_year, end_year):
        if year is None and allow_none:
            continue
        if isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999:
            raise ValueError(f"{label}年份必须是 1 至 9999 的整数。")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError(f"{label}起始年份不能大于终止年份。")


def validate_parameters(start_year, end_year, temp_threshold, duration_threshold,
                        longitude, latitude, method, percentile,
                        value_scale, value_offset):
    """校验经纬度、输出年份、阈值、判定模式及温度换算参数。"""
    if longitude is None or latitude is None:
        raise ValueError("请先在脚本末尾填写 LONGITUDE（经度）和 LATITUDE（纬度）。")
    if not (math.isfinite(longitude) and -180 <= longitude <= 180):
        raise ValueError("LONGITUDE 必须是 -180 至 180 之间的有限数值。")
    if not (math.isfinite(latitude) and -90 <= latitude <= 90):
        raise ValueError("LATITUDE 必须是 -90 至 90 之间的有限数值。")
    validate_year_range(start_year, end_year, "输出")
    if not math.isfinite(temp_threshold):
        raise ValueError("TEMP_THRESHOLD 必须是有限数值，单位为摄氏度。")
    if (isinstance(duration_threshold, bool) or not isinstance(duration_threshold, int)
            or duration_threshold < 1):
        raise ValueError("DURATION_THRESHOLD 必须是正整数。")
    if method not in ("absolute", "relative"):
        raise ValueError("METHOD 只能是 'absolute' 或 'relative'。")
    if THRESHOLD_KIND == "absolute" and method != "absolute":
        raise ValueError("热浪 1 仅使用固定阈值；组合阈值请运行热浪 2 或热浪 3。")
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("PERCENTILE 必须大于 0 且不超过 100。")
    if not math.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("VALUE_SCALE 必须是正的有限数值。")
    if not math.isfinite(value_offset):
        raise ValueError("VALUE_OFFSET 必须是有限数值。")


def list_daily_tifs(input_folder, start_year=None, end_year=None):
    """从一级四位年份目录收集影像，校验日期、目录年份及重复日期。"""
    validate_year_range(start_year, end_year, "读取", allow_none=True)
    folder = Path(input_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"输入根目录不存在：{folder}")
    records = []
    seen_dates = set()
    for year_folder in sorted(folder.iterdir()):
        if not (year_folder.is_dir() and len(year_folder.name) == 4
                and year_folder.name.isdigit()):
            continue
        year = int(year_folder.name)
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        for path in year_folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
                continue
            try:
                if len(path.stem) != 8 or not path.stem.isdigit():
                    raise ValueError
                date = datetime.datetime.strptime(path.stem, "%Y%m%d").date()
            except ValueError as exc:
                raise ValueError(f"TIFF 文件名必须为 YYYYMMDD.tif：{path}") from exc
            if date.year != year:
                raise ValueError(f"影像日期与年份文件夹不一致：{path}")
            if date in seen_dates:
                raise ValueError(f"存在重复日期 {date} 的 TIFF：{path}")
            seen_dates.add(date)
            records.append((date, path))
    if not records:
        raise FileNotFoundError("指定年份范围的 YYYY 子文件夹中没有逐日 TIFF。")
    return sorted(records, key=lambda item: item[0])


def spatial_reference(projection):
    """解析影像投影信息，并固定为经度在前、纬度在后的坐标轴顺序。"""
    if not projection:
        raise ValueError("TIFF 缺少坐标系，无法可靠地用经纬度定位像元。")
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(projection) != 0:
        raise ValueError("无法解析 TIFF 的坐标系。")
    if hasattr(srs, "SetAxisMappingStrategy"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def locate_pixel(dataset, longitude, latitude):
    """将输入经纬度转换到影像坐标系，通过逆仿射变换定位包含该点的像元。"""
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
            f"经纬度 ({longitude}, {latitude}) 不在栅格范围内；"
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
    """所有参与计算的年份、日期须使用同一空间网格。"""
    if (dataset.RasterYSize, dataset.RasterXSize) != (
            reference["rows"], reference["cols"]):
        raise ValueError(f"栅格行列数与首幅影像不一致：{path}")
    geo = dataset.GetGeoTransform(can_return_null=True)
    if geo is None or not np.allclose(geo, reference["geo"], rtol=0, atol=1e-12):
        raise ValueError(f"栅格仿射变换与首幅影像不一致：{path}")
    projection = dataset.GetProjection()
    if projection != reference["projection"]:
        if not spatial_reference(projection).IsSame(
                spatial_reference(reference["projection"])):
            raise ValueError(f"栅格坐标系与首幅影像不一致：{path}")


def in_year_range(date, start_year, end_year):
    """判断日期是否处于给定年份范围；未设置的端点不作限制。"""
    return ((start_year is None or date.year >= start_year)
            and (end_year is None or date.year <= end_year))


def read_point_series(input_folder, longitude, latitude,
                      start_year=None, end_year=None,
                      value_scale=0.1, value_offset=0.0, input_nodata=-32768,
                      baseline_start_year=None, baseline_end_year=None,
                      include_baseline=False):
    """逐文件读取单像元窗口，先剔除缺测编码，再以单精度浮点数换算温度。"""
    records = list_daily_tifs(
        input_folder, None if include_baseline else start_year,
        None if include_baseline else end_year,
    )
    if not any(in_year_range(date, start_year, end_year) for date, _ in records):
        raise FileNotFoundError("输出年份内没有逐日 TIFF。")
    if include_baseline:
        validate_year_range(baseline_start_year, baseline_end_year, "基准期", True)
        if not any(in_year_range(date, baseline_start_year, baseline_end_year)
                   for date, _ in records):
            raise FileNotFoundError("CRTT 基准期内没有逐日 TIFF。")
        records = [(date, path) for date, path in records
                   if in_year_range(date, start_year, end_year)
                   or in_year_range(date, baseline_start_year, baseline_end_year)]
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
            value = np.float32(values[0, 0])
            valid = bool(np.isfinite(value))
            # 缺测编码属于原始值，必须在转换温度单位前剔除。
            for nodata in (band.GetNoDataValue(), input_nodata):
                if nodata is not None and value == np.float32(nodata):
                    valid = False
            if valid and band.GetMaskFlags() != gdal.GMF_ALL_VALID:
                mask_band = band.GetMaskBand()
                mask = mask_band.ReadAsArray(col, row, 1, 1)
                if mask is None:
                    raise RuntimeError(f"有效数据掩膜读取失败：{path}")
                valid = bool(mask[0, 0] != 0)
            if valid:
                converted = value * np.float32(value_scale) + np.float32(value_offset)
                if np.isfinite(converted):
                    series[index] = converted
        finally:
            mask_band = band = dataset = None
        if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(records):
            print(f"读取进度：{index + 1}/{len(records)}，日期 {date}")
    return series, dates, reference


def dates_are_continuous(previous_date, current_date, allow_missing_leap_day=True):
    """判断两日期是否连续，并按设置处理缺失闰日的特殊情况。"""
    if (current_date - previous_date).days == 1:
        return True
    return (allow_missing_leap_day and previous_date.year == current_date.year
            and calendar.isleap(previous_date.year)
            and (previous_date.month, previous_date.day) == (2, 28)
            and (current_date.month, current_date.day) == (3, 1))


def find_heatwave_events(series, dates, thresholds, duration_threshold,
                         break_on_date_gap=True, allow_missing_leap_day=True):
    """返回每次合格事件；持续天数为实际高温日条数，未观测闰日不计入。"""
    values = np.asarray(series)
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError("温度序列必须是一维，且与日期列表等长。")
    if any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError("日期必须严格递增，不能重复。")
    if (isinstance(duration_threshold, bool) or not isinstance(duration_threshold, int)
            or duration_threshold < 1):
        raise ValueError("最小持续天数必须是正整数。")
    threshold_values = np.asarray(thresholds)
    if threshold_values.ndim != 0 and threshold_values.shape != values.shape:
        raise ValueError("阈值必须是标量，或与温度序列等长的一维数组。")
    # 保留单精度温度与标量阈值的比较方式，避免改为双精度后
    # 改变恰好处于百分位阈值舍入边界的高温日判定。
    comparison_thresholds = thresholds if threshold_values.ndim == 0 else threshold_values
    hot = (np.isfinite(values) & np.isfinite(threshold_values)
           & (values >= comparison_thresholds))
    events = []
    start = end = None
    duration = 0

    def finish_run():
        """结束当前连续高温段，仅保存达到最短持续时间的事件。"""
        if start is not None and duration >= duration_threshold:
            events.append({"start_date": start, "end_date": end,
                           "duration_days": duration})

    for date, is_hot in zip(dates, hot):
        if is_hot:
            continuous = (start is not None and (not break_on_date_gap
                          or dates_are_continuous(end, date, allow_missing_leap_day)))
            if continuous:
                end = date
                duration += 1
            else:
                finish_run()
                start = end = date
                duration = 1
        else:
            finish_run()
            start = end = None
            duration = 0
    finish_run()
    return events


def date_to_doy(date):
    """返回从零开始的气候日序号（0 至 364）；2 月 29 日并入 2 月 28 日。"""
    doy = date.timetuple().tm_yday
    if calendar.isleap(date.year) and (date.month > 2 or (date.month, date.day) == (2, 29)):
        doy -= 1
    return doy - 1


def compute_daily_percentile_threshold(series, dates, percentile,
                                       baseline_start_year=None, baseline_end_year=None):
    """使用基准期内同气候日的有效样本计算百分位阈值，以单精度浮点数存储。"""
    validate_year_range(baseline_start_year, baseline_end_year, "基准期", True)
    values = np.asarray(series)
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError("温度序列必须是一维，且与日期列表等长。")
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("PERCENTILE 必须大于 0 且不超过 100。")
    groups = {}
    for index, date in enumerate(dates):
        if in_year_range(date, baseline_start_year, baseline_end_year):
            groups.setdefault(date_to_doy(date), []).append(index)
    if not groups:
        raise ValueError("CRTT 基准期内没有日期记录。")
    thresholds = np.full(365, np.nan, dtype=np.float32)
    for doy, indices in groups.items():
        samples = values[indices]
        samples = samples[np.isfinite(samples)]
        if samples.size:
            thresholds[doy] = np.percentile(samples, percentile)
    return thresholds


def detect_events(series, dates, start_year, end_year, temp_threshold,
                  duration_threshold, method=None, percentile=90,
                  baseline_start_year=None, baseline_end_year=None,
                  break_on_date_gap=True, allow_missing_leap_day=True):
    """逐年识别热浪事件；组合阈值模式须同时达到相对阈值和固定温度阈值。"""
    method = DEFAULT_METHOD if method is None else method
    validate_year_range(start_year, end_year, "输出")
    if method not in ("absolute", "relative"):
        raise ValueError("METHOD 只能是 'absolute' 或 'relative'。")
    if THRESHOLD_KIND == "absolute" and method != "absolute":
        raise ValueError("热浪 1 仅使用固定阈值。")
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("PERCENTILE 必须大于 0 且不超过 100。")
    values = np.asarray(series)
    if values.ndim != 1 or len(values) != len(dates):
        raise ValueError("温度序列必须是一维，且与日期列表等长。")
    if any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError("日期必须严格递增，不能重复。")
    daily_thresholds = None
    if method == "relative" and THRESHOLD_KIND == "CRTT":
        daily_thresholds = compute_daily_percentile_threshold(
            values, dates, percentile, baseline_start_year, baseline_end_year,
        )
    year_indices = {}
    for index, date in enumerate(dates):
        if start_year <= date.year <= end_year:
            year_indices.setdefault(date.year, []).append(index)
    events = []
    for year in range(start_year, end_year + 1):
        indices = year_indices.get(year, [])
        if not indices:
            print(f"警告：{year} 年没有数据，跳过。")
            continue
        year_dates = [dates[index] for index in indices]
        year_values = values[indices]
        thresholds = float(temp_threshold)
        if daily_thresholds is not None:
            doys = np.asarray([date_to_doy(date) for date in year_dates], dtype=int)
            thresholds = np.maximum(daily_thresholds[doys], thresholds)
        elif method == "relative" and THRESHOLD_KIND == "ARTT":
            valid = year_values[np.isfinite(year_values)]
            artt = np.percentile(valid, percentile) if valid.size else np.nan
            thresholds = max(artt, thresholds)
        year_events = find_heatwave_events(
            year_values, year_dates, thresholds, duration_threshold,
            break_on_date_gap, allow_missing_leap_day,
        )
        events.extend(year_events)
        print(f"  {year} 年：识别到 {len(year_events)} 次热浪事件。")
    return events


def write_events_csv(output_folder, events, longitude, latitude,
                     start_year, end_year, label):
    """以带字节顺序标记的 UTF-8 编码写出事件表；无事件时保留表头。"""
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
                             event["end_date"].isoformat(), event["duration_days"]])
    print(f"共输出 {len(events)} 次热浪事件：{path}")
    return path


def main(input_folder, output_folder, start_year, end_year,
         temp_threshold, duration_threshold, longitude, latitude,
         method=None, percentile=90,
         value_scale=0.1, value_offset=0.0, input_nodata=-32768,
         baseline_start_year=None, baseline_end_year=None,
         break_on_date_gap=True, allow_missing_leap_day=True):
    """返回事件表路径；气候逐日阈值的基准期独立于输出期，空值表示不限制该端年份。"""
    method = DEFAULT_METHOD if method is None else method
    validate_parameters(start_year, end_year, temp_threshold, duration_threshold,
                        longitude, latitude, method, percentile, value_scale, value_offset)
    include_baseline = THRESHOLD_KIND == "CRTT" and method == "relative"
    if include_baseline:
        validate_year_range(baseline_start_year, baseline_end_year, "基准期", True)
    series, dates, _ = read_point_series(
        input_folder, longitude, latitude, start_year, end_year,
        value_scale, value_offset, input_nodata,
        baseline_start_year, baseline_end_year, include_baseline,
    )
    selected = [i for i, date in enumerate(dates) if start_year <= date.year <= end_year]
    n_valid = int(np.count_nonzero(np.isfinite(series[selected])))
    print(f"输出年份内：{len(selected)} 个日文件，其中 {n_valid} 天为有效像元值。")
    if n_valid == 0:
        print("提示：该位置全部缺测，将输出仅含表头的 CSV。")
    print("事件按年拆分；持续天数为实际高温影像日数。")
    if not break_on_date_gap:
        print("日期缺口不中断事件；NoData 和非高温日仍中断事件。")
    elif allow_missing_leap_day:
        print("缺日中断事件，但缺失闰年 2 月 29 日允许连续（不计缺失日天数）。")
    else:
        print("严格日历连续：任何缺日均中断事件。")
    events = detect_events(
        series, dates, start_year, end_year, temp_threshold, duration_threshold,
        method, percentile, baseline_start_year, baseline_end_year,
        break_on_date_gap, allow_missing_leap_day,
    )
    label = "absolute" if method == "absolute" else f"{THRESHOLD_KIND}_P{percentile:g}"
    label += f"_LST{temp_threshold:g}_D{duration_threshold}"
    if include_baseline:
        baseline_years = [date.year for date in dates
                          if in_year_range(date, baseline_start_year, baseline_end_year)]
        label += f"_B{min(baseline_years)}-{max(baseline_years)}"
    if not break_on_date_gap:
        label += "_ignoreDateGaps"
    elif not allow_missing_leap_day:
        label += "_strictCalendar"
    return write_events_csv(output_folder, events, longitude, latitude,
                            start_year, end_year, label)


if __name__ == "__main__":
    # ========== 用户参数：运行前填写路径、经度和纬度 ==========
    INPUT_FOLDER = r""  # 填写输入根目录，其中按四位年份设置子文件夹
    OUTPUT_FOLDER = r""  # 填写事件表输出文件夹；重复运行会覆盖同名结果
    LONGITUDE = None  # 填写目标经度（WGS84 十进制度，东经为正）
    LATITUDE = None  # 填写目标纬度（WGS84 十进制度，北纬为正）
    START_YEAR = 2003  # 需要输出的起始年份（包含该年）
    END_YEAR = 2024  # 需要输出的终止年份（包含该年）
    TEMP_THRESHOLD = 45.0  # 固定温度阈值（摄氏度），达到或超过即满足该判据
    DURATION_THRESHOLD = 3  # 最少连续高温影像日数；缺失闰日不计入
    VALUE_SCALE = 0.1  # 温度缩放系数；原始值 450 对应 45.0 摄氏度
    VALUE_OFFSET = 0.0  # 温度换算偏移量（摄氏度）
    INPUT_NODATA = -32768  # 备用原始缺测编码，在缩放前过滤，同时检查影像缺测值和掩膜
    ALLOW_MISSING_LEAP_DAY = True  # 是否允许缺失闰日连接；设为 False 时要求严格日历连续
    # ==============================================
    main(
        INPUT_FOLDER, OUTPUT_FOLDER, START_YEAR, END_YEAR,
        TEMP_THRESHOLD, DURATION_THRESHOLD, LONGITUDE, LATITUDE,
        value_scale=VALUE_SCALE, value_offset=VALUE_OFFSET,
        input_nodata=INPUT_NODATA,
        allow_missing_leap_day=ALLOW_MISSING_LEAP_DAY,
    )
