"""Compare weekly intermittent-demand forecasts for bridge spare parts."""

from collections import defaultdict
from datetime import date, timedelta
from math import ceil, isclose, ulp
from statistics import mean, pvariance


MIN_TRAIN_WEEKS = 8
MAX_LEAD_WEEKS = 26
MAX_QUANTITY = 1_000_000_000


def sample_rows() -> list[dict]:
    """Fixed work-order demand ledger with a common closing week."""
    parts = (
        ('HB-014', '东闸桥', '液压接头密封包', 4, 2, 1,
         {1: 2, 6: 3, 11: 2, 17: 3, 20: 2, 27: 3, 32: 2, 36: 3, 41: 2, 45: 3}),
        ('HB-027', '东闸桥', '限位开关', 6, 1, 2,
         {1: 3, 5: 2, 10: 4, 14: 3, 19: 2, 23: 4, 28: 3, 32: 2}),
        ('HB-041', '西泵桥', '导向轮轴承', 5, 3, 2,
         {2: 1, 8: 12, 13: 2, 18: 1, 24: 18, 31: 2, 38: 1, 44: 16}),
        ('HB-052', '西泵桥', '编码器安装支架', 3, 1, 5,
         {0: 1, 8: 1, 15: 2, 22: 1, 30: 1, 39: 2, 47: 1}),
    )
    start = date(2025, 12, 22)
    return [
        {
            'sku': sku,
            'bridge': bridge,
            'part': part,
            'week': (start + timedelta(weeks=index)).isoformat(),
            'demand': events.get(index, 0),
            # Synthetic weekly balances with work-order drawdowns.
            'on_hand': max(0, closing_stock + (47 - index) % 6 - events.get(index, 0))
            if index < 47 else closing_stock,
            'lead_weeks': lead,
            'reserve': reserve,
        }
        for sku, bridge, part, lead, reserve, closing_stock, events in parts
        for index in range(48)
    ]


def _validate_row(row):
    for key, label in (('sku', '备件编号'), ('bridge', '桥位'), ('part', '备件名称')):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'{label}为空，请补齐备件台账后重新导入。')
    sku = row['sku']
    raw_week = row.get('week')
    try:
        week = date.fromisoformat(raw_week)
        if week.isoformat() != raw_week or week.weekday() != 0:
            raise ValueError
    except (ValueError, TypeError) as error:
        raise ValueError(f'{sku} 的周次 {raw_week} 无效，请填写 YYYY-MM-DD 格式的周一日期。') from error
    for key, label, lower, upper in (
        ('demand', '周需求量', 0, MAX_QUANTITY),
        ('on_hand', '期末可用库存', 0, MAX_QUANTITY),
        ('lead_weeks', '采购提前期', 1, MAX_LEAD_WEEKS),
        ('reserve', '最低保有量', 0, MAX_QUANTITY),
    ):
        value = row.get(key)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{sku} {week} 的{label}无效，请填写 {lower} 至 {upper} 的整数。')
    return week


def _group_rows(rows):
    groups = defaultdict(dict)
    for row in rows:
        week = _validate_row(row)
        ledger = groups[row['sku']]
        if week in ledger:
            raise ValueError(f"{row['sku']} {week} 存在重复周，请汇总需求并保留一条期末库存记录。")
        ledger[week] = row

    series = {}
    for sku, ledger in sorted(groups.items()):
        weeks = sorted(ledger)
        closing = ledger[weeks[-1]]
        for week in weeks:
            row = ledger[week]
            if (row['bridge'], row['part']) != (closing['bridge'], closing['part']):
                raise ValueError(f'{sku} {week} 的桥位或名称不一致，请核对备件编号及主数据。')
        for previous, current in zip(weeks, weeks[1:]):
            if (current - previous).days != 7:
                missing = previous + timedelta(weeks=1)
                raise ValueError(f'{sku} 缺少 {missing} 周，请补齐台账并为确认无需求的周填写 0。')
        series[sku] = [ledger[week] for week in weeks]

    cutoff = max(history[-1]['week'] for history in series.values())
    for sku, history in series.items():
        if history[-1]['week'] != cutoff:
            raise ValueError(f'{sku} 缺少截至 {cutoff} 的同期库存，请补齐结账周需求及库存。')
    return series, cutoff


def _settings(options):
    alpha = options.get('alpha', 0.20)
    beta = options.get('beta', 0.15)
    for label, value in (('需求量与间隔平滑 α', alpha), ('需求发生概率平滑 β', beta)):
        if type(value) not in (float, int) or not 0.01 <= value <= 0.99:
            raise ValueError(f'请将{label}设为 0.01 至 0.99 的数值。')
    origins = options.get('holdout_origins', 8)
    if type(origins) is not int or not 3 <= origins <= 12:
        raise ValueError('请将滚动留出窗口数设为 3 至 12 的整数。')
    idle_weeks = options.get('idle_weeks', 8)
    if type(idle_weeks) is not int or not 2 <= idle_weeks <= 26:
        raise ValueError('请将停用核查阈值设为 2 至 26 周的整数。')
    return alpha, beta, origins, idle_weeks


def _forecast_states(demand, alpha, beta):
    """After k observations, states[k] predicts each of the next weeks."""
    states = [(0.0, 0.0)]
    previous_event = 0
    size = interval = probability = sba = 0.0
    for week, quantity in enumerate(demand, start=1):
        if quantity:
            gap = week - previous_event
            if previous_event:
                size += alpha * (quantity - size)
                interval += alpha * (gap - interval)
                probability += beta * (1.0 - probability)
            else:
                size = quantity
                interval = gap
                probability = 1.0 / gap
            sba = (1.0 - alpha / 2.0) * size / interval
            previous_event = week
        else:
            probability *= 1.0 - beta
        states.append((sba, size * probability))
    return states


def _rolling_holdout(demand, states, lead, origins):
    """Use the last `origins` complete lead-time windows, one week apart."""
    last_origin = len(demand) - lead
    first_origin = last_origin - origins + 1
    errors = [[], []]
    positive_windows = 0
    for origin in range(first_origin, last_origin + 1):
        actual = sum(demand[origin:origin + lead])
        positive_windows += actual > 0
        for rate, model_errors in zip(states[origin], errors):
            model_errors.append(abs(lead * rate - actual))
    return mean(errors[0]), mean(errors[1]), positive_windows, first_origin


def _analyze_part(ledger, alpha, beta, origins, idle_weeks):
    closing = ledger[-1]
    sku, stock, lead, reserve = (closing[key] for key in ('sku', 'on_hand', 'lead_weeks', 'reserve'))
    minimum = MIN_TRAIN_WEEKS + lead + origins - 1
    if len(ledger) < minimum:
        raise ValueError(f'{sku} 仅有 {len(ledger)} 周，当前提前期和回测设置至少需要 {minimum} 周。'
                         '请补齐连续周或减少留出窗口数。')
    demand = [row['demand'] for row in ledger]
    states = _forecast_states(demand, alpha, beta)
    sba_mae, tsb_mae, positive_windows, first_origin = _rolling_holdout(demand, states, lead, origins)
    sba, tsb = (rate * lead for rate in states[-1])
    use_tsb = tsb_mae < sba_mae and not isclose(tsb_mae, sba_mae, rel_tol=0, abs_tol=1e-9)
    selected = tsb if use_tsb else sba
    # Remove floating-point noise at a positive integer before rounding up.
    nearest_unit = round(selected)
    if nearest_unit > 0 and abs(selected - nearest_unit) <= 4 * ulp(selected):
        selected = nearest_unit
    reorder_point = ceil(selected) + reserve
    gap = max(reorder_point - stock, 0)
    if gap:
        stock_signal = '低于补货点'
    elif stock == reorder_point and reorder_point > 0:
        stock_signal = '到达补货点'
    else:
        stock_signal = '未触发补货点'

    events = [index for index, quantity in enumerate(demand) if quantity > 0]
    positives = [demand[index] for index in events]
    trailing_zeros = len(demand) - 1 - events[-1] if events else len(demand)
    adi = len(demand) / len(events) if events else None
    cv2 = pvariance(positives) / mean(positives) ** 2 if len(events) >= 2 else None
    lumpy = cv2 is not None and adi >= 1.32 and cv2 >= 0.49
    flags = []
    if len(events) < 2:
        flags.append('正需求样本不足')
    if lumpy:
        flags.append('批量波动（Lumpy）')
    if trailing_zeros >= idle_weeks:
        flags.append('长期零需求，核查停用')
    if positive_windows == 0:
        flags.append('留出期全零，复核选模')

    return {
        '备件编号': sku,
        '桥位': closing['bridge'],
        '备件名称': closing['part'],
        '库存结账周': closing['week'],
        '提前期（周）': lead,
        '期末可用库存（件）': stock,
        '最低保有量（件）': reserve,
        '采用模型': 'TSB' if use_tsb else 'Croston SBA',
        'SBA提前期需求（件）': sba,
        'TSB提前期需求（件）': tsb,
        '补货点（件）': reorder_point,
        '库存缺口（件）': gap,
        '库存信号': stock_signal,
        '人工复核提示': '；'.join(flags) or '常规复核',
        'SBA留出MAE（件）': sba_mae,
        'TSB留出MAE（件）': tsb_mae,
        '首个留出周': ledger[first_origin]['week'],
        '有需求的留出窗口数': positive_windows,
        '连续零需求（周）': trailing_zeros,
        'ADI（周）': adi,
        'CV²': cv2,
    }


def compute(rows: list[dict], options: dict) -> dict:
    if not rows:
        raise ValueError('请上传包含备件编号、周需求和期末库存的 CSV。')
    alpha, beta, origins, idle_weeks = _settings(options)
    series, cutoff = _group_rows(rows)
    results = [_analyze_part(ledger, alpha, beta, origins, idle_weeks) for ledger in series.values()]
    results.sort(key=lambda row: (-row['库存缺口（件）'], row['备件编号']))
    triggers = sum(row['库存信号'] != '未触发补货点' for row in results)
    reviews = sum(row['人工复核提示'] != '常规复核' for row in results)
    return {
        'headline': f'{cutoff} 结账周：{triggers} 种备件触及补货点，{reviews} 种备件有专项复核提示。',
        'metrics': {
            '备件种数': len(results),
            '触及补货点数': triggers,
            '专项复核数': reviews,
            '每种备件留出窗口数': origins,
            '采用 TSB 数': sum(row['采用模型'] == 'TSB' for row in results),
        },
        'rows': results,
        'can_publish': True,
        'notes': [
            f'SBA 用 α={alpha:g} 平滑正需求量与间隔，再乘 (1−α/2) 修正。TSB 同样平滑正需求量，'
            f'并以 β={beta:g} 逐周更新发生概率，零需求周使概率衰减。两者在首个正需求周初始化，'
            '初始间隔为从台账首周起计的周数，初始概率取该间隔的倒数。',
            f'回测取最近 {origins} 个完整提前期窗口，每次推进一周，训练至少 8 周且只用留出周之前的需求。'
            '两种模型都用当时的周需求率乘提前期，与窗口内实际总需求比较，按 MAE 较低者选模，'
            '并列用 SBA。窗口可能重叠。无正需求训练段的预测为 0，留出期全零时须复核评分。',
            '按全史更新选中模型，补货点取提前期预计件数向上取整再加最低保有量。缺口为补货点减同期可用库存，'
            '最低记 0。补货点为正且库存等于该点时也提示复核。信号仅供盘点和采购复核，采购量还需核对在途和调拨。',
            'ADI 为总周数除以正需求周数，CV² 为正需求量总体方差除以均值平方。'
            'ADI ≥ 1.32 且 CV² ≥ 0.49 标为 Lumpy，集中检修的大批量领用可超过均值预测，请核对检修工单。',
            f'连续 {idle_weeks} 周零需求触发停用核查。请区分退役、缺货未领、替代件和工单漏记。'
            '低 TSB 预测不足以判定停用或清退，关键件保有量由维护规程复核。',
        ],
        'chart': {
            'label': '提前期需求估计（按库存缺口排列）',
            'unit': '件',
            'points': [
                {'label': f"{row['备件编号']} · {model}", 'value': row[f'{model}提前期需求（件）']}
                for row in results[:6] for model in ('SBA', 'TSB')
            ],
        },
    }
