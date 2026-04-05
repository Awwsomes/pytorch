import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution, basinhopping
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import multiprocessing as mp
import time

# 设置中文字体和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置参数 ---
EXCEL_FILE_PATH = "D:\\Pytorch\\yolov5-master\\src\\R3.xlsx" # 【请修改】包含r和v数据的Excel文件路径
R_COLUMN_NAME = 'r'      # 水平射程r所在的列名
V_COLUMN_NAME = 'v'      # 篮球出射速度v所在的列名
HEADER_ROW = 0           # Excel中表头所在的行号 (第一行为0)

# --- 物理常量配置 ---
LAUNCH_HEIGHT = 1.5      # 篮球发射高度 (单位: 米)
BASKET_HEIGHT = 2.43     # 篮筐高度 (单位: 米)
LAUNCH_ANGLE_DEG = 68.0  # 篮球发射俯仰角 (单位: 度)
GRAVITY = 9.788          # 重力加速度 (单位: m/s²)

LAUNCH_ANGLE_RAD = np.deg2rad(LAUNCH_ANGLE_DEG)
DELTA_H = BASKET_HEIGHT - LAUNCH_HEIGHT # 发射点与篮筐的高度差

# 并行计算配置
MAX_WORKERS = min(mp.cpu_count(), 8)  # 限制最大进程数
ENABLE_PARALLEL = True  # 是否启用并行计算

# --- 模型函数定义 ---

def power_3param_func(r, a, b, c):
    return a * np.power(r, b) + c

def power_2param_func(r, a, b):
    return a * np.power(r, b)

def linear_basic_func(r, a, b):
    return a * r + b

def polynomial_2_func(r, a, b, c):
    return a * r**2 + b * r + c

def polynomial_3_func(r, a, b, c, d):
    return a * r**3 + b * r**2 + c * r + d

def ballistic_ideal_func(r, k):
    """
    理想弹道模型: v = k * sqrt(r² + Δh²) / cos(θ)
    基于理想抛物运动，适合固定角度发射的情况
    """
    cos_theta = np.cos(LAUNCH_ANGLE_RAD)
    total_distance = np.sqrt(r**2 + DELTA_H**2)
    return k * total_distance / cos_theta

def ballistic_drag_linear_func(r, a, b):
    """
    考虑线性空气阻力的弹道模型: v = a * sqrt(r² + Δh²) + b
    适合中低速情况下的空气阻力修正
    """
    total_distance = np.sqrt(r**2 + DELTA_H**2)
    return a * total_distance + b

def ballistic_drag_quadratic_func(r, a, b, c):
    """
    考虑二次空气阻力的弹道模型: v = a * (r² + Δh²) + b * sqrt(r² + Δh²) + c
    适合高速情况下的空气阻力修正
    """
    total_distance = np.sqrt(r**2 + DELTA_H**2)
    return a * (r**2 + DELTA_H**2) + b * total_distance + c

def motor_linear_func(r, k_motor, v_base):
    """
    电机线性控制模型: v = k_motor * r + v_base
    假设电机输出与距离成线性关系
    """
    return k_motor * r + v_base

def motor_pwm_func(r, v_max, k_pwm, offset):
    """
    电机PWM控制模型: v = v_max * (1 - exp(-k_pwm * r)) + offset
    模拟电机PWM控制的非线性响应特性
    """
    with np.errstate(over='ignore'):
        exp_term = np.clip(-k_pwm * r, -700, 700)
        return v_max * (1 - np.exp(exp_term)) + offset

def motor_torque_func(r, torque_const, r_ref, v_min):
    """
    电机扭矩控制模型: v = torque_const * sqrt(r / r_ref) + v_min
    基于电机需要提供的扭矩与距离的平方根关系
    """
    with np.errstate(invalid='ignore'):
        if np.any(r <= 0):
            return np.full_like(r, np.inf)
        return torque_const * np.sqrt(r / r_ref) + v_min

def ballistic_energy_func(r, E_const, efficiency):
    """
    能量守恒弹道模型: v = sqrt(2 * E_const * (r² + Δh²) * efficiency)
    基于篮球需要的动能与飞行距离的关系
    """
    with np.errstate(invalid='ignore'):
        total_distance_sq = r**2 + DELTA_H**2
        energy_term = 2 * E_const * total_distance_sq * efficiency
        if np.any(energy_term <= 0):
            return np.full_like(r, np.inf)
        return np.sqrt(energy_term)

def empirical_sqrt_func(r, a, b, c):
    """
    经验平方根模型: v = a * sqrt(r + b) + c
    常用于描述物理系统中的非线性关系
    """
    with np.errstate(invalid='ignore'):
        sqrt_arg = r + b
        if np.any(sqrt_arg <= 0):
            return np.full_like(r, np.inf)
        return a * np.sqrt(sqrt_arg) + c

def exponential_func(r, a, b, c):
    # 添加溢出保护
    with np.errstate(over='ignore'):
        exp_term = np.clip(b * r, -700, 700)  # 防止exp溢出
        return a * np.exp(exp_term) + c

def logarithmic_func(r, a, b, c):
    # 确保log参数为正
    with np.errstate(invalid='ignore'):
        log_arg = b * r + 1
        if np.any(log_arg <= 0):
            return np.full_like(r, np.inf)
        return a * np.log(log_arg) + c

def rational_func(r, a, b, c, d):
    """
    有理函数模型: v = (a * r + b) / (c * r + d)
    适合描述饱和特性
    """
    denominator = c * r + d
    # 避免分母为零
    if np.any(np.abs(denominator) < 1e-10):
        return np.full_like(r, np.inf)
    return (a * r + b) / denominator

# --- 数据处理模块 ---
def load_data():
    """从Excel文件加载数据"""
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, header=HEADER_ROW)
        if R_COLUMN_NAME not in df.columns:
            raise ValueError(f"列名 '{R_COLUMN_NAME}' 未找到。")
        if V_COLUMN_NAME not in df.columns:
            raise ValueError(f"列名 '{V_COLUMN_NAME}' 未找到。")
        r_data = df[R_COLUMN_NAME].values
        v_data = df[V_COLUMN_NAME].values
        
        print(f"从 '{EXCEL_FILE_PATH}' 读取成功，共 {len(r_data)} 个原始数据项。")
        if len(r_data) != len(v_data):
            raise ValueError("r和v数据长度不一致。")
        
        return r_data, v_data
        
    except Exception as e:
        print(f"读取或校验数据时出错: {e}")
        exit()

def preprocess_data(r_data, v_data):
    """数据预处理：移除空值、排序、对重复r值取平均"""
    r_array = np.array(r_data, dtype=float)
    v_array = np.array(v_data, dtype=float)
    nan_mask = np.isnan(r_array) | np.isnan(v_array)
    
    if np.any(nan_mask):
        r_array = r_array[~nan_mask]
        v_array = v_array[~nan_mask]
        print(f"预处理：移除了 {np.sum(nan_mask)} 行包含空值的数据。")
    
    sort_indices = np.argsort(r_array)
    r_sorted = r_array[sort_indices]
    v_sorted = v_array[sort_indices]
    
    unique_r_vals = np.unique(r_sorted)
    if len(unique_r_vals) < len(r_sorted):
        print("预处理：对重复r值对应的v值取平均。")
        v_aggregated = [np.mean(v_sorted[r_sorted == ur]) for ur in unique_r_vals]
        r_processed = unique_r_vals
        v_processed = np.array(v_aggregated)
    else:
        r_processed = r_sorted
        v_processed = v_sorted
    
    print(f"用于拟合的数据点数量: {len(r_processed)}")
    return r_processed, v_processed

def detect_outliers(r_data, v_data, method='iqr', threshold=1.5):
    """检测异常值"""
    outlier_indices = []
    
    if method == 'iqr':
        for data in [r_data, v_data]:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = np.where((data < lower) | (data > upper))[0]
            outlier_indices.extend(outliers)
    
    elif method == 'zscore':
        for data in [r_data, v_data]:
            z_scores = np.abs(stats.zscore(data))
            outliers = np.where(z_scores > threshold)[0]
            outlier_indices.extend(outliers)
    
    return np.unique(outlier_indices)

def clean_outliers(r_data, v_data, threshold=1.5):
    """清理异常值"""
    outlier_indices = detect_outliers(r_data, v_data, method='iqr', threshold=threshold)
    
    if len(outlier_indices) > 0:
        print(f"检测到 {len(outlier_indices)} 个潜在异常值")
        r_clean = np.delete(r_data, outlier_indices)
        v_clean = np.delete(v_data, outlier_indices)
        
        if len(r_clean) >= 3:
            print(f"移除异常值后剩余 {len(r_clean)} 个数据点")
            return r_clean, v_clean, outlier_indices
        else:
            print("移除异常值后数据点不足，保留原始数据")
    
    return r_data, v_data, []

# --- 模型配置模块 ---
def get_models():
    """获取所有拟合模型配置"""
    return {
        # 基础数学模型
        'linear_basic': {
            'func': linear_basic_func,
            'params': ['a', 'b'],
            'description': '线性模型: v = a*r + b',
            'category': '基础数学模型'
        },
        'polynomial_2': {
            'func': polynomial_2_func,
            'params': ['a', 'b', 'c'],
            'description': '二次多项式: v = a*r² + b*r + c',
            'category': '基础数学模型'
        },
        'polynomial_3': {
            'func': polynomial_3_func,
            'params': ['a', 'b', 'c', 'd'],
            'description': '三次多项式: v = a*r³ + b*r² + c*r + d',
            'category': '基础数学模型'
        },
        'power_2param': {
            'func': power_2param_func,
            'params': ['a', 'b'],
            'description': '幂函数: v = a*r^b',
            'category': '基础数学模型'
        },
        'power_3param': {
            'func': power_3param_func,
            'params': ['a', 'b', 'c'],
            'description': '幂函数+常数: v = a*r^b + c',
            'category': '基础数学模型'
        },
        
        # 弹道物理模型
        'ballistic_ideal': {
            'func': ballistic_ideal_func,
            'params': ['k'],
            'description': '理想弹道: v = k*√(r²+Δh²)/cos(θ)',
            'category': '弹道物理模型'
        },
        'ballistic_drag_linear': {
            'func': ballistic_drag_linear_func,
            'params': ['a', 'b'],
            'description': '线性阻力弹道: v = a*√(r²+Δh²) + b',
            'category': '弹道物理模型'
        },
        'ballistic_drag_quadratic': {
            'func': ballistic_drag_quadratic_func,
            'params': ['a', 'b', 'c'],
            'description': '二次阻力弹道: v = a*(r²+Δh²) + b*√(r²+Δh²) + c',
            'category': '弹道物理模型'
        },
        'ballistic_energy': {
            'func': ballistic_energy_func,
            'params': ['E_const', 'efficiency'],
            'description': '能量弹道: v = √(2*E*(r²+Δh²)*η)',
            'category': '弹道物理模型'
        },
        
        # 电机控制模型
        'motor_linear': {
            'func': motor_linear_func,
            'params': ['k_motor', 'v_base'],
            'description': '电机线性: v = k*r + v_base',
            'category': '电机控制模型'
        },
        'motor_pwm': {
            'func': motor_pwm_func,
            'params': ['v_max', 'k_pwm', 'offset'],
            'description': '电机PWM: v = v_max*(1-exp(-k*r)) + offset',
            'category': '电机控制模型'
        },
        'motor_torque': {
            'func': motor_torque_func,
            'params': ['torque_const', 'r_ref', 'v_min'],
            'description': '电机扭矩: v = k*√(r/r_ref) + v_min',
            'category': '电机控制模型'
        },
        
        # 通用经验模型
        'empirical_sqrt': {
            'func': empirical_sqrt_func,
            'params': ['a', 'b', 'c'],
            'description': '平方根模型: v = a*√(r+b) + c',
            'category': '经验模型'
        },
        'exponential': {
            'func': exponential_func,
            'params': ['a', 'b', 'c'],
            'description': '指数模型: v = a*exp(b*r) + c',
            'category': '经验模型'
        },
        'logarithmic': {
            'func': logarithmic_func,
            'params': ['a', 'b', 'c'],
            'description': '对数模型: v = a*ln(b*r + 1) + c',
            'category': '经验模型'
        },
        'rational': {
            'func': rational_func,
            'params': ['a', 'b', 'c', 'd'],
            'description': '有理函数: v = (a*r + b)/(c*r + d)',
            'category': '经验模型'
        }
    }

def get_fitting_strategies():
    """获取拟合策略配置"""
    return [
        {'name': 'curve_fit_默认', 'method': 'curve_fit'},
        {'name': 'curve_fit_TRF', 'method': 'curve_fit', 'method_opt': 'trf'},
        {'name': 'curve_fit_Dogbox', 'method': 'curve_fit', 'method_opt': 'dogbox'},
        {'name': '全局优化_Basin', 'method': 'global_basin'},
    ]

def get_linear_fitting_strategies():
    """获取线性模型专用的拟合策略"""
    return [
        {'name': '专用线性拟合', 'method': 'special_linear'},
    ]

# --- 拟合参数估计模块 ---
def get_robust_initial_guess(model_name, r_data, v_data, models):
    """为不同模型提供鲁棒的初值估计"""
    try:
        r_mean, r_std = np.mean(r_data), np.std(r_data)
        v_mean, v_std = np.mean(v_data), np.std(v_data)
        v_min, v_max = np.min(v_data), np.max(v_data)
        
        if model_name.startswith('linear_') or model_name == 'motor_linear':
            coeffs = np.polyfit(r_data, v_data, 1)
            return coeffs.tolist()
        elif model_name == 'ballistic_ideal':
            # 基于理想弹道公式的初值估计
            cos_theta = np.cos(LAUNCH_ANGLE_RAD)
            total_dist_mean = np.sqrt(r_mean**2 + DELTA_H**2)
            k_est = v_mean * cos_theta / total_dist_mean
            return [k_est]
        elif model_name == 'ballistic_drag_linear':
            total_dist_mean = np.sqrt(r_mean**2 + DELTA_H**2)
            a_est = v_mean / total_dist_mean
            return [a_est, v_min * 0.1]
        elif model_name == 'ballistic_drag_quadratic':
            total_dist_mean = np.sqrt(r_mean**2 + DELTA_H**2)
            return [v_mean / (r_mean**2 + DELTA_H**2), v_mean / total_dist_mean, 0.0]
        elif model_name == 'ballistic_energy':
            total_dist_sq_mean = r_mean**2 + DELTA_H**2
            E_est = v_mean**2 / (2 * total_dist_sq_mean)
            return [E_est, 1.0]
        elif model_name == 'motor_pwm':
            return [v_max * 1.2, 1.0 / r_mean, v_min]
        elif model_name == 'motor_torque':
            return [v_std, r_mean, v_min]
        elif model_name == 'empirical_sqrt':
            return [v_std, 0.0, v_mean]
        elif model_name in ['power_3param', 'power_2param']:
            if np.all(r_data > 0) and np.all(v_data > 0):
                log_r = np.log(r_data)
                log_v = np.log(v_data)
                coeffs = np.polyfit(log_r, log_v, 1)
                b_est = coeffs[0]
                a_est = np.exp(coeffs[1])
                
                if model_name == 'power_3param':
                    return [a_est, b_est, v_min * 0.1]
                else:
                    return [a_est, b_est]
        elif model_name.startswith('polynomial_'):
            degree = int(model_name.split('_')[1])
            coeffs = np.polyfit(r_data, v_data, degree)
            return coeffs.tolist()
        elif model_name == 'exponential':
            if np.all(v_data > 0):
                log_v = np.log(v_data - v_min + 1)
                coeffs = np.polyfit(r_data, log_v, 1)
                return [np.exp(coeffs[1]), coeffs[0], v_min]
            else:
                return [v_std, 0.1, v_mean]
        elif model_name == 'logarithmic':
            return [v_std, 1.0, v_mean]
        elif model_name == 'rational':
            return [v_mean/r_mean, v_mean, 1.0, 1.0]
        else:
            return [1.0] * len(models[model_name]['params'])
            
    except Exception:
        return [1.0] * len(models[model_name]['params'])

def get_parameter_bounds(model_name, models):
    """为不同模型设置合理的参数边界"""
    bounds_dict = {
        'linear_basic': ([-1000, -1000], [1000, 1000]),
        'motor_linear': ([-1000, -1000], [1000, 1000]),
        'polynomial_2': ([-1000, -1000, -1000], [1000, 1000, 1000]),
        'polynomial_3': ([-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]),
        'power_2param': ([-1000, -10], [1000, 10]),
        'power_3param': ([-1000, -10, -1000], [1000, 10, 1000]),
        'ballistic_ideal': ([0.001], [1000]),
        'ballistic_drag_linear': ([0.001, -1000], [1000, 1000]),
        'ballistic_drag_quadratic': ([-1000, -1000, -1000], [1000, 1000, 1000]),
        'ballistic_energy': ([0.001, 0.001], [1000, 10.0]),
        'motor_pwm': ([0.001, 0.001, -1000], [1000, 10, 1000]),
        'motor_torque': ([0.001, 0.001, -1000], [1000, 100, 1000]),
        'empirical_sqrt': ([-1000, -100, -1000], [1000, 100, 1000]),
        'exponential': ([-1000, -10, -1000], [1000, 10, 1000]),
        'logarithmic': ([-1000, 0.001, -1000], [1000, 100, 1000]),
        'rational': ([-1000, -1000, -100, 0.001], [1000, 1000, 100, 100])
    }
    
    default_bounds = ([-1000] * len(models[model_name]['params']), 
                     [1000] * len(models[model_name]['params']))
    return bounds_dict.get(model_name, default_bounds)

# --- 模型评估模块 ---
def calculate_model_criteria(y_true, y_pred, n_params, n_data):
    """计算AIC, BIC和调整R²"""
    mse = np.mean((y_true - y_pred)**2)
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    
    if tss == 0: return {'r2': 1, 'adj_r2': 1, 'aic': -np.inf, 'bic': -np.inf, 'mse': 0}

    r2 = 1 - rss/tss
    # 修正：当数据点少于参数时，adj_r2可能无意义
    if n_data - n_params - 1 <= 0:
        adj_r2 = r2
    else:
        adj_r2 = 1 - (1 - r2) * (n_data - 1) / (n_data - n_params - 1)
    
    # 修正：避免log(0)
    if mse <= 0: mse = 1e-9
    log_likelihood = -0.5 * n_data * (np.log(2 * np.pi) + np.log(mse) + 1)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_data) - 2 * log_likelihood
    
    return {'r2': r2, 'adj_r2': adj_r2, 'aic': aic, 'bic': bic, 'mse': mse}

# --- 拟合执行模块 ---
def fit_single_model_strategy(args):
    """单个模型策略的拟合函数，用于并行计算"""
    model_name, model_info, strategy, r_data, v_data, models = args
    
    try:
        # 只有线性模型才使用专用线性拟合方法
        if strategy['method'] == 'special_linear' and not model_name.startswith('linear_'):
            return None
            
        model_func = model_info['func']
        n_params = len(model_info['params'])
        
        p0 = get_robust_initial_guess(model_name, r_data, v_data, models)
        bounds = get_parameter_bounds(model_name, models)
        
        # 执行拟合
        if strategy['method'] == 'curve_fit':
            method_kw = strategy.get('method_opt', 'lm')
            if method_kw == 'lm':
                popt, _ = curve_fit(model_func, r_data, v_data, p0=p0, maxfev=20000)
            else:
                popt, _ = curve_fit(model_func, r_data, v_data, p0=p0, bounds=bounds, method=method_kw, maxfev=20000)
        
        elif strategy['method'] == 'global_basin':
            def objective(params):
                try:
                    predicted = model_func(r_data, *params)
                    if np.any(np.isinf(predicted)) or np.any(np.isnan(predicted)): return 1e12
                    return np.mean((v_data - predicted)**2)
                except:
                    return 1e12
            
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': list(zip(bounds[0], bounds[1]))}
            result = basinhopping(objective, p0, niter=50, minimizer_kwargs=minimizer_kwargs, stepsize=0.5)
            if not result.success: return None
            popt = result.x
        
        elif strategy['method'] == 'special_linear':
            # 专用线性拟合，只适用于线性模型
            if model_name == 'linear_basic':
                X = r_data.reshape(-1, 1)
                reg = LinearRegression()
                reg.fit(X, v_data)
                popt = [reg.coef_[0], reg.intercept_]
            else:
                return None
        
        # 评估拟合结果
        v_pred = model_func(r_data, *popt)
        if np.any(~np.isfinite(v_pred)): return None
        
        criteria = calculate_model_criteria(v_data, v_pred, n_params, len(v_data))
        if criteria['r2'] < -10: return None # 过滤掉极差的拟合
        
        return {
            'model_name': model_name,
            'model_desc': model_info['description'],
            'model_category': model_info['category'],
            'strategy': strategy['name'],
            'params': popt,
            'param_names': model_info['params'],
            'model_func': model_func,
            **criteria
        }
    except Exception:
        return None

def run_fitting(models, strategies, r_data, v_data):
    """执行拟合，自动选择并行或串行"""
    fitting_tasks = []
    linear_strategies = get_linear_fitting_strategies()
    
    for model_name, model_info in models.items():
        # 为所有模型添加通用策略
        for strategy in strategies:
            fitting_tasks.append((model_name, model_info, strategy, r_data, v_data, models))
        
        # 只为线性模型添加专用线性拟合策略
        if model_name.startswith('linear_'):
            for strategy in linear_strategies:
                fitting_tasks.append((model_name, model_info, strategy, r_data, v_data, models))
    
    print(f"\n准备执行 {len(fitting_tasks)} 个拟合任务...")
    
    if ENABLE_PARALLEL and len(fitting_tasks) > 1:
        print(f"启动并行拟合，使用 {MAX_WORKERS} 个进程...")
        start_time = time.time()
        try:
            with mp.Pool(processes=MAX_WORKERS) as pool:
                results = pool.map(fit_single_model_strategy, fitting_tasks)
            results = [r for r in results if r is not None]
            print(f"并行拟合完成，耗时 {time.time() - start_time:.2f} 秒。成功 {len(results)}/{len(fitting_tasks)} 个任务。")
        except Exception as e:
            print(f"并行计算失败: {e}。回退到串行模式。")
            results = run_serial_fitting(fitting_tasks)
    else:
        results = run_serial_fitting(fitting_tasks)
        
    return results

def run_serial_fitting(tasks):
    """串行执行拟合任务"""
    print("使用串行模式进行拟合...")
    start_time = time.time()
    results = [fit_single_model_strategy(task) for task in tasks]
    results = [r for r in results if r is not None]
    print(f"串行拟合完成，耗时 {time.time() - start_time:.2f} 秒。")
    return results

# --- 输出与可视化模块 ---
def generate_formula_string(model_name, params):
    """生成具体的拟合公式字符串"""
    p = params
    if model_name == 'linear_basic' or model_name == 'motor_linear':
        return f"v = {p[0]:.4f} * r + {p[1]:.4f}"
    elif model_name == 'polynomial_2':
        return f"v = {p[0]:.4f}*r² + {p[1]:.4f}*r + {p[2]:.4f}"
    elif model_name == 'polynomial_3':
        return f"v = {p[0]:.4f}*r³ + {p[1]:.4f}*r² + {p[2]:.4f}*r + {p[3]:.4f}"
    elif model_name == 'power_2param':
        return f"v = {p[0]:.4f} * r^({p[1]:.4f})"
    elif model_name == 'power_3param':
        return f"v = {p[0]:.4f} * r^({p[1]:.4f}) + {p[2]:.4f}"
    elif model_name == 'ballistic_ideal':
        cos_theta = np.cos(LAUNCH_ANGLE_RAD)
        return f"v = {p[0]:.4f} * √(r² + {DELTA_H:.2f}²) / {cos_theta:.3f}"
    elif model_name == 'ballistic_drag_linear':
        return f"v = {p[0]:.4f} * √(r² + {DELTA_H:.2f}²) + {p[1]:.4f}"
    elif model_name == 'ballistic_drag_quadratic':
        return f"v = {p[0]:.4f}*(r² + {DELTA_H:.2f}²) + {p[1]:.4f}*√(r² + {DELTA_H:.2f}²) + {p[2]:.4f}"
    elif model_name == 'ballistic_energy':
        return f"v = √(2*{p[0]:.4f}*(r² + {DELTA_H:.2f}²)*{p[1]:.4f})"
    elif model_name == 'motor_pwm':
        return f"v = {p[0]:.4f}*(1-exp(-{p[1]:.4f}*r)) + {p[2]:.4f}"
    elif model_name == 'motor_torque':
        return f"v = {p[0]:.4f}*√(r/{p[1]:.4f}) + {p[2]:.4f}"
    elif model_name == 'empirical_sqrt':
        return f"v = {p[0]:.4f}*√(r+{p[1]:.4f}) + {p[2]:.4f}"
    elif model_name == 'exponential':
        return f"v = {p[0]:.4f}*exp({p[1]:.4f}*r) + {p[2]:.4f}"
    elif model_name == 'logarithmic':
        return f"v = {p[0]:.4f}*ln({p[1]:.4f}*r + 1) + {p[2]:.4f}"
    elif model_name == 'rational':
        return f"v = ({p[0]:.4f}*r + {p[1]:.4f})/({p[2]:.4f}*r + {p[3]:.4f})"
    return "未知模型"

def print_model_ranking(results):
    """输出模型排行榜"""
    if not results:
        print("没有成功拟合的模型结果。")
        return None
    
    best_per_model = {}
    for res in results:
        name = res['model_name']
        if name not in best_per_model or res['aic'] < best_per_model[name]['aic']:
            best_per_model[name] = res
            
    sorted_results = sorted(best_per_model.values(), key=lambda x: x['aic'])
    
    print("\n" + "="*120)
    print("=== 模型排行榜 (综合AIC表现, AIC越小越好) ===")
    print("="*120)
    for i, res in enumerate(sorted_results):
        formula = generate_formula_string(res['model_name'], res['params'])
        print(f"第 {i+1:2d} 名: {res['model_desc']} [{res['model_category']}]")
        print(f"        最佳策略: {res['strategy']}")
        print(f"        拟合公式: {formula}")
        print(f"        评价指标: AIC={res['aic']:.2f} | R²={res['r2']:.4f} | Adj R²={res['adj_r2']:.4f} | MSE={res['mse']:.6f}")
        print("-" * 120)
    
    return sorted_results

def create_visualization(r_data, v_data, outlier_indices, r_orig_proc, v_orig_proc, best_model_info, all_results):
    """创建可视化图表"""
    if not best_model_info:
        print("无最佳模型信息，无法绘图。")
        return

    plt.figure(figsize=(14, 10))
    
    # 子图1: 主要拟合结果
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(r_data, v_data, label='拟合数据点', color='blue', marker='o', s=60, alpha=0.8, zorder=5)
    
    if len(outlier_indices) > 0:
        outlier_r = r_orig_proc[outlier_indices]
        outlier_v = v_orig_proc[outlier_indices]
        ax1.scatter(outlier_r, outlier_v, label='识别的异常值', color='red', marker='x', s=100, zorder=6)

    # 拟合曲线和置信区间
    r_plot = np.linspace(min(r_data), max(r_data), 400)
    v_fit_plot = best_model_info['model_func'](r_plot, *best_model_info['params'])
    ax1.plot(r_plot, v_fit_plot, label=f"最佳拟合曲线 (AIC={best_model_info['aic']:.1f})", color='red', lw=2.5)

    # 计算98%置信区间
    try:
        # 使用bootstrap方法计算置信区间
        n_bootstrap = 200
        v_bootstrap = []
        
        print("正在计算98%置信区间...")
        
        for i in range(n_bootstrap):
            # 随机重采样
            indices = np.random.choice(len(r_data), len(r_data), replace=True)
            r_boot = r_data[indices]
            v_boot = v_data[indices]
            
            try:
                # 重新拟合模型
                p0 = get_robust_initial_guess(best_model_info['model_name'], r_boot, v_boot, get_models())
                bounds = get_parameter_bounds(best_model_info['model_name'], get_models())
                
                # 尝试拟合
                if len(bounds[0]) > 1:  # 有边界约束的模型
                    popt_boot, _ = curve_fit(best_model_info['model_func'], r_boot, v_boot, 
                                           p0=p0, bounds=bounds, method='trf', maxfev=5000)
                else:
                    popt_boot, _ = curve_fit(best_model_info['model_func'], r_boot, v_boot, 
                                           p0=p0, maxfev=5000)
                
                # 计算预测值
                v_boot_pred = best_model_info['model_func'](r_plot, *popt_boot)
                
                # 检查预测值是否有效
                if np.all(np.isfinite(v_boot_pred)):
                    v_bootstrap.append(v_boot_pred)
                    
            except Exception:
                continue
        
        # 如果有足够的bootstrap样本，计算置信区间
        if len(v_bootstrap) > 20:
            v_bootstrap = np.array(v_bootstrap)
            # 98%置信区间对应1%和99%分位数
            v_lower = np.percentile(v_bootstrap, 1, axis=0)
            v_upper = np.percentile(v_bootstrap, 99, axis=0)
            
            # 绘制置信区间
            ax1.fill_between(r_plot, v_lower, v_upper, 
                           alpha=0.2, color='red', 
                           label=f'98%置信区间 (基于{len(v_bootstrap)}次bootstrap)')
            print(f"成功计算置信区间，使用了{len(v_bootstrap)}次有效bootstrap样本")
        else:
            print(f"bootstrap样本数量不足({len(v_bootstrap)})，无法计算可靠的置信区间")
            
    except Exception as e:
        print(f"计算置信区间时出错: {e}")
        
        # 备用方案：使用残差标准差估计置信区间
        try:
            v_pred_data = best_model_info['model_func'](r_data, *best_model_info['params'])
            residuals_std = np.std(v_data - v_pred_data)
            
            # 98%置信区间约为2.576个标准差
            confidence_multiplier = 2.576
            v_lower_simple = v_fit_plot - confidence_multiplier * residuals_std
            v_upper_simple = v_fit_plot + confidence_multiplier * residuals_std
            
            ax1.fill_between(r_plot, v_lower_simple, v_upper_simple, 
                           alpha=0.15, color='orange', 
                           label='98%置信区间 (基于残差标准差)')
            print("使用残差标准差方法计算置信区间")
        except Exception as e2:
            print(f"备用置信区间计算也失败: {e2}")

    formula_str = generate_formula_string(best_model_info['model_name'], best_model_info['params'])
    info_text = (f"最佳模型: {best_model_info['model_desc']}\n"
                 f"类别: {best_model_info['model_category']}\n"
                 f"公式: {formula_str}\n"
                 f"R² = {best_model_info['r2']:.4f}, AIC = {best_model_info['aic']:.1f}")
    ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=9, va='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
    
    ax1.set_xlabel('水平射程 r (m)')
    ax1.set_ylabel('出射速度 v (m/s)')
    ax1.set_title('r-v 关系拟合结果 (含98%置信区间)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.5)

    # 子图2: 残差图
    ax2 = plt.subplot(2, 2, 2)
    v_pred_on_data = best_model_info['model_func'](r_data, *best_model_info['params'])
    residuals = v_data - v_pred_on_data
    ax2.scatter(r_data, residuals, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--')
    
    # 添加残差的98%置信带
    residual_std = np.std(residuals)
    confidence_multiplier = 2.576  # 98%置信区间
    ax2.axhline(y=confidence_multiplier * residual_std, color='orange', linestyle=':', alpha=0.7, label='+98%置信线')
    ax2.axhline(y=-confidence_multiplier * residual_std, color='orange', linestyle=':', alpha=0.7, label='-98%置信线')
    
    ax2.set_xlabel('水平射程 r (m)')
    ax2.set_ylabel('残差 (真实值 - 预测值)')
    ax2.set_title('最佳模型残差图')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # 子图3: 残差Q-Q图
    ax3 = plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    ax3.set_title('残差Q-Q图 (检验正态性)')
    ax3.get_lines()[0].set_markerfacecolor('blue')
    ax3.get_lines()[0].set_markeredgecolor('blue')
    ax3.get_lines()[1].set_color('red')
    ax3.grid(True, alpha=0.5)
    
    # 添加正态性检验信息
    try:
        from scipy.stats import shapiro, normaltest
        # Shapiro-Wilk检验
        shapiro_stat, shapiro_p = shapiro(residuals)
        # D'Agostino正态性检验
        dagostino_stat, dagostino_p = normaltest(residuals)
        
        normality_text = f"Shapiro-Wilk: p={shapiro_p:.4f}\nD'Agostino: p={dagostino_p:.4f}"
        ax3.text(0.05, 0.95, normality_text, transform=ax3.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))
    except:
        pass

    # 子图4: 模型比较
    ax4 = plt.subplot(2, 2, 4)
    sorted_results = print_model_ranking(all_results)
    if sorted_results:
        top_results = sorted_results[:7] # 最多显示前7名
        model_names = [res['model_desc'].split(':')[0] for res in top_results]
        aic_values = [res['aic'] for res in top_results]
        bars = ax4.barh(model_names, aic_values, color='skyblue')
        bars[0].set_color('salmon') # 高亮最佳模型
        ax4.set_xlabel('AIC 值 (越小越好)')
        ax4.set_title('模型性能排行榜')
        ax4.invert_yaxis() # 让最好的模型在顶部
        ax4.grid(True, axis='x', alpha=0.5)

    plt.tight_layout()
    plt.show()


# --- 主程序 ---
def main():
    # 只在主进程中打印初始化信息
    print(f"系统CPU核心数: {mp.cpu_count()}, 将使用 {MAX_WORKERS} 个进程进行并行拟合")
    print(f"物理参数配置: 发射高度={LAUNCH_HEIGHT}m, 目标高度={BASKET_HEIGHT}m, 发射角度={LAUNCH_ANGLE_DEG}°, 重力={GRAVITY}m/s²")
    
    # 1. 数据加载与预处理
    r_data_raw, v_data_raw = load_data()
    r_processed, v_processed = preprocess_data(r_data_raw, v_data_raw)
    
    # 2. 异常值处理
    r_for_fit, v_for_fit, outlier_indices = clean_outliers(r_processed, v_processed)
    
    if len(r_for_fit) <= 2:
        print("\n错误：有效数据点不足，无法进行拟合。请检查数据。")
        return
    
    # 3. 获取模型和策略并执行拟合
    models = get_models()
    strategies = get_fitting_strategies()
    all_results = run_fitting(models, strategies, r_for_fit, v_for_fit)

    if not all_results:
        print("\n错误：所有模型都未能成功拟合。请检查数据或模型定义。")
        return
        
    # 4. 分析结果，选出最佳模型
    best_model_info = min(all_results, key=lambda x: x['aic'])
    
    # 5. 输出最佳模型信息
    print("\n" + "="*60)
    print("🏆 最佳拟合模型 🏆")
    print("="*60)
    print(f"模型类别: {best_model_info['model_category']}")
    print(f"模型描述: {best_model_info['model_desc']}")
    print(f"最佳策略: {best_model_info['strategy']}")
    formula_str = generate_formula_string(best_model_info['model_name'], best_model_info['params'])
    print(f"拟合公式: {formula_str}")
    print(f"模型参数:")
    for name, val in zip(best_model_info['param_names'], best_model_info['params']):
        print(f"  {name} = {val:.6f}")
    print(f"AIC = {best_model_info['aic']:.2f}, R² = {best_model_info['r2']:.4f}")
    print("="*60)

    # 6. 可视化
    create_visualization(r_for_fit, v_for_fit, outlier_indices, r_processed, v_processed, best_model_info, all_results)

    # 7. C语言数组输出
    print("\n--- 原始数据的C语言数组格式 ---")
    r_c_array = ", ".join([f"{val:.3f}f" for val in r_data_raw])
    v_c_array = ", ".join([f"{val:.3f}f" for val in v_data_raw])
    print(f"const float raw_r_data[] = {{ {r_c_array} }};")
    print(f"const float raw_v_data[] = {{ {v_c_array} }};")
    print(f"const int RAW_DATA_SIZE = {len(r_data_raw)};")


if __name__ == '__main__':
    main()