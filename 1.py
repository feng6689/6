import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 增强版水印去除算法 ====================

def remove_watermark_advanced(image_path):
    """
    高级水印去除算法：
    1. 多尺度分析
    2. 频域滤波（检测周期性水印）
    3. 自适应背景建模
    4. 增强形态学操作
    5. 多级图像修复
    6. 背景颜色一致性处理
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = img.shape[:2]
    print(f"图片尺寸: {width}x{height}")
    
    # 转换为不同颜色空间
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 保存原始图片用于后续修复
    original_img = img.copy()
    
    # ==================== 第一步：检测几何图形区域 ====================
    print("\n[1/8] 检测几何图形区域...")
    shape_mask = detect_geometric_shapes(img, hsv)
    cv2.imwrite('debug_step1_shape_mask.jpg', shape_mask)
    
    # ==================== 第二步：估计背景颜色 ====================
    print("\n[2/8] 估计背景颜色...")
    background_color, background_mask = estimate_background(img, gray, shape_mask)
    print(f"估计的背景颜色 (BGR): {background_color}")
    cv2.imwrite('debug_step2_background_mask.jpg', background_mask)
    
    # ==================== 第三步：多尺度颜色差异检测 ====================
    print("\n[3/8] 多尺度颜色差异检测...")
    multi_scale_mask = multi_scale_color_detection(img, background_color, shape_mask)
    cv2.imwrite('debug_step3_multi_scale_mask.jpg', multi_scale_mask)
    
    # ==================== 第四步：频域分析（检测周期性水印） ====================
    print("\n[4/8] 频域分析检测周期性水印...")
    frequency_mask = frequency_domain_analysis(gray, shape_mask)
    cv2.imwrite('debug_step4_frequency_mask.jpg', frequency_mask)
    
    # ==================== 第五步：纹理和边缘分析 ====================
    print("\n[5/8] 纹理和边缘分析...")
    texture_mask = texture_edge_analysis(gray, shape_mask)
    cv2.imwrite('debug_step5_texture_mask.jpg', texture_mask)
    
    # ==================== 第六步：LAB颜色空间分析 ====================
    print("\n[6/8] LAB颜色空间分析...")
    lab_mask = lab_color_analysis(lab, background_color, shape_mask)
    cv2.imwrite('debug_step6_lab_mask.jpg', lab_mask)
    
    # ==================== 第七步：合并所有检测结果 ====================
    print("\n[7/8] 合并所有检测结果...")
    combined_mask = combine_masks([
        multi_scale_mask,
        frequency_mask,
        texture_mask,
        lab_mask
    ])
    
    # 增强掩码
    combined_mask = enhance_mask(combined_mask)
    cv2.imwrite('debug_step7_combined_mask.jpg', combined_mask)
    
    # ==================== 第八步：多级图像修复 ====================
    print("\n[8/8] 多级图像修复...")
    result = multi_stage_inpainting(original_img, combined_mask, shape_mask)
    
    # ==================== 背景颜色一致性处理 ====================
    print("\n[额外步骤] 背景颜色一致性处理...")
    result = ensure_background_consistency(result, shape_mask, background_color)
    
    # 保存最终结果
    cv2.imwrite('1_processed.jpg', result)
    print("\n已保存处理后的图片: 1_processed.jpg")
    
    return result, shape_mask, background_color

def detect_geometric_shapes(img, hsv):
    """
    检测几何图形（红、绿、蓝）
    """
    # 红色范围（两个区间）
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    
    # 绿色范围
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # 蓝色范围
    lower_blue = np.array([95, 40, 40])
    upper_blue = np.array([145, 255, 255])
    
    # 创建掩码
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 合并
    shape_mask = cv2.bitwise_or(red_mask, green_mask)
    shape_mask = cv2.bitwise_or(shape_mask, blue_mask)
    
    # 膨胀以覆盖边缘
    kernel = np.ones((20, 20), np.uint8)
    shape_mask = cv2.dilate(shape_mask, kernel, iterations=2)
    
    return shape_mask

def estimate_background(img, gray, shape_mask):
    """
    估计背景颜色
    """
    # 创建非几何图形区域的掩码
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 如果非几何图形区域太小，使用图像的中心区域
    if cv2.countNonZero(non_shape_mask) < (img.shape[0] * img.shape[1] * 0.2):
        h, w = img.shape[:2]
        center_region = img[h//4:3*h//4, w//4:3*w//4]
        background_color = np.mean(center_region, axis=(0, 1)).astype(int)
        
        # 创建中心区域掩码
        center_mask = np.zeros(gray.shape, dtype=np.uint8)
        center_mask[h//4:3*h//4, w//4:3*w//4] = 255
        background_mask = cv2.bitwise_and(center_mask, non_shape_mask)
    else:
        # 计算非几何图形区域的平均颜色
        background_color = cv2.mean(img, mask=non_shape_mask)[:3]
        background_color = np.array(background_color).astype(int)
        background_mask = non_shape_mask
    
    return background_color, background_mask

def multi_scale_color_detection(img, background_color, shape_mask):
    """
    多尺度颜色差异检测
    """
    height, width = img.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 在多个尺度上检测
    scales = [1.0, 0.8, 0.6, 0.4]
    
    for scale in scales:
        # 缩放图像
        new_size = (int(width * scale), int(height * scale))
        img_scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        shape_mask_scaled = cv2.resize(shape_mask, new_size, interpolation=cv2.INTER_NEAREST)
        
        # 计算颜色差异
        color_diff = np.sqrt(np.sum((img_scaled - background_color) ** 2, axis=2))
        
        # 归一化
        color_diff_normalized = ((color_diff - color_diff.min()) / 
                                (color_diff.max() - color_diff.min() + 1e-10) * 255).astype(np.uint8)
        
        # 多阈值处理
        thresholds = [10, 15, 20, 25]
        for thresh in thresholds:
            _, diff_mask = cv2.threshold(color_diff_normalized, thresh, 255, cv2.THRESH_BINARY)
            
            # 排除几何图形区域
            diff_mask = cv2.bitwise_and(diff_mask, cv2.bitwise_not(shape_mask_scaled))
            
            # 缩放回原始大小
            diff_mask_original = cv2.resize(diff_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # 合并
            combined_mask = cv2.bitwise_or(combined_mask, diff_mask_original)
    
    # 形态学操作增强
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    return combined_mask

def frequency_domain_analysis(gray, shape_mask):
    """
    频域分析 - 检测周期性水印
    """
    height, width = gray.shape[:2]
    
    # 创建非几何图形区域的掩码
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 应用掩码
    masked_gray = cv2.bitwise_and(gray, gray, mask=non_shape_mask)
    
    # 傅里叶变换
    f = np.fft.fft2(masked_gray)
    fshift = np.fft.fftshift(f)
    
    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 归一化
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 检测高频区域（可能是水印）
    # 创建中心掩码（排除直流分量和低频）
    center_radius = min(height, width) // 10
    cy, cx = height // 2, width // 2
    
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # 高频区域掩码
    high_freq_mask = (dist_from_center > center_radius).astype(np.uint8) * 255
    
    # 检测幅度谱中的亮点
    _, bright_spots = cv2.threshold(magnitude_normalized, 180, 255, cv2.THRESH_BINARY)
    bright_spots = cv2.bitwise_and(bright_spots, high_freq_mask)
    
    # 形态学操作增强
    kernel = np.ones((5, 5), np.uint8)
    bright_spots = cv2.dilate(bright_spots, kernel, iterations=2)
    
    # 如果检测到高频亮点，尝试在空间域中找到对应区域
    # 这里我们简化处理，直接使用边缘检测作为补充
    frequency_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 使用不同的边缘检测方法
    edges_canny = cv2.Canny(gray, 30, 100)
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_sobel = cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges_sobel = cv2.threshold(edges_sobel, 30, 255, cv2.THRESH_BINARY)
    
    # 拉普拉斯边缘检测
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges_laplacian = np.absolute(edges_laplacian)
    edges_laplacian = cv2.normalize(edges_laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges_laplacian = cv2.threshold(edges_laplacian, 20, 255, cv2.THRESH_BINARY)
    
    # 合并所有边缘检测结果
    frequency_mask = cv2.bitwise_or(edges_canny, edges_sobel)
    frequency_mask = cv2.bitwise_or(frequency_mask, edges_laplacian)
    
    # 排除几何图形区域
    frequency_mask = cv2.bitwise_and(frequency_mask, non_shape_mask)
    
    # 形态学操作增强文本状区域
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    
    # 水平文本增强
    horizontal_edges = cv2.morphologyEx(frequency_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    horizontal_edges = cv2.morphologyEx(horizontal_edges, cv2.MORPH_OPEN, horizontal_kernel)
    
    # 垂直文本增强
    vertical_edges = cv2.morphologyEx(frequency_mask, cv2.MORPH_CLOSE, vertical_kernel)
    vertical_edges = cv2.morphologyEx(vertical_edges, cv2.MORPH_OPEN, vertical_kernel)
    
    # 合并
    frequency_mask = cv2.bitwise_or(horizontal_edges, vertical_edges)
    
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    frequency_mask = cv2.dilate(frequency_mask, kernel, iterations=2)
    
    return frequency_mask

def texture_edge_analysis(gray, shape_mask):
    """
    纹理和边缘分析
    """
    height, width = gray.shape[:2]
    texture_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建非几何图形区域的掩码
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 方法1：局部标准差（纹理检测）
    window_sizes = [5, 7, 9]
    for window_size in window_sizes:
        # 计算局部均值
        local_mean = cv2.blur(gray, (window_size, window_size))
        
        # 计算局部方差
        local_sq_mean = cv2.blur(gray.astype(np.float32)**2, (window_size, window_size))
        local_var = local_sq_mean - local_mean.astype(np.float32)**2
        
        # 计算标准差
        local_std = np.sqrt(local_var)
        
        # 归一化
        local_std_normalized = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 阈值处理
        _, std_mask = cv2.threshold(local_std_normalized, 15, 255, cv2.THRESH_BINARY)
        
        # 排除几何图形区域
        std_mask = cv2.bitwise_and(std_mask, non_shape_mask)
        
        # 合并
        texture_mask = cv2.bitwise_or(texture_mask, std_mask)
    
    # 方法2：自适应阈值
    adaptive_methods = [
        (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
        (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 3),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 3),
    ]
    
    for method, block_size, C in adaptive_methods:
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, method, cv2.THRESH_BINARY_INV, block_size, C
        )
        
        # 排除几何图形区域
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, non_shape_mask)
        
        # 合并
        texture_mask = cv2.bitwise_or(texture_mask, adaptive_thresh)
    
    # 方法3：多尺度边缘检测
    for sigma in [1.0, 1.5, 2.0]:
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 20, 80)
        
        # 排除几何图形区域
        edges = cv2.bitwise_and(edges, non_shape_mask)
        
        # 合并
        texture_mask = cv2.bitwise_or(texture_mask, edges)
    
    # 形态学操作增强
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
    
    # 水平文本
    horizontal_texture = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    horizontal_texture = cv2.morphologyEx(horizontal_texture, cv2.MORPH_OPEN, horizontal_kernel)
    
    # 垂直文本
    vertical_texture = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, vertical_kernel)
    vertical_texture = cv2.morphologyEx(vertical_texture, cv2.MORPH_OPEN, vertical_kernel)
    
    # 合并
    texture_mask = cv2.bitwise_or(horizontal_texture, vertical_texture)
    
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    texture_mask = cv2.dilate(texture_mask, kernel, iterations=2)
    
    return texture_mask

def lab_color_analysis(lab, background_color, shape_mask):
    """
    LAB颜色空间分析
    """
    height, width = lab.shape[:2]
    
    # 转换背景颜色到LAB空间
    background_bgr = np.uint8([[background_color]])
    background_lab = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2LAB)[0][0]
    
    # 分离LAB通道
    L, A, B = cv2.split(lab)
    
    # 创建非几何图形区域的掩码
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 计算每个通道与背景的差异
    l_diff = np.abs(L.astype(np.float32) - background_lab[0])
    a_diff = np.abs(A.astype(np.float32) - background_lab[1])
    b_diff = np.abs(B.astype(np.float32) - background_lab[2])
    
    # 合并差异
    total_diff = l_diff + a_diff * 1.5 + b_diff * 1.5  # 给颜色通道更高权重
    
    # 归一化
    total_diff_normalized = cv2.normalize(total_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 多阈值处理
    lab_mask = np.zeros((height, width), dtype=np.uint8)
    thresholds = [8, 12, 16, 20]
    
    for thresh in thresholds:
        _, diff_mask = cv2.threshold(total_diff_normalized, thresh, 255, cv2.THRESH_BINARY)
        
        # 排除几何图形区域
        diff_mask = cv2.bitwise_and(diff_mask, non_shape_mask)
        
        # 合并
        lab_mask = cv2.bitwise_or(lab_mask, diff_mask)
    
    # 形态学操作
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    
    # 增强文本状区域
    lab_horizontal = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    lab_horizontal = cv2.morphologyEx(lab_horizontal, cv2.MORPH_OPEN, horizontal_kernel)
    
    lab_vertical = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE, vertical_kernel)
    lab_vertical = cv2.morphologyEx(lab_vertical, cv2.MORPH_OPEN, vertical_kernel)
    
    lab_mask = cv2.bitwise_or(lab_horizontal, lab_vertical)
    
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    lab_mask = cv2.dilate(lab_mask, kernel, iterations=2)
    
    return lab_mask

def combine_masks(masks):
    """
    合并多个掩码
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = cv2.bitwise_or(combined, mask)
    
    return combined

def enhance_mask(mask):
    """
    增强掩码
    """
    # 形态学闭运算（连接断开的区域）
    kernel = np.ones((7, 7), np.uint8)
    enhanced = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    enhanced = cv2.dilate(enhanced, kernel, iterations=2)
    
    # 移除小的噪点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enhanced, connectivity=8)
    
    # 过滤掉小的连通区域
    min_area = 50
    filtered = np.zeros_like(enhanced)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    
    # 再次膨胀
    kernel = np.ones((7, 7), np.uint8)
    filtered = cv2.dilate(filtered, kernel, iterations=1)
    
    return filtered

def multi_stage_inpainting(img, mask, shape_mask):
    """
    多级图像修复
    """
    result = img.copy()
    
    # 检查掩码是否有内容
    if cv2.countNonZero(mask) == 0:
        print("警告：未检测到水印区域，使用备用方法...")
        # 使用更激进的方法检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        non_shape_mask = cv2.bitwise_not(shape_mask)
        
        # 自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, non_shape_mask)
        
        # 形态学操作
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, horizontal_kernel)
        adaptive_thresh = cv2.dilate(adaptive_thresh, np.ones((7, 7), np.uint8), iterations=2)
        
        mask = adaptive_thresh
        cv2.imwrite('debug_fallback_mask.jpg', mask)
    
    # 第一阶段：使用Telea算法
    print("  - 第一阶段：Telea算法修复...")
    radius = 7
    result = cv2.inpaint(result, mask, radius, cv2.INPAINT_TELEA)
    cv2.imwrite('debug_inpaint_stage1.jpg', result)
    
    # 第二阶段：使用NS算法（如果可用）
    try:
        print("  - 第二阶段：NS算法修复...")
        result = cv2.inpaint(result, mask, radius, cv2.INPAINT_NS)
        cv2.imwrite('debug_inpaint_stage2.jpg', result)
    except:
        print("    NS算法不可用，跳过")
    
    # 第三阶段：对非几何图形区域进行平滑处理
    print("  - 第三阶段：背景平滑处理...")
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 膨胀非几何图形掩码，确保覆盖边缘
    wide_shape_mask = cv2.dilate(shape_mask, np.ones((25, 25), np.uint8), iterations=2)
    wide_non_shape_mask = cv2.bitwise_not(wide_shape_mask)
    
    # 应用多尺度平滑
    for sigma in [3, 5, 7]:
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)
        result = np.where(wide_non_shape_mask[:, :, np.newaxis] > 0, blurred, result)
    
    # 中值滤波
    median_blurred = cv2.medianBlur(result, 7)
    result = np.where(wide_non_shape_mask[:, :, np.newaxis] > 0, median_blurred, result)
    
    cv2.imwrite('debug_inpaint_stage3.jpg', result)
    
    return result

def ensure_background_consistency(img, shape_mask, target_color):
    """
    确保背景颜色一致
    """
    height, width = img.shape[:2]
    
    # 膨胀几何图形掩码
    wide_shape_mask = cv2.dilate(shape_mask, np.ones((30, 30), np.uint8), iterations=2)
    wide_non_shape_mask = cv2.bitwise_not(wide_shape_mask)
    
    # 计算当前背景颜色
    if cv2.countNonZero(wide_non_shape_mask) > 0:
        current_background = cv2.mean(img, mask=wide_non_shape_mask)[:3]
        current_background = np.array(current_background).astype(int)
    else:
        current_background = target_color
    
    # 计算颜色调整量
    color_adjustment = target_color.astype(np.float32) - current_background.astype(np.float32)
    
    # 应用颜色调整（只调整非几何图形区域）
    result = img.copy().astype(np.float32)
    
    # 对非几何图形区域应用颜色调整
    for c in range(3):
        result[:, :, c] = np.where(
            wide_non_shape_mask > 0,
            np.clip(result[:, :, c] + color_adjustment[c], 0, 255),
            result[:, :, c]
        )
    
    result = result.astype(np.uint8)
    
    # 额外的平滑处理
    # 对远离几何图形的区域应用更强的平滑
    # 创建距离变换
    dist_transform = cv2.distanceTransform(wide_non_shape_mask, cv2.DIST_L2, 3)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    
    # 多尺度模糊
    for sigma in [5, 9, 13]:
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)
        
        # 只对距离较远的区域应用更强的模糊
        weight = (dist_transform > 0.3).astype(np.float32)
        weight = cv2.GaussianBlur(weight, (15, 15), 0)
        
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - weight) + blurred[:, :, c] * weight
    
    result = result.astype(np.uint8)
    
    return result

# ==================== 形状检测函数 ====================

def detect_shapes(image, shape_mask, background_color):
    """
    检测图片中的几何图形（三角形、矩形、圆形），并统计它们的数量和颜色
    """
    img_with_contours = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edged = cv2.Canny(blurred, 50, 150)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化计数器
    triangle_count = 0
    rectangle_count = 0
    circle_count = 0
    red_count = 0
    green_count = 0
    blue_count = 0
    
    shapes_info = []
    
    # 遍历每个轮廓
    for contour in contours:
        # 过滤小轮廓
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        # 近似轮廓
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        x, y, w, h = cv2.boundingRect(approx)
        
        # 提取轮廓区域的颜色
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        mean_color = cv2.mean(image, mask=mask)[:3]
        mean_color_bgr = np.array(mean_color)
        
        # 判断颜色
        roi = image[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_mask = mask[y:y+h, x:x+w]
        
        if cv2.countNonZero(roi_mask) > 0:
            mean_hsv = cv2.mean(roi_hsv, mask=roi_mask)[:3]
        else:
            mean_hsv = [0, 0, 0]
        
        h_val, s, v = mean_hsv
        
        color_name = "未知"
        color_bgr = (0, 0, 0)
        
        # 红色
        if (0 <= h_val <= 15 or 165 <= h_val <= 180) and s > 40 and v > 40:
            color_name = "红色"
            color_bgr = (0, 0, 255)
            red_count += 1
        # 绿色
        elif 35 <= h_val <= 85 and s > 40 and v > 40:
            color_name = "绿色"
            color_bgr = (0, 255, 0)
            green_count += 1
        # 蓝色
        elif 95 <= h_val <= 145 and s > 40 and v > 40:
            color_name = "蓝色"
            color_bgr = (255, 0, 0)
            blue_count += 1
        else:
            # 备用方法
            b, g, r = mean_color_bgr
            if r > g and r > b and r > 100:
                color_name = "红色"
                color_bgr = (0, 0, 255)
                red_count += 1
            elif g > r and g > b and g > 100:
                color_name = "绿色"
                color_bgr = (0, 255, 0)
                green_count += 1
            elif b > r and b > g and b > 100:
                color_name = "蓝色"
                color_bgr = (255, 0, 0)
                blue_count += 1
        
        # 判断形状
        shape_name = "未知"
        
        # 三角形
        if len(approx) == 3:
            shape_name = "三角形"
            triangle_count += 1
            cv2.drawContours(img_with_contours, [approx], -1, color_bgr, 2)
            cv2.putText(img_with_contours, f"{color_name}{shape_name}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            shapes_info.append({"shape": "三角形", "color": color_name})
        
        # 矩形
        elif len(approx) == 4:
            shape_name = "矩形"
            rectangle_count += 1
            cv2.drawContours(img_with_contours, [approx], -1, color_bgr, 2)
            cv2.putText(img_with_contours, f"{color_name}{shape_name}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            shapes_info.append({"shape": "矩形", "color": color_name})
        
        # 圆形
        else:
            if peri > 0:
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity > 0.6:
                    shape_name = "圆形"
                    circle_count += 1
                    cv2.drawContours(img_with_contours, [approx], -1, color_bgr, 2)
                    
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = x + w//2, y + h//2
                    
                    cv2.putText(img_with_contours, f"{color_name}{shape_name}", (cX-50, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                    shapes_info.append({"shape": "圆形", "color": color_name})
    
    # 保存带有轮廓的图片
    cv2.imwrite('1_with_shapes.jpg', img_with_contours)
    print("已检测形状并保存为1_with_shapes.jpg")
    
    # 打印统计结果
    print(f"\n形状统计:")
    print(f"三角形数量: {triangle_count}")
    print(f"矩形数量: {rectangle_count}")
    print(f"圆形数量: {circle_count}")
    
    print(f"\n颜色统计:")
    print(f"红色图形数量: {red_count}")
    print(f"绿色图形数量: {green_count}")
    print(f"蓝色图形数量: {blue_count}")
    
    return {
        "triangle": triangle_count,
        "rectangle": rectangle_count,
        "circle": circle_count,
        "red": red_count,
        "green": green_count,
        "blue": blue_count,
        "shapes_info": shapes_info
    }

# ==================== 图表生成函数 ====================

def create_shape_histogram(counts, output_path='2.jpg'):
    """
    生成形状数量的直方图
    """
    shapes = ['三角形', '矩形', '圆形']
    counts_list = [counts['triangle'], counts['rectangle'], counts['circle']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(shapes, counts_list, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_title('几何图形数量统计', fontsize=16, fontweight='bold')
    ax.set_xlabel('图形类型', fontsize=14)
    ax.set_ylabel('数量', fontsize=14)
    
    max_count = max(counts_list)
    ax.set_ylim(0, max_count + 2 if max_count > 0 else 5)
    
    for bar, count in zip(bars, counts_list):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成形状数量直方图并保存为{output_path}")

def create_color_pie_chart(counts, output_path='3.jpg'):
    """
    生成颜色占比的饼状图
    """
    colors = ['红色', '绿色', '蓝色']
    counts_list = [counts['red'], counts['green'], counts['blue']]
    
    filtered_colors = []
    filtered_counts = []
    pie_colors = []
    
    for color, count in zip(colors, counts_list):
        if count > 0:
            filtered_colors.append(color)
            filtered_counts.append(count)
            if color == '红色':
                pie_colors.append('#FF0000')
            elif color == '绿色':
                pie_colors.append('#00FF00')
            elif color == '蓝色':
                pie_colors.append('#0000FF')
    
    if not filtered_colors:
        filtered_colors = ['无数据']
        filtered_counts = [1]
        pie_colors = ['#CCCCCC']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    wedges, texts, autotexts = ax.pie(
        filtered_counts,
        labels=filtered_colors,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        explode=[0.05] * len(filtered_colors)
    )
    
    ax.set_title('几何图形颜色占比统计', fontsize=16, fontweight='bold', pad=20)
    ax.axis('equal')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成颜色占比饼状图并保存为{output_path}")

# ==================== 主函数 ====================

def main():
    print("=" * 70)
    print("高级水印去除与几何图形分析系统")
    print("=" * 70)
    
    # 第一步：去除水印（使用高级算法）
    print("\n" + "=" * 70)
    print("第一步：高级水印去除")
    print("=" * 70)
    processed_image, shape_mask, background_color = remove_watermark_advanced('1.jpg')
    
    # 第二步：检测形状和颜色
    print("\n" + "=" * 70)
    print("第二步：检测几何图形")
    print("=" * 70)
    counts = detect_shapes(processed_image, shape_mask, background_color)
    
    # 第三步：生成直方图
    print("\n" + "=" * 70)
    print("第三步：生成形状数量直方图")
    print("=" * 70)
    create_shape_histogram(counts, '2.jpg')
    
    # 第四步：生成饼状图
    print("\n" + "=" * 70)
    print("第四步：生成颜色占比饼状图")
    print("=" * 70)
    create_color_pie_chart(counts, '3.jpg')
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print("\n生成的主要文件:")
    print("- 1_processed.jpg: 去除水印后的图片")
    print("- 1_with_shapes.jpg: 标记了形状和颜色的图片")
    print("- 2.jpg: 形状数量直方图")
    print("- 3.jpg: 颜色占比饼状图")
    print("\n调试文件（用于分析水印检测效果）:")
    print("- debug_step1_shape_mask.jpg: 几何图形掩码")
    print("- debug_step2_background_mask.jpg: 背景掩码")
    print("- debug_step3_multi_scale_mask.jpg: 多尺度颜色差异掩码")
    print("- debug_step4_frequency_mask.jpg: 频域/边缘检测掩码")
    print("- debug_step5_texture_mask.jpg: 纹理分析掩码")
    print("- debug_step6_lab_mask.jpg: LAB颜色空间掩码")
    print("- debug_step7_combined_mask.jpg: 最终合并的水印掩码")
    print("- debug_inpaint_stage1.jpg: 第一阶段修复结果")
    print("- debug_inpaint_stage2.jpg: 第二阶段修复结果")
    print("- debug_inpaint_stage3.jpg: 第三阶段修复结果")
    print("- debug_fallback_mask.jpg: 备用检测掩码（如适用）")

if __name__ == "__main__":
    main()
