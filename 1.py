import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 第一步：读取并处理图片，去除水印（改进版）
def remove_watermark_improved(image_path):
    """
    改进版水印去除算法：
    1. 检测整个图像中与背景颜色不同的区域
    2. 排除几何图形区域
    3. 使用形态学操作增强文本状水印
    4. 使用图像修复技术去除水印
    5. 平滑处理确保背景颜色一致
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片尺寸
    height, width = img.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算背景颜色的平均值
    # 使用图像中心区域来估计背景颜色，因为几何图形可能分布在各处
    # 创建一个掩码，排除明显的彩色区域
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义高饱和度区域（可能是几何图形）
    _, saturation, _ = cv2.split(hsv)
    high_saturation_mask = (saturation > 50).astype(np.uint8) * 255
    
    # 膨胀高饱和度区域，以覆盖几何图形的边缘
    kernel = np.ones((10, 10), np.uint8)
    high_saturation_mask = cv2.dilate(high_saturation_mask, kernel, iterations=2)
    
    # 计算背景颜色（使用非高饱和度区域）
    background_mask = cv2.bitwise_not(high_saturation_mask)
    
    # 如果背景区域太小，使用整个图像的中心区域
    if cv2.countNonZero(background_mask) < (height * width * 0.3):
        center_region = img[height//4:3*height//4, width//4:3*width//4]
        background_color = np.mean(center_region, axis=(0, 1)).astype(int)
    else:
        # 计算背景区域的平均颜色
        background_color = cv2.mean(img, mask=background_mask)[:3]
        background_color = np.array(background_color).astype(int)
    
    print(f"估计的背景颜色 (BGR): {background_color}")
    
    # 检测所有与背景颜色不同的区域
    # 计算每个像素与背景颜色的欧氏距离
    color_diff = np.sqrt(np.sum((img - background_color) ** 2, axis=2))
    
    # 归一化差异值
    color_diff_normalized = ((color_diff - color_diff.min()) / 
                              (color_diff.max() - color_diff.min() + 1e-10) * 255).astype(np.uint8)
    
    # 应用阈值，检测差异较大的区域
    _, diff_mask = cv2.threshold(color_diff_normalized, 20, 255, cv2.THRESH_BINARY)
    
    # 现在需要排除几何图形区域
    # 首先检测几何图形（使用HSV颜色范围）
    # 红色范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 绿色范围
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # 蓝色范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # 创建几何图形的掩码
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 合并所有几何图形的掩码
    shape_mask = cv2.bitwise_or(red_mask, green_mask)
    shape_mask = cv2.bitwise_or(shape_mask, blue_mask)
    
    # 膨胀几何图形掩码，以覆盖边缘
    kernel = np.ones((15, 15), np.uint8)
    shape_mask = cv2.dilate(shape_mask, kernel, iterations=3)
    
    # 从差异掩码中排除几何图形区域
    watermark_candidate = cv2.bitwise_and(diff_mask, cv2.bitwise_not(shape_mask))
    
    # 现在，检测文本状的水印
    # 使用形态学操作来增强文本区域
    # 创建结构元素，适应文本的形状
    # 对于水平文本，使用水平方向的结构元素
    # 对于垂直文本，使用垂直方向的结构元素
    
    # 首先尝试水平结构元素
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    detected_horizontal = cv2.morphologyEx(watermark_candidate, cv2.MORPH_CLOSE, horizontal_kernel)
    detected_horizontal = cv2.morphologyEx(detected_horizontal, cv2.MORPH_OPEN, horizontal_kernel)
    
    # 尝试垂直结构元素
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
    detected_vertical = cv2.morphologyEx(watermark_candidate, cv2.MORPH_CLOSE, vertical_kernel)
    detected_vertical = cv2.morphologyEx(detected_vertical, cv2.MORPH_OPEN, vertical_kernel)
    
    # 合并水平和垂直检测结果
    detected_text = cv2.bitwise_or(detected_horizontal, detected_vertical)
    
    # 使用通用结构元素进行进一步处理
    kernel = np.ones((5, 5), np.uint8)
    detected_text = cv2.dilate(detected_text, kernel, iterations=2)
    detected_text = cv2.erode(detected_text, kernel, iterations=1)
    
    # 现在，我们还需要检测可能的半透明水印
    # 半透明水印通常会导致图像的亮度或对比度变化
    # 使用拉普拉斯算子检测边缘和纹理变化
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    laplacian_normalized = ((laplacian_abs - laplacian_abs.min()) / 
                             (laplacian_abs.max() - laplacian_abs.min() + 1e-10) * 255).astype(np.uint8)
    
    # 阈值处理，检测高纹理区域
    _, texture_mask = cv2.threshold(laplacian_normalized, 15, 255, cv2.THRESH_BINARY)
    
    # 排除几何图形区域
    texture_mask = cv2.bitwise_and(texture_mask, cv2.bitwise_not(shape_mask))
    
    # 形态学操作增强文本状区域
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, horizontal_kernel)
    texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, horizontal_kernel)
    texture_mask = cv2.dilate(texture_mask, kernel, iterations=1)
    
    # 合并所有检测到的水印候选区域
    watermark_mask = cv2.bitwise_or(detected_text, texture_mask)
    
    # 最后，检查图像角落和边缘的小区域（通常水印会出现在这些位置）
    corner_size = min(height, width) // 10
    edge_width = min(height, width) // 30
    
    # 定义需要检查的区域
    regions_to_check = [
        # 四个角落
        (0, corner_size, 0, corner_size),
        (0, corner_size, width - corner_size, width),
        (height - corner_size, height, 0, corner_size),
        (height - corner_size, height, width - corner_size, width),
        # 四条边的中间区域
        (0, edge_width, corner_size, width - corner_size),  # 顶部中间
        (height - edge_width, height, corner_size, width - corner_size),  # 底部中间
        (corner_size, height - corner_size, 0, edge_width),  # 左侧中间
        (corner_size, height - corner_size, width - edge_width, width)  # 右侧中间
    ]
    
    # 检查每个区域
    for (y1, y2, x1, x2) in regions_to_check:
        # 获取区域
        region_gray = gray[y1:y2, x1:x2]
        region_color = img[y1:y2, x1:x2]
        
        # 计算区域的标准差
        std_dev = np.std(region_gray)
        
        # 如果标准差超过阈值，可能有水印
        if std_dev > 10:
            # 计算与背景颜色的差异
            region_diff = np.sqrt(np.sum((region_color - background_color) ** 2, axis=2))
            
            # 归一化并阈值化
            region_diff_normalized = ((region_diff - region_diff.min()) / 
                                      (region_diff.max() - region_diff.min() + 1e-10) * 255).astype(np.uint8)
            _, region_mask = cv2.threshold(region_diff_normalized, 15, 255, cv2.THRESH_BINARY)
            
            # 排除几何图形区域
            region_shape_mask = shape_mask[y1:y2, x1:x2]
            region_mask = cv2.bitwise_and(region_mask, cv2.bitwise_not(region_shape_mask))
            
            # 形态学操作
            region_mask = cv2.dilate(region_mask, np.ones((3, 3), np.uint8), iterations=2)
            
            # 应用到水印掩码
            watermark_mask[y1:y2, x1:x2] = cv2.bitwise_or(watermark_mask[y1:y2, x1:x2], region_mask)
    
    # 保存中间结果用于调试
    cv2.imwrite('debug_diff_mask.jpg', diff_mask)
    cv2.imwrite('debug_shape_mask.jpg', shape_mask)
    cv2.imwrite('debug_watermark_candidate.jpg', watermark_candidate)
    cv2.imwrite('debug_detected_text.jpg', detected_text)
    cv2.imwrite('debug_texture_mask.jpg', texture_mask)
    cv2.imwrite('debug_final_watermark_mask.jpg', watermark_mask)
    print("已保存调试图片，用于分析水印检测效果")
    
    # 现在使用图像修复技术去除水印
    # 首先尝试使用Telea修复算法
    result = img.copy()
    
    # 检查水印掩码是否有内容
    if cv2.countNonZero(watermark_mask) > 0:
        # 应用inpaint
        radius = 5
        result = cv2.inpaint(result, watermark_mask, radius, cv2.INPAINT_TELEA)
        print("已使用Telea算法修复水印区域")
    else:
        print("警告：未检测到明显的水印区域，尝试使用替代方法")
        
        # 如果没有检测到水印，尝试更激进的方法
        # 检测所有非几何图形区域中的高变化区域
        # 使用自适应阈值
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 排除几何图形区域
        adaptive_thresh = cv2.bitwise_and(adaptive_thresh, cv2.bitwise_not(shape_mask))
        
        # 形态学操作
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, horizontal_kernel)
        adaptive_thresh = cv2.dilate(adaptive_thresh, np.ones((3, 3), np.uint8), iterations=1)
        
        cv2.imwrite('debug_adaptive_thresh.jpg', adaptive_thresh)
        
        # 应用inpaint
        if cv2.countNonZero(adaptive_thresh) > 0:
            result = cv2.inpaint(result, adaptive_thresh, 3, cv2.INPAINT_TELEA)
            print("已使用自适应阈值方法修复水印区域")
    
    # 为了确保背景颜色一致，对非几何图形区域进行处理
    # 创建非几何图形区域的掩码
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 对非几何图形区域应用以下处理：
    # 1. 首先使用轻微的模糊
    blurred = cv2.GaussianBlur(result, (7, 7), 0)
    
    # 2. 然后使用中值滤波去除噪点
    median_blurred = cv2.medianBlur(blurred, 5)
    
    # 3. 只在非几何图形区域应用处理
    # 创建一个更宽的掩码，包括几何图形的边缘
    wide_shape_mask = cv2.dilate(shape_mask, np.ones((20, 20), np.uint8), iterations=2)
    wide_non_shape_mask = cv2.bitwise_not(wide_shape_mask)
    
    # 应用处理
    result = np.where(wide_non_shape_mask[:, :, np.newaxis] > 0, median_blurred, result)
    
    # 最后，确保背景颜色完全一致
    # 计算处理后的背景颜色
    processed_background_color = cv2.mean(result, mask=wide_non_shape_mask)[:3]
    processed_background_color = np.array(processed_background_color).astype(int)
    
    print(f"处理后的背景颜色 (BGR): {processed_background_color}")
    
    # 对非几何图形区域进行颜色调整，确保与平均背景颜色一致
    # 计算颜色差异
    color_adjustment = processed_background_color - background_color
    
    # 只调整非几何图形区域
    # 注意：这里我们使用更保守的方法，因为直接调整颜色可能会导致不自然的结果
    # 相反，我们使用羽化边缘的方法来平滑过渡
    
    # 创建羽化边缘的掩码
    # 首先创建一个距离变换
    dist_transform = cv2.distanceTransform(wide_non_shape_mask, cv2.DIST_L2, 3)
    
    # 归一化距离变换
    dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    
    # 应用高斯平滑到距离变换
    dist_transform_smoothed = cv2.GaussianBlur(dist_transform_normalized, (15, 15), 0)
    
    # 只对远离几何图形的区域应用颜色调整
    # 这里我们不直接调整颜色，而是使用更平滑的方法
    # 实际上，前面的模糊和中值滤波应该已经足够
    
    # 保存最终处理后的图片
    cv2.imwrite('1_processed.jpg', result)
    print("已去除水印并保存为1_processed.jpg")
    
    return result, shape_mask, processed_background_color

# 第二步：检测并统计几何图形的数量和颜色
def detect_shapes(image, shape_mask, background_color):
    """
    检测图片中的几何图形（三角形、矩形、圆形），并统计它们的数量和颜色
    """
    # 复制图片用于绘制
    img_with_contours = image.copy()
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edged = cv2.Canny(blurred, 50, 150)
    
    # 形态学操作，连接断开的边缘
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    edged = cv2.erode(edged, kernel, iterations=1)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化计数器
    triangle_count = 0
    rectangle_count = 0
    circle_count = 0
    
    # 初始化颜色统计
    red_count = 0
    green_count = 0
    blue_count = 0
    
    # 形状和颜色的详细信息
    shapes_info = []
    
    # 遍历每个轮廓
    for contour in contours:
        # 计算轮廓面积，过滤掉太小的轮廓
        area = cv2.contourArea(contour)
        if area < 100:  # 阈值可以调整
            continue
        
        # 近似轮廓
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(approx)
        
        # 提取轮廓区域的颜色
        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # 计算该区域的平均颜色
        mean_color = cv2.mean(image, mask=mask)[:3]
        mean_color_bgr = np.array(mean_color)
        
        # 判断颜色（红、蓝、绿）
        # 转换为HSV进行更准确的颜色判断
        # 先提取该区域的图像
        roi = image[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 创建ROI的掩码
        roi_mask = mask[y:y+h, x:x+w]
        
        # 计算ROI内的平均HSV值
        if cv2.countNonZero(roi_mask) > 0:
            mean_hsv = cv2.mean(roi_hsv, mask=roi_mask)[:3]
        else:
            mean_hsv = [0, 0, 0]
        
        h, s, v = mean_hsv
        
        # 判断颜色
        color_name = "未知"
        color_bgr = (0, 0, 0)
        
        # 红色（两个范围）
        if (0 <= h <= 10 or 170 <= h <= 180) and s > 50 and v > 50:
            color_name = "红色"
            color_bgr = (0, 0, 255)
            red_count += 1
        # 绿色
        elif 40 <= h <= 80 and s > 50 and v > 50:
            color_name = "绿色"
            color_bgr = (0, 255, 0)
            green_count += 1
        # 蓝色
        elif 100 <= h <= 140 and s > 50 and v > 50:
            color_name = "蓝色"
            color_bgr = (255, 0, 0)
            blue_count += 1
        else:
            # 如果颜色不明确，尝试使用BGR值判断
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
            
            # 绘制轮廓
            cv2.drawContours(img_with_contours, [approx], -1, color_bgr, 2)
            cv2.putText(img_with_contours, f"{color_name}{shape_name}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            shapes_info.append({"shape": "三角形", "color": color_name})
        
        # 矩形或正方形
        elif len(approx) == 4:
            # 计算宽高比
            aspect_ratio = float(w) / h
            
            # 如果宽高比接近1，则是正方形，否则是矩形
            # 但用户要求统计矩形，所以都算矩形
            shape_name = "矩形"
            rectangle_count += 1
            
            # 绘制轮廓
            cv2.drawContours(img_with_contours, [approx], -1, color_bgr, 2)
            cv2.putText(img_with_contours, f"{color_name}{shape_name}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            
            shapes_info.append({"shape": "矩形", "color": color_name})
        
        # 圆形
        else:
            # 使用圆形度来判断
            # 圆形度 = 4π * 面积 / 周长²
            # 完美的圆形圆形度为1
            if peri > 0:
                circularity = 4 * np.pi * area / (peri * peri)
                
                # 如果圆形度接近1，则认为是圆形
                if circularity > 0.7:
                    shape_name = "圆形"
                    circle_count += 1
                    
                    # 绘制轮廓
                    cv2.drawContours(img_with_contours, [approx], -1, color_bgr, 2)
                    
                    # 计算圆心和半径
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

# 第三步：生成直方图（形状数量）
def create_shape_histogram(counts, output_path='2.jpg'):
    """
    生成形状数量的直方图，横轴为三角形、矩形、圆形，纵轴为数量
    横纵轴名称必须为汉字
    """
    # 准备数据
    shapes = ['三角形', '矩形', '圆形']
    counts_list = [counts['triangle'], counts['rectangle'], counts['circle']]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置柱状图颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 绘制柱状图
    bars = ax.bar(shapes, counts_list, color=colors, edgecolor='black', linewidth=1.5)
    
    # 设置标题和标签（全部使用汉字）
    ax.set_title('几何图形数量统计', fontsize=16, fontweight='bold')
    ax.set_xlabel('图形类型', fontsize=14)
    ax.set_ylabel('数量', fontsize=14)
    
    # 设置y轴范围
    max_count = max(counts_list)
    ax.set_ylim(0, max_count + 2 if max_count > 0 else 5)
    
    # 在柱状图上方添加数值标签
    for bar, count in zip(bars, counts_list):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 设置网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 设置字体
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成形状数量直方图并保存为{output_path}")

# 第四步：生成饼状图（颜色占比）
def create_color_pie_chart(counts, output_path='3.jpg'):
    """
    生成颜色占比的饼状图，红色占比部分用红色，绿色用绿色，蓝色用蓝色
    """
    # 准备数据
    colors = ['红色', '绿色', '蓝色']
    counts_list = [counts['red'], counts['green'], counts['blue']]
    
    # 过滤掉数量为0的颜色
    filtered_colors = []
    filtered_counts = []
    pie_colors = []
    
    for color, count in zip(colors, counts_list):
        if count > 0:
            filtered_colors.append(color)
            filtered_counts.append(count)
            if color == '红色':
                pie_colors.append('#FF0000')  # 红色
            elif color == '绿色':
                pie_colors.append('#00FF00')  # 绿色
            elif color == '蓝色':
                pie_colors.append('#0000FF')  # 蓝色
    
    # 如果没有数据，添加默认数据
    if not filtered_colors:
        filtered_colors = ['无数据']
        filtered_counts = [1]
        pie_colors = ['#CCCCCC']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制饼状图
    wedges, texts, autotexts = ax.pie(
        filtered_counts,
        labels=filtered_colors,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        explode=[0.05] * len(filtered_colors)  # 轻微分离每个部分
    )
    
    # 设置标题
    ax.set_title('几何图形颜色占比统计', fontsize=16, fontweight='bold', pad=20)
    
    # 确保饼状图是圆形
    ax.axis('equal')
    
    # 调整自动百分比文本的颜色，使其在深色背景上更清晰
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已生成颜色占比饼状图并保存为{output_path}")

# 主函数
def main():
    print("=" * 60)
    print("开始处理图片（改进版水印去除）...")
    print("=" * 60)
    
    # 第一步：去除水印（使用改进版算法）
    print("\n[第一步] 去除水印...")
    processed_image, shape_mask, background_color = remove_watermark_improved('1.jpg')
    
    # 第二步：检测形状和颜色
    print("\n[第二步] 检测几何图形...")
    counts = detect_shapes(processed_image, shape_mask, background_color)
    
    # 第三步：生成直方图
    print("\n[第三步] 生成形状数量直方图...")
    create_shape_histogram(counts, '2.jpg')
    
    # 第四步：生成饼状图
    print("\n[第四步] 生成颜色占比饼状图...")
    create_color_pie_chart(counts, '3.jpg')
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("- 1_processed.jpg: 去除水印后的图片")
    print("- 1_with_shapes.jpg: 标记了形状和颜色的图片")
    print("- 2.jpg: 形状数量直方图")
    print("- 3.jpg: 颜色占比饼状图")
    print("\n调试文件（用于分析水印检测效果）:")
    print("- debug_diff_mask.jpg: 颜色差异掩码")
    print("- debug_shape_mask.jpg: 几何图形掩码")
    print("- debug_watermark_candidate.jpg: 水印候选区域")
    print("- debug_detected_text.jpg: 检测到的文本区域")
    print("- debug_texture_mask.jpg: 纹理掩码")
    print("- debug_final_watermark_mask.jpg: 最终水印掩码")
    print("- debug_adaptive_thresh.jpg: 自适应阈值掩码（如适用）")

if __name__ == "__main__":
    main()
