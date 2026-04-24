import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 第一步：读取并处理图片，去除水印
def remove_watermark(image_path):
    """
    去除图片中的AI生成水印，并保持背景颜色一致
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片尺寸
    height, width = img.shape[:2]
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算背景颜色的平均值（假设背景是均匀的）
    # 我们可以通过计算图片中心区域的颜色来估计背景颜色
    center_region = img[height//4:3*height//4, width//4:3*width//4]
    background_color = np.mean(center_region, axis=(0, 1)).astype(int)
    
    print(f"估计的背景颜色 (BGR): {background_color}")
    
    # 检测水印区域 - 假设水印在角落或边缘，并且颜色与背景不同
    # 我们可以通过颜色差异来检测水印
    # 首先，创建一个掩码，标记与背景颜色差异较大的区域
    # 但要排除几何图形（它们颜色更鲜艳）
    
    # 转换为HSV颜色空间，更好地检测颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义几何图形的颜色范围（红、蓝、绿）
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
    
    # 形态学操作，填充小的空洞
    kernel = np.ones((5, 5), np.uint8)
    shape_mask = cv2.dilate(shape_mask, kernel, iterations=2)
    shape_mask = cv2.erode(shape_mask, kernel, iterations=1)
    
    # 现在，寻找可能的水印区域
    # 水印通常是文本，颜色较浅或较深，位于角落
    # 我们可以检查图片的四个角落区域
    
    # 定义角落区域的大小
    corner_size = min(height, width) // 8
    
    # 检查四个角落
    corners = [
        (0, corner_size, 0, corner_size),  # 左上角
        (0, corner_size, width - corner_size, width),  # 右上角
        (height - corner_size, height, 0, corner_size),  # 左下角
        (height - corner_size, height, width - corner_size, width)  # 右下角
    ]
    
    # 创建水印掩码
    watermark_mask = np.zeros_like(gray)
    
    for (y1, y2, x1, x2) in corners:
        # 获取角落区域
        corner_region = gray[y1:y2, x1:x2]
        
        # 计算区域的标准差，判断是否有水印（水印区域通常有更多变化）
        std_dev = np.std(corner_region)
        
        # 如果标准差较大，可能有水印
        if std_dev > 20:  # 阈值可以调整
            # 在这个区域中，寻找与背景差异大但不是几何图形的区域
            corner_bgr = img[y1:y2, x1:x2]
            corner_hsv = hsv[y1:y2, x1:x2]
            
            # 计算与背景颜色的差异
            color_diff = np.sum(np.abs(corner_bgr - background_color), axis=2)
            
            # 寻找差异较大的区域（可能是水印）
            _, diff_mask = cv2.threshold(color_diff, 30, 255, cv2.THRESH_BINARY)
            
            # 排除几何图形区域
            corner_shape_mask = shape_mask[y1:y2, x1:x2]
            diff_mask = cv2.bitwise_and(diff_mask.astype(np.uint8), cv2.bitwise_not(corner_shape_mask))
            
            # 应用到水印掩码
            watermark_mask[y1:y2, x1:x2] = diff_mask
    
    # 也检查图片边缘，不只是角落
    edge_width = min(height, width) // 20
    
    # 检查边缘区域
    edges = [
        (0, edge_width, 0, width),  # 顶部边缘
        (height - edge_width, height, 0, width),  # 底部边缘
        (0, height, 0, edge_width),  # 左侧边缘
        (0, height, width - edge_width, width)  # 右侧边缘
    ]
    
    for (y1, y2, x1, x2) in edges:
        edge_region = gray[y1:y2, x1:x2]
        std_dev = np.std(edge_region)
        
        if std_dev > 15:
            edge_bgr = img[y1:y2, x1:x2]
            color_diff = np.sum(np.abs(edge_bgr - background_color), axis=2)
            _, diff_mask = cv2.threshold(color_diff, 25, 255, cv2.THRESH_BINARY)
            
            edge_shape_mask = shape_mask[y1:y2, x1:x2]
            diff_mask = cv2.bitwise_and(diff_mask.astype(np.uint8), cv2.bitwise_not(edge_shape_mask))
            
            watermark_mask[y1:y2, x1:x2] = cv2.bitwise_or(watermark_mask[y1:y2, x1:x2], diff_mask)
    
    # 形态学操作，增强水印区域
    kernel = np.ones((3, 3), np.uint8)
    watermark_mask = cv2.dilate(watermark_mask, kernel, iterations=2)
    
    # 使用修复算法去除水印
    # 使用inpaint方法
    # radius参数决定了修复区域的大小
    radius = 3
    
    # 创建一个与原图相同的副本用于修复
    result = img.copy()
    
    # 应用inpaint
    result = cv2.inpaint(result, watermark_mask, radius, cv2.INPAINT_TELEA)
    
    # 为了确保背景颜色一致，我们可以对非几何图形区域进行平滑处理
    # 创建一个掩码，标记非几何图形区域
    non_shape_mask = cv2.bitwise_not(shape_mask)
    
    # 对非几何图形区域应用轻微的模糊，使背景更均匀
    blurred = cv2.GaussianBlur(result, (5, 5), 0)
    
    # 只在非几何图形区域应用模糊
    result = np.where(non_shape_mask[:, :, np.newaxis] > 0, blurred, result)
    
    # 保存处理后的图片
    cv2.imwrite('1_processed.jpg', result)
    print("已去除水印并保存为1_processed.jpg")
    
    return result, shape_mask, background_color

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
    print("开始处理图片...")
    print("=" * 60)
    
    # 第一步：去除水印
    print("\n[第一步] 去除水印...")
    processed_image, shape_mask, background_color = remove_watermark('1.jpg')
    
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

if __name__ == "__main__":
    main()
