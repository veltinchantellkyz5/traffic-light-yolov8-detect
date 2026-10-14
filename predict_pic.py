import sys
import warnings
import time
import os
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from ultralytics import YOLO

warnings.filterwarnings('ignore')

class TrafficLightDetectWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = "runs/detect/train/weights/best.pt"
        self.save_path = "runs/predict/pic_result/result"
        self.total_stats_txt_path = os.path.join(self.save_path, "灯色分类统计总表.txt")
        self.temp_dir = "temp_preprocess"
        self.img_path = ""
        self.model = None

        # 窗口初始大小
        self.setWindowTitle('红绿灯检测工具')
        self.resize(1500, 800)

        self.init_gui()
        self.preload_model()

    def init_gui(self):
        """初始化GUI界面布局"""
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 1. 上半部分：原始图片 + 检测结果图片
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setSpacing(40)

        # 原始图片区域
        ori_group = QtWidgets.QGroupBox("原始图片")
        ori_group.setStyleSheet('''
            QGroupBox {
                font-size: 20px; 
                font-weight: bold;
                color: #333;
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        ''')
        ori_layout = QtWidgets.QVBoxLayout(ori_group)
        self.ori_img_label = QtWidgets.QLabel("请选择图片")
        self.ori_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ori_img_label.setStyleSheet('''
            border: 2px dashed #999;
            border-radius: 8px;
            background-color: #f8f8f8;
            font-size: 18px; 
            color: #666;
            min-height: 500px; 
        ''')
        self.ori_img_label.setScaledContents(False)
        ori_layout.addWidget(self.ori_img_label)
        top_layout.addWidget(ori_group, stretch=2)

        # 检测结果图片区域
        detect_group = QtWidgets.QGroupBox("检测结果")
        detect_group.setStyleSheet('''
            QGroupBox {
                font-size: 20px;
                font-weight: bold;
                color: #333;
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        ''')
        detect_layout = QtWidgets.QVBoxLayout(detect_group)
        self.detect_img_label = QtWidgets.QLabel("检测结果将显示于此")
        self.detect_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.detect_img_label.setStyleSheet('''
            border: 2px dashed #999;
            border-radius: 8px;
            background-color: #f8f8f8;
            font-size: 18px;
            color: #666;
            min-height: 500px;
        ''')
        self.detect_img_label.setScaledContents(False)
        detect_layout.addWidget(self.detect_img_label)
        top_layout.addWidget(detect_group, stretch=2)

        main_layout.addLayout(top_layout)

        # 2. 下半部分：输出日志 + 功能按钮
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.setSpacing(40)

        # 检测日志输出框
        log_group = QtWidgets.QGroupBox("检测日志")
        log_group.setStyleSheet('''
            QGroupBox {
                font-size: 20px;
                font-weight: bold;
                color: #333;
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        ''')
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_browser = QtWidgets.QTextBrowser()
        self.log_browser.setStyleSheet('''
            font-size: 16px; 
            font-family: "Microsoft YaHei"; 
            background-color: #f0f0f0; 
            border: 2px solid #ccc; 
            border-radius: 8px;
            padding: 10px;
        ''')
        log_layout.addWidget(self.log_browser)
        bottom_layout.addWidget(log_group, stretch=3)

        # 功能按钮区域
        btn_group = QtWidgets.QGroupBox("操作功能")
        btn_group.setStyleSheet('''
            QGroupBox {
                font-size: 20px;
                font-weight: bold;
                color: #333;
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        ''')
        btn_layout = QtWidgets.QVBoxLayout(btn_group)
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(QtCore.Qt.AlignTop)

        # 选择图片按钮
        self.select_img_btn = QtWidgets.QPushButton("选择图片")
        self.select_img_btn.setFixedHeight(50)
        self.select_img_btn.setStyleSheet('''
            QPushButton {
                font-size: 18px; 
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        ''')
        self.select_img_btn.clicked.connect(self.select_image)
        btn_layout.addWidget(self.select_img_btn)

        # 置信度阈值调节
        conf_widget = QtWidgets.QWidget()
        conf_layout = QtWidgets.QHBoxLayout(conf_widget)
        conf_layout.setSpacing(15)
        conf_label = QtWidgets.QLabel("置信度阈值：")
        conf_label.setStyleSheet("font-size: 18px;")
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(25)
        self.conf_label = QtWidgets.QLabel("0.25")
        self.conf_label.setStyleSheet("font-size: 18px; min-width: 50px;")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        btn_layout.addWidget(conf_widget)

        # 开始检测按钮
        self.detect_btn = QtWidgets.QPushButton("开始检测")
        self.detect_btn.setFixedHeight(50)
        self.detect_btn.setStyleSheet('''
            QPushButton {
                font-size: 18px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        ''')
        self.detect_btn.clicked.connect(self.run_detection)
        btn_layout.addWidget(self.detect_btn)

        # 清空按钮
        self.clear_btn = QtWidgets.QPushButton("清空内容")
        self.clear_btn.setFixedHeight(50)
        self.clear_btn.setStyleSheet('''
            QPushButton {
                font-size: 18px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        ''')
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.clear_btn)

        bottom_layout.addWidget(btn_group, stretch=1)

        main_layout.addLayout(bottom_layout)

    def preload_model(self):
        """预加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to("cpu")  # 兼容CPU环境
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】模型加载成功：{self.model_path}")
        except Exception as e:
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】模型加载失败：{str(e)}")
            QMessageBox.warning(self, "警告", f"模型加载失败：{str(e)}")

    def select_image(self):
        """选择本地图片并显示原始预览"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择红绿灯图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        self.img_path = file_path
        img_name = os.path.basename(file_path)

        # 显示原始图片
        try:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.ori_img_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ori_img_label.setPixmap(pixmap)
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】已选择图片：{img_name}")
        except Exception as e:
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】图片加载失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"图片加载失败：{str(e)}")

    def run_detection(self):
        """执行红绿灯检测核心逻辑"""
        if not self.img_path:
            QMessageBox.warning(self, "提示", "请先选择一张红绿灯图片！")
            return
        if not self.model:
            QMessageBox.warning(self, "提示", "模型未加载成功，请检查模型路径！")
            return

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # 图像预处理
        img = cv2.imread(self.img_path)
        if img is None:
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】错误：无法读取图像！")
            QMessageBox.critical(self, "错误", "无法读取图像，请检查图片是否损坏！")
            return
        img_denoised = cv2.GaussianBlur(img, (5, 5), 0)

        # 保存临时文件
        temp_img_name = os.path.basename(self.img_path)
        temp_img_path = os.path.join(self.temp_dir, temp_img_name)
        cv2.imwrite(temp_img_path, img_denoised)
        self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】预处理完成，临时文件保存至：{temp_img_path}")

        # 推理
        conf_threshold = self.conf_slider.value() / 100
        self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】开始检测，置信度阈值：{conf_threshold}")
        try:
            results = self.model.predict(
                source=temp_img_path,
                imgsz=640,
                conf=conf_threshold,
                save=True,
                project=os.path.dirname(self.save_path),
                name=os.path.basename(self.save_path),
                exist_ok=True,
                verbose=False
            )
            result_img = results[0].plot()
        except Exception as e:
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】检测失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"检测失败：{str(e)}")
            return

        # 显示检测结果
        try:
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(result_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.detect_img_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.detect_img_label.setPixmap(pixmap)
        except Exception as e:
            self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】结果图片预览失败：{str(e)}")
            detect_result_path = os.path.join(self.save_path, temp_img_name)
            cv2.imwrite(detect_result_path, result_img)
            QMessageBox.warning(self, "提示", f"结果图片预览失败，已保存至：{detect_result_path}")

        # 统计
        class_map = {0: "绿灯(green)", 1: "红灯(red)", 2: "黄灯(yellow)"}
        light_statistics = {k: {"count": 0, "confidences": []} for k in class_map.values()}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            light_type = class_map.get(cls_id, "未知类别")
            if light_type in light_statistics:
                light_statistics[light_type]["count"] += 1
                light_statistics[light_type]["confidences"].append(round(conf_score, 4))

        # 日志输出
        self.log_browser.append("=" * 80)
        self.log_browser.append(f"测试图像：{temp_img_name}")
        self.log_browser.append("【灯色分类统计结果】")
        total = 0
        for light, stats in light_statistics.items():
            count = stats["count"]
            conf_list = stats["confidences"]
            avg_conf = round(sum(conf_list) / len(conf_list), 4) if conf_list else 0.0
            total += count
            self.log_browser.append(f"  - {light}：")
            self.log_browser.append(f"    数量：{count} 个")
            self.log_browser.append(f"    置信度列表：{conf_list}")
            self.log_browser.append(f"    平均置信度：{avg_conf}")
        self.log_browser.append(f"  - 检测到红绿灯总数量：{total} 个")
        self.log_browser.append("=" * 80)

        # 追加统计总表
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        stats_text = f"\n{'=' * 80}\n【检测时间】：{current_time}\n【测试图像】：{temp_img_name}\n【置信度阈值】：{conf_threshold}\n【灯色分类统计结果】\n"
        for light, stats in light_statistics.items():
            stats_text += f"  - {light}：数量 {stats['count']} 个，置信度列表 {stats['confidences']}，平均置信度 {round(sum(stats['confidences']) / len(stats['confidences']), 4) if stats['confidences'] else 0.0}\n"
        stats_text += f"  - 红绿灯总数量：{total} 个\n{'=' * 80}\n"
        with open(self.total_stats_txt_path, "a", encoding="utf-8") as f:
            f.write(stats_text)

        # 完成提示
        detect_result_path = os.path.join(self.save_path, temp_img_name)
        self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】推理完成！检测结果保存至：{detect_result_path}")
        self.log_browser.append(
            f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】统计结果已追加到总表：{self.total_stats_txt_path}")
        QMessageBox.information(self, "成功", "检测完成！统计结果已追加到总表，详情请查看日志框。")

    def clear_all(self):
        """清空所有内容"""
        self.img_path = ""
        self.ori_img_label.clear()
        self.ori_img_label.setText("请选择图片")
        self.detect_img_label.clear()
        self.detect_img_label.setText("检测结果将显示于此")
        self.log_browser.clear()
        self.log_browser.append(f"【{time.strftime('%Y-%m-%d %H:%M:%S')}】已清空内容，等待选择新图片...")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TrafficLightDetectWindow()
    window.show()
    sys.exit(app.exec_())
