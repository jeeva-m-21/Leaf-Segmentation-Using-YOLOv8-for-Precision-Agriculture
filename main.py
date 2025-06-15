import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QMessageBox, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QTimer, QRect
import cv2
from detector import detect_leaves

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #202020; border-radius: 15px;")
        self.setFixedSize(400, 300)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 30, 20, 30)
        self.layout.setSpacing(20)
        self.layout.setAlignment(Qt.AlignCenter)
        self.setLayout(self.layout)

        self.icon_label = QLabel("\U0001F343")
        self.icon_label.setFixedSize(120, 120)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFont(QFont("Arial", 72))
        self.icon_label.setStyleSheet("color: #55bb55;")
        self.layout.addWidget(self.icon_label)

        self.logo_label = QLabel("Leaf Counter")
        self.logo_label.setFont(QFont("Arial", 26, QFont.Bold))
        self.logo_label.setStyleSheet("color: white;")
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.logo_label)

        self.loading_label = QLabel("Loading")
        self.loading_label.setFont(QFont("Arial", 16))
        self.loading_label.setStyleSheet("color: white;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.loading_label)

        self.dot_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_dots)
        self.timer.start(500)

    def animate_dots(self):
        self.dot_count = (self.dot_count + 1) % 4
        dots = '.' * self.dot_count
        self.loading_label.setText(f"Loading{dots}")

    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        geo = screen.geometry()
        x = (geo.width() - self.width()) // 2
        y = (geo.height() - self.height()) // 2
        self.move(x, y)

class LeafCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leaf Segmentation Model")
        self.setGeometry(100, 100, 820, 680)
        self.setAcceptDrops(True)
        self.image_path = None
        self.result_img = None
        self.setup_ui()
        self.apply_dark_theme()

    def setup_ui(self):
        self.toolbar = QLabel("Leaf Segmentation")
        self.toolbar.setFixedHeight(40)
        self.toolbar.setAlignment(Qt.AlignCenter)
        self.toolbar.setStyleSheet("background-color: black; color: #55bb55; font: bold 18px 'Arial';")

        self.image_label = QLabel("Upload or drag & drop an image to begin.", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(750, 520)
        self.image_label.setStyleSheet("border: 2px solid #555; background-color: #222; color: #aaa; border-radius: 10px;")

        self.upload_button = self.create_button("\U0001F4C1 Upload Image", self.upload_image)
        self.count_button = self.create_button("\U0001F9AE Count Leaves", self.count_leaves)
        self.save_button = self.create_button("\U0001F4BE Save Result", self.save_image)
        self.save_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        button_layout.addStretch()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.count_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()

        self.count_label = QLabel("Leaf Count: N/A")
        self.pixel_label = QLabel("Total Leaf Pixels: N/A")
        self.image_size_label = QLabel("Image Size: N/A")

        for label in (self.count_label, self.pixel_label, self.image_size_label):
            label.setFont(QFont("Arial", 14, QFont.Bold))
            label.setStyleSheet("color: #f9d342; margin-top: 6px;")
            label.setAlignment(Qt.AlignCenter)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addSpacing(8)
        main_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        main_layout.addSpacing(12)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.count_label)
        main_layout.addWidget(self.pixel_label)
        main_layout.addWidget(self.image_size_label)
        main_layout.addStretch()

        self.setLayout(main_layout)

    def create_button(self, text, slot):
        button = QPushButton(text)
        button.setStyleSheet("""
            QPushButton {
                background-color: #383838;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 22px;
                font-size: 14px;
                font-weight: 600;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #55bb55;
            }
            QPushButton:pressed {
                background-color: #3d8b3d;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
        """)
        button.clicked.connect(slot)
        button.setCursor(Qt.PointingHandCursor)
        return button

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.Highlight, QColor(85, 187, 85))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)

        app.setPalette(dark_palette)
        app.setStyle("Fusion")

    def upload_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file:
            self.load_image(file)

    def load_image(self, file):
        self.image_path = file
        pixmap = QPixmap(file).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.count_label.setText("Leaf Count: N/A")
        self.pixel_label.setText("Total Leaf Pixels: N/A")
        self.image_size_label.setText("Image Size: N/A")
        self.save_button.setEnabled(False)

    def count_leaves(self):
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please upload or drag & drop an image first.")
            return

        result_img, count, pixel_count, (w, h) = detect_leaves(self.image_path)
        self.result_img = result_img
        qimg = self.convert_cv_qt(result_img)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
        self.count_label.setText(f"Leaf Count: {count}")
        self.pixel_label.setText(f"Total Leaf Pixels: {pixel_count}")
        self.image_size_label.setText(f"Image Size: {w}x{h} ({w*h} pixels)")
        self.save_button.setEnabled(True)

    def save_image(self):
        if self.result_img is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPG Files (*.jpg)")
            if save_path:
                cv2.imwrite(save_path, self.result_img)
                QMessageBox.information(self, "Saved", "Image saved successfully.")

    def convert_cv_qt(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            local_file = urls[0].toLocalFile()
            if local_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.load_image(local_file)
                event.acceptProposedAction()
            else:
                QMessageBox.warning(self, "Invalid File", "Please drop a valid image file (.png, .jpg, .jpeg).")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = LoadingScreen()
    splash.center_on_screen()
    splash.show()

    window = LeafCounterApp()

    def show_main():
        splash.timer.stop()
        splash.close()
        qr = window.frameGeometry()
        cp = app.primaryScreen().geometry().center()
        qr.moveCenter(cp)
        window.move(qr.topLeft())
        window.show()

    QTimer.singleShot(2000, show_main)

    sys.exit(app.exec_())
