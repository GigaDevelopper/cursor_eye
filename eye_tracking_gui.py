import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import pyautogui
import dlib
from cursor import get_eye_position, detect_pupil, control_cursor, MovingAverageFilter, draw_eye_landmarks

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Основные параметры окна и инициализация элементов GUI
        self.setWindowTitle("Eye Tracking Interface")
        self.setGeometry(200, 200, 800, 600)
        widget = QWidget(self)
        self.layout = QVBoxLayout()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)
        self.start_button = QPushButton("Start Tracking", self)
        self.start_button.clicked.connect(self.start_tracking)
        self.layout.addWidget(self.start_button)

        # Таймер и камера
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)

        # Фильтры и размеры экрана
        self.left_eye_filter = MovingAverageFilter(window_size=5)
        self.right_eye_filter = MovingAverageFilter(window_size=5)
        self.screen_width, self.screen_height = pyautogui.size()

        # dlib компоненты
        self.predictor_path = './model/eye_landmark.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def start_tracking(self):
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()
            self.start_button.setText("Start Tracking")
        else:
            self.cap.open(0)
            self.timer.start(20)
            self.start_button.setText("Stop Tracking")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                for face in faces:
                    shape = self.predictor(gray, face)
                    draw_eye_landmarks(frame, shape)
                    left_eye, right_eye = get_eye_position(shape)
                    left_eye_frame = self.get_eye_frame(frame, left_eye)
                    right_eye_frame = self.get_eye_frame(frame, right_eye)

                    if left_eye_frame is not None and right_eye_frame is not None:
                        left_pupil = detect_pupil(left_eye_frame)
                        right_pupil = detect_pupil(right_eye_frame)

                        # Проверяем, что оба зрачка обнаружены
                        if left_pupil is not None and right_pupil is not None:
                            left_pupil_smooth = self.left_eye_filter.update(left_pupil)
                            right_pupil_smooth = self.right_eye_filter.update(right_pupil)
                            control_cursor(left_pupil_smooth, right_pupil_smooth, self.screen_width, self.screen_height)
                        else:
                            # Обработка ситуации, когда один или оба зрачка не обнаружены
                            print("Зрачок(и) не обнаружен(ы)")

                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
                pix = QPixmap.fromImage(image)
                self.video_label.setPixmap(
                    pix.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
            except Exception as e:
                print(f"Error: {e}")

    def get_eye_frame(self, frame, eye_points):
        if len(eye_points) == 6:
            x_min, x_max = max(eye_points[:, 0].min(), 0), min(eye_points[:, 0].max(), frame.shape[1])
            y_min, y_max = max(eye_points[:, 1].min(), 0), min(eye_points[:, 1].max(), frame.shape[0])
            return frame[y_min:y_max, x_min:x_max]
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
