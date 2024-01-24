import cv2
import dlib
import numpy as np
import pyautogui
import time
import tkinter as tk
from tkinter import Toplevel

# Инициализация Tkinter
root = tk.Tk()
root.withdraw()  # Скрываем основное окно Tkinter

class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return np.mean(self.values, axis=0)

# Экземпляры фильтра для левого и правого глаза
left_eye_filter = MovingAverageFilter(window_size=5)
right_eye_filter = MovingAverageFilter(window_size=5)


def map_eye_position_to_screen(eye_position, screen_width, screen_height):
    # Проверка на допустимость координат
    x = min(max(eye_position[0], 0), screen_width)
    y = min(max(eye_position[1], 0), screen_height)

    # Масштабирование координат зрачка
    x_scaled = np.interp(x, [0, screen_width], [0, screen_width])
    y_scaled = np.interp(y, [0, screen_height], [0, screen_height])
    return int(x_scaled), int(y_scaled)


def control_cursor(left_pupil, right_pupil, screen_width, screen_height, sensitivity=0.5):
    # Добавляем проверку на None
    if left_pupil is None or right_pupil is None:
        return

    # Вычисляем среднее положение зрачков
    avg_pupil_position = np.mean([left_pupil, right_pupil], axis=0)

    # Управление курсором с учетом чувствительности
    x, y = map_eye_position_to_screen(avg_pupil_position, screen_width, screen_height)
    print(x,y)
    pyautogui.moveTo(x * sensitivity, y * sensitivity, duration=0.1)


def detect_pupil(eye_frame):
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    # Улучшаем контраст для выделения зрачка
    contrasted = cv2.equalizeHist(gray)

    # Применяем пороговое значение
    _, threshold = cv2.threshold(contrasted, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Поиск контура с максимальной площадью, предполагая, что это зрачок
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y))
    return None


def get_eye_position(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    return left_eye, right_eye

def draw_eye_landmarks(frame, landmarks):
    # Рисуем точки глаз на кадре
    for n in range(36, 48):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

def get_eye_frame(frame, eye_points):
    x_min, x_max = eye_points[:, 0].min(), eye_points[:, 0].max()
    y_min, y_max = eye_points[:, 1].min(), eye_points[:, 1].max()
    return frame[y_min:y_max, x_min:x_max]

