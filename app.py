#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import sys
import threading
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyttsx3

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    # Membuat parser argumen untuk mengambil input dari command line
    parser = argparse.ArgumentParser()

    # Menentukan parameter argumen
    parser.add_argument("--device", type=int, default=0)  # Pilih perangkat kamera
    parser.add_argument("--width", help='Lebar kamera', type=int, default=960)
    parser.add_argument("--height", help='Tinggi kamera', type=int, default=540)

    # Mode pengaturan tambahan
    parser.add_argument('--use_static_image_mode', action='store_true')  # Gunakan mode gambar statis
    parser.add_argument("--min_detection_confidence",
                        help='Tingkat kepercayaan minimum deteksi',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='Tingkat kepercayaan minimum pelacakan',
                        type=int,
                        default=0.5)

    args = parser.parse_args()  # Parsing argumen
    return args  # Mengembalikan nilai argumen


def speak_text(engine, text):
    """Fungsi untuk menjalankan teks-ke-suara di thread terpisah."""
    engine.say(text)  # Memasukkan teks ke engine
    engine.runAndWait()  # Menjalankan engine


def main():
    # Parsing argumen
    args = get_args()

    # Mengatur parameter kamera
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Mengatur parameter model
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True  # Mengaktifkan bounding rectangle

    # Inisialisasi kamera
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)  # Mengatur lebar kamera
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)  # Mengatur tinggi kamera

    # Memuat model MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Memuat model klasifikasi titik kunci
    keypoint_classifier = KeyPointClassifier()

    # Membaca label klasifikasi
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # Menghitung FPS
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Timer untuk menampilkan output di terminal
    last_output_time = time.time()
    hand_sign_text = ""  # Inisialisasi teks isyarat tangan

    # Daftar untuk menyimpan isyarat tangan yang terdeteksi
    detected_hand_signs = []

    # Variabel untuk melacak waktu pembaruan isyarat tangan
    last_update_time = time.time()

    # Penundaan awal sebelum memulai (contoh: 3 detik)
    print("Memulai dalam 3 detik...")
    time.sleep(3)  # Penundaan 3 detik
    print("Deteksi dimulai...")

    # Inisialisasi engine pyttsx3 untuk teks-ke-suara
    engine = pyttsx3.init()

    while True:
        # Menghitung FPS
        fps = cvFpsCalc.get()

        # Proses penanganan tombol
        key = cv.waitKey(1)
        if key == 27:  # ESC untuk keluar
            break
        if key == ord('c') or key == ord('C'):  # 'C' untuk menghapus panel
            detected_hand_signs = []  # Menghapus daftar isyarat tangan
            print("Panel dihapus!")  # Menampilkan pesan ke terminal
        if key == ord('s') or key == ord('S'):  # 'S' untuk membaca semua isyarat tanpa jeda
            if detected_hand_signs:
                # Menggabungkan semua isyarat tangan menjadi satu string
                concatenated_signs = "".join(detected_hand_signs)
                print(f"Membaca semua isyarat tangan: {concatenated_signs}")

                # Memulai TTS di thread terpisah
                tts_thread = threading.Thread(target=speak_text, args=(engine, concatenated_signs))
                tts_thread.start()

        # Menangkap frame kamera
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Membalik tampilan kamera
        debug_image = copy.deepcopy(image)

        # Implementasi deteksi tangan
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Konversi ke RGB
        image.flags.writeable = False
        results = hands.process(image)  # Proses deteksi tangan
        image.flags.writeable = True

        hand_sign_id = -1  # Default jika tidak ada tangan yang terdeteksi

        # Proses dan tampilkan hasil
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Menghitung landmark
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Konversi ke koordinat relatif atau normalisasi
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Klasifikasi isyarat tangan
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Bagian menggambar
                debug_image = draw_landmarks(debug_image, landmark_list)

        # Menampilkan ID isyarat tangan pada frame
        if hand_sign_id != -1:
            hand_sign_text = f"{keypoint_classifier_labels[hand_sign_id]}"
            cv.putText(debug_image, hand_sign_text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            # Periksa apakah sudah 2 detik sejak pembaruan terakhir
            current_time = time.time()
            if current_time - last_update_time >= 2.0:  # Setiap 2 detik
                # Menambahkan teks isyarat tangan yang terdeteksi ke daftar
                detected_hand_signs.append(hand_sign_text)
                last_update_time = current_time

                # Membaca isyarat tangan yang terdeteksi di thread terpisah
                tts_thread = threading.Thread(target=speak_text, args=(engine, hand_sign_text))
                tts_thread.start()

                # Menampilkan isyarat tangan di terminal
                if current_time - last_output_time >= 2.0:  # Setiap 2 detik
                    sys.stdout.write(f"{hand_sign_text}")  # Tambahkan spasi untuk pemisahan
                    sys.stdout.flush()  # Tampilkan output segera
                    last_output_time = current_time

        # Membuat panel/frame di bawah tampilan kamera untuk menampilkan teks secara horizontal
        height, width, _ = debug_image.shape
        panel_height = 100  # Tinggi panel
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)  # Panel hitam

        # Penempatan teks isyarat tangan secara horizontal di panel
        x_offset = 10  # Posisi horizontal awal
        for sign in detected_hand_signs:
            cv.putText(panel, sign, (x_offset, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
            x_offset += len(sign) * 30  # Sesuaikan posisi horizontal berdasarkan panjang teks

        # Menggabungkan tampilan kamera (atas) dan panel (bawah)
        combined_image = np.vstack((debug_image, panel))

        # Menampilkan gambar gabungan
        cv.imshow('Hand Gesture Recognition', combined_image)

    # Melepas kamera dan menutup jendela
    cap.release()
    cv.destroyAllWindows()
    
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def calc_bounding_rect(image, landmarks):
    # Mengambil lebar dan tinggi gambar
    image_width, image_height = image.shape[1], image.shape[0]

    # Inisialisasi array untuk menyimpan koordinat landmark
    landmark_array = np.empty((0, 2), int)

    # Loop melalui setiap landmark untuk menghitung koordinat piksel
    for _, landmark in enumerate(landmarks.landmark):
        # Konversi koordinat landmark dari normalisasi (0-1) ke piksel
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        # Simpan koordinat landmark sebagai array
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    # Hitung kotak pembatas (bounding rectangle) berdasarkan array landmark
    x, y, w, h = cv.boundingRect(landmark_array)

    # Kembalikan koordinat kotak pembatas [x_min, y_min, x_max, y_max]
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    # Mengambil lebar dan tinggi gambar
    image_width, image_height = image.shape[1], image.shape[0]

    # Inisialisasi daftar untuk menyimpan koordinat landmark
    landmark_point = []

    # Loop melalui setiap landmark untuk menghitung koordinat piksel
    for _, landmark in enumerate(landmarks.landmark):
        # Konversi koordinat landmark dari normalisasi (0-1) ke piksel
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        # Tambahkan koordinat landmark ke dalam daftar
        landmark_point.append([landmark_x, landmark_y])

    # Kembalikan daftar koordinat landmark
    return landmark_point


def pre_process_landmark(landmark_list):
    # Membuat salinan dari daftar landmark untuk diproses
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Konversi koordinat menjadi relatif terhadap titik pertama
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:  # Landmark pertama menjadi titik dasar
            base_x, base_y = landmark_point[0], landmark_point[1]

        # Hitung koordinat relatif terhadap titik dasar
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Mengubah daftar dua dimensi menjadi satu dimensi
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalisasi nilai koordinat
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):  # Fungsi untuk normalisasi
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    # Kembalikan daftar landmark yang sudah dinormalisasi
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    # Mengambil lebar dan tinggi gambar
    image_width, image_height = image.shape[1], image.shape[0]

    # Membuat salinan dari riwayat titik untuk diproses
    temp_point_history = copy.deepcopy(point_history)

    # Konversi koordinat menjadi relatif terhadap titik pertama
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:  # Titik pertama menjadi titik dasar
            base_x, base_y = point[0], point[1]

        # Hitung koordinat relatif terhadap titik dasar dan normalisasi
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Mengubah daftar dua dimensi menjadi satu dimensi
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    # Kembalikan daftar riwayat titik yang sudah diproses
    return temp_point_history


def logging_csv(number, mode, landmark_list):
    # Jika mode 0, tidak melakukan apa-apa
    if mode == 0:
        pass

    # Jika mode 1 dan nomor berada dalam rentang 0-9
    if mode == 1 and (0 <= number <= 9):
        # Menentukan lokasi file CSV untuk menyimpan data
        csv_path = 'model/keypoint_classifier/keypoint.csv'

        # Membuka file CSV dan menambahkan baris baru
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            # Menulis nomor dan landmark ke file
            writer.writerow([number, *landmark_list])
    
    # Tidak ada nilai yang dikembalikan
    return


def draw_landmarks(image, landmark_point):
    # Gambar garis dan titik-titik landmark pada gambar

    if len(landmark_point) > 0:
        # Jempol
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Jari telunjuk
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Jari tengah
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Jari manis
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Jari kelingking
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Telapak tangan
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Menambahkan titik-titik landmark
    for index, landmark in enumerate(landmark_point):
        radius = 5 if index < 4 else 8  # Menyesuaikan radius untuk jari
        color = (255, 255, 0)
        thickness = -1
        cv.circle(image, (landmark[0], landmark[1]), radius, color, thickness)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    # Gambar kotak pembatas jika diaktifkan
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


if __name__ == '__main__':
    main()  # Jalankan fungsi utama
