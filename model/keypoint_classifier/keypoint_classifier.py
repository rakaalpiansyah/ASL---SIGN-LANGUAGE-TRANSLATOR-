#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    # Konstruktor untuk menginisialisasi interpreter TensorFlow Lite
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',  # Path ke model TFLite
        num_threads=1,  # Jumlah thread yang digunakan untuk inferensi
    ):
        # Memuat model TensorFlow Lite
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        # Mengalokasikan tensor untuk model
        self.interpreter.allocate_tensors()

        # Mendapatkan detail input dan output tensor
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # Fungsi untuk melakukan inferensi menggunakan landmark yang diberikan
    def __call__(self, landmark_list):
        # Mendapatkan indeks tensor input dari model
        input_details_tensor_index = self.input_details[0]['index']
        
        # Mengatur input tensor dengan landmark_list yang diberikan
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32)  # Mengubah landmark_list menjadi numpy array dengan tipe data float32
        )
        
        # Menjalankan inferensi (eksekusi model)
        self.interpreter.invoke()

        # Mendapatkan indeks tensor output dari model
        output_details_tensor_index = self.output_details[0]['index']

        # Mengambil hasil inferensi dari tensor output
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Mengambil indeks kelas dengan nilai tertinggi (argmax)
        result_index = np.argmax(np.squeeze(result))

        # Mengembalikan indeks kelas hasil inferensi
        return result_index
