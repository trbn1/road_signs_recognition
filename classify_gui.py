# -*- coding: utf-8 -*-

"""
This is a GUI for the classify.py script.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QGridLayout, QGroupBox,
                             QLabel, QPushButton)

from classify import classify


class ClassifyGUI(QDialog):
    """Create GUI."""
    def __init__(self, parent=None):
        super(ClassifyGUI, self).__init__(parent)

        self.model_location = 'models/predict_signs_model_50.pkl'
        self.results_location = 'test_images/results_en.txt'
        self.file_location = None
        self.result = None

        self.create_top_left_group_box()
        self.create_top_right_group_box()
        self.create_bottom_left_buttons()
        self.create_bottom_right_group_box()

        bottom_left_layout = QGridLayout()
        bottom_left_layout.addWidget(self.choose_model_button, 1, 0)
        bottom_left_layout.addWidget(self.choose_file_button, 1, 1)
        bottom_left_layout.addWidget(self.choose_results_button, 2, 0)
        bottom_left_layout.addWidget(self.recognize_sign_button, 2, 1)

        main_layout = QGridLayout()
        main_layout.addWidget(self.top_left_group_box, 1, 0)
        main_layout.addWidget(self.top_left_image, 1, 0)
        main_layout.addWidget(self.top_right_group_box, 1, 1)
        main_layout.addLayout(bottom_left_layout, 2, 0)
        main_layout.addWidget(self.bottom_right_group_box, 2, 1)
        main_layout.addWidget(self.bottom_right_text, 2, 1)
        main_layout.setRowStretch(1, 1)
        main_layout.setRowStretch(2, 0)
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        self.setLayout(main_layout)

        self.setWindowTitle("Road Signs Recognition")


    def create_top_left_group_box(self):
        """Create a box for input image."""
        self.top_left_group_box = QGroupBox('Input image:')
        self.top_left_group_box.setMinimumSize(300, 300)
        self.top_left_image = QLabel()
        self.top_left_image.setMargin(30)
        self.top_left_image.setAlignment(Qt.AlignCenter)
        self.top_left_image.setScaledContents(True)


    def create_top_right_group_box(self):
        """Create a box for recognized sign image."""
        self.top_right_group_box = QGroupBox('Recognized sign image:')
        self.top_right_group_box.setMinimumSize(300, 300)
        self.top_right_image = QLabel()
        self.top_right_image.setMargin(30)
        self.top_right_image.setAlignment(Qt.AlignCenter)
        self.top_right_image.setScaledContents(True)


    def create_bottom_left_buttons(self):
        """Create buttons."""
        self.choose_model_button = QPushButton('Choose model')
        self.choose_model_button .setMinimumSize(50, 30)
        self.choose_model_button.clicked.connect(self.choose_model)
        self.choose_results_button = QPushButton('Choose results names')
        self.choose_results_button.clicked.connect(self.choose_results)
        self.choose_results_button.setMinimumSize(50, 30)
        self.choose_file_button = QPushButton('Choose file')
        self.choose_file_button .setMinimumSize(50, 30)
        self.choose_file_button.clicked.connect(self.choose_file)
        self.recognize_sign_button = QPushButton('Recognize sign')
        self.recognize_sign_button.clicked.connect(self.recognize_sign)
        self.recognize_sign_button.setMinimumSize(50, 30)


    def create_bottom_right_group_box(self):
        """Create a box for recognized sign name."""
        self.bottom_right_group_box = QGroupBox('Recognized sign name:')
        self.bottom_right_group_box.setMinimumSize(200, 60)
        self.bottom_right_text = QLabel()
        self.bottom_right_text.setAlignment(Qt.AlignCenter)


    def choose_model(self):
        """Open file dialog on button click and choose model file."""
        file_name = QFileDialog.getOpenFileName()
        if file_name:
            self.model_location = file_name[0]


    def choose_results(self):
        """Open file dialog on button click and choose results road signs names."""
        file_name = QFileDialog.getOpenFileName()
        if file_name:
            self.results_location = file_name[0]


    def choose_file(self):
        """Open file dialog on button click and choose image file."""
        file_name = QFileDialog.getOpenFileName()
        if file_name:
            self.file_location = file_name[0]
            self.top_left_image.setPixmap(QPixmap(self.file_location))


    def recognize_sign(self):
        """Recognize road sign on a given image file."""
        if self.model_location and self.file_location and self.results_location:
            self.bottom_right_text.setText('')
            result = classify(self.model_location, self.file_location, self.results_location)
            self.bottom_right_text.setText(result)


if __name__ == '__main__':
    import sys

    APP = QApplication(sys.argv)
    GUI = ClassifyGUI()
    GUI.show()
    sys.exit(APP.exec_())
