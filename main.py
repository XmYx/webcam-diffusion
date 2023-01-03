import os
import sys

import psutil
import pynvml
from PySide6.QtCore import QTextStream, Qt, QFile, QIODevice, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QMainWindow, QDockWidget, QFileDialog, QMessageBox, QMdiArea, QMdiSubWindow, \
    QLabel, QTextEdit, QGridLayout, QSizePolicy


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the memory monitor dock widget
        self.memory_monitor = MemoryMonitor(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.memory_monitor)
        # Create the MDI area
        self.mdi_area = QMdiArea(self)
        self.setCentralWidget(self.mdi_area)

        # Set up the File menu
        self.file_menu = self.menuBar().addMenu("File")
        self.open_action = self.file_menu.addAction("Open")
        self.open_action.triggered.connect(self.open_file)

    def open_file(self):
        # Show a file dialog to choose a Python file
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Python File", "", "Python Files (*.py)")

        # If a file was chosen, open it in a new MDI sub window
        if file_name:
            # Open the file and read its contents
            file = QFile(file_name)
            if file.open(QIODevice.ReadOnly | QIODevice.Text):
                stream = QTextStream(file)
                text = stream.readAll()
                file.close()

                # Create a new sub window to display the contents of the file
                sub_window = QMdiSubWindow()
                sub_window.setAttribute(Qt.WA_DeleteOnClose)
                sub_window.setWindowTitle(file_name)
                sub_window.setWidget(QTextEdit(text))
                self.mdi_area.addSubWindow(sub_window)
                sub_window.show()
            else:
                QMessageBox.warning(self, "Error", "Could not open file")
class MemoryMonitor(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Memory Monitor", parent)

        # Initialize NVML
        pynvml.nvmlInit()

        # Create labels to display the memory usage
        self.cpu_label = QLabel("CPU memory usage:")
        self.cpu_value_label = QLabel()
        self.gpu_label = QLabel("GPU memory usage:")
        self.gpu_value_label = QLabel()

        # Set the size policies of the labels to expanding
        self.cpu_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.cpu_value_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.gpu_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.gpu_value_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Set up a timer to refresh the memory usage display every 1 second
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(1000)

        # Use a grid layout to lay out the labels
        layout = QGridLayout()
        layout.addWidget(self.cpu_label, 0, 0)
        layout.addWidget(self.cpu_value_label, 0, 1)
        layout.addWidget(self.gpu_label, 1, 0)
        layout.addWidget(self.gpu_value_label, 1, 1)
        self.setLayout(layout)

        self.refresh()
    def refresh(self):
        # Get the current CPU and GPU memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        cpu_memory_usage = memory_info.rss / 1024 / 1024

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_usage = info.used / 1024 / 1024

        # Update the labels with the current memory usage
        self.cpu_value_label.setText("{:.2f} MB".format(cpu_memory_usage))
        self.gpu_value_label.setText("{:.2f} MB".format(gpu_memory_usage))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
