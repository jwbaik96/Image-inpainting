import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDesktopWidget, QVBoxLayout, QPushButton, QHBoxLayout
from painter import *


class MyApp(QMainWindow):
    def __init__(self, network, device):
        super(MyApp, self).__init__()
        self.network = network
        self.device = device
        self.initUI()

    def initUI(self):
        # Status bar
        self.statusBar().showMessage('Ready')

        # QWidget
        self.table_widget = MainWidget(self)
        self.setCentralWidget(self.table_widget)

        self.setWindowTitle('Inpainting Gui')
        self.resize(600, 400)
        self.move_center()
        self.show()

    def move_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class MainWidget(QWidget):
    def __init__(self, mainApp):
        super(MainWidget, self).__init__()
        self.mainApp = mainApp
        self.result_window = Result()
        self.canvas = Painter(mainApp, self.result_window)
        self.initUI()

    def initUI(self):
        # Vertical menu Layout
        open_img = QPushButton('Open')
        open_img.clicked.connect(self.canvas.image_open)
        Pen_1 = QPushButton('narrow Pen')
        Pen_1.clicked.connect(self.canvas.small_pen)
        Pen_2 = QPushButton('broad Pen')
        Pen_2.clicked.connect(self.canvas.big_pen)
        erase = QPushButton('Erase')
        erase.clicked.connect(self.canvas.erase)
        inpaint_bt = QPushButton('Inpaint')
        inpaint_bt.clicked.connect(self.canvas.save_draw)

        Menu = QVBoxLayout()
        Menu.addStretch(1)
        Menu.addWidget(open_img)
        Menu.addStretch(1)
        Menu.addWidget(Pen_1)
        Menu.addWidget(Pen_2)
        Menu.addWidget(erase)
        Menu.addStretch(1)
        Menu.addWidget(inpaint_bt)
        Menu.addStretch(3)

        # Vertical Image Layout
        img_layout = QVBoxLayout()
        img_layout.addStretch(1)
        img_layout.addWidget(self.canvas)
        img_layout.addStretch(1)

        # Vertical Result Layout
        result_layout = QVBoxLayout()
        result_layout.addStretch(1)
        result_layout.addWidget(self.result_window)
        result_layout.addStretch(1)


        # Horizon Layout
        hbox = QHBoxLayout()
        hbox.addStretch(0.2)
        hbox.addLayout(Menu)
        hbox.addStretch(0.5)
        hbox.addLayout(img_layout)
        hbox.addStretch(0.5)
        hbox.addLayout(result_layout)
        hbox.addStretch(0.5)

        self.setLayout(hbox)


def Execute_gui(network, device):
    app = QApplication(sys.argv)
    ex = MyApp(network, device)
    sys.exit(app.exec_())

