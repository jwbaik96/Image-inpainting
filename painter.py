import os
import torch
from PyQt5 import QtGui, QtCore, QtWidgets
from PIL import Image
from torchvision import transforms


class Color(object):
    def __init__(self, nR=0, nG=0, nB=0):
        self.R = nR
        self.G = nG
        self.B = nB


class Point(object):
    def __init__(self, nX=0, nY=0):
        self.X = nX
        self.Y = nY

    def Set(self, nX, nY):
        self.X = nX
        self.Y = nY


class Shape(object):
    def __init__(self, location=Point(0, 0), width=1, color=Color(255, 255, 255), number=0):
        self.Location = location
        self.Width = width
        self.Color = color
        self.ShapeNumber = number


class Shapes(object):
    def __init__(self):
        self.Shapes = []

    def NumberOfShapes(self):
        return len(self.Shapes)

    def NewShape(self, location=Point(0, 0), width=4, color=Color(255, 255, 255), number=0):
        Sh = Shape(location, width, color, number)
        self.Shapes.append(Sh)

    def getShape(self, Index):
        return self.Shapes[Index]


class Painter(QtWidgets.QLabel):
    def __init__(self, mainApp, result_widget):
        super(Painter, self).__init__()
        self.mainApp = mainApp
        self.result_widget = result_widget

        pixmap = QtGui.QPixmap(256, 256)
        pixmap.fill(QtCore.Qt.gray)
        self.setPixmap(pixmap)

        # mask map
        self.map = QtGui.QImage(256, 256, QtGui.QImage.Format_RGB32)
        self.map.fill(QtCore.Qt.black)

        self.DrawingShape = Shapes()
        self.CurrentWidth = 6
        self.MouseLocation = Point(0, 0)
        self.LastPosition = Point(0, 0)
        self.CurrentColor = Color(255, 255, 255)
        self.ShapeNum = 0

        self.IsPainting = False

    def mousePressEvent(self, event):
        self.IsPainting = True
        self.ShapeNum += 1
        self.LastPosition = Point(0, 0)

    def mouseMoveEvent(self, event):
        if self.IsPainting:
            self.MouseLocation = Point(event.x(), event.y())

            if self.LastPosition.X != self.MouseLocation.X or self.LastPosition.Y != self.MouseLocation.Y:
                self.LastPosition = Point(event.x(), event.y())
                self.DrawingShape.NewShape(self.LastPosition, self.CurrentWidth, self.CurrentColor, self.ShapeNum)
            self.draw_lines()

    def mouseReleaseEvent(self, event):
        self.IsPainting = False

    def draw_lines(self):
        painter = QtGui.QPainter(self.pixmap())
        painter.RenderHint(QtGui.QPainter.Antialiasing)

        for i in range(self.DrawingShape.NumberOfShapes() - 1):
            T = self.DrawingShape.getShape(i)
            T1 = self.DrawingShape.getShape(i + 1)

            if T.ShapeNumber == T1.ShapeNumber:
                pen = QtGui.QPen(QtGui.QColor(0, 0, 0), T.Width, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(T.Location.X, T.Location.Y, T1.Location.X, T1.Location.Y)
        painter.end()
        self.update()

    def image_open(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Select the image')
        if fname[0] == '':
            pixmap = QtGui.QPixmap(256, 256)
            pixmap.fill(QtCore.Qt.gray)
        else:
            pixmap = QtGui.QPixmap(fname[0])
            pixmap = pixmap.scaled(256, 256)
            pixmap.save('./result/GT.jpg')

        self.setPixmap(pixmap)
        self.DrawingShape = Shapes()

        # map 초기화
        self.map = QtGui.QImage(256, 256, QtGui.QImage.Format_RGB32)
        self.map.fill(QtCore.Qt.black)

    def erase(self):
        if os.path.exists('./result/GT.jpg'):
            pixmap = QtGui.QPixmap('./result/GT.jpg')
        else:
            pixmap = QtGui.QPixmap(256, 256)
            pixmap.fill(QtCore.Qt.gray)
        pixmap = pixmap.scaled(256, 256)
        self.setPixmap(pixmap)
        self.DrawingShape = Shapes()

        # map 초기화
        self.map = QtGui.QImage(256, 256, QtGui.QImage.Format_RGB32)
        self.map.fill(QtCore.Qt.black)

    def save_draw(self):
        self.mainApp.statusBar().showMessage('Inpainting Process is running. Wait a minute!')
        painter = QtGui.QPainter(self.map)
        painter.begin(self.map)

        for i in range(self.DrawingShape.NumberOfShapes() - 1):
            T = self.DrawingShape.getShape(i)
            T1 = self.DrawingShape.getShape(i + 1)

            if T.ShapeNumber == T1.ShapeNumber:
                pen = QtGui.QPen(QtGui.QColor(T.Color.R, T.Color.G, T.Color.B), T.Width, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(T.Location.X, T.Location.Y, T1.Location.X, T1.Location.Y)
        painter.end()
        self.map.save('./result/mask.png')
        self.inpainting()

    def small_pen(self):
        self.CurrentWidth = 6

    def big_pen(self):
        self.CurrentWidth = 12

    def inpainting(self):
        # Transform
        transform = transforms.ToTensor()
        PIL_transform = transforms.ToPILImage('RGB')

        img = transform(Image.open('./result/GT.jpg').convert('RGB')).unsqueeze(dim=0).to(self.mainApp.device)
        mask = transform(Image.open('./result/mask.png').convert('L')).unsqueeze(dim=0).to(self.mainApp.device)
        coarse_img, refine_img = self.mainApp.network(img, mask)
        self.mainApp.network.zero_buffer()

        masked_img = img * (1.0 - mask) + mask
        masked_img = PIL_transform(masked_img.squeeze(dim=0).to(torch.device('cpu')))
        masked_img.save('./result/masked_img.jpg')

        coarse_img = PIL_transform(coarse_img.squeeze(dim=0).to(torch.device('cpu')))
        coarse_img.save('./result/coarse.jpg')

        complete_img = img * (1.0 - mask) + mask * refine_img
        complete_img = PIL_transform(complete_img.squeeze(dim=0).to(torch.device('cpu')))
        complete_img.save('./result/result.jpg')

        self.mainApp.statusBar().showMessage('Ready')
        self.result_widget.update_result()


class Result(QtWidgets.QLabel):
    def __init__(self):
        super(Result, self).__init__()
        self.initUI()

    def initUI(self):
        pixmap = QtGui.QPixmap(256, 256)
        pixmap.fill(QtCore.Qt.gray)
        self.setPixmap(pixmap)

    def update_result(self):
        pixmap = QtGui.QPixmap('result/result.jpg')
        self.setPixmap(pixmap)
