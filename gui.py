from model import Trainer
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from controller import Controller


class ClickableLabel(QtWidgets.QLabel):
    def __init__(self, widget):
        super(ClickableLabel, self).__init__(widget)
        self.main = widget

    def mousePressEvent(self, event):
        self.main.show_details(event.pos())


class Gui(QMainWindow):

    def __init__(self):
        super().__init__()

        self.controller = Controller()

        self.setFixedSize(1744, 660)
        self.setWindowTitle("Viewer")

        self.shape_selector = QtWidgets.QComboBox(self)
        self.shape_selector.addItems(['bench', 'chair'])
        self.train_button = QtWidgets.QPushButton(self)
        self.train_button.setText('Predict')
        self.summary_pane = ClickableLabel(self)
        self.true_img_pane = QtWidgets.QLabel(self)
        self.pred_img_pane = QtWidgets.QLabel(self)
        self.true_legend = QtWidgets.QLabel(self)
        self.pred_legend = QtWidgets.QLabel(self)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.task_description = QtWidgets.QLabel(self)

        self.shape_selector.adjustSize()
        self.train_button.adjustSize()
        self.task_description.setFixedSize(150, 20)

        self.shape_selector.move(10, 10)
        self.train_button.move(945, 10)
        self.summary_pane.move(1054, 10)
        self.true_img_pane.move(10, 70)
        self.pred_img_pane.move(532, 70)
        self.true_legend.move(240, 585)
        self.pred_legend.move(760, 585)
        self.progress_bar.move(10, 620)
        self.task_description.move(120, 625)

        self.shape_selector.activated[str].connect(self.show_summary)
        self.train_button.clicked.connect(self.train)
        self.controller.cluster_maker.progress_signal.connect(self.update_progress_bar)

        self.task_thread = None

        self.show()

        self.re_cluster()

    def __enable_interaction(self):
        self.shape_selector.setDisabled(False)
        self.train_button.setDisabled(False)

    def __disable_interaction(self):
        self.shape_selector.setDisabled(True)
        self.train_button.setDisabled(True)

    def __pil_to_rgb_pixmap(self, img):
        data = img.tobytes('raw', 'RGB')
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def __pil_to_rgba_pixmap(self, img):
        data = img.tobytes('raw', 'RGBA')
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def show_summary(self):
        shape_name = self.shape_selector.currentText()
        summary_img = self.controller.get_summary(shape_name)
        pixmap = self.__pil_to_rgba_pixmap(summary_img)
        self.summary_pane.setPixmap(pixmap)
        self.summary_pane.adjustSize()
        self.true_img_pane.clear()
        self.true_legend.clear()
        self.pred_img_pane.clear()
        self.pred_legend.clear()

    def show_details(self, pos):
        shape_name = self.shape_selector.currentText()
        i = pos.x() // 138
        j = pos.y() // 64
        if pos.x() > i * 138 + 128:
            return
        index = str(i) + str(j)
        true_img, pred_img = self.controller.get_details(shape_name, int(index))
        true_pixmap = self.__pil_to_rgb_pixmap(true_img)
        pred_pixmap = self.__pil_to_rgb_pixmap(pred_img)
        self.true_img_pane.setPixmap(true_pixmap)
        self.pred_img_pane.setPixmap(pred_pixmap)
        self.true_img_pane.adjustSize()
        self.pred_img_pane.adjustSize()
        self.true_legend.setText('Original')
        self.pred_legend.setText('Procedural')

    def __train_finished(self):
        self.progress_bar.hide()
        self.task_description.setText('')
        self.train_button.setText('Predict')
        if self.task_thread.is_model_dirty:
            self.re_cluster()
        self.task_thread = None

    def train(self):
        if isinstance(self.task_thread, Trainer):
            self.task_thread.stop()
        else:
            self.progress_bar.reset()
            self.progress_bar.show()
            self.task_description.setText('Training')
            self.train_button.setText('Stop')
            self.task_thread = Trainer(self.shape_selector.currentText())
            self.task_thread.progress_signal.connect(self.update_progress_bar)
            self.task_thread.finished.connect(self.__train_finished)
            self.task_thread.start()

    def __cluster_finished(self):
        self.progress_bar.hide()
        self.task_description.setText('')
        self.__enable_interaction()
        self.show_summary()

    def re_cluster(self):
        self.progress_bar.reset()
        self.progress_bar.show()
        self.task_description.setText('Preparing shapes')
        self.__disable_interaction()
        self.controller.make_clusters()
        self.controller.cluster_maker.finished.connect(self.__cluster_finished)

    def update_progress_bar(self, value: int):
        self.progress_bar.setValue(value)
