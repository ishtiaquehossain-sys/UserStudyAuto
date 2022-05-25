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

        self.setFixedSize(1800, 800)
        self.setWindowTitle("Viewer")

        self.shape_selector = QtWidgets.QComboBox(self)
        self.shape_selector.addItems(['bench', 'chair', 'sofa', 'table'])
        self.train_button = QtWidgets.QPushButton(self)
        self.train_button.setText('Train')
        self.num_cluster_label = QtWidgets.QLabel(self)
        self.num_cluster_label.setText('Number of clusters:')
        self.num_cluster_input = QtWidgets.QLineEdit(self)
        self.num_cluster_input.setText(str(self.controller.cluster_maker.num_clusters))
        self.cluster_button = QtWidgets.QPushButton(self)
        self.cluster_button.setText('Cluster')
        self.summary_pane = ClickableLabel(self)
        self.true_img_pane = QtWidgets.QLabel(self)
        self.pred_img_pane = QtWidgets.QLabel(self)
        self.true_legend = QtWidgets.QLabel(self)
        self.pred_legend = QtWidgets.QLabel(self)
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.task_description = QtWidgets.QLabel(self)
        self.task_description.setText('Clustering')

        self.shape_selector.adjustSize()
        self.train_button.adjustSize()
        self.num_cluster_label.adjustSize()
        self.num_cluster_input.setMaximumWidth(30)
        self.num_cluster_input.setMaxLength(2)
        self.num_cluster_input.adjustSize()
        self.cluster_button.adjustSize()

        self.shape_selector.move(50, 50)
        self.train_button.move(130, 50)
        self.num_cluster_label.move(820, 55)
        self.num_cluster_input.move(960, 50)
        self.cluster_button.move(1000, 50)
        self.summary_pane.move(1090, 50)
        self.true_img_pane.move(50, 100)
        self.pred_img_pane.move(570, 100)
        self.true_legend.move(280, 620)
        self.pred_legend.move(790, 620)
        self.progress_bar.move(50, 730)
        self.task_description.move(160, 730)

        self.shape_selector.activated[str].connect(self.show_summary)
        self.train_button.clicked.connect(self.train)
        self.cluster_button.clicked.connect(self.re_cluster)
        self.controller.cluster_maker.progress_signal.connect(self.update_progress_bar)
        self.controller.cluster_maker.finished.connect(self.__cluster_finished)

        self.task_thread = None

        self.show()

        self.re_cluster(False)

    def __enable_interaction(self):
        self.shape_selector.setDisabled(False)
        self.train_button.setDisabled(False)
        self.cluster_button.setDisabled(False)

    def __disable_interaction(self):
        self.shape_selector.setDisabled(True)
        self.train_button.setDisabled(True)
        self.cluster_button.setDisabled(True)

    def __pil_to_pixmap(self, img):
        data = img.tobytes('raw', 'RGB')
        qim = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qim)
        return pixmap

    def show_summary(self):
        shape_name = self.shape_selector.currentText()
        summary_img = self.controller.get_summary(shape_name)
        pixmap = self.__pil_to_pixmap(summary_img)
        self.summary_pane.setPixmap(pixmap)
        self.summary_pane.adjustSize()
        self.true_img_pane.clear()
        self.true_legend.clear()
        self.pred_img_pane.clear()
        self.pred_legend.clear()

    def show_details(self, pos):
        shape_name = self.shape_selector.currentText()
        index = str(pos.y()//128) + str(pos.x()//64)
        true_img, pred_img = self.controller.get_details(shape_name, int(index))
        true_pixmap = self.__pil_to_pixmap(true_img)
        pred_pixmap = self.__pil_to_pixmap(pred_img)
        self.true_img_pane.setPixmap(true_pixmap)
        self.pred_img_pane.setPixmap(pred_pixmap)
        self.true_img_pane.adjustSize()
        self.pred_img_pane.adjustSize()
        self.true_legend.setText('Original')
        self.pred_legend.setText('Procedural')

    def __train_finished(self):
        self.progress_bar.hide()
        self.task_description.setText('')
        self.train_button.setText('Train')
        if self.task_thread.is_model_dirty:
            self.re_cluster(False)
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

    def re_cluster(self, cluster_only: bool = True):
        if self.task_thread is not None:
            self.task_thread.stop()
        self.progress_bar.reset()
        self.progress_bar.show()
        self.task_description.setText('Clustering')
        self.__disable_interaction()
        self.controller.cluster_maker.num_clusters = int(self.num_cluster_input.text())
        if cluster_only:
            self.controller.remake_clusters()
        else:
            self.controller.remake_images_clusters()

    def update_progress_bar(self, value: int):
        self.progress_bar.setValue(value)
