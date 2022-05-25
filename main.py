import sys
from gui import Gui
from PyQt5.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    ex = Gui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
