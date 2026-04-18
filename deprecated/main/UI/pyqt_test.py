import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My PyQt App')
        self.setGeometry(100, 100, 300, 200)

        btn = QPushButton('Click Me', self)
        btn.clicked.connect(self.showMessage)
        btn.resize(btn.sizeHint())
        btn.move(100, 80)

    def showMessage(self):
        QMessageBox.information(self, 'Message', 'Button Clicked!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
