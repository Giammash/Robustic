from mindmove.gui.mindmove import MindMove
# from mindmove import device_interfaces
import sys
from qdarkstyle import load_stylesheet
from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet(qt_api="pyside6"))
    main_window = MindMove()
    main_window.show()
    sys.exit(app.exec())
