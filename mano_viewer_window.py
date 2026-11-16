"""Native Qt WebEngine window, isolated from the editor's Tk event loop."""
import sys


def main():
    from PySide6.QtCore import QSocketNotifier, QUrl
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebEngineWidgets import QWebEngineView

    app = QApplication(sys.argv[:1])
    window = QWebEngineView()
    window.setWindowTitle('MANO 三维视图')
    window.resize(1000, 760)
    window.setMinimumSize(420, 360)
    notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Read)

    def command(*args):
        if not sys.stdin.readline():
            app.quit()
            return
        window.showNormal()
        window.raise_()
        window.activateWindow()

    def loaded(ok):
        print('READY' if ok else 'FAILED', flush=True)

    notifier.activated.connect(command)
    window.loadFinished.connect(loaded)
    window.load(QUrl(sys.argv[1]))
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
