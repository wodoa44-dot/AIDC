"""Render the local HTML with installed Qt WebEngine; no external URLs required."""
import json,os,sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QTWEBENGINE_DISABLE_SANDBOX']='1'
os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--no-sandbox --disable-gpu --disable-dev-shm-usage'
os.environ['QT_QUICK_BACKEND']='software'
from PyQt5.QtCore import QUrl,QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
app=QApplication([]);out=Path(sys.argv[1]).resolve();view=QWebEngineView();view.resize(1440,1000);view.show()
completed=False

def finish(ok):
    def inspect():
        js="""JSON.stringify({documentWidth:document.documentElement.scrollWidth,viewport:window.innerWidth,images:Array.from(document.images).map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0,width:i.width,naturalWidth:i.naturalWidth})),title:document.title,bodyHeight:document.body.scrollHeight})"""
        def received(value):
            global completed
            info=json.loads(value);info['load_ok']=bool(ok);info['horizontal_overflow']=info['documentWidth']>info['viewport']+2
            info['all_images_loaded']=all(x['loaded'] for x in info['images']);info['renderer']='QtWebEngine actual HTML rendering'
            view.grab().save(str(out/'report_render_top.png'))
            (out/'render_verification.json').write_text(json.dumps(info,indent=2))
            def bottom():
                view.grab().save(str(out/'report_render_bottom.png'));completed=True;app.exit(0 if ok and info['all_images_loaded'] and not info['horizontal_overflow'] else 2)
            view.page().runJavaScript('window.scrollTo(0, document.body.scrollHeight)')
            QTimer.singleShot(800,bottom)
        view.page().runJavaScript(js,received)
    QTimer.singleShot(1500,inspect)
view.loadFinished.connect(finish)
view.load(QUrl.fromLocalFile(str(out/'report_ko.html')))
QTimer.singleShot(45000,lambda:app.exit(3))
sys.exit(app.exec_())
