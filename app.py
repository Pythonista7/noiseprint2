from flask import Flask
from flask import render_template, request
import batchprocess
app = Flask(__name__)

@app.route("/")
def imageupload():
  return render_template("upload.html")

@app.route("/cameramodel", methods = ['GET', 'POST'])
def processimage():
     if request.method == 'POST':
        f = request.files['file']
        batchprocess.start_search(f)
        return 'file uploaded successfully'

if __name__ == "__main__":
  app.run(debug = True)