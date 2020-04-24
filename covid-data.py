

from flask import Flask,render_template,request

from sklearn.externals import joblib

app  =  Flask(__name__)

model  =  joblib.load("corona_model-final"
                      ".pkl")

@app.route('/')

def check():
   return render_template('index.html')


@app.route('/',methods  =  ['POST'])
def submit_data():
    if request.method  ==  'POST':
        age  =  float(request.form['age'])

        result  =  str(model.predict([[age]]))
    return render_template('index.html',var  =  result)


if __name__  ==  "__main__":
   app.run(debug=True)