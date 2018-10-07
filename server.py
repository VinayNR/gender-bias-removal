from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import os
import compiled2
# from flask_mysqldb import MySQL
# from connectdb import connection
# import test
 
app = Flask(__name__)
 
@app.route('/')
def home():
    # if not session.get('logged_in'):
    return render_template('login.html')
    # else:
    #     return render_template('graphs.html')
 
@app.route('/result', methods=['POST'])
def do_admin_login():
    text = request.form['plot']
    movie_name = test.getImage(request.form['movie-name'])
    # print(text)
    newtext = compiled.main(movie_name, text)
    return render_template('result.html', variable=text, variable1=newtext)

 
if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='127.0.0.1', port=4000)
