from flask import Flask, render_template, request
from wtforms import Form, DecimalField, validators

app = Flask(__name__)


class EntryForm(Form):
    x_entry = DecimalField('x:',
                           places=10,
                           validators=[validators.NumberRange(-1e10, 1e10)])
    y_entry = DecimalField('y:',
                           places=10,
                           validators=[validators.NumberRange(-1e10, 1e10)])

@app.route('/')
def index():
    form = EntryForm(request.form)
    return render_template('entry.html', form=form, z='')

@app.route('/results', methods=['POST'])
def results():
    form = EntryForm(request.form)
    z = ''
    if request.method == 'POST' and form.validate():
        x = request.form['x_entry']
        y = request.form['y_entry']
        z = float(x) + float(y)
    return render_template('entry.html', form=form, z=z)

if __name__ == '__main__':
    app.run(debug=True)