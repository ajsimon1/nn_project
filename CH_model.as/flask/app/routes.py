from flask import render_template, flash
from app import app
from app.forms import UserInputForm

@app.route('/', methods=['GET','POST'])
@app.route('/index', methods=['GET','POST'])
def index():
    form = UserInputForm()
    if form.validate_on_submit:
        flash('Running {} network on problem {} with {} repeats...'.format(
            form.model_type.data,
            form.problem.data,
            form.repeats.data,
        ))
    return render_template('index.html', title='Neural Net',form=form)
