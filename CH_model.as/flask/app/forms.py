from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, SelectField, IntegerField
from wtforms import validators

class UserInputForm(FlaskForm):
    model_type = SelectField('Select Model:',
                            choices=[('i','Intact'),
                                    ('l', 'Lesioned'),
                                    ('s', 'Scopolamine'),
                                    ('p', 'Physostygimine'),],
                            validators=[validators.DataRequired()],)
    problem = SelectField('Select Problem:',
                            choices=[('one', '1'),],
                            validators=[validators.DataRequired()],)
    repeats = IntegerField('Enter Repeats:',
                            validators=[validators.length(min=1, max=1),])
    run = SubmitField('Run Model')
