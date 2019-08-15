from wtforms import Form, StringField, validators, DateField, IntegerField
from math import pi


class InputForm(Form):
    ID = StringField(
        label='', default="1",
        validators=[validators.InputRequired(), validators.length(max=10)])
    Span1_start = DateField(
        label='YYYY-MM-DD',
        validators=[validators.InputRequired()],
        format='%Y-%m-%d')
    Span1_end = DateField(
        label='YYYY-MM-DD',
        validators=[validators.InputRequired()],
        format='%Y-%m-%d')
    Span2_start = DateField(
        label='YYYY-MM-DD',
        validators=[validators.InputRequired()],
        format='%Y-%m-%d')
    Span2_end = DateField(
        label='YYYY-MM-DD',
        validators=[validators.InputRequired()],
        format='%Y-%m-%d')
