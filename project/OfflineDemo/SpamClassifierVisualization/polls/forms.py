from django import forms

class AddForm(forms.Form):
	message = forms.CharField(max_length=50)
	CHOICES = (('SVM', 'SVM'),('MP', 'MP'),('LR', 'LR'),('DT', 'DT'),('GBDT','GBDT'))
	method = forms.ChoiceField(choices=CHOICES)

