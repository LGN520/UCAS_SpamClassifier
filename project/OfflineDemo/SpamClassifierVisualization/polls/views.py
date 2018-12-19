# from django.shortcuts import render

# from django.http import HttpResponse

import sys

import os
from SpamClassifierVisualization.settings import BASE_DIR
sys.path.append(os.path.join(BASE_DIR,'../../SpamClassifier/'))

from SpamClassifier import SpamClassifier
from SpamClassifier import ClassifierAssistant

from django.shortcuts import render, render_to_response
from django.http import HttpResponse, HttpResponseRedirect

from django.http import HttpResponse

from .forms import AddForm

SC_SVM = SpamClassifier(type=ClassifierAssistant.SVM)
SC_MP = SpamClassifier(type=ClassifierAssistant.MP)
SC_LR = SpamClassifier(type=ClassifierAssistant.LR)
SC_DT = SpamClassifier(type=ClassifierAssistant.DT)
SC_GBDT = SpamClassifier(type=ClassifierAssistant.GBDT)

def index(request):
    if request.method == 'POST':  

        form = AddForm(request.POST)  

        if form.is_valid():  
            a = form.cleaned_data['message']
            b = form.cleaned_data['method']
            b = str(b)
            if b == "SVM":
                result = SC_SVM.predict(str(a))
            if b == "MP":
                result = SC_MP.predict(str(a))
            if b == "LR":
                result = SC_LR.predict(str(a))
            if b == "DT":
                result = SC_DT.predict(str(a))
            if b == "GBDT":
                result = SC_GBDT.predict(str(a))
            
            if str(result) == "0":
                ret = "message不是垃圾短信"
            if str(result) == "1":
                ret = "message是垃圾短信"
            return HttpResponse(ret)
    else: 
        form = AddForm()
    return render(request, 'index.html', {'form': form})

