from django.shortcuts import render, redirect
from cnnClassifier.pipeline.stage_03_prediction import PredictionPipeline
from django.core.files.storage import FileSystemStorage
media = 'media'
import os 



from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User


# Create your views here.

def index(request):
    if request.method == 'POST' and request.FILES['upload']:
        f = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(f.name,f)
        file_url = fss.url(file)
        pred_img = os.path.join(media, file)
        pred = PredictionPipeline(pred_img)
        result = pred.predict()

        context = {
            'file_url':file_url,
            'result':result,
            'pred_img':pred_img

        }
        return render(request, 'index.html', context)
        
    return render(request, 'index.html')