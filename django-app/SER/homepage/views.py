from django.http import HttpResponse
from django.shortcuts import render
from SER.settings import BASE_DIR, MEDIA_ROOT
from homepage.models import Document
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import os
from django.template import loader
from .sermodel.model import predict


# Create your views here.


def index(request):
    return render(request, "homepage/index.html")


def upload(request):
    if request.method == "POST":
        try:
            myfile = request.FILES["audiofile"]
        except:
            return JsonResponse("Didn't recieve any file!", safe=False)
        fs = FileSystemStorage()
        filepath = fs.save(myfile.name, myfile)
        filename = os.path.basename(filepath)
        result = predict(filepath)
        context = {
            "result": result,
        }
        return render(request, "homepage/result.html", context)

    return render(request, "homepage/nu.html")


def nu(request):
    return render(request, "homepage/nu.html")


def contact(request):
    return render(request, "homepage/contact.html")
