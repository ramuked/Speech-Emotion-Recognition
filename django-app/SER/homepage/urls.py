import imp
from operator import index
from . import views
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

app_name = "homepage"
urlpatterns = [
    path("", views.index, name="index"),
    path("upload", views.upload, name="upload"),
    path("nu", views.nu, name="nu"),
    path("contact", views.contact, name="contact"),
]
