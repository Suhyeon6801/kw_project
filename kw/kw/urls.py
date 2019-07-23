"""kw URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import myapp.views
from myapp.views import GraphData

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',myapp.views.main, name="main"),
    path('index',myapp.views.index, name='index'),
    path('predict',myapp.views.predict, name='predict'),
    path('story',myapp.views.story, name='story'),
    path('api/graph/data/',GraphData.as_view()),#class base view를 쓰기 때문에 사용(as_view)
    path("api/graph/data/usa", myapp.views.getJsonUsa, name="json_usa"),
]
