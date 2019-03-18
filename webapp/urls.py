from django.urls import path, re_path
import webapp.views as views

urlpatterns = \
    [
        path('', views.index),
        re_path(r'^(?P<path>.*)/$', views.index, name='index'),
    ]
