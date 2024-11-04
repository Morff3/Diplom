from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_speech, name='classify_speech'),
    path('generation/', views.generation_view, name='generation'),
    path('generate_audio/', views.generate_audio, name='generate_audio'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
]
