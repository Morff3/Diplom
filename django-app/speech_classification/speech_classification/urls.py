from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static  # Импортируйте эту функцию

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('classifier.urls')),  # Подключаем URLs из приложения classifier
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Добавьте эту строку для обслуживания медиафайлов в режиме разработки
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
