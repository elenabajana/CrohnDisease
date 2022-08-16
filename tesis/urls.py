
from django.contrib import admin
from django.urls import path
from tesis import views
from django.conf import settings
from django.conf.urls.static import static
from tesis.views import FormularioAgregarfactorView, resultadog, resultadoarbol, editarfactor


urlpatterns = [
    path('', views.login, name="login"),
    path('inicio/', views.inicio, name="inicio"),
    path('informacion',views.informacion, name="informacion"),
    path('contactanos',views.contactanos, name="contactanos"),
    path('logout/', views.logout, name="logout"),
    path('agregarfactor', FormularioAgregarfactorView.agregarfactor, name="agregarfactor"),
    path('editarfactor/', editarfactor, name="editarfactor"),
    path('resultadog/', resultadog, name='resultadog'),
    path('resultadoarbol/', resultadoarbol, name='resultadoarbol'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
