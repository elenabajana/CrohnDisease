from django.contrib import admin
from django.contrib.auth.models import Group
# Register your models here.
from tesis.models import enfermedadcrohn
from import_export import resources
from import_export.admin import ImportExportModelAdmin

class enfermedadcrohnResource(resources.ModelResource):
    class Meta:
        model = enfermedadcrohn

class enfermedadcrohnAdmin(ImportExportModelAdmin, admin.ModelAdmin):

    list_display = ("id","numero_eventos","imc","altura","sexo","edad","peso","tipo_tratamiento")
    search_fields = ("numero_eventos", "sexo")
    list_filter = ('sexo',)
    resource_class = enfermedadcrohnResource
    list_per_page = 10

admin.site.register(enfermedadcrohn,enfermedadcrohnAdmin)
admin.site.site_header= 'Sistema de Administración'
admin.site.site_title= 'Hola Nuevamente'
admin.site.index_title= 'Bienvenido Administrador'