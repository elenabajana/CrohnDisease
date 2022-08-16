from django.db import models

# Create your models here.
class enfermedadcrohn(models.Model):
    numero_eventos = models.CharField(max_length=30, blank=True, null=True)
    imc = models.CharField(max_length=30, blank=True, null=True)
    altura = models.CharField(max_length=30, blank=True, null=True)
    pais = models.CharField(max_length=30, blank=True, null=True)
    sexo = models.CharField(max_length=30, blank=True, null=True)
    edad = models.CharField(max_length=30, blank=True, null=True)
    peso = models.CharField(max_length=30, blank=True, null=True)
    tipo_tratamiento = models.CharField(default="",max_length=30, blank=True, null=True)

    class Meta:
        verbose_name_plural = "CROHN"

    def __str__(self):
        return self.sexo