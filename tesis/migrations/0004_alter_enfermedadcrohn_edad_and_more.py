# Generated by Django 4.0.1 on 2022-02-22 08:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tesis', '0003_alter_enfermedadcrohn_edad_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='enfermedadcrohn',
            name='edad',
            field=models.CharField(blank=True, max_length=30, null=True),
        ),
        migrations.AlterField(
            model_name='enfermedadcrohn',
            name='sexo',
            field=models.CharField(blank=True, max_length=30, null=True),
        ),
    ]
