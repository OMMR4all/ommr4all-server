from django.db import migrations, models


def insert_data(apps, schema_editor):
    BookStyle = apps.get_model('database', 'BookStyle')
    french14 = BookStyle(id='french14', name='F 14')
    french14.save()


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('database', '0004_to_pcgts_version_1'),
    ]

    operations = [
        migrations.CreateModel(
            name='BookStyle',
            fields=[
                ('id', models.CharField(max_length=255, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255, unique=True)),
            ],
        ),
        migrations.RunPython(insert_data),
    ]
