from django.db import migrations, models


class Migration(migrations.Migration):
    """A class with base symbol type 'other' has no sub type, so the column may be empty."""

    dependencies = [
        ('database', '0016_alter_globalpermissions_options_symbolclass'),
    ]

    operations = [
        migrations.AlterField(
            model_name='symbolclass',
            name='base_sub_type',
            field=models.CharField(blank=True, default='', max_length=32),
        ),
    ]
