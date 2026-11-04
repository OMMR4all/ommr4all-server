from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0017_symbolclass_base_sub_type_blank'),
    ]

    operations = [
        migrations.AddField(
            model_name='pageindex',
            name='content',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
