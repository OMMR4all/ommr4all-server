from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0011_alter_bookdocumentsindex_file_mtime_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='pageindex',
            name='comments',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='pageindex',
            name='comments_count',
            field=models.IntegerField(default=0),
        ),
    ]
